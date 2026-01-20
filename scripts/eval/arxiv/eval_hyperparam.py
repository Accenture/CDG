#!/usr/bin/env python3
"""
Hyperparameter tuning for CDG method (alpha, beta, position_pct).

Sweeps over parameter combinations and generates accuracy plots.
Results are saved to JSON for later analysis.

Usage:
    # Sweep beta with fixed alpha values
    python scripts/eval/eval_hyperparam.py --sweep beta

    # Sweep position_pct
    python scripts/eval/eval_hyperparam.py --sweep position_pct

    # Use 256-trace subset runs
    python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256

    # Filter by model/dataset
    python scripts/eval/eval_hyperparam.py --sweep beta --model qwen32b --dataset aime2025

    # Save results and figure
    python scripts/eval/eval_hyperparam.py --sweep beta --output results.json --figure figure.png
"""

import os
import sys
import json
import argparse
import fnmatch
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import load_run_results, evaluate_cdg, DEFAULT_PARAMS
from eval_cache import EvalCache
from hyperparam_config import HyperparamConfig, find_best_params_across_datasets

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Figure generation disabled.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE
SUBSET_SUBDIR = "subset_trace"

# Parameter sweep ranges
ALPHA_VALUES = [0.5, 1.0]
BETA_VALUES = list(range(10, 55, 5))  # [10, 15, 20, 25, 30, 35, 40, 45, 50]
POSITION_PCT_VALUES = [10, 20, 30]


def parse_run_id(run_id: str) -> dict:
    """Parse run_id to extract model, dataset, budget, subset info."""
    import re
    datasets = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025']

    result = {
        'model': None,
        'dataset': None,
        'budget': None,
        'subset_size': None,
        'version': None,
        'is_subset': False,
        'run_id': run_id,
    }

    # Check for subset pattern
    subset_match = re.search(r'_subset(\d+)v(\d+)$', run_id)
    if subset_match:
        result['subset_size'] = int(subset_match.group(1))
        result['version'] = int(subset_match.group(2))
        result['is_subset'] = True
        base_run_id = run_id[:subset_match.start()]
    else:
        base_run_id = run_id

    for dataset in datasets:
        if f'_{dataset}_' in base_run_id:
            parts = base_run_id.split(f'_{dataset}_')
            if len(parts) == 2:
                result['model'] = parts[0]
                result['dataset'] = dataset
                result['budget'] = parts[1]
                break

    return result


def discover_runs(results_dir: str, trace_count: int = None,
                  model_filter: str = None, dataset_filter: str = None,
                  patterns: str = None) -> list:
    """
    Discover runs matching criteria.

    Args:
        results_dir: Base results directory
        trace_count: If specified, find subset runs with this trace count (or full 512 runs)
        model_filter: Filter by model name
        dataset_filter: Filter by dataset name
        patterns: Comma-separated patterns to filter runs (e.g., "deepseek8b*,qwq32b*")
    """
    runs = []
    base = Path(results_dir)

    if not base.exists():
        return runs

    # Parse patterns into a list
    pattern_list = []
    if patterns:
        pattern_list = [p.strip() for p in patterns.split(',')]

    # Check full runs (512 traces)
    if trace_count is None or trace_count == 512:
        for item in base.iterdir():
            if item.is_dir() and item.name not in ['subset_trace', '__pycache__', '.eval_cache']:
                pkl_files = list(item.glob("*.pkl"))
                if pkl_files:
                    info = parse_run_id(item.name)
                    # Skip runs that couldn't be properly parsed
                    if info['model'] is None or info['dataset'] is None:
                        continue
                    if model_filter and info['model'] != model_filter:
                        continue
                    if dataset_filter and info['dataset'] != dataset_filter:
                        continue
                    # Check patterns filter
                    if pattern_list:
                        if not any(fnmatch.fnmatch(item.name, p) for p in pattern_list):
                            continue
                    runs.append({
                        'run_id': item.name,
                        'path': str(item),
                        'info': info,
                        'trace_count': 512,
                    })

    # Check subset runs
    if trace_count is not None and trace_count != 512:
        subset_base = base / SUBSET_SUBDIR
        if subset_base.exists():
            for item in subset_base.iterdir():
                if item.is_dir():
                    pkl_files = list(item.glob("*.pkl"))
                    if pkl_files:
                        info = parse_run_id(item.name)
                        # Skip runs that couldn't be properly parsed
                        if info['model'] is None or info['dataset'] is None:
                            continue
                        if info['subset_size'] != trace_count:
                            continue
                        if model_filter and info['model'] != model_filter:
                            continue
                        if dataset_filter and info['dataset'] != dataset_filter:
                            continue
                        # Check patterns filter
                        if pattern_list:
                            if not any(fnmatch.fnmatch(item.name, p) for p in pattern_list):
                                continue
                        runs.append({
                            'run_id': item.name,
                            'path': str(item),
                            'info': info,
                            'trace_count': trace_count,
                        })

    return runs


def evaluate_with_params(run_info: dict, alpha: float, beta: float,
                         position_pct: int = 20, cache: EvalCache = None) -> dict:
    """Evaluate a single run with specific parameters."""
    params = {
        'alpha': alpha,
        'beta': beta,
        'position_pct': position_pct,
        'tail_window': DEFAULT_PARAMS['tail_window'],
        'bottom_window': DEFAULT_PARAMS['bottom_window'],
        'bottom_pct': DEFAULT_PARAMS['bottom_pct'],
    }

    # Check cache first
    run_id = run_info['run_id']
    if cache is not None:
        cached = cache.load_result(run_id, 'cdg', params)
        if cached is not None:
            return {
                'run_id': run_id,
                'model': run_info['info']['model'],
                'dataset': run_info['info']['dataset'],
                'trace_count': run_info['trace_count'],
                'alpha': alpha,
                'beta': beta,
                'position_pct': position_pct,
                'correct': cached['correct'],
                'total': cached['total'],
                'accuracy': cached['accuracy'],
                'cached': True,
            }

    # Load results (sequential to avoid thread issues if called in parallel)
    results_by_qid = load_run_results(run_info['path'], use_threads=False)

    if not results_by_qid:
        return None

    result = evaluate_cdg(results_by_qid, params)

    # Save to cache
    if cache is not None:
        cache.save_result(run_id, 'cdg', params, result)

    return {
        'run_id': run_id,
        'model': run_info['info']['model'],
        'dataset': run_info['info']['dataset'],
        'trace_count': run_info['trace_count'],
        'alpha': alpha,
        'beta': beta,
        'position_pct': position_pct,
        'correct': result['correct'],
        'total': result['total'],
        'accuracy': result['accuracy'],
        'cached': False,
    }


def check_cache_status(runs: list, alpha_values: list, beta_values: list,
                       position_pct: int, cache: EvalCache) -> tuple:
    """
    Check how many results are already cached.

    Returns (cached_count, total_count)
    """
    if cache is None:
        return 0, len(runs) * len(alpha_values) * len(beta_values)

    cached_count = 0
    for run in runs:
        for alpha in alpha_values:
            for beta in beta_values:
                params = {
                    'alpha': alpha,
                    'beta': beta,
                    'position_pct': position_pct,
                    'tail_window': DEFAULT_PARAMS['tail_window'],
                    'bottom_window': DEFAULT_PARAMS['bottom_window'],
                    'bottom_pct': DEFAULT_PARAMS['bottom_pct'],
                }
                cached = cache.load_result(run['run_id'], 'cdg', params)
                if cached is not None:
                    cached_count += 1

    total = len(runs) * len(alpha_values) * len(beta_values)
    return cached_count, total


def sweep_beta(runs: list, alpha_values: list = None, beta_values: list = None,
               position_pct: int = 20, num_workers: int = None,
               cache: EvalCache = None) -> list:
    """
    Sweep over beta values for given alpha values.

    Returns list of result dicts.
    """
    if alpha_values is None:
        alpha_values = ALPHA_VALUES
    if beta_values is None:
        beta_values = BETA_VALUES

    # Check cache status first
    total_tasks = len(runs) * len(alpha_values) * len(beta_values)
    if cache is not None:
        print(f"Checking cache status...")
        cached_count, total = check_cache_status(runs, alpha_values, beta_values, position_pct, cache)
        need_compute = total - cached_count
        print(f"Cache: {cached_count}/{total} found, need to compute {need_compute}")
        if cached_count == total:
            print("All results cached!")
    else:
        print(f"Cache disabled, will compute all {total_tasks} evaluations")

    # Build task list
    tasks = []
    for run in runs:
        for alpha in alpha_values:
            for beta in beta_values:
                tasks.append((run, alpha, beta, position_pct))

    print(f"Running {len(tasks)} evaluations ({len(runs)} runs x {len(alpha_values)} alphas x {len(beta_values)} betas)...")

    # Determine number of workers (conservative for SMB drives)
    if num_workers is None:
        import multiprocessing as mp
        num_workers = min(4, mp.cpu_count(), len(tasks))

    results = []
    cached_count = 0

    def process_task(task):
        run, alpha, beta, pct = task
        return evaluate_with_params(run, alpha, beta, pct, cache=cache)

    if num_workers > 1 and len(tasks) > 1:
        # Parallel processing
        print(f"Using {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(executor.map(process_task, tasks))
            for i, result in enumerate(futures):
                if result:
                    if result.get('cached'):
                        cached_count += 1
                    results.append(result)
                if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                    print(f"  Progress: {i+1}/{len(tasks)} (cached: {cached_count})")
    else:
        # Sequential processing
        for i, task in enumerate(tasks):
            result = process_task(task)
            if result:
                if result.get('cached'):
                    cached_count += 1
                results.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                print(f"  Progress: {i+1}/{len(tasks)} (cached: {cached_count})")

    if cached_count > 0:
        print(f"Loaded {cached_count}/{len(tasks)} results from cache")

    return results


def sweep_position_pct(runs: list, position_pct_values: list = None,
                       alpha: float = 0.5, beta: float = 10,
                       num_workers: int = None, cache: EvalCache = None) -> list:
    """Sweep over position_pct values."""
    if position_pct_values is None:
        position_pct_values = POSITION_PCT_VALUES

    tasks = []
    for run in runs:
        for pct in position_pct_values:
            tasks.append((run, alpha, beta, pct))

    print(f"Running {len(tasks)} evaluations...")

    # Determine number of workers (conservative for SMB drives)
    if num_workers is None:
        import multiprocessing as mp
        num_workers = min(4, mp.cpu_count(), len(tasks))

    results = []
    cached_count = 0

    def process_task(task):
        run, a, b, pct = task
        return evaluate_with_params(run, a, b, pct, cache=cache)

    if num_workers > 1 and len(tasks) > 1:
        print(f"Using {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(executor.map(process_task, tasks))
            for i, result in enumerate(futures):
                if result:
                    if result.get('cached'):
                        cached_count += 1
                    results.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    print(f"  Progress: {i+1}/{len(tasks)} (cached: {cached_count})")
    else:
        for i, task in enumerate(tasks):
            result = process_task(task)
            if result:
                if result.get('cached'):
                    cached_count += 1
                results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  Progress: {i+1}/{len(tasks)} (cached: {cached_count})")

    return results


def aggregate_results(results: list, group_by: str = 'beta') -> dict:
    """
    Aggregate results by parameter.

    Returns: {(model, dataset, alpha): {param_value: {'mean': x, 'std': x, 'values': [...]}}}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for r in results:
        if group_by == 'beta':
            key = (r['model'], r['dataset'], r['alpha'])
            param_val = r['beta']
        elif group_by == 'position_pct':
            key = (r['model'], r['dataset'], r['alpha'], r['beta'])
            param_val = r['position_pct']
        else:
            key = (r['model'], r['dataset'])
            param_val = r.get(group_by, 0)

        grouped[key][param_val].append(r['accuracy'])

    # Compute mean/std
    aggregated = {}
    for key, param_data in grouped.items():
        aggregated[key] = {}
        for param_val, accuracies in param_data.items():
            if HAS_NUMPY:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
            else:
                mean_acc = sum(accuracies) / len(accuracies)
                std_acc = 0
            aggregated[key][param_val] = {
                'mean': mean_acc,
                'std': std_acc,
                'values': accuracies,
                'n': len(accuracies),
            }

    return aggregated


def print_results_table(aggregated: dict, param_name: str = 'beta'):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print(f"HYPERPARAMETER SWEEP: {param_name.upper()}")
    print("=" * 100)

    def sort_key(item):
        """Sort key that handles None values by converting to empty string."""
        key = item[0]
        if isinstance(key, tuple):
            return tuple('' if x is None else x for x in key)
        return '' if key is None else key

    for key, param_data in sorted(aggregated.items(), key=sort_key):
        if len(key) == 3:
            model, dataset, alpha = key
            print(f"\n{model} - {dataset} (alpha={alpha})")
        else:
            print(f"\n{key}")

        print("-" * 80)

        # Header
        param_values = sorted(param_data.keys())
        header = f"{'Param':<10}"
        for pv in param_values:
            header += f"{pv:<12}"
        print(header)

        # Values
        row = f"{'Acc':<10}"
        for pv in param_values:
            data = param_data[pv]
            if data['n'] > 1:
                row += f"{100*data['mean']:.1f}+/-{100*data['std']:.1f}  "
            else:
                row += f"{100*data['mean']:.1f}%       "
        print(row)

        # Find best
        best_param = max(param_data.items(), key=lambda x: x[1]['mean'])
        print(f"Best: {param_name}={best_param[0]} ({100*best_param[1]['mean']:.1f}%)")

    print("=" * 100)


def create_beta_sweep_figure(aggregated: dict, output_path: str = None,
                             title: str = None, show: bool = True):
    """
    Create beta sweep figure.

    X: beta values
    Y: Accuracy
    Color: alpha values
    Subplots: model-dataset combinations
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create figure: matplotlib not available")
        return

    # Group by (model, dataset) for subplots
    subplot_data = defaultdict(dict)
    for (model, dataset, alpha), param_data in aggregated.items():
        subplot_data[(model, dataset)][alpha] = param_data

    n_plots = len(subplot_data)
    if n_plots == 0:
        print("No data to plot")
        return

    # Grid layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Colors for alpha values
    alpha_colors = {0.5: 'blue', 1.0: 'red', 0.25: 'green', 0.75: 'orange'}

    def safe_sort_key(item):
        """Sort key that handles None values."""
        key = item[0]
        if isinstance(key, tuple):
            return tuple('' if x is None else x for x in key)
        return '' if key is None else key

    for idx, ((model, dataset), alpha_data) in enumerate(sorted(subplot_data.items(), key=safe_sort_key)):
        ax = axes[idx]

        for alpha, param_data in sorted(alpha_data.items()):
            color = alpha_colors.get(alpha, 'gray')
            beta_values = sorted(param_data.keys())
            accuracies = [100 * param_data[b]['mean'] for b in beta_values]
            stds = [100 * param_data[b]['std'] for b in beta_values]

            ax.errorbar(beta_values, accuracies, yerr=stds if any(stds) else None,
                       label=f'α={alpha}', color=color, marker='o',
                       capsize=3, linewidth=2, markersize=6)

        ax.set_xlabel('β (gradient weight)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{model} - {dataset}', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis
        ax.set_ylim(0, 100)

    # Hide unused
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('CDG Hyperparameter Sweep: β vs Accuracy', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


def save_results(results: list, output_path: str):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results),
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for CDG method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sweep beta                           # Sweep beta with default alphas
  %(prog)s --sweep beta --trace_count 256         # Use 256-trace subset runs
  %(prog)s --sweep position_pct                   # Sweep position_pct
  %(prog)s --sweep beta --output results.json     # Save results
  %(prog)s --sweep beta --figure plot.png         # Save figure
        """
    )

    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Base directory for results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--sweep', type=str, required=False,
                        choices=['beta', 'position_pct'],
                        help='Parameter to sweep (required unless --show-config)')
    parser.add_argument('--trace_count', type=int, default=512,
                        help='Trace count to use (default: 512, use 256 for faster tuning)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4, increase if using local SSD)')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model name')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset name')
    parser.add_argument('--patterns', type=str, default=None,
                        help='Filter runs by multiple comma-separated patterns (e.g., "deepseek8b*,qwq32b*")')

    # Parameter ranges
    parser.add_argument('--alphas', type=str, default='0.5,1.0',
                        help='Comma-separated alpha values (default: 0.5,1.0)')
    parser.add_argument('--betas', type=str, default=None,
                        help='Comma-separated beta values (e.g., 1,5,10,15,20,25,30). Overrides beta_min/max/step if provided.')
    parser.add_argument('--beta_min', type=int, default=10,
                        help='Min beta value (default: 10)')
    parser.add_argument('--beta_max', type=int, default=50,
                        help='Max beta value (default: 50)')
    parser.add_argument('--beta_step', type=int, default=5,
                        help='Beta step size (default: 5)')
    parser.add_argument('--position_pct', type=int, default=20,
                        help='Position percentile for gradient (default: 20)')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--figure', '-f', type=str, default=None,
                        help='Save figure to file')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom figure title')

    # Cache options
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable result caching')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cache before running')

    # Hyperparameter config options
    parser.add_argument('--save-config', action='store_true',
                        help='Save best hyperparameters per model to config file')
    parser.add_argument('--show-config', action='store_true',
                        help='Show current hyperparameter config and exit')

    # Debug mode
    parser.add_argument('--fast', action='store_true',
                        help='Fast debug mode: 1 alpha, 2 betas, 1 model (results will be inaccurate)')

    args = parser.parse_args()

    # Apply fast mode overrides
    if args.fast:
        print("=" * 60)
        print("FAST DEBUG MODE - Results will be inaccurate!")
        print("=" * 60)
        args.alphas = "0.5"
        args.beta_min = 10
        args.beta_max = 20
        args.beta_step = 10  # Only 10, 20
        args.model = args.model or "qwen32b"  # Default to first model found
        args.trace_count = 512  # Use full runs (faster to find than subsets)

    # Initialize hyperparam config
    hyperparam_config = HyperparamConfig(args.results_dir)

    # Handle --show-config
    if args.show_config:
        hyperparam_config.print_summary()
        return

    # Validate --sweep is required for actual sweep operations
    if not args.sweep:
        parser.error("--sweep is required (choose from: beta, position_pct)")

    # Initialize cache
    cache = EvalCache(args.results_dir, enabled=not args.no_cache)

    if args.clear_cache:
        print("Clearing cache...")
        cache.clear_results()
        print("Cache cleared.")

    # Parse alpha values
    alpha_values = [float(a.strip()) for a in args.alphas.split(',')]

    # Build beta values - use --betas if provided, otherwise use range
    if args.betas:
        beta_values = [int(b.strip()) for b in args.betas.split(',')]
    else:
        beta_values = list(range(args.beta_min, args.beta_max + 1, args.beta_step))

    print(f"Hyperparameter Sweep: {args.sweep}")
    print(f"Trace count: {args.trace_count}")
    print(f"Alpha values: {alpha_values}")
    if args.sweep == 'beta':
        print(f"Beta range: {beta_values}")
    print(f"Position percentile: {args.position_pct}")
    print()

    # Discover runs
    runs = discover_runs(
        args.results_dir,
        trace_count=args.trace_count,
        model_filter=args.model,
        dataset_filter=args.dataset,
        patterns=args.patterns
    )

    if not runs:
        print(f"No runs found with trace_count={args.trace_count}")
        print("Available subset sizes in subset_trace/ directory:")
        subset_dir = Path(args.results_dir) / SUBSET_SUBDIR
        if subset_dir.exists():
            sizes = set()
            for item in subset_dir.iterdir():
                info = parse_run_id(item.name)
                if info['subset_size']:
                    sizes.add(info['subset_size'])
            print(f"  {sorted(sizes)}")
        return

    print(f"Found {len(runs)} runs")
    for r in runs[:5]:
        print(f"  {r['run_id']}")
    if len(runs) > 5:
        print(f"  ... and {len(runs) - 5} more")
    print()

    # Run sweep
    if not args.no_cache:
        print(f"Cache: enabled (use --no-cache to disable)")
    print()

    if args.sweep == 'beta':
        results = sweep_beta(
            runs,
            alpha_values=alpha_values,
            beta_values=beta_values,
            position_pct=args.position_pct,
            num_workers=args.workers,
            cache=cache
        )
        aggregated = aggregate_results(results, group_by='beta')
        print_results_table(aggregated, param_name='beta')

        if HAS_MATPLOTLIB and (args.figure or not args.no_show):
            create_beta_sweep_figure(
                aggregated,
                output_path=args.figure,
                title=args.title,
                show=not args.no_show
            )

    elif args.sweep == 'position_pct':
        results = sweep_position_pct(
            runs,
            position_pct_values=POSITION_PCT_VALUES,
            alpha=alpha_values[0],
            beta=beta_values[len(beta_values)//2],  # middle beta
            num_workers=args.workers,
            cache=cache
        )
        aggregated = aggregate_results(results, group_by='position_pct')
        print_results_table(aggregated, param_name='position_pct')

    # Save results to JSON
    if args.output:
        save_results(results, args.output)

    # Save best hyperparameters per model to config
    if args.save_config and results:
        print("\n" + "=" * 70)
        print("SAVING BEST HYPERPARAMETERS PER MODEL")
        print("=" * 70)

        # Find unique models in results
        models = list(set(r.get('model') for r in results if r.get('model')))

        for model in sorted(models):
            best = find_best_params_across_datasets(results, model)
            if best:
                hyperparam_config.set_model_params(
                    model=model,
                    alpha=best['alpha'],
                    beta=best['beta'],
                    position_pct=best['position_pct'],
                    accuracy=best['accuracy'],
                    tuning_details={
                        'correct': best['correct'],
                        'total': best['total'],
                        'datasets': best.get('datasets', []),
                        'sweep_type': args.sweep,
                        'trace_count': args.trace_count,
                    }
                )
                print(f"  {model}: alpha={best['alpha']}, beta={best['beta']}, "
                      f"position_pct={best['position_pct']} "
                      f"(accuracy={100*best['accuracy']:.1f}%)")

        hyperparam_config.save()
        hyperparam_config.print_summary()


if __name__ == "__main__":
    main()
