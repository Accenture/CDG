#!/usr/bin/env python3
"""
Evaluate subsampled trace experiments and generate scaling analysis.

Evaluates runs at different trace counts (8, 16, 32, 64, 128, 256, 512)
and produces:
1. Table: Accuracy by trace count and method
2. Figure: Accuracy vs trace count with methods as different colors/shapes

Usage:
    # Evaluate all subsampled runs and generate figure
    python scripts/eval/eval_subsample.py

    # Evaluate specific model/dataset
    python scripts/eval/eval_subsample.py --model qwen32b --dataset aime2025

    # Save figure to file
    python scripts/eval/eval_subsample.py --output figure.png

    # Custom results directory
    python scripts/eval/eval_subsample.py --results_dir /path/to/results
"""

import os
import sys
import re
import argparse
import fnmatch
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import load_run_results, evaluate_with_method, get_available_methods, EVAL_METHODS, DEFAULT_PARAMS
from eval_cache import EvalCache
from hyperparam_config import HyperparamConfig

# Try to import matplotlib (optional for figure generation)
try:
    import matplotlib.pyplot as plt
    import matplotlib.markers as mmarkers
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Figure generation disabled.")

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Default paths
DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE
SUBSET_SUBDIR = "subset_trace"

# Trace counts to analyze (including full 512)
TRACE_COUNTS = [8, 16, 32, 64, 128, 256, 512]


def parse_run_id(run_id: str) -> dict:
    """
    Parse run_id to extract model, dataset, budget, subset info.

    Formats:
        Full run: {model}_{dataset}_{budget}
            e.g., qwen32b_aime2025_512
        Subset run: {model}_{dataset}_{budget}_subset{size}v{version}
            e.g., qwen32b_aime2025_512_subset32v1

    Returns:
        dict with model, dataset, budget, subset_size, version, is_subset
    """
    # Known datasets
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
        # Remove subset suffix for base parsing
        base_run_id = run_id[:subset_match.start()]
    else:
        base_run_id = run_id

    # Parse base run_id
    for dataset in datasets:
        if f'_{dataset}_' in base_run_id:
            parts = base_run_id.split(f'_{dataset}_')
            if len(parts) == 2:
                result['model'] = parts[0]
                result['dataset'] = dataset
                result['budget'] = parts[1]
                break

    return result


def discover_runs(results_dir: str, subset_dir: str = None, patterns: str = None) -> dict:
    """
    Discover all runs including subsampled versions.

    Args:
        results_dir: Base results directory
        subset_dir: Subdirectory for subset runs
        patterns: Comma-separated patterns to filter runs (e.g., "deepseek8b*,qwq32b*")

    Returns:
        dict: {
            'full': [run_ids...],  # Full 512 trace runs
            'subset': [run_ids...],  # Subsampled runs
        }
    """
    runs = {'full': [], 'subset': []}
    base = Path(results_dir)

    if not base.exists():
        return runs

    # Parse patterns into a list
    pattern_list = []
    if patterns:
        pattern_list = [p.strip() for p in patterns.split(',')]

    # Find full runs in base directory
    for item in base.iterdir():
        if item.is_dir() and item.name not in ['subset_trace', '__pycache__', '.eval_cache']:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                # Check patterns filter
                if pattern_list:
                    if not any(fnmatch.fnmatch(item.name, p) for p in pattern_list):
                        continue
                runs['full'].append(item.name)

    # Find subset runs
    subset_base = base / (subset_dir or SUBSET_SUBDIR)
    if subset_base.exists():
        for item in subset_base.iterdir():
            if item.is_dir():
                pkl_files = list(item.glob("*.pkl"))
                if pkl_files:
                    # Check patterns filter
                    if pattern_list:
                        if not any(fnmatch.fnmatch(item.name, p) for p in pattern_list):
                            continue
                    runs['subset'].append(item.name)

    return runs


def evaluate_single_run(args: tuple) -> tuple:
    """Worker function for parallel run evaluation."""
    # Unpack args - hyperparam_config is optional
    if len(args) == 5:
        run_id, run_path, method, params, hyperparam_config = args
    else:
        run_id, run_path, method, params = args
        hyperparam_config = None

    # IMPORTANT: use_threads=False to avoid nested ThreadPoolExecutor
    results_by_qid = load_run_results(run_path, use_threads=False)

    if not results_by_qid:
        return run_id, method, None

    # Get model-specific params if hyperparam_config provided
    info = parse_run_id(run_id)
    eval_params = params.copy()

    if hyperparam_config is not None and info['model']:
        model_params = hyperparam_config.get_model_params(info['model'], use_defaults=False)
        if model_params:
            eval_params.update(model_params)

    result = evaluate_with_method(results_by_qid, method, eval_params)

    return run_id, method, {
        'info': info,
        'result': result,
    }


def evaluate_all_runs(results_dir: str, methods: list = None,
                      model_filter: str = None, dataset_filter: str = None,
                      patterns: str = None, num_workers: int = None,
                      params: dict = None, cache: EvalCache = None,
                      hyperparam_config: HyperparamConfig = None) -> dict:
    """
    Evaluate all runs with specified methods.

    Args:
        results_dir: Base results directory
        methods: List of methods to evaluate (None = all)
        model_filter: Filter by model name
        dataset_filter: Filter by dataset name
        patterns: Comma-separated patterns to filter runs
        num_workers: Max parallel workers (capped at 8 to avoid thread issues)
        params: Method parameters dict
        cache: EvalCache for caching results
        hyperparam_config: HyperparamConfig for model-specific params (optional)

    Returns:
        dict: {run_id: {method: result, ...}, ...}
    """
    if methods is None:
        methods = get_available_methods()

    if params is None:
        params = DEFAULT_PARAMS.copy()

    runs = discover_runs(results_dir, patterns=patterns)
    all_run_ids = runs['full'] + runs['subset']

    # Apply filters
    filtered_runs = []
    for run_id in all_run_ids:
        info = parse_run_id(run_id)
        if model_filter and info['model'] != model_filter:
            continue
        if dataset_filter and info['dataset'] != dataset_filter:
            continue
        filtered_runs.append(run_id)

    if not filtered_runs:
        print(f"No runs found matching filters (model={model_filter}, dataset={dataset_filter})")
        return {}

    total_tasks = len(filtered_runs) * len(methods)
    print(f"Found {len(filtered_runs)} runs to evaluate with {len(methods)} methods ({total_tasks} total tasks)")

    all_results = defaultdict(dict)

    # Check cache for existing results
    cached_count = 0
    tasks_to_run = []

    for run_id in filtered_runs:
        info = parse_run_id(run_id)
        if info['is_subset']:
            run_path = os.path.join(results_dir, SUBSET_SUBDIR, run_id)
        else:
            run_path = os.path.join(results_dir, run_id)

        # Get model-specific params for cache key
        run_params = params.copy()
        if hyperparam_config is not None and info['model']:
            model_params = hyperparam_config.get_model_params(info['model'], use_defaults=False)
            if model_params:
                run_params.update(model_params)

        for method in methods:
            if cache is not None:
                cached_result = cache.load_result(run_id, method, run_params)
                if cached_result is not None:
                    all_results[run_id][method] = {'info': info, 'result': cached_result}
                    cached_count += 1
                    continue
            tasks_to_run.append((run_id, run_path, method, params, hyperparam_config))

    if cached_count > 0:
        print(f"Loaded {cached_count}/{total_tasks} results from cache")

    if not tasks_to_run:
        print("All results cached, nothing to compute")
        return dict(all_results)

    print(f"Computing {len(tasks_to_run)} remaining tasks...")

    # Cap workers to avoid thread exhaustion
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(tasks_to_run), 8)

    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        from concurrent.futures import as_completed

        future_to_task = {
            executor.submit(evaluate_single_run, task): task
            for task in tasks_to_run
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            run_id = task[0]
            method = task[2]

            try:
                run_id, method, data = future.result()
                if data is not None:
                    all_results[run_id][method] = data
                    # Save to cache with model-specific params
                    if cache is not None:
                        info = data['info']
                        cache_params = params.copy()
                        if hyperparam_config is not None and info['model']:
                            model_params = hyperparam_config.get_model_params(info['model'], use_defaults=False)
                            if model_params:
                                cache_params.update(model_params)
                        cache.save_result(run_id, method, cache_params, data['result'])
            except Exception as e:
                print(f"  ERROR processing {run_id}/{method}: {e}")

            completed += 1
            if completed % 20 == 0 or completed == len(tasks_to_run):
                print(f"  Progress: {completed}/{len(tasks_to_run)} tasks")

    return dict(all_results)


def aggregate_by_trace_count(all_results: dict) -> dict:
    """
    Aggregate results by (model, dataset, trace_count, method).

    For subsampled runs, averages across versions.

    Returns:
        dict: {(model, dataset): {trace_count: {method: {'mean': x, 'std': x, 'values': [...]}}}}
    """
    # Group by (model, dataset, trace_count, method)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_id, methods_results in all_results.items():
        for method, data in methods_results.items():
            info = data['info']
            model = info['model']
            dataset = info['dataset']
            result = data['result']
            accuracy = result['accuracy']

            if info['is_subset']:
                trace_count = info['subset_size']
            else:
                # Full run - use budget as trace count
                try:
                    trace_count = int(info['budget'])
                except:
                    trace_count = 512

            grouped[(model, dataset)][trace_count][method].append(accuracy)

    # Compute mean and std
    aggregated = {}
    for (model, dataset), trace_data in grouped.items():
        aggregated[(model, dataset)] = {}
        for trace_count, method_data in trace_data.items():
            aggregated[(model, dataset)][trace_count] = {}
            for method, values in method_data.items():
                if HAS_NUMPY:
                    mean_acc = np.mean(values)
                    std_acc = np.std(values) if len(values) > 1 else 0
                else:
                    mean_acc = sum(values) / len(values)
                    std_acc = 0
                aggregated[(model, dataset)][trace_count][method] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'values': values,
                    'n': len(values),
                }

    return aggregated


def print_table(aggregated: dict, methods: list):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("SUBSAMPLE EVALUATION RESULTS")
    print("=" * 100)

    for (model, dataset), trace_data in sorted(aggregated.items()):
        print(f"\n{model} - {dataset}")
        print("-" * 100)

        # Header
        header = f"{'Traces':<10}"
        for method in methods:
            method_name = EVAL_METHODS[method]['name']
            header += f"{method_name:<20}"
        print(header)
        print("-" * 100)

        # Rows
        for trace_count in sorted(trace_data.keys()):
            row = f"{trace_count:<10}"
            for method in methods:
                if method in trace_data[trace_count]:
                    data = trace_data[trace_count][method]
                    if data['n'] > 1:
                        cell = f"{100*data['mean']:.1f}% (+/-{100*data['std']:.1f})"
                    else:
                        cell = f"{100*data['mean']:.1f}%"
                else:
                    cell = "-"
                row += f"{cell:<20}"
            print(row)

        print("-" * 100)

    print("=" * 100)


def create_figure(aggregated: dict, methods: list, output_path: str = None,
                  title: str = None, show: bool = True):
    """
    Create accuracy vs trace count figure.

    X-axis: Trace count (8 -> 512)
    Y-axis: Accuracy
    Color/Shape: Methods
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create figure: matplotlib not available")
        return

    # Define colors and markers for methods
    method_styles = {
        'majority': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'deepconf_mean': {'color': 'orange', 'marker': 's', 'linestyle': '--'},
        'deepconf_tail': {'color': 'green', 'marker': '^', 'linestyle': '-.'},
        'deepconf_bottom': {'color': 'purple', 'marker': 'v', 'linestyle': ':'},
        'gradient': {'color': 'red', 'marker': 'D', 'linestyle': '-'},
        'cdg': {'color': 'black', 'marker': '*', 'linestyle': '--'},
    }

    # Create figure with subplots for each (model, dataset) combination
    n_plots = len(aggregated)
    if n_plots == 0:
        print("No data to plot")
        return

    # Determine grid size
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, ((model, dataset), trace_data) in enumerate(sorted(aggregated.items())):
        ax = axes[idx]

        for method in methods:
            style = method_styles.get(method, {'color': 'gray', 'marker': 'x', 'linestyle': ':'})
            method_name = EVAL_METHODS[method]['name']

            # Collect data points
            x_vals = []
            y_vals = []
            y_errs = []

            for trace_count in sorted(trace_data.keys()):
                if method in trace_data[trace_count]:
                    data = trace_data[trace_count][method]
                    x_vals.append(trace_count)
                    y_vals.append(100 * data['mean'])
                    y_errs.append(100 * data['std'])

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs if any(y_errs) else None,
                           label=method_name,
                           color=style['color'],
                           marker=style['marker'],
                           linestyle=style['linestyle'],
                           capsize=3,
                           markersize=8,
                           linewidth=2)

        ax.set_xlabel('Number of Traces', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{model} - {dataset}', fontsize=14)
        ax.set_xscale('log', base=2)
        ax.set_xticks(TRACE_COUNTS)
        ax.set_xticklabels([str(t) for t in TRACE_COUNTS])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        # Set y-axis limits with some padding
        ax.set_ylim(0, 100)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Accuracy vs Number of Traces', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Evaluate subsampled trace experiments')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Base directory for results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model name')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset name')
    parser.add_argument('--patterns', type=str, default=None,
                        help='Filter runs by multiple comma-separated patterns (e.g., "deepseek8b*,qwq32b*")')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods (default: all)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save figure to file (e.g., figure.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure (useful for headless servers)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom figure title')
    parser.add_argument('--list', action='store_true',
                        help='List available runs without evaluating')

    # Method parameters
    parser.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'],
                        help=f'CDG count dampening exponent (default: {DEFAULT_PARAMS["alpha"]})')
    parser.add_argument('--beta', type=float, default=DEFAULT_PARAMS['beta'],
                        help=f'Gradient weight (default: {DEFAULT_PARAMS["beta"]})')
    parser.add_argument('--position_pct', type=int, default=DEFAULT_PARAMS['position_pct'],
                        help=f'Percentile for gradient computation (default: {DEFAULT_PARAMS["position_pct"]})')
    parser.add_argument('--tail_window', type=int, default=DEFAULT_PARAMS['tail_window'],
                        help=f'Window size for tail confidence (default: {DEFAULT_PARAMS["tail_window"]})')
    parser.add_argument('--bottom_window', type=int, default=DEFAULT_PARAMS['bottom_window'],
                        help=f'Window size for bottom window (default: {DEFAULT_PARAMS["bottom_window"]})')
    parser.add_argument('--bottom_pct', type=float, default=DEFAULT_PARAMS['bottom_pct'],
                        help=f'Bottom percentile for bottom window (default: {DEFAULT_PARAMS["bottom_pct"]})')

    # Cache options
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable result caching')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cache before running')

    # Hyperparameter config options
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Use model-specific tuned hyperparameters from config file')

    # Debug mode
    parser.add_argument('--fast', action='store_true',
                        help='Fast debug mode: 2 methods, 1 model (results will be inaccurate)')

    args = parser.parse_args()

    # Apply fast mode overrides
    if args.fast:
        print("=" * 60)
        print("FAST DEBUG MODE - Results will be inaccurate!")
        print("=" * 60)
        args.methods = "majority,cdg"
        args.model = args.model or "qwen32b"

    # Initialize hyperparam config
    hyperparam_config = HyperparamConfig(args.results_dir)

    # Initialize cache
    cache = EvalCache(args.results_dir, enabled=not args.no_cache)

    if args.clear_cache:
        print("Clearing cache...")
        cache.clear_results()
        print("Cache cleared.")

    # Build params dict
    params = {
        'alpha': args.alpha,
        'beta': args.beta,
        'position_pct': args.position_pct,
        'tail_window': args.tail_window,
        'bottom_window': args.bottom_window,
        'bottom_pct': args.bottom_pct,
    }

    # Parse methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    else:
        methods = get_available_methods()

    # List mode
    if args.list:
        runs = discover_runs(args.results_dir, patterns=args.patterns)
        print(f"Full runs ({len(runs['full'])}):")
        for run in sorted(runs['full']):
            info = parse_run_id(run)
            print(f"  {run:<50} ({info['model']}, {info['dataset']})")
        print(f"\nSubset runs ({len(runs['subset'])}):")
        for run in sorted(runs['subset'])[:20]:
            info = parse_run_id(run)
            print(f"  {run:<50} (size={info['subset_size']}, v={info['version']})")
        if len(runs['subset']) > 20:
            print(f"  ... and {len(runs['subset']) - 20} more")
        print(f"\nAvailable methods: {', '.join(get_available_methods())}")
        return

    # Evaluate all runs
    print("Evaluating runs...")
    print(f"Methods: {', '.join(methods)}")
    if args.use_tuned_params:
        print(f"Parameters: using model-specific tuned hyperparameters")
        tuned_models = hyperparam_config.get_all_models()
        if tuned_models:
            print(f"  Tuned models: {', '.join(tuned_models)}")
    else:
        print(f"Parameters: alpha={params['alpha']}, beta={params['beta']}")
    if not args.no_cache:
        print(f"Cache: enabled (use --no-cache to disable)")

    hp_config = hyperparam_config if args.use_tuned_params else None
    all_results = evaluate_all_runs(
        args.results_dir,
        methods=methods,
        model_filter=args.model,
        dataset_filter=args.dataset,
        patterns=args.patterns,
        params=params,
        cache=cache,
        hyperparam_config=hp_config
    )

    if not all_results:
        print("No results found!")
        return

    # Aggregate by trace count
    aggregated = aggregate_by_trace_count(all_results)

    # Print table
    print_table(aggregated, methods)

    # Create figure
    if HAS_MATPLOTLIB:
        create_figure(
            aggregated,
            methods,
            output_path=args.output,
            title=args.title,
            show=not args.no_show
        )
    else:
        print("\nTo generate figures, install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
