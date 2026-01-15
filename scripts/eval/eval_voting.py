"""
Unified voting evaluation script for LLM trace results.

Evaluates inference results using multiple voting methods and computes accuracy
against ground truth answers. Supports single method, all methods comparison,
and batch evaluation across runs.

Available Methods:
    - majority: Simple majority voting by count
    - deepconf_mean: Sum of mean confidence per answer
    - deepconf_tail: Sum of tail (last N tokens) confidence
    - deepconf_bottom: Sum of sliding window min confidence
    - gradient: Sum of (mean_conf + beta * gradient)
    - cdg: Count-Dampened Gradient: count^alpha * mean(conf + beta * gradient)

Usage:
    # List all available runs
    python scripts/eval/eval_voting.py --list

    # Evaluate single run with default method (majority)
    python scripts/eval/eval_voting.py --run_id qwen32b_aime2025_512

    # Evaluate with specific method
    python scripts/eval/eval_voting.py --run_id qwen32b_aime2025_512 --method cdg

    # Evaluate ALL runs with all methods (comparison table)
    python scripts/eval/eval_voting.py --all --all-methods

    # Evaluate runs matching pattern with CDG method
    python scripts/eval/eval_voting.py --pattern "*aime2025*" --method cdg

    # Custom CDG parameters
    python scripts/eval/eval_voting.py --all --method cdg --alpha 0.5 --beta 10

    # Show per-question details
    python scripts/eval/eval_voting.py --run_id qwen32b_aime2025_512 -v

Expected run_id format: {model}_{dataset}_{budget}
    Examples: qwen32b_aime2025_512, gptoss20b_hmmt2025_512, deepseek8b_bruno2025_512
"""
import os
import sys
import argparse
import fnmatch
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import (
    load_run_results,
    evaluate_with_method,
    get_available_methods,
    get_method_info,
    EVAL_METHODS,
    DEFAULT_PARAMS,
)
from eval_cache import EvalCache
from hyperparam_config import HyperparamConfig

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE


def parse_run_id(run_id: str) -> dict:
    """
    Parse run_id to extract model, dataset, budget.
    Expected format: {model}_{dataset}_{budget}
    """
    datasets = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025']

    for dataset in datasets:
        if f'_{dataset}_' in run_id:
            parts = run_id.split(f'_{dataset}_')
            if len(parts) == 2:
                return {
                    'model': parts[0],
                    'dataset': dataset,
                    'budget': parts[1],
                    'run_id': run_id,
                }

    return {'model': run_id, 'dataset': '?', 'budget': '?', 'run_id': run_id}


def discover_runs(results_dir: str, pattern: str = None, patterns: str = None) -> list:
    """Discover all run directories.

    Args:
        results_dir: Base directory for results
        pattern: Single pattern to filter runs (e.g., "qwen32b_*")
        patterns: Comma-separated patterns to filter runs (e.g., "deepseek8b*,qwq32b*")
    """
    runs = []
    base = Path(results_dir)

    if not base.exists():
        return runs

    # Parse patterns into a list
    pattern_list = []
    if pattern:
        pattern_list.append(pattern)
    if patterns:
        pattern_list.extend([p.strip() for p in patterns.split(',')])

    for item in base.iterdir():
        if item.is_dir() and item.name not in ['subset_trace', '__pycache__', '.eval_cache']:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                # If patterns specified, check if run matches ANY pattern
                if pattern_list:
                    if not any(fnmatch.fnmatch(item.name, p) for p in pattern_list):
                        continue
                runs.append(item.name)

    return sorted(runs)


def evaluate_single_run(args: tuple) -> tuple:
    """Worker function for parallel run evaluation."""
    # Unpack args - hyperparam_config is optional (for model-specific params)
    if len(args) == 5:
        run_id, results_dir, method, params, hyperparam_config = args
    else:
        run_id, results_dir, method, params = args
        hyperparam_config = None

    run_path = os.path.join(results_dir, run_id)

    # IMPORTANT: use_threads=False to avoid nested ThreadPoolExecutor
    # which causes "can't start new thread" errors
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

    return run_id, method, {'info': info, 'result': result}


def evaluate_all_runs(results_dir: str, methods: list, params: dict,
                      pattern: str = None, patterns: str = None,
                      num_workers: int = None,
                      cache: EvalCache = None,
                      hyperparam_config: HyperparamConfig = None) -> dict:
    """
    Evaluate all runs with specified methods.

    Uses caching to skip already-computed results and shows progress.

    Args:
        results_dir: Base results directory
        methods: List of method names to evaluate
        params: Parameters dict for evaluation methods
        pattern: Optional glob pattern to filter runs
        patterns: Optional comma-separated patterns to filter runs
        num_workers: Max parallel workers (None = auto, capped at 8 to avoid thread issues)
        cache: EvalCache instance for caching results
        hyperparam_config: HyperparamConfig for model-specific params (optional)

    Returns:
        {run_id: {method: {'info': ..., 'result': ...}, ...}, ...}
    """
    runs = discover_runs(results_dir, pattern, patterns)

    if not runs:
        print(f"No runs found in {results_dir}")
        return {}

    total_tasks = len(runs) * len(methods)
    print(f"Found {len(runs)} runs to evaluate with {len(methods)} method(s) ({total_tasks} total tasks)")

    all_results = defaultdict(dict)

    # Check cache for existing results
    cached_count = 0
    tasks_to_run = []

    for run_id in runs:
        info = parse_run_id(run_id)

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
            tasks_to_run.append((run_id, results_dir, method, params, hyperparam_config))

    if cached_count > 0:
        print(f"Loaded {cached_count}/{total_tasks} results from cache")

    if not tasks_to_run:
        print("All results cached, nothing to compute")
        return dict(all_results)

    print(f"Computing {len(tasks_to_run)} remaining tasks...")

    # Cap workers to avoid thread exhaustion (each worker loads ~30 pkl files sequentially)
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(tasks_to_run), 8)

    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(evaluate_single_run, task): task
            for task in tasks_to_run
        }

        # Process as they complete with progress
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
            if completed % 10 == 0 or completed == len(tasks_to_run):
                print(f"  Progress: {completed}/{len(tasks_to_run)} tasks completed")

    return dict(all_results)


def print_summary_table(all_results: dict, methods: list):
    """Print summary table with all methods as columns."""
    if not all_results:
        print("No results to display")
        return

    # Group by dataset
    by_dataset = defaultdict(list)
    for run_id, methods_data in all_results.items():
        if methods_data:
            first_method = list(methods_data.keys())[0]
            dataset = methods_data[first_method]['info']['dataset']
            by_dataset[dataset].append((run_id, methods_data))

    # Calculate column widths
    method_width = max(12, max(len(EVAL_METHODS[m]['name']) for m in methods) + 2)

    # Print header
    print("\n" + "=" * 100)
    print("VOTING METHODS EVALUATION")
    print("=" * 100)

    # Table header
    header = f"{'Model':<15} {'Dataset':<10} {'Budget':<8}"
    for method in methods:
        method_name = EVAL_METHODS[method]['name']
        header += f" {method_name:^{method_width}}"
    print(f"\n{header}")
    print("-" * len(header))

    # Sort datasets
    dataset_order = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025', '?']
    sorted_datasets = sorted(by_dataset.keys(), key=lambda x: dataset_order.index(x) if x in dataset_order else 99)

    for dataset in sorted_datasets:
        runs = by_dataset[dataset]
        runs.sort(key=lambda x: x[1][list(x[1].keys())[0]]['info']['model'])

        for run_id, methods_data in runs:
            info = methods_data[list(methods_data.keys())[0]]['info']
            row = f"{info['model']:<15} {info['dataset']:<10} {info['budget']:<8}"

            for method in methods:
                if method in methods_data:
                    result = methods_data[method]['result']
                    acc_str = f"{result['correct']}/{result['total']} ({100*result['accuracy']:.1f}%)"
                else:
                    acc_str = "N/A"
                row += f" {acc_str:^{method_width}}"

            print(row)

        if dataset != sorted_datasets[-1]:
            print("-" * len(header))

    print("=" * 100)

    # Best method summary
    if len(methods) > 1:
        print("\nBest method per dataset:")
        for dataset in sorted_datasets:
            runs = by_dataset[dataset]
            method_totals = defaultdict(lambda: {'correct': 0, 'total': 0})

            for run_id, methods_data in runs:
                for method, data in methods_data.items():
                    result = data['result']
                    method_totals[method]['correct'] += result['correct']
                    method_totals[method]['total'] += result['total']

            if method_totals:
                best_method = max(method_totals.items(),
                                  key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0)
                acc = best_method[1]['correct'] / best_method[1]['total'] if best_method[1]['total'] > 0 else 0
                print(f"  {dataset}: {EVAL_METHODS[best_method[0]]['name']} ({100*acc:.1f}%)")


def print_single_run(run_id: str, methods_data: dict, verbose: bool = False):
    """Print results for a single run."""
    info = list(methods_data.values())[0]['info']

    print("\n" + "=" * 70)
    print(f"EVALUATION: {run_id}")
    print("=" * 70)
    print(f"Model:    {info['model']}")
    print(f"Dataset:  {info['dataset']}")
    print(f"Budget:   {info['budget']}")
    print("-" * 70)

    # Results per method
    for method, data in methods_data.items():
        result = data['result']
        method_name = EVAL_METHODS[method]['name']
        print(f"{method_name}: {result['correct']}/{result['total']} ({100*result['accuracy']:.1f}%)")

    print("=" * 70)

    # Per-question breakdown for first method
    if verbose:
        method = list(methods_data.keys())[0]
        result = methods_data[method]['result']

        print(f"\nPer-Question Details ({EVAL_METHODS[method]['name']}):")
        print(f"{'QID':<5} {'GT':<12} {'Predicted':<12} {'Traces':<8} {'Result':<8}")
        print("-" * 60)

        for q in result['per_question']:
            mark = "OK" if q['correct'] else "X"
            gt = str(q['ground_truth'])[:10]
            pred = str(q['predicted'])[:10]
            num_traces = q.get('num_traces', q.get('num_valid_answers', '?'))
            print(f"{q['qid']:<5} {gt:<12} {pred:<12} {num_traces:<8} {mark}")

        print("-" * 60)

        # Wrong answers
        wrong = [q for q in result['per_question'] if not q['correct']]
        if wrong:
            print(f"\nWRONG ANSWERS ({len(wrong)} questions):")
            for q in wrong:
                print(f"  QID {q['qid']}: GT={q['ground_truth']}, Predicted={q['predicted']}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate voting methods on LLM trace results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                              # List available runs
  %(prog)s --run_id qwen32b_aime2025_512       # Evaluate single run (majority)
  %(prog)s --all --method cdg                  # All runs with CDG method
  %(prog)s --all --all-methods                 # Compare all methods
  %(prog)s --pattern "*aime2025*" --method cdg # Pattern match with CDG
        """
    )

    # Run selection
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Base directory for results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Evaluate single run by ID')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all runs')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Filter runs by pattern (e.g., "qwen32b_*", "*aime2025*")')
    parser.add_argument('--patterns', type=str, default=None,
                        help='Filter runs by multiple comma-separated patterns (e.g., "deepseek8b*,qwq32b*")')
    parser.add_argument('--list', action='store_true',
                        help='List available runs without evaluating')

    # Method selection
    parser.add_argument('--method', type=str, default='majority',
                        choices=get_available_methods(),
                        help='Voting method to use (default: majority)')
    parser.add_argument('--all-methods', action='store_true',
                        help='Evaluate with all methods and compare')

    # CDG/Gradient parameters
    parser.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'],
                        help=f'CDG count dampening exponent (default: {DEFAULT_PARAMS["alpha"]})')
    parser.add_argument('--beta', type=float, default=DEFAULT_PARAMS['beta'],
                        help=f'Gradient weight (default: {DEFAULT_PARAMS["beta"]})')
    parser.add_argument('--position_pct', type=int, default=DEFAULT_PARAMS['position_pct'],
                        help=f'Percentile for gradient computation (default: {DEFAULT_PARAMS["position_pct"]})')
    parser.add_argument('--tail_window', type=int, default=DEFAULT_PARAMS['tail_window'],
                        help=f'Window size for tail confidence (default: {DEFAULT_PARAMS["tail_window"]})')
    parser.add_argument('--bottom_window', type=int, default=DEFAULT_PARAMS['bottom_window'],
                        help=f'Window size for bottom window method (default: {DEFAULT_PARAMS["bottom_window"]})')
    parser.add_argument('--bottom_pct', type=float, default=DEFAULT_PARAMS['bottom_pct'],
                        help=f'Bottom percentile for bottom window (default: {DEFAULT_PARAMS["bottom_pct"]})')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-question details')

    # Cache options
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable result caching (recompute everything)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached results before running')
    parser.add_argument('--cache-stats', action='store_true',
                        help='Show cache statistics and exit')

    # Hyperparameter config options
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Use model-specific tuned hyperparameters from config file')
    parser.add_argument('--show-tuned-params', action='store_true',
                        help='Show tuned hyperparameters config and exit')

    # Debug mode
    parser.add_argument('--fast', action='store_true',
                        help='Fast debug mode: 2 methods, 1 run pattern (results will be inaccurate)')

    args = parser.parse_args()

    # Apply fast mode overrides
    if args.fast:
        print("=" * 60)
        print("FAST DEBUG MODE - Results will be inaccurate!")
        print("=" * 60)
        args.all_methods = False
        args.method = "majority"  # Will be overridden to include cdg below
        args.pattern = args.pattern or "*qwen32b*aime2025*"

    # Initialize hyperparam config
    hyperparam_config = HyperparamConfig(args.results_dir)

    # Handle --show-tuned-params
    if args.show_tuned_params:
        hyperparam_config.print_summary()
        return

    # Initialize cache
    cache = EvalCache(args.results_dir, enabled=not args.no_cache)

    # Handle cache-related commands
    if args.cache_stats:
        stats = cache.stats()
        print("Cache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

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

    # Determine methods to use
    if args.fast:
        methods = ['majority', 'cdg']  # Fast mode: only 2 methods
    elif args.all_methods:
        methods = get_available_methods()
    else:
        methods = [args.method]

    # List mode
    if args.list:
        runs = discover_runs(args.results_dir, args.pattern, args.patterns)
        print(f"Available runs in {args.results_dir}:")
        print(f"\n{'Run ID':<45} {'Model':<15} {'Dataset':<10}")
        print("-" * 70)
        for run in runs:
            info = parse_run_id(run)
            print(f"{run:<45} {info['model']:<15} {info['dataset']:<10}")
        print(f"\nTotal: {len(runs)} runs")
        print(f"\nAvailable methods: {', '.join(get_available_methods())}")
        return

    # Evaluate all runs (--all, --pattern, or --patterns)
    if args.all or args.pattern or args.patterns:
        print(f"Methods: {', '.join(EVAL_METHODS[m]['name'] for m in methods)}")
        if args.use_tuned_params:
            print(f"Parameters: using model-specific tuned hyperparameters")
            tuned_models = hyperparam_config.get_all_models()
            if tuned_models:
                print(f"  Tuned models: {', '.join(tuned_models)}")
            else:
                print(f"  WARNING: No tuned params found, using defaults")
        else:
            print(f"Parameters: alpha={params['alpha']}, beta={params['beta']}, position_pct={params['position_pct']}")
        if not args.no_cache:
            print(f"Cache: enabled (use --no-cache to disable)")
        print()

        # Pass hyperparam_config only if --use-tuned-params is set
        hp_config = hyperparam_config if args.use_tuned_params else None
        all_results = evaluate_all_runs(args.results_dir, methods, params, args.pattern,
                                        args.patterns, cache=cache, hyperparam_config=hp_config)
        print_summary_table(all_results, methods)
        return

    # Evaluate single run
    if args.run_id:
        results_dir = os.path.join(args.results_dir, args.run_id)
        print(f"Loading results from: {results_dir}")
        results_by_qid = load_run_results(results_dir)
        print(f"Loaded {len(results_by_qid)} questions")

        if not results_by_qid:
            print("No results found!")
            return

        # Evaluate with all requested methods
        methods_data = {}
        for method in methods:
            result = evaluate_with_method(results_by_qid, method, params)
            methods_data[method] = {
                'info': parse_run_id(args.run_id),
                'result': result,
            }

        print_single_run(args.run_id, methods_data, args.verbose)
        return

    # No run specified - show help
    parser.print_help()
    print("\n\nAvailable runs:")
    runs = discover_runs(args.results_dir)
    for run in runs[:10]:
        print(f"  {run}")
    if len(runs) > 10:
        print(f"  ... and {len(runs) - 10} more")
    print("\nUse --run_id <name> or --all")


if __name__ == "__main__":
    main()
