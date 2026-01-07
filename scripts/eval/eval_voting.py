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
from concurrent.futures import ThreadPoolExecutor
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


def discover_runs(results_dir: str, pattern: str = None) -> list:
    """Discover all run directories."""
    runs = []
    base = Path(results_dir)

    if not base.exists():
        return runs

    for item in base.iterdir():
        if item.is_dir() and item.name not in ['subset_trace', '__pycache__']:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                if pattern and not fnmatch.fnmatch(item.name, pattern):
                    continue
                runs.append(item.name)

    return sorted(runs)


def evaluate_single_run(args: tuple) -> tuple:
    """Worker function for parallel run evaluation."""
    run_id, results_dir, method, params = args
    run_path = os.path.join(results_dir, run_id)
    results_by_qid = load_run_results(run_path)

    if not results_by_qid:
        return run_id, method, None

    result = evaluate_with_method(results_by_qid, method, params)
    info = parse_run_id(run_id)

    return run_id, method, {'info': info, 'result': result}


def evaluate_all_runs(results_dir: str, methods: list, params: dict,
                      pattern: str = None, num_workers: int = None) -> dict:
    """Evaluate all runs with specified methods in parallel."""
    runs = discover_runs(results_dir, pattern)

    if not runs:
        print(f"No runs found in {results_dir}")
        return {}

    print(f"Found {len(runs)} runs to evaluate with {len(methods)} method(s)...")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(runs) * len(methods), 16)

    # Build task list
    tasks = []
    for run_id in runs:
        for method in methods:
            tasks.append((run_id, results_dir, method, params))

    all_results = defaultdict(dict)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(evaluate_single_run, tasks)
        for run_id, method, data in results:
            if data is not None:
                all_results[run_id][method] = data

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

    args = parser.parse_args()

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
    if args.all_methods:
        methods = get_available_methods()
    else:
        methods = [args.method]

    # List mode
    if args.list:
        runs = discover_runs(args.results_dir, args.pattern)
        print(f"Available runs in {args.results_dir}:")
        print(f"\n{'Run ID':<45} {'Model':<15} {'Dataset':<10}")
        print("-" * 70)
        for run in runs:
            info = parse_run_id(run)
            print(f"{run:<45} {info['model']:<15} {info['dataset']:<10}")
        print(f"\nTotal: {len(runs)} runs")
        print(f"\nAvailable methods: {', '.join(get_available_methods())}")
        return

    # Evaluate all runs
    if args.all or args.pattern:
        print(f"Methods: {', '.join(EVAL_METHODS[m]['name'] for m in methods)}")
        print(f"Parameters: alpha={params['alpha']}, beta={params['beta']}")
        print()

        all_results = evaluate_all_runs(args.results_dir, methods, params, args.pattern)
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
