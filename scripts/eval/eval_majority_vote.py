"""
Simple majority vote evaluation script.

Evaluates inference results using majority voting and computes accuracy
against ground truth answers. Supports single run evaluation or batch
evaluation across all runs with summary tables.

Usage:
    # List all available runs (no evaluation)
    python scripts/eval/eval_majority_vote.py --list

    # Evaluate ALL runs and show summary table
    python scripts/eval/eval_majority_vote.py --all

    # Evaluate single run by ID
    python scripts/eval/eval_majority_vote.py --run_id qwen32b_aime2025_512

    # Evaluate single run with per-question details
    python scripts/eval/eval_majority_vote.py --run_id qwen32b_aime2025_512 -v

    # Filter runs by pattern (supports wildcards)
    python scripts/eval/eval_majority_vote.py --pattern "qwen32b_*"
    python scripts/eval/eval_majority_vote.py --pattern "*aime2025*"
    python scripts/eval/eval_majority_vote.py --pattern "gptoss*"

    # Custom results directory
    python scripts/eval/eval_majority_vote.py --all --results_dir /path/to/results

Expected run_id format: {model}_{dataset}_{budget}
    Examples: qwen32b_aime2025_512, gptoss20b_hmmt2025_512, deepseek8b_bruno2025_512

Output:
    --all mode produces a summary table grouped by dataset with aggregate stats
    --run_id mode shows per-question breakdown and wrong answer analysis
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
    evaluate_majority_vote as evaluate,
    majority_vote,
    equal_func,
)

# Default results directory
DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE


def parse_run_id(run_id: str) -> dict:
    """
    Parse run_id to extract model, dataset, budget.
    Expected format: {model}_{dataset}_{budget}
    Examples:
        qwen32b_aime2025_512 -> model=qwen32b, dataset=aime2025, budget=512
        gptoss20b_aime2025_512 -> model=gptoss20b, dataset=aime2025, budget=512
        deepseek8b_hmmt2025_512 -> model=deepseek8b, dataset=hmmt2025, budget=512
    """
    # Known datasets
    datasets = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025']

    for dataset in datasets:
        if f'_{dataset}_' in run_id:
            parts = run_id.split(f'_{dataset}_')
            if len(parts) == 2:
                model = parts[0]
                budget = parts[1]
                return {
                    'model': model,
                    'dataset': dataset,
                    'budget': budget,
                    'run_id': run_id,
                }

    # Fallback: just return run_id as-is
    return {
        'model': run_id,
        'dataset': '?',
        'budget': '?',
        'run_id': run_id,
    }


def discover_runs(results_dir: str, pattern: str = None) -> list:
    """Discover all run directories."""
    runs = []
    base = Path(results_dir)

    if not base.exists():
        return runs

    for item in base.iterdir():
        if item.is_dir():
            # Skip known non-run directories
            if item.name in ['subset_trace', '__pycache__']:
                continue

            # Check if it has pickle files
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                # Apply pattern filter if specified
                if pattern and not fnmatch.fnmatch(item.name, pattern):
                    continue
                runs.append(item.name)

    return sorted(runs)


def evaluate_single_run(args: tuple) -> tuple:
    """Worker function for parallel run evaluation."""
    run_id, results_dir = args
    run_path = os.path.join(results_dir, run_id)
    results_by_qid = load_run_results(run_path, quiet=True)

    if not results_by_qid:
        return run_id, None

    result = evaluate(results_by_qid)
    info = parse_run_id(run_id)

    return run_id, {
        'info': info,
        'result': result,
    }


def evaluate_all_runs(results_dir: str, pattern: str = None, num_workers: int = None) -> dict:
    """Evaluate all runs in parallel."""
    runs = discover_runs(results_dir, pattern)

    if not runs:
        print(f"No runs found in {results_dir}")
        return {}

    print(f"Found {len(runs)} runs to evaluate...")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(runs))

    all_results = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(run_id, results_dir) for run_id in runs]
        results = executor.map(evaluate_single_run, args_list)

        for run_id, data in results:
            if data is not None:
                all_results[run_id] = data

    return all_results


def print_summary_table(all_results: dict):
    """Print a summary table of all results."""
    if not all_results:
        print("No results to display")
        return

    # Group by dataset
    by_dataset = defaultdict(list)
    for run_id, data in all_results.items():
        dataset = data['info']['dataset']
        by_dataset[dataset].append((run_id, data))

    # Print header
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (Majority Vote)")
    print("=" * 80)

    # Print table header
    print(f"\n{'Model':<15} {'Dataset':<12} {'Budget':<8} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
    print("-" * 80)

    # Sort datasets
    dataset_order = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025', '?']
    sorted_datasets = sorted(by_dataset.keys(), key=lambda x: dataset_order.index(x) if x in dataset_order else 99)

    for dataset in sorted_datasets:
        runs = by_dataset[dataset]
        # Sort by model name
        runs.sort(key=lambda x: x[1]['info']['model'])

        for run_id, data in runs:
            info = data['info']
            result = data['result']
            acc_str = f"{100 * result['accuracy']:.1f}%"
            print(f"{info['model']:<15} {info['dataset']:<12} {info['budget']:<8} {result['correct']:<10} {result['total']:<8} {acc_str:<10}")

        # Add separator between datasets
        if dataset != sorted_datasets[-1]:
            print("-" * 80)

    print("=" * 80)

    # Summary by model (aggregate across datasets)
    print("\n" + "=" * 80)
    print("AGGREGATE BY MODEL")
    print("=" * 80)

    by_model = defaultdict(lambda: {'correct': 0, 'total': 0})
    for run_id, data in all_results.items():
        model = data['info']['model']
        result = data['result']
        by_model[model]['correct'] += result['correct']
        by_model[model]['total'] += result['total']

    print(f"\n{'Model':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)

    for model in sorted(by_model.keys()):
        stats = by_model[model]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{model:<15} {stats['correct']:<10} {stats['total']:<10} {100*acc:.1f}%")

    print("=" * 80)


def print_single_run(result: dict, run_id: str, verbose: bool = False):
    """Print results for a single run."""
    info = parse_run_id(run_id)

    print("\n" + "=" * 60)
    print(f"MAJORITY VOTE EVALUATION: {run_id}")
    print("=" * 60)
    print(f"Model:    {info['model']}")
    print(f"Dataset:  {info['dataset']}")
    print(f"Budget:   {info['budget']}")
    print("-" * 60)
    print(f"Correct:  {result['correct']} / {result['total']}")
    print(f"Accuracy: {100 * result['accuracy']:.1f}%")
    print("=" * 60)

    # Per-question breakdown
    if verbose or result['total'] <= 50:
        print(f"\n{'QID':<5} {'GT':<12} {'Predicted':<12} {'Traces':<8} {'Result':<8}")
        print("-" * 60)

        for q in result['per_question']:
            mark = "OK" if q['correct'] else "X"
            gt = str(q['ground_truth'])[:10]
            pred = str(q['predicted'])[:10]
            print(f"{q['qid']:<5} {gt:<12} {pred:<12} {q['num_valid_answers']:<8} {mark}")

        print("-" * 60)
        print(f"Accuracy: {100 * result['accuracy']:.1f}%")

    # Show wrong answers
    wrong = [q for q in result['per_question'] if not q['correct']]
    if wrong and verbose:
        print(f"\n{'=' * 60}")
        print(f"WRONG ANSWERS ({len(wrong)} questions)")
        print("=" * 60)
        for q in wrong:
            print(f"\nQID {q['qid']}:")
            print(f"  Ground truth: {q['ground_truth']}")
            print(f"  Predicted:    {q['predicted']}")
            print(f"  Top answers:  {q['top_answers']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate majority vote accuracy')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Base directory for results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Evaluate single run by ID')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all runs and show summary table')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Filter runs by pattern (e.g., "qwen32b_*", "*aime2025*")')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-question details')
    parser.add_argument('--list', action='store_true',
                        help='List available runs without evaluating')
    args = parser.parse_args()

    # List mode
    if args.list:
        runs = discover_runs(args.results_dir, args.pattern)
        print(f"Available runs in {args.results_dir}:")
        for run in runs:
            info = parse_run_id(run)
            print(f"  {run:<40} (model={info['model']}, dataset={info['dataset']})")
        return

    # Evaluate all runs
    if args.all or args.pattern:
        all_results = evaluate_all_runs(args.results_dir, args.pattern)
        print_summary_table(all_results)
        return

    # Evaluate single run
    if args.run_id:
        results_dir = os.path.join(args.results_dir, args.run_id)
    else:
        # If no run_id and not --all, show help
        parser.print_help()
        print("\n\nAvailable runs:")
        runs = discover_runs(args.results_dir)
        for run in runs[:10]:
            print(f"  {run}")
        if len(runs) > 10:
            print(f"  ... and {len(runs) - 10} more")
        print("\nUse --run_id <name> or --all")
        return

    print(f"Loading results from: {results_dir}")
    results_by_qid = load_run_results(results_dir)
    print(f"Loaded {len(results_by_qid)} questions")

    if not results_by_qid:
        print("No results found!")
        return

    result = evaluate(results_by_qid)
    print_single_run(result, args.run_id, args.verbose)


if __name__ == "__main__":
    main()
