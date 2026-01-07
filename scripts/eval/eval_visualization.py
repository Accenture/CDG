"""
Visualization tools for confidence analysis in LLM traces.

Analyzes correlation between confidence and position in traces,
comparing correct vs wrong traces to identify patterns.

Usage:
    # Analyze single run
    python scripts/eval/eval_visualization.py --run_id qwen32b_aime2025_512

    # Analyze with custom bins
    python scripts/eval/eval_visualization.py --run_id qwen32b_aime2025_512 --num_bins 20

    # Save figure without displaying
    python scripts/eval/eval_visualization.py --run_id qwen32b_aime2025_512 --output figure.png --no-show

    # Analyze multiple runs
    python scripts/eval/eval_visualization.py --pattern "*aime2025*"
"""
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import load_run_results, equal_func

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Figure generation disabled.")

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE


def divide_trace_into_bins(confs: list, num_bins: int = 10) -> list:
    """
    Divide a trace into N bins and compute mean confidence per bin.

    Returns:
        bin_means: List of mean confidence for each bin
    """
    if not confs:
        return [0.0] * num_bins

    n = len(confs)
    bin_size = n / num_bins

    bin_means = []
    for i in range(num_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)

        if end_idx > n:
            end_idx = n

        if start_idx >= n:
            bin_means.append(np.nan)
        else:
            bin_conf = confs[start_idx:end_idx]
            bin_means.append(np.mean(bin_conf))

    return bin_means


def analyze_confidence_position(results_by_qid: dict, num_bins: int = 10) -> dict:
    """
    Analyze confidence vs position for traces.

    Returns:
        dict with correct/wrong trace statistics by position
    """
    correct_traces_by_position = defaultdict(list)
    wrong_traces_by_position = defaultdict(list)

    for qid, results_list in results_by_qid.items():
        # Get ground truth and traces
        ground_truth = None
        all_traces = []

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not ground_truth:
            continue

        for trace in all_traces:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is None or len(confs) < 10:
                continue

            try:
                is_correct = equal_func(str(answer), str(ground_truth))
            except:
                is_correct = str(answer) == str(ground_truth)

            bin_means = divide_trace_into_bins(confs, num_bins)

            target = correct_traces_by_position if is_correct else wrong_traces_by_position
            for pos, mean_conf in enumerate(bin_means, 1):
                if not np.isnan(mean_conf):
                    target[pos].append(mean_conf)

    # Compute statistics
    positions = list(range(1, num_bins + 1))

    stats = {
        'positions': positions,
        'correct': {'means': [], 'stds': [], 'counts': []},
        'wrong': {'means': [], 'stds': [], 'counts': []},
    }

    for pos in positions:
        for category, trace_data in [('correct', correct_traces_by_position),
                                      ('wrong', wrong_traces_by_position)]:
            confs = trace_data[pos]
            stats[category]['means'].append(np.mean(confs) if confs else np.nan)
            stats[category]['stds'].append(np.std(confs) if confs else np.nan)
            stats[category]['counts'].append(len(confs))

    return stats


def print_analysis(stats: dict, run_name: str):
    """Print confidence analysis results."""
    positions = stats['positions']
    num_bins = len(positions)

    print('=' * 90)
    print(f'CONFIDENCE VS POSITION ANALYSIS - {run_name}')
    print('=' * 90)

    # Summary table
    print('\nMEAN CONFIDENCE BY POSITION (1=beginning, N=end):')
    print('=' * 90)
    print(f'{"Position":<12} {"Correct Traces":<25} {"Wrong Traces":<25} {"Difference"}')
    print('-' * 90)

    for i, pos in enumerate(positions):
        c_mean = stats['correct']['means'][i]
        c_std = stats['correct']['stds'][i]
        w_mean = stats['wrong']['means'][i]
        w_std = stats['wrong']['stds'][i]

        c_str = f'{c_mean:5.2f} +/- {c_std:4.2f}' if not np.isnan(c_mean) else 'N/A'
        w_str = f'{w_mean:5.2f} +/- {w_std:4.2f}' if not np.isnan(w_mean) else 'N/A'

        if not np.isnan(c_mean) and not np.isnan(w_mean):
            diff = c_mean - w_mean
            diff_str = f'{diff:+5.2f}'
        else:
            diff_str = 'N/A'

        print(f'{pos:<12} {c_str:<25} {w_str:<25} {diff_str}')

    # Sample counts
    print()
    if stats['correct']['counts'][0] > 0:
        print(f'Correct traces: ~{stats["correct"]["counts"][0]:,} per position')
    if stats['wrong']['counts'][0] > 0:
        print(f'Wrong traces:   ~{stats["wrong"]["counts"][0]:,} per position')

    # Key observations
    print()
    print('=' * 90)
    print('KEY OBSERVATIONS:')
    print('=' * 90)

    c_means = [m for m in stats['correct']['means'] if not np.isnan(m)]
    w_means = [m for m in stats['wrong']['means'] if not np.isnan(m)]

    if c_means and w_means:
        c_start = np.mean(stats['correct']['means'][:2])
        c_end = np.mean(stats['correct']['means'][-2:])
        w_start = np.mean(stats['wrong']['means'][:2])
        w_end = np.mean(stats['wrong']['means'][-2:])

        print(f'\n1. Overall confidence levels:')
        print(f'   Correct traces: {np.mean(c_means):.2f}')
        print(f'   Wrong traces:   {np.mean(w_means):.2f}')
        print(f'   Difference:     {np.mean(c_means) - np.mean(w_means):+.2f}')

        print(f'\n2. Confidence trends:')
        print(f'   Correct: {c_start:.2f} -> {c_end:.2f} (change: {c_end - c_start:+.2f})')
        print(f'   Wrong:   {w_start:.2f} -> {w_end:.2f} (change: {w_end - w_start:+.2f})')

        if c_end > c_start and w_end < w_start:
            print(f'   -> Correct traces GAIN confidence towards the end')
            print(f'   -> Wrong traces LOSE confidence towards the end')
            print(f'   -> This supports gradient-based weighting!')

    print()


def create_figure(stats: dict, run_name: str, output_path: str = None, show: bool = True):
    """Create confidence vs position visualization."""
    if not HAS_MATPLOTLIB:
        print("Cannot create figure: matplotlib not available")
        return None

    positions = stats['positions']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean confidence by position
    ax1 = axes[0]
    ax1.errorbar(positions, stats['correct']['means'], yerr=stats['correct']['stds'],
                 marker='o', markersize=8, linewidth=2, capsize=5,
                 label='Correct', color='green', alpha=0.8)
    ax1.errorbar(positions, stats['wrong']['means'], yerr=stats['wrong']['stds'],
                 marker='s', markersize=8, linewidth=2, capsize=5,
                 label='Wrong', color='red', alpha=0.8)
    ax1.set_xlabel('Position in Trace (1=beginning)', fontsize=12)
    ax1.set_ylabel('Mean Token-Level Confidence', fontsize=12)
    ax1.set_title(f'{run_name}: Confidence vs Position', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(positions)

    # Plot 2: Difference
    ax2 = axes[1]
    diffs = [stats['correct']['means'][i] - stats['wrong']['means'][i]
             for i in range(len(positions))]
    colors = ['green' if not np.isnan(d) and d > 0 else 'red' for d in diffs]
    ax2.bar(positions, diffs, color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Position in Trace (1=beginning)', fontsize=12)
    ax2.set_ylabel('Confidence Difference (Correct - Wrong)', fontsize=12)
    ax2.set_title(f'{run_name}: Confidence Gap by Position', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(positions)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {output_path}')

    if show:
        plt.show()

    return fig


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
                if pattern and not __import__('fnmatch').fnmatch(item.name, pattern):
                    continue
                runs.append(item.name)

    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence vs position in LLM traces')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Base directory for results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID to analyze')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Pattern to match run IDs')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins to divide traces into (default: 10)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save figure to file')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure')
    parser.add_argument('--list', action='store_true',
                        help='List available runs')

    args = parser.parse_args()

    # List mode
    if args.list:
        runs = discover_runs(args.results_dir, args.pattern)
        print(f"Available runs in {args.results_dir}:")
        for run in runs:
            print(f"  {run}")
        return

    # Get runs to analyze
    if args.run_id:
        runs = [args.run_id]
    elif args.pattern:
        runs = discover_runs(args.results_dir, args.pattern)
    else:
        parser.print_help()
        print("\nUse --run_id <name> or --pattern <pattern>")
        return

    if not runs:
        print("No runs found!")
        return

    # Analyze each run
    for run_id in runs:
        run_path = os.path.join(args.results_dir, run_id)
        print(f"\nLoading {run_id}...")

        results_by_qid = load_run_results(run_path)

        if not results_by_qid:
            print(f"  No results found in {run_path}")
            continue

        print(f"  Loaded {len(results_by_qid)} questions")

        # Analyze
        stats = analyze_confidence_position(results_by_qid, args.num_bins)
        print_analysis(stats, run_id)

        # Create figure
        if HAS_MATPLOTLIB:
            output_path = args.output
            if output_path and len(runs) > 1:
                # Add run_id to filename for multiple runs
                base, ext = os.path.splitext(output_path)
                output_path = f"{base}_{run_id}{ext}"

            create_figure(stats, run_id, output_path, show=not args.no_show)


if __name__ == '__main__':
    main()
