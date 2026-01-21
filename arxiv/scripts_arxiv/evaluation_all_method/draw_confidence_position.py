"""
Analyze correlation between confidence and position in traces

Compare correct vs wrong traces to see if confidence patterns differ by position.
This script reads from any folder containing pickle files with trace data.

Usage:
    python draw_confidence_position.py --data_dir /path/to/data [--output_dir /path/to/output] [--num_bins 10]
"""
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))
from dynasor.core.evaluator import math_equal


def quick_parse(text: str) -> str:
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def divide_trace_into_bins(confs, num_bins=10):
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


def analyze_confidence_position(data_dir, output_dir=None, num_bins=10):
    """
    Analyze confidence vs position for traces in data_dir.

    Args:
        data_dir: Directory containing pickle files
        output_dir: Directory to save output figure (defaults to data_dir)
        num_bins: Number of bins to divide traces into
    """
    results_dir = Path(data_dir)

    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all pickle files
    pkl_files = sorted(results_dir.glob('qid*.pkl'))

    # If no pkl files found, try subdirectories
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    if not pkl_files:
        print(f'No pickle files found in {results_dir}')
        sys.exit(1)

    # Extract dataset name from path
    dataset_name = results_dir.name

    print('='*90)
    print(f'CONFIDENCE VS POSITION ANALYSIS - {dataset_name}')
    print('='*90)
    print(f'Results directory: {results_dir}')
    print(f'Found {len(pkl_files)} pickle files')
    print()

    # Collect data across all questions
    correct_traces_by_position = defaultdict(list)
    wrong_traces_by_position = defaultdict(list)

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f'Warning: Could not load {pkl_file}: {e}')
            continue

        gt = data['ground_truth']

        for trace in data['all_traces']:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is None or len(confs) < 10:
                continue

            is_correct = equal_func(str(answer), str(gt))

            # Divide trace into bins
            bin_means = divide_trace_into_bins(confs, num_bins)

            if is_correct:
                for pos, mean_conf in enumerate(bin_means, 1):
                    if not np.isnan(mean_conf):
                        correct_traces_by_position[pos].append(mean_conf)
            else:
                for pos, mean_conf in enumerate(bin_means, 1):
                    if not np.isnan(mean_conf):
                        wrong_traces_by_position[pos].append(mean_conf)

    # Check if we have enough data
    if not correct_traces_by_position and not wrong_traces_by_position:
        print('No valid traces found with sufficient data.')
        print('Traces need at least 10 confidence values to be analyzed.')
        sys.exit(1)

    # Compute statistics
    positions = list(range(1, num_bins + 1))

    correct_means = []
    correct_stds = []
    correct_counts = []

    wrong_means = []
    wrong_stds = []
    wrong_counts = []

    for pos in positions:
        # Correct traces
        correct_confs = correct_traces_by_position[pos]
        correct_means.append(np.mean(correct_confs) if correct_confs else np.nan)
        correct_stds.append(np.std(correct_confs) if correct_confs else np.nan)
        correct_counts.append(len(correct_confs))

        # Wrong traces
        wrong_confs = wrong_traces_by_position[pos]
        wrong_means.append(np.mean(wrong_confs) if wrong_confs else np.nan)
        wrong_stds.append(np.std(wrong_confs) if wrong_confs else np.nan)
        wrong_counts.append(len(wrong_confs))

    # Print summary table
    print('MEAN CONFIDENCE BY POSITION (1=beginning, 10=end):')
    print('='*90)
    print(f'{"Position":<12} {"Correct Traces":<25} {"Wrong Traces":<25} {"Difference"}')
    print(f'{"":12} {"Mean +/- Std":<25} {"Mean +/- Std":<25}')
    print('-'*90)

    for i, pos in enumerate(positions):
        correct_str = f'{correct_means[i]:5.2f} +/- {correct_stds[i]:4.2f}' if not np.isnan(correct_means[i]) else 'N/A'
        wrong_str = f'{wrong_means[i]:5.2f} +/- {wrong_stds[i]:4.2f}' if not np.isnan(wrong_means[i]) else 'N/A'

        if not np.isnan(correct_means[i]) and not np.isnan(wrong_means[i]):
            diff = correct_means[i] - wrong_means[i]
            diff_str = f'{diff:+5.2f}'
            if diff > 0.5:
                diff_str += ' [+]'
            elif diff < -0.5:
                diff_str += ' [-]'
        else:
            diff_str = 'N/A'

        print(f'{pos:<12} {correct_str:<25} {wrong_str:<25} {diff_str}')

    print()
    print('Sample counts per position:')
    if correct_counts[0] > 0:
        print(f'  Correct traces: ~{correct_counts[0]:,} per position')
    if wrong_counts[0] > 0:
        print(f'  Wrong traces:   ~{wrong_counts[0]:,} per position')

    print()
    print('='*90)
    print('KEY OBSERVATIONS:')
    print('='*90)

    # Compute trends (filter out NaN values)
    valid_correct_means = [m for m in correct_means if not np.isnan(m)]
    valid_wrong_means = [m for m in wrong_means if not np.isnan(m)]

    if valid_correct_means and valid_wrong_means:
        correct_start = np.mean(correct_means[:2])  # First 20%
        correct_end = np.mean(correct_means[-2:])   # Last 20%
        wrong_start = np.mean(wrong_means[:2])
        wrong_end = np.mean(wrong_means[-2:])

        print(f'\n1. Overall confidence levels:')
        print(f'   Correct traces: {np.mean(valid_correct_means):.2f} (averaged across all positions)')
        print(f'   Wrong traces:   {np.mean(valid_wrong_means):.2f}')
        print(f'   Difference:     {np.mean(valid_correct_means) - np.mean(valid_wrong_means):+.2f}')

        print(f'\n2. Confidence trends from beginning to end:')
        print(f'   Correct traces: {correct_start:.2f} -> {correct_end:.2f} (change: {correct_end - correct_start:+.2f})')
        print(f'   Wrong traces:   {wrong_start:.2f} -> {wrong_end:.2f} (change: {wrong_end - wrong_start:+.2f})')

        if correct_end > correct_start and wrong_end < wrong_start:
            print(f'   -> Correct traces gain confidence towards the end')
            print(f'   -> Wrong traces lose confidence towards the end')
        elif correct_end > correct_start:
            print(f'   -> Both gain confidence, but correct traces gain more')
        elif wrong_end < wrong_start:
            print(f'   -> Wrong traces lose confidence towards the end')

        print(f'\n3. Position with largest difference:')
        diffs_with_idx = [(i, correct_means[i] - wrong_means[i])
                          for i in range(num_bins)
                          if not np.isnan(correct_means[i]) and not np.isnan(wrong_means[i])]
        if diffs_with_idx:
            max_diff_idx, max_diff = max(diffs_with_idx, key=lambda x: x[1])
            print(f'   Position {max_diff_idx + 1}: Correct={correct_means[max_diff_idx]:.2f}, Wrong={wrong_means[max_diff_idx]:.2f}, Diff={max_diff:+.2f}')

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean confidence by position
    ax1 = axes[0]
    ax1.errorbar(positions, correct_means, yerr=correct_stds,
                 marker='o', markersize=8, linewidth=2, capsize=5,
                 label='Correct', color='green', alpha=0.8)
    ax1.errorbar(positions, wrong_means, yerr=wrong_stds,
                 marker='s', markersize=8, linewidth=2, capsize=5,
                 label='Wrong', color='red', alpha=0.8)
    ax1.set_xlabel('Position in Trace (1=beginning, 10=end)', fontsize=12)
    ax1.set_ylabel('Mean Token-Level Confidence', fontsize=12)
    ax1.set_title(f'{dataset_name}: Confidence vs Position', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(positions)

    # Plot 2: Difference between correct and wrong
    ax2 = axes[1]
    diffs = [correct_means[i] - wrong_means[i] for i in range(num_bins)]
    colors = ['green' if not np.isnan(d) and d > 0 else 'red' for d in diffs]
    ax2.bar(positions, diffs, color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Position in Trace (1=beginning, 10=end)', fontsize=12)
    ax2.set_ylabel('Confidence Difference (Correct - Wrong)', fontsize=12)
    ax2.set_title(f'{dataset_name}: Confidence Gap by Position', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(positions)

    plt.tight_layout()
    output_path = output_dir / f'confidence_position_{dataset_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nFigure saved to: {output_path}')

    print()
    print('='*90)
    print('INTERPRETATION:')
    print('='*90)

    if valid_correct_means and valid_wrong_means:
        avg_diff = np.nanmean(diffs)
        if avg_diff > 0.5:
            print(f'\n[+] Correct traces have consistently HIGHER confidence across all positions')
            print(f'    Average difference: {avg_diff:+.2f}')
            print(f'    This suggests correct reasoning maintains high confidence throughout')
        elif avg_diff < -0.5:
            print(f'\n[-] Wrong traces have HIGHER confidence on average')
            print(f'    This is concerning - suggests overconfidence in incorrect reasoning')
        else:
            print(f'\n[=] Confidence levels are SIMILAR between correct and wrong traces')
            print(f'    Average difference: {avg_diff:+.2f}')
            print(f'    This explains why confidence-based weighting has limited impact')

        if abs(correct_end - correct_start) > 1.0:
            print(f'\n[^] Correct traces show significant confidence change across positions')
            print(f'    Position-dependent weighting might be beneficial')
        elif abs(wrong_end - wrong_start) > 1.0:
            print(f'\n[v] Wrong traces show significant confidence change across positions')
            print(f'    Wrong traces become less/more confident towards the end')

    print()
    print(f'Analysis complete for {dataset_name}')
    print()

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence vs position in LLM traces')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing pickle files with trace data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output figure (defaults to data_dir)')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins to divide traces into (default: 10)')

    args = parser.parse_args()

    analyze_confidence_position(args.data_dir, args.output_dir, args.num_bins)


if __name__ == '__main__':
    main()
