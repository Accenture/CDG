#!/usr/bin/env python3
"""
Histogram Figures: Correct vs Wrong Answer Distributions.

Generates overlapping histograms comparing different confidence metrics
for traces that lead to correct vs wrong answers.

Metrics compared:
1. Majority vote (count-based, shown as bar chart)
2. Mean confidence across all tokens (DeepConf "mean conf")
3. Tail confidence (DeepConf "tail conf")
4. CDG score (with fixed parameters)
5. Confidence gradient: Conf_tail - Conf_start (10%, the beta term)

Usage:
    # Generate histograms for default model/dataset
    python scripts/eval/fig_histogram.py

    # Specify model and dataset
    python scripts/eval/fig_histogram.py --model qwen32b --dataset aime2025

    # Custom output directory
    python scripts/eval/fig_histogram.py --output-dir results/fig_histogram

    # Custom CDG parameters
    python scripts/eval/fig_histogram.py --alpha 0.5 --beta 20
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import (
    load_run_results, compute_gradient, compute_tail_conf,
    equal_func, DEFAULT_PARAMS
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Error: Required dependencies missing: {e}")
    print("Install with: pip install matplotlib numpy scipy")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE
OUTPUT_BASE = Path(__file__).parent.parent.parent / "results" / "fig_histogram"

# Default model/dataset for paper figure
DEFAULT_MODEL = 'qwen32b'
DEFAULT_DATASET = 'aime2025'

# Models and datasets
MODELS = ['qwen32b', 'deepseek8b', 'gemma3_27b', 'gptoss20b']
DATASETS = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025']

# Pretty names
MODEL_NAMES = {
    'qwen32b': 'Qwen3 32B',
    'deepseek8b': 'DeepSeek R1 8B',
    'gemma3_27b': 'Gemma 3 27B',
    'gptoss20b': 'GPT-OSS 20B',
}

DATASET_NAMES = {
    'aime2025': 'AIME 2025',
    'aime2024': 'AIME 2024',
    'hmmt2025': 'HMMT 2025',
    'bruno2025': 'Bruno 2025',
}


# =============================================================================
# Data Extraction
# =============================================================================

def extract_trace_metrics(results_dir: str, model: str, dataset: str,
                          alpha: float = 0.5, beta: float = 10,
                          position_pct: int = 10) -> dict:
    """
    Extract per-trace metrics for correct vs wrong answers.

    Returns:
        {
            'correct': {
                'counts': [n1, n2, ...],  # count per answer group
                'mean_conf': [c1, c2, ...],
                'tail_conf': [t1, t2, ...],
                'gradient': [g1, g2, ...],
                'cdg_score': [s1, s2, ...],
            },
            'wrong': { ... same structure ... }
        }
    """
    # Find matching run
    base = Path(results_dir)
    run_path = None

    for item in base.iterdir():
        if item.is_dir() and f'{model}_{dataset}_' in item.name:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                run_path = str(item)
                break

    if not run_path:
        print(f"No run found for {model}_{dataset}")
        return None

    print(f"Loading data from: {run_path}")
    results_by_qid = load_run_results(run_path, use_threads=True)

    if not results_by_qid:
        return None

    # Extract metrics per trace
    metrics = {
        'correct': {
            'counts': [],
            'mean_conf': [],
            'tail_conf': [],
            'gradient': [],
            'cdg_score': [],
        },
        'wrong': {
            'counts': [],
            'mean_conf': [],
            'tail_conf': [],
            'gradient': [],
            'cdg_score': [],
        },
        'meta': {
            'model': model,
            'dataset': dataset,
            'alpha': alpha,
            'beta': beta,
            'position_pct': position_pct,
            'n_questions': 0,
            'n_correct_traces': 0,
            'n_wrong_traces': 0,
        }
    }

    for qid in sorted(results_by_qid.keys()):
        results_list = results_by_qid[qid]

        all_traces = []
        ground_truth = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        metrics['meta']['n_questions'] += 1

        # Group traces by answer
        answer_groups = defaultdict(list)
        for trace in all_traces:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is not None and len(confs) >= 10:
                mean_conf = float(np.mean(confs))
                gradient = compute_gradient(confs, position_pct)
                tail_conf = compute_tail_conf(confs, window=2048)

                answer_groups[answer].append({
                    'mean_conf': mean_conf,
                    'gradient': gradient,
                    'tail_conf': tail_conf,
                })

        # Compute metrics per answer group
        for answer, traces_list in answer_groups.items():
            count = len(traces_list)

            # Mean of trace-level metrics
            mean_confs = [t['mean_conf'] for t in traces_list]
            gradients = [t['gradient'] for t in traces_list]
            tail_confs = [t['tail_conf'] for t in traces_list]

            # CDG score for this answer group
            trace_scores = [t['mean_conf'] + beta * t['gradient'] for t in traces_list]
            cdg_score = (count ** alpha) * np.mean(trace_scores)

            # Check if answer is correct
            try:
                is_correct = equal_func(str(answer), str(ground_truth))
            except:
                is_correct = str(answer) == str(ground_truth)

            key = 'correct' if is_correct else 'wrong'

            # Store metrics for each trace in the group
            for t in traces_list:
                metrics[key]['counts'].append(count)
                metrics[key]['mean_conf'].append(t['mean_conf'])
                metrics[key]['tail_conf'].append(t['tail_conf'])
                metrics[key]['gradient'].append(t['gradient'])
                # CDG score is per-group, but we assign to each trace for histogram
                metrics[key]['cdg_score'].append(cdg_score / count)  # Normalize by count for per-trace

            if is_correct:
                metrics['meta']['n_correct_traces'] += len(traces_list)
            else:
                metrics['meta']['n_wrong_traces'] += len(traces_list)

    return metrics


# =============================================================================
# Figure Generation
# =============================================================================

def create_histogram_subplot(ax, correct_data: list, wrong_data: list,
                             title: str, xlabel: str, n_bins: int = 50,
                             show_legend: bool = True):
    """Create a single histogram subplot comparing correct vs wrong distributions."""

    # Determine bin range
    all_data = correct_data + wrong_data
    if not all_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    vmin, vmax = np.percentile(all_data, [1, 99])  # Clip outliers

    bins = np.linspace(vmin, vmax, n_bins + 1)

    # Plot histograms
    ax.hist(correct_data, bins=bins, alpha=0.6, label='Correct', color='#2ca02c',
            density=True, edgecolor='darkgreen', linewidth=0.5)
    ax.hist(wrong_data, bins=bins, alpha=0.6, label='Wrong', color='#d62728',
            density=True, edgecolor='darkred', linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=11)

    if show_legend:
        ax.legend(loc='upper right', fontsize=9)

    # Add statistical summary
    if len(correct_data) > 1 and len(wrong_data) > 1:
        # t-test
        t_stat, p_val = stats.ttest_ind(correct_data, wrong_data)

        # Means
        correct_mean = np.mean(correct_data)
        wrong_mean = np.mean(wrong_data)

        stats_text = f'Correct: {correct_mean:.3f}\nWrong: {wrong_mean:.3f}\np={p_val:.2e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_count_histogram(ax, correct_counts: list, wrong_counts: list,
                           title: str = 'Answer Count Distribution'):
    """Create histogram for majority vote counts."""
    if not correct_counts and not wrong_counts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # Count frequencies
    correct_counter = Counter(correct_counts)
    wrong_counter = Counter(wrong_counts)

    all_counts = set(correct_counts + wrong_counts)
    x_vals = sorted(all_counts)

    correct_freqs = [correct_counter.get(x, 0) / len(correct_counts) if correct_counts else 0 for x in x_vals]
    wrong_freqs = [wrong_counter.get(x, 0) / len(wrong_counts) if wrong_counts else 0 for x in x_vals]

    width = 0.35
    x_pos = np.arange(len(x_vals))

    ax.bar(x_pos - width/2, correct_freqs, width, label='Correct', color='#2ca02c', alpha=0.7)
    ax.bar(x_pos + width/2, wrong_freqs, width, label='Wrong', color='#d62728', alpha=0.7)

    ax.set_xlabel('Answer Count (Majority Vote)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.legend(loc='upper right', fontsize=9)

    # Log scale for x if many values
    if len(x_vals) > 20:
        ax.set_xticks(x_pos[::5])
        ax.set_xticklabels([str(x) for x in x_vals[::5]])


def create_combined_figure(metrics: dict, output_dir: Path, n_bins: int = 50):
    """Create combined figure with all 5 metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    model = metrics['meta']['model']
    dataset = metrics['meta']['dataset']
    model_name = MODEL_NAMES.get(model, model)
    dataset_name = DATASET_NAMES.get(dataset, dataset)

    # 1. Majority vote (count distribution)
    create_count_histogram(
        axes[0],
        metrics['correct']['counts'],
        metrics['wrong']['counts'],
        title='(a) Majority Vote (Answer Count)'
    )

    # 2. Mean confidence
    create_histogram_subplot(
        axes[1],
        metrics['correct']['mean_conf'],
        metrics['wrong']['mean_conf'],
        title='(b) Mean Confidence',
        xlabel='Mean Token Confidence',
        n_bins=n_bins
    )

    # 3. Tail confidence
    create_histogram_subplot(
        axes[2],
        metrics['correct']['tail_conf'],
        metrics['wrong']['tail_conf'],
        title='(c) Tail Confidence (DeepConf)',
        xlabel='Tail Token Confidence',
        n_bins=n_bins
    )

    # 4. CDG score
    create_histogram_subplot(
        axes[3],
        metrics['correct']['cdg_score'],
        metrics['wrong']['cdg_score'],
        title='(d) CDG Score (Ours)',
        xlabel='CDG Score (per trace)',
        n_bins=n_bins
    )

    # 5. Gradient (Conf_tail - Conf_start)
    create_histogram_subplot(
        axes[4],
        metrics['correct']['gradient'],
        metrics['wrong']['gradient'],
        title=r'(e) Confidence Gradient ($\beta$ term)',
        xlabel='Gradient: Tail 10% - Head 10%',
        n_bins=n_bins
    )

    # Hide 6th subplot, add summary
    axes[5].axis('off')
    summary_text = f"""Summary Statistics

Model: {model_name}
Dataset: {dataset_name}

Questions: {metrics['meta']['n_questions']}
Correct traces: {metrics['meta']['n_correct_traces']}
Wrong traces: {metrics['meta']['n_wrong_traces']}

CDG Parameters:
  alpha = {metrics['meta']['alpha']}
  beta = {metrics['meta']['beta']}
  position_pct = {metrics['meta']['position_pct']}%
"""
    axes[5].text(0.1, 0.9, summary_text, transform=axes[5].transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle(f'Correct vs Wrong Answer Distributions: {model_name} on {dataset_name}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    filename = f'histogram_{model}_{dataset}'
    output_path = output_dir / f'{filename}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def create_individual_histograms(metrics: dict, output_dir: Path, n_bins: int = 50):
    """Create individual histogram figures for each metric."""
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(parents=True, exist_ok=True)

    model = metrics['meta']['model']
    dataset = metrics['meta']['dataset']

    metric_configs = [
        ('majority_vote', 'counts', 'Answer Count', 'count'),
        ('mean_conf', 'mean_conf', 'Mean Token Confidence', 'hist'),
        ('tail_conf', 'tail_conf', 'Tail Token Confidence', 'hist'),
        ('cdg_score', 'cdg_score', 'CDG Score', 'hist'),
        ('gradient', 'gradient', 'Confidence Gradient (Tail 10% - Head 10%)', 'hist'),
    ]

    for name, key, xlabel, plot_type in metric_configs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if plot_type == 'count':
            create_count_histogram(
                ax,
                metrics['correct'][key],
                metrics['wrong'][key],
                title=f'Correct vs Wrong: {xlabel}'
            )
        else:
            create_histogram_subplot(
                ax,
                metrics['correct'][key],
                metrics['wrong'][key],
                title=f'Correct vs Wrong: {xlabel}',
                xlabel=xlabel,
                n_bins=n_bins
            )

        plt.tight_layout()

        output_path = individual_dir / f'{name}_{model}_{dataset}.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved individual figures to {individual_dir}")


def save_numerical_data(metrics: dict, output_dir: Path):
    """Save numerical data to JSON."""
    # Compute summary statistics
    data = {
        'timestamp': datetime.now().isoformat(),
        'meta': metrics['meta'],
        'statistics': {}
    }

    for category in ['correct', 'wrong']:
        data['statistics'][category] = {}
        for metric in ['counts', 'mean_conf', 'tail_conf', 'gradient', 'cdg_score']:
            values = metrics[category][metric]
            if values:
                data['statistics'][category][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n': len(values),
                    'percentiles': {
                        '25': float(np.percentile(values, 25)),
                        '75': float(np.percentile(values, 75)),
                    }
                }

    # Add statistical tests
    data['tests'] = {}
    for metric in ['mean_conf', 'tail_conf', 'gradient', 'cdg_score']:
        correct = metrics['correct'][metric]
        wrong = metrics['wrong'][metric]
        if len(correct) > 1 and len(wrong) > 1:
            t_stat, p_val = stats.ttest_ind(correct, wrong)
            data['tests'][metric] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05),
            }

    output_path = output_dir / f"histogram_data_{metrics['meta']['model']}_{metrics['meta']['dataset']}.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved numerical data to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate histogram comparison figures for correct vs wrong answers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Base directory for experiment results')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_BASE),
                        help='Output directory for figures')

    # Model/dataset selection
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=MODELS,
                        help=f'Model to analyze (default: {DEFAULT_MODEL})')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, choices=DATASETS,
                        help=f'Dataset to analyze (default: {DEFAULT_DATASET})')

    # CDG parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='CDG alpha parameter (default: 0.5)')
    parser.add_argument('--beta', type=float, default=10,
                        help='CDG beta parameter (default: 10)')
    parser.add_argument('--position-pct', type=int, default=10,
                        help='Percentile for gradient calculation (default: 10)')

    # Figure options
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip individual figures')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures')

    # Generate for all combinations
    parser.add_argument('--all', action='store_true',
                        help='Generate for all model-dataset combinations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Histogram Figures: Correct vs Wrong Distributions")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    if args.all:
        # Generate for all combinations
        for model in MODELS:
            for dataset in DATASETS:
                print(f"\nProcessing {model} - {dataset}...")

                metrics = extract_trace_metrics(
                    args.results_dir, model, dataset,
                    alpha=args.alpha, beta=args.beta, position_pct=args.position_pct
                )

                if metrics:
                    create_combined_figure(metrics, output_dir, n_bins=args.bins)
                    save_numerical_data(metrics, output_dir)

                    if not args.no_individual:
                        create_individual_histograms(metrics, output_dir, n_bins=args.bins)
    else:
        # Single model-dataset
        print(f"Processing {args.model} - {args.dataset}...")

        metrics = extract_trace_metrics(
            args.results_dir, args.model, args.dataset,
            alpha=args.alpha, beta=args.beta, position_pct=args.position_pct
        )

        if metrics:
            create_combined_figure(metrics, output_dir, n_bins=args.bins)
            save_numerical_data(metrics, output_dir)

            if not args.no_individual:
                create_individual_histograms(metrics, output_dir, n_bins=args.bins)

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
