#!/usr/bin/env python3
"""
Generate 1x4 histogram figures showing correct vs wrong answer distributions.

Figure (a): Mean Confidence (per trace)
Figure (b): Tail Confidence (per trace)
Figure (c): CDG Score (per answer)
Figure (d): Confidence Gradient (per trace) - Multi-model comparison

Generates figures for all model × dataset combinations.

Usage:
    python figure_histogram.py                           # Generate all figures
    python figure_histogram.py --model deepseek8b        # Single model, all datasets
    python figure_histogram.py --dataset aime2024        # All models, single dataset
    python figure_histogram.py --model deepseek8b --dataset aime2024  # Single figure
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Import config for paths
from config import OUTPUT_DIRS, FIGURES_DIR, MODEL_NAMES, DATASET_NAMES


# =============================================================================
# Constants
# =============================================================================

MODEL_KEYS = ['deepseek8b', 'gptoss20b', 'gemma3_27b', 'qwq32b']
MODEL_DISPLAY = ['DeepSeek-8B', 'GPT-OSS-20B', 'Gemma-27B', 'QWQ-32B']
DATASET_KEYS = ['aime2024', 'aime2025', 'bruno2025', 'hmmt2025']

COLOR_CORRECT = '#2ca02c'  # Green
COLOR_WRONG = '#d62728'    # Red
MODEL_COLORS = ['#9467bd', '#e377c2', '#17becf', '#bcbd22']


# =============================================================================
# Setup
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    return plt


def load_histogram_cache(model: str, dataset: str) -> dict:
    """Load cached histogram metrics for a model-dataset pair."""
    cache_dir = OUTPUT_DIRS['plot_distributions'] / 'cache'
    cache_file = cache_dir / f'{model}_{dataset}_metrics.json'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}\nRun plot_distributions.py first.")

    with open(cache_file, 'r') as f:
        return json.load(f)


# =============================================================================
# Panel Drawing Functions
# =============================================================================

def plot_histogram_panel(ax, data_correct, data_wrong, xlabel, ylabel=None,
                         legend_loc='upper left', use_sci_notation=False):
    """Plot a single histogram panel comparing correct vs wrong distributions."""
    ax.hist(data_correct, bins=50, alpha=0.6, color=COLOR_CORRECT,
            label='Correct', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(data_wrong, bins=50, alpha=0.6, color=COLOR_WRONG,
            label='Wrong', density=True, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')

    ax.legend(loc=legend_loc, framealpha=0.9, edgecolor='gray', fontsize=10,
              handlelength=1.0, handletextpad=0.3, borderpad=0.3)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim([0, None])

    if use_sci_notation:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(9)


def plot_gradient_panel(ax, dataset: str):
    """Plot the gradient direction analysis panel (d) for all 4 models on a given dataset."""
    # Load data for all 4 models
    models_data = {}
    for model_key in MODEL_KEYS:
        try:
            models_data[model_key] = load_histogram_cache(model_key, dataset)
        except FileNotFoundError:
            print(f"Warning: Cache not found for {model_key}/{dataset}, skipping")

    # Calculate percentages for each model
    pct_correct_positive = []
    pct_wrong_negative = []
    available_models = []

    for model_key, display_name in zip(MODEL_KEYS, MODEL_DISPLAY):
        if model_key not in models_data:
            continue
        available_models.append(display_name)
        correct_grad = np.array(models_data[model_key]['correct']['gradient'])
        wrong_grad = np.array(models_data[model_key]['wrong']['gradient'])

        pct_correct_positive.append((correct_grad > 0).sum() / len(correct_grad) * 100)
        pct_wrong_negative.append((wrong_grad < 0).sum() / len(wrong_grad) * 100)

    # Create grouped bar chart
    x = np.arange(2)  # 2 groups: correct positive, wrong negative
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width][:len(available_models)]

    for i, model_name in enumerate(available_models):
        values = [pct_correct_positive[i], pct_wrong_negative[i]]
        bars = ax.bar(x + offsets[i], values, width, label=model_name,
                      color=MODEL_COLORS[i], alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add percentage labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel(r'(d) Direction of $\nabla C_{\ell}$ (4 models)', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r'Correct ($\nabla C_{\ell} > 0$)', r'Wrong ($\nabla C_{\ell} < 0$)'])
    ax.set_ylim([0, 115])
    ax.legend(loc='upper right', ncol=2, framealpha=0.9,
              edgecolor='gray', fontsize=10, handlelength=1.0, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.2, columnspacing=0.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_single_histogram_figure(model: str, dataset: str, output_dir: Path = None) -> Path:
    """Generate a single 1x4 histogram figure for a given model and dataset."""
    plt = setup_matplotlib()
    import matplotlib.gridspec as gridspec

    # Default output directory
    if output_dir is None:
        output_dir = FIGURES_DIR / 'histograms'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for this model-dataset pair
    try:
        data = load_histogram_cache(model, dataset)
    except FileNotFoundError as e:
        print(f"Skipping {model}/{dataset}: {e}")
        return None

    correct = data['correct']
    wrong = data['wrong']
    meta = data['meta']

    # Create figure with two visual clusters: (a,b,c) and (d)
    fig = plt.figure(figsize=(14, 3.45), dpi=300)

    # Outer grid: [abc cluster] [gap] [d panel]
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1.2], wspace=0.12)

    # Inner grid for a, b, c
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.15)

    ax1 = fig.add_subplot(inner_gs[0])
    ax2 = fig.add_subplot(inner_gs[1])
    ax3 = fig.add_subplot(inner_gs[2])
    ax4 = fig.add_subplot(outer_gs[1])

    # Panel (a): Mean Confidence
    plot_histogram_panel(ax1, correct['mean_conf'], wrong['mean_conf'],
                         xlabel='(a) Mean Confidence (per trace)',
                         ylabel='Density', legend_loc='upper left')

    # Panel (b): Tail Confidence
    plot_histogram_panel(ax2, correct['tail_conf'], wrong['tail_conf'],
                         xlabel='(b) Tail Confidence (per trace)',
                         ylabel=None, legend_loc='upper right')

    # Panel (c): CDG Score
    plot_histogram_panel(ax3, correct['cdg_score'], wrong['cdg_score'],
                         xlabel='(c) CDG Score (per answer)',
                         ylabel=None, legend_loc='upper right', use_sci_notation=True)

    # Panel (d): Gradient Direction Analysis (all models on this dataset)
    plot_gradient_panel(ax4, dataset)

    # Minimal margins
    plt.subplots_adjust(left=0.045, right=0.99, top=0.88, bottom=0.22)

    # Save figure
    model_display = MODEL_NAMES.get(model, model)
    dataset_display = DATASET_NAMES.get(dataset, dataset)

    output_path = output_dir / f'histogram_{model}_{dataset}.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)

    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Generated: {model_display} × {dataset_display}")
    print(f"  PDF: {output_path}")
    print(f"  PNG: {png_path}")
    print(f"  Questions: {meta['n_questions']}, Correct: {meta['n_correct_answers']}, Wrong: {meta['n_wrong_answers']}")

    plt.close(fig)
    return output_path


def generate_all_histogram_figures(models: list = None, datasets: list = None, output_dir: Path = None):
    """Generate histogram figures for all specified model × dataset combinations."""
    if models is None:
        models = MODEL_KEYS
    if datasets is None:
        datasets = DATASET_KEYS

    print(f"Generating {len(models)} × {len(datasets)} = {len(models) * len(datasets)} figures...")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print()

    generated = []
    for model in models:
        for dataset in datasets:
            result = generate_single_histogram_figure(model, dataset, output_dir)
            if result:
                generated.append(result)
            print()

    print(f"Done! Generated {len(generated)} figures.")
    return generated


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate histogram figures for paper')
    parser.add_argument('--model', '-m', type=str, default=None,
                        choices=MODEL_KEYS,
                        help='Specific model (default: all models)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        choices=DATASET_KEYS,
                        help='Specific dataset (default: all datasets)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: results/figures/histograms)')
    args = parser.parse_args()

    models = [args.model] if args.model else None
    datasets = [args.dataset] if args.dataset else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    generate_all_histogram_figures(models, datasets, output_dir)
