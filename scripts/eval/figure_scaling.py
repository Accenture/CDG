#!/usr/bin/env python3
"""
Generate 1x4 figure showing scaling trends and beta ablation.

Figure (a): DeepSeek-R1-8B on AIME 2025 (trace scaling)
Figure (b): Gemma-3-27B on AIME 2024 (trace scaling)
Figure (c): QWQ-32B on BRUNO 2025 (trace scaling)
Figure (d): DeepSeek-8B beta ablation (alpha=0.5)

This figure is designed for a cross-column ICML paper (wide, rectangular format).

Usage:
    python figure_scaling.py                    # Generate figure
    python figure_scaling.py --output fig.pdf  # Custom output path
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Import config for paths
from config import OUTPUT_DIRS, FIGURES_DIR, TRACE_COUNTS, BETA_VALUES


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    return plt


def load_scaling_cache() -> dict:
    """Load cached scaling results."""
    cache_file = OUTPUT_DIRS['compare_voting_methods'] / 'cache.json'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}\nRun compare_voting_methods.py first.")

    with open(cache_file, 'r') as f:
        return json.load(f)


def load_sweep_cache() -> dict:
    """Load cached hyperparameter sweep results."""
    cache_file = OUTPUT_DIRS['cdg_hyperparam_sweep'] / 'sweep_cache.json'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}\nRun cdg_hyperparam_sweep.py first.")

    with open(cache_file, 'r') as f:
        return json.load(f)


def get_scaling_data(cache_data: dict, model: str, dataset: str) -> dict:
    """Extract scaling data for a specific model-dataset pair."""
    results = cache_data.get('results', cache_data)  # Handle both formats

    data = {'majority': [], 'mean_weighted': [], 'top10_tail': [], 'cdg': []}

    for trace_count in TRACE_COUNTS:
        for method in data.keys():
            entry = next((r for r in results
                         if r['model'] == model and r['dataset'] == dataset
                         and r['trace_count'] == trace_count and r['method'] == method), None)
            if entry:
                data[method].append(entry['accuracy'] * 100)
            else:
                data[method].append(None)

    return data


def generate_scaling_figure(output_path: Path = None):
    """Generate the 1x4 scaling figure."""
    plt = setup_matplotlib()
    import matplotlib.gridspec as gridspec

    # Default output path
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / 'trace_scaling_figure.pdf'

    # Load cached data
    scaling_cache = load_scaling_cache()
    sweep_cache = load_sweep_cache()

    # Create figure with GridSpec for clustering: (a,b,c) and (d)
    fig = plt.figure(figsize=(14, 3.8), dpi=300)

    # Outer grid: [abc cluster] [d panel] - abc 20% wider
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[2.9, 1.0], wspace=0.08)

    # Inner grid for a, b, c
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.10)

    axes = [
        fig.add_subplot(inner_gs[0]),
        fig.add_subplot(inner_gs[1]),
        fig.add_subplot(inner_gs[2]),
        fig.add_subplot(outer_gs[1])
    ]

    # Color scheme - professional colors
    colors = {
        'Majority': '#1f77b4',        # Blue
        'DeepConf-Mean': '#ff7f0e',   # Orange
        'DeepConf-Tail': '#2ca02c',   # Green
        'CDG': '#d62728'              # Red (highlight)
    }

    markers = {
        'Majority': 'o',
        'DeepConf-Mean': 's',
        'DeepConf-Tail': '^',
        'CDG': 'D'
    }

    method_map = {
        'majority': ('Majority', 'Majority'),
        'mean_weighted': ('DeepConf-Mean', 'DeepConf-Mean'),
        'top10_tail': ('DeepConf-Tail', 'DeepConf-Tail'),
        'cdg': ('CDG', 'CDG (ours)')
    }

    def plot_scaling(ax, model, dataset, title, ylim, show_ylabel=False, show_legend=False):
        """Plot scaling data for a model-dataset pair."""
        data = get_scaling_data(scaling_cache, model, dataset)

        for method, values in data.items():
            style_key, label = method_map[method]
            linewidth = 2.0 if method == 'cdg' else 1.5
            ax.plot(TRACE_COUNTS, values, label=label,
                   color=colors[style_key], marker=markers[style_key], linewidth=linewidth)

        ax.set_xscale('log', base=2)
        ax.set_xticks(TRACE_COUNTS)
        ax.set_xticklabels([str(t) for t in TRACE_COUNTS])
        ax.set_ylim(ylim)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if show_legend:
            ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
                     handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)
        if show_ylabel:
            ax.set_ylabel('Accuracy (%)', fontweight='bold')

    # Section title for abc cluster (centered over wider abc area)
    fig.text(0.36, 0.90, 'Trace Scaling', ha='center', fontweight='bold', fontsize=14)

    # Figure 1: DeepSeek-R1-8B on AIME 2025
    ax1 = axes[0]
    plot_scaling(ax1, 'deepseek8b', 'aime2025', '(a) DeepSeek-R1-8B on AIME 2025', [78, 95], show_ylabel=True, show_legend=True)
    ax1.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Figure 2: Gemma-3-27B on AIME 2024
    ax2 = axes[1]
    plot_scaling(ax2, 'gemma3_27b', 'aime2024', '(b) Gemma-3-27B on AIME 2024', [38, 62])
    ax2.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Figure 3: QWQ-32B on BRUNO 2025
    ax3 = axes[2]
    plot_scaling(ax3, 'qwq32b', 'bruno2025', '(c) QWQ-32B on BRUNO 2025', [78, 92])
    ax3.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Figure 4: DeepSeek-8B Beta Ablation
    ax4 = axes[3]

    # Filter sweep data for deepseek8b with alpha=0.5, position_pct=10
    sweep_results = sweep_cache.get('results', sweep_cache)
    deepseek_sweep = [
        entry for entry in sweep_results
        if entry['model'] == 'deepseek8b' and entry['alpha'] == 0.5 and entry['position_pct'] == 10
    ]

    # Organize by dataset
    datasets_beta = {
        'aime2024': [],
        'aime2025': [],
        'bruno2025': [],
        'hmmt2025': []
    }

    for dataset in datasets_beta.keys():
        for beta in BETA_VALUES:
            entry = next((e for e in deepseek_sweep if e['dataset'] == dataset and e['beta'] == beta), None)
            if entry:
                datasets_beta[dataset].append(entry['accuracy'] * 100)
            else:
                datasets_beta[dataset].append(None)

    # Colors for each dataset
    dataset_colors = {
        'aime2024': '#9467bd',    # Purple
        'aime2025': '#e377c2',    # Pink
        'bruno2025': '#17becf',   # Cyan
        'hmmt2025': '#bcbd22'     # Yellow-green
    }

    dataset_labels = {
        'aime2024': 'AIME2024',
        'aime2025': 'AIME2025',
        'bruno2025': 'BRUNO2025',
        'hmmt2025': 'HMMT2025'
    }

    # Plot each dataset
    for dataset, accuracies in datasets_beta.items():
        ax4.plot(BETA_VALUES, accuracies,
                 color=dataset_colors[dataset], marker='o', linewidth=1.5, markersize=5,
                 label=dataset_labels[dataset])

    # Add vertical line at beta=10 (optimal for DeepSeek)
    ax4.axvline(x=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Section title for d panel (centered over narrower d area)
    fig.text(0.86, 0.90, r'$\beta$ Search', ha='center', fontweight='bold', fontsize=14)

    # Vertical dashed line separating abc cluster from d panel
    from matplotlib.lines import Line2D
    vline = Line2D([0.73, 0.73], [0.05, 0.92], transform=fig.transFigure,
                   color='gray', linewidth=1.2, linestyle='--')
    fig.add_artist(vline)

    ax4.set_xlabel(r'CDG weight $\beta$', fontweight='bold')
    ax4.set_title(r'(d) DeepSeek-R1-8B ($\alpha$=0.5)', fontweight='bold')
    ax4.set_xticks(BETA_VALUES)
    ax4.set_xticklabels([str(b) for b in BETA_VALUES])
    ax4.set_ylim([60, 100])
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.legend(loc='lower center', ncol=2, framealpha=0.9, edgecolor='gray',
              handlelength=1.2, handletextpad=0.3, borderpad=0.3, labelspacing=0.2, columnspacing=0.8)

    # Adjust layout - room for section titles at top
    plt.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.14)

    # Save figure
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")

    # Also save as PNG for preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"PNG preview saved to: {png_path}")

    plt.close(fig)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate scaling figure for paper')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for figure (default: results/figures/trace_scaling_figure.pdf)')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_scaling_figure(output_path)
