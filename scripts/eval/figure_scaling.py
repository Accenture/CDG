#!/usr/bin/env python3
"""
Generate 1x4 figure showing scaling trends and beta ablation.

Figure (a): DeepSeek-R1-8B on AIME 2025 (trace scaling)
Figure (b): Gemma-3-27B on AIME 2024 (trace scaling)
Figure (c): QWQ-32B on BRUMO 2025 (trace scaling)
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

# Theoretical r values per model (for optimal beta range calculation)
MODEL_R_VALUES = {
    'deepseek8b': 7.87,
    'gptoss20b': 8.70,
    'gemma3_27b': 6.64,
    'qwq32b': 4.39,
}

def get_optimal_beta_range(model: str) -> tuple:
    """Compute optimal beta range as [r/2, 1.2*r] from theoretical r value."""
    r = MODEL_R_VALUES[model]
    return (r / 2, 1.5 * r)


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    return plt


def load_scaling_cache() -> dict:
    """Load cached scaling results."""
    cache_file = OUTPUT_DIRS['exp_voting_methods'] / 'cache.json'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}\nRun exp_voting_methods.py first.")

    with open(cache_file, 'r') as f:
        return json.load(f)


def load_sweep_cache() -> dict:
    """Load cached hyperparameter sweep results."""
    cache_file = OUTPUT_DIRS['exp_cdg_sweep'] / 'sweep_cache.json'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}\nRun exp_cdg_sweep.py first.")

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

    # Outer grid: [abc cluster] [d panel] - d ~12% wider to fit legend
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[2.78, 1.12], wspace=0.08)

    # Inner grid for a, b, c
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.12)

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

    def plot_scaling(ax, model, dataset, ylim, show_ylabel=False, show_legend=False):
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
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if show_legend:
            ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
                     handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)
        if show_ylabel:
            ax.set_ylabel('Accuracy (%)', fontweight='bold')

    # Section title for abc cluster (centered over wider abc area)
    fig.text(0.365, 0.92, 'Trace Scaling', ha='center', fontweight='bold', fontsize=16)

    # Figure 1: DeepSeek-R1-8B on AIME 2025
    ax1 = axes[0]
    plot_scaling(ax1, 'deepseek8b', 'aime2025', [78, 95], show_ylabel=True, show_legend=True)
    ax1.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Figure 2: Gemma-3-27B on AIME 2024
    ax2 = axes[1]
    plot_scaling(ax2, 'gemma3_27b', 'aime2024', [38, 62])
    ax2.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Figure 3: QWQ-32B on BRUMO 2025
    ax3 = axes[2]
    plot_scaling(ax3, 'qwq32b', 'brumo2025', [78, 92])
    ax3.set_xlabel(r'Number of Traces $L$', fontweight='bold')

    # Subtitles at bottom (below x-axis labels)
    fig.text(0.15, 0.01, '(a) DeepSeek-R1-8B on AIME 2025', ha='center', fontweight='bold', fontsize=13)
    fig.text(0.365, 0.01, '(b) Gemma-3-27B on AIME 2024', ha='center', fontweight='bold', fontsize=13)
    fig.text(0.58, 0.01, '(c) QWQ-32B on BRUMO 2025', ha='center', fontweight='bold', fontsize=13)

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
        'brumo2025': [],
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
        'brumo2025': '#17becf',   # Cyan
        'hmmt2025': '#bcbd22'     # Yellow-green
    }

    dataset_labels = {
        'aime2024': 'AIME2024',
        'aime2025': 'AIME2025',
        'brumo2025': 'BRUMO2025',
        'hmmt2025': 'HMMT2025'
    }

    # Add gradient band showing theoretical optimal beta range for deepseek8b
    # Darkest in center, fading towards edges using non-overlapping slices
    beta_min, beta_max = get_optimal_beta_range('deepseek8b')
    beta_center = (beta_min + beta_max) / 2
    half_width = (beta_max - beta_min) / 2

    # Create gradient with non-overlapping slices
    n_slices = 50
    max_alpha = 0.20
    for i in range(n_slices):
        # Position from left edge to right edge
        left = beta_min + (i / n_slices) * (beta_max - beta_min)
        right = beta_min + ((i + 1) / n_slices) * (beta_max - beta_min)
        # Distance from center (0 at center, 1 at edges)
        slice_center = (left + right) / 2
        dist_from_center = abs(slice_center - beta_center) / half_width
        # Alpha is highest at center, fades with power 3
        alpha = max_alpha * (1 - dist_from_center ** 3)
        ax4.axvspan(left, right, alpha=alpha, color='#2ca02c', linewidth=0)

    # Plot each dataset
    for dataset, accuracies in datasets_beta.items():
        ax4.plot(BETA_VALUES, accuracies,
                 color=dataset_colors[dataset], marker='o', linewidth=1.5, markersize=5,
                 label=dataset_labels[dataset])

    # Section title for d panel (centered over narrower d area)
    fig.text(0.855, 0.92, r'$\beta$ Search', ha='center', fontweight='bold', fontsize=16)

    # Vertical dashed line separating abc cluster from d panel
    from matplotlib.lines import Line2D
    vline = Line2D([0.7025, 0.7025], [0.05, 0.92], transform=fig.transFigure,
                   color='gray', linewidth=2, linestyle='--')
    fig.add_artist(vline)

    ax4.set_xlabel(r'CDG weight $\beta$', fontweight='bold')

    # Subtitle at bottom for panel d
    fig.text(0.855, 0.01, r'(d) DeepSeek-R1-8B ($\alpha$=0.5)', ha='center', fontweight='bold', fontsize=13)
    ax4.set_xticks(BETA_VALUES)
    ax4.set_xticklabels([str(b) for b in BETA_VALUES])
    ax4.set_ylim([60, 100])
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    # Add custom patch for optimal range legend
    from matplotlib.patches import Patch
    handles, labels = ax4.get_legend_handles_labels()
    optimal_patch = Patch(facecolor='#2ca02c', alpha=0.20, label='Optimal range')
    handles.append(optimal_patch)
    labels.append('Optimal range')
    ax4.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.9, edgecolor='gray',
              handlelength=1.2, handletextpad=0.3, borderpad=0.3, labelspacing=0.2, columnspacing=0.8)

    # Adjust layout - room for section titles at top and subtitles at bottom
    plt.subplots_adjust(left=0.045, right=0.99, top=0.88, bottom=0.20)

    # Save figure
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")

    # # Also save as PNG for preview
    # png_path = output_path.with_suffix('.png')
    # plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    # print(f"PNG preview saved to: {png_path}")

    plt.close(fig)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate scaling figure for paper')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for figure (default: results/figures/trace_scaling_figure.pdf)')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_scaling_figure(output_path)
