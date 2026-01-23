#!/usr/bin/env python3
"""
Generate appendix figures extending figure_scaling.py.

Figure 3 abc full: 4x4 grid of trace scaling (4 models x 4 datasets)
Figure 3 d full: 1x4 beta ablation (one panel per model, 4 datasets each)

Usage:
    python figure_scaling_appendix.py                # Generate all figures
    python figure_scaling_appendix.py --type scaling # Only 4x4 scaling grid
    python figure_scaling_appendix.py --type beta    # Only beta ablation
"""

import argparse
import json
from pathlib import Path

from config import OUTPUT_DIRS, APPENDIX_DIR, TRACE_COUNTS, BETA_VALUES

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_ORDER = ['deepseek8b', 'gemma3_27b', 'qwq32b', 'gptoss20b']
DATASET_ORDER = ['aime2024', 'aime2025', 'brumo2025', 'hmmt2025']

MODEL_DISPLAY = {
    'deepseek8b': 'DeepSeek-R1-8B',
    'gemma3_27b': 'Gemma-3-27B',
    'gptoss20b': 'gpt-oss-20B',
    'qwq32b': 'QWQ-32B',
}

DATASET_DISPLAY = {
    'aime2024': 'AIME 2024',
    'aime2025': 'AIME 2025',
    'brumo2025': 'BRUMO 2025',
    'hmmt2025': 'HMMT 2025',
}

# Color scheme for scaling methods
SCALING_COLORS = {
    'Majority': '#1f77b4',
    'DeepConf-Mean': '#ff7f0e',
    'DeepConf-Tail': '#2ca02c',
    'CDG': '#d62728',
}

SCALING_MARKERS = {
    'Majority': 'o',
    'DeepConf-Mean': 's',
    'DeepConf-Tail': '^',
    'CDG': 'D',
}

METHOD_MAP = {
    'majority': ('Majority', 'Majority'),
    'mean_weighted': ('DeepConf-Mean', 'DeepConf-Mean'),
    'top10_tail': ('DeepConf-Tail', 'DeepConf-Tail'),
    'cdg': ('CDG', 'CDG (ours)'),
}

# Colors for datasets in beta ablation
DATASET_COLORS = {
    'aime2024': '#9467bd',
    'aime2025': '#e377c2',
    'brumo2025': '#17becf',
    'hmmt2025': '#bcbd22',
}

# Theoretical r values per model (for optimal beta range calculation)
MODEL_R_VALUES = {
    'deepseek8b': 7.87,
    'gptoss20b': 8.70,
    'gemma3_27b': 6.64,
    'qwq32b': 4.39,
}


# ============================================================================
# MATPLOTLIB SETUP
# ============================================================================

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
    plt.rcParams['lines.markersize'] = 5
    return plt


# ============================================================================
# DATA LOADING
# ============================================================================

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
    results = cache_data.get('results', cache_data)
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


def get_optimal_beta_range(model: str) -> tuple:
    """Compute optimal beta range as [r/2, 1.5*r] from theoretical r value."""
    r = MODEL_R_VALUES[model]
    return (r / 2, 1.5 * r)


# ============================================================================
# PANEL PLOTTING FUNCTIONS
# ============================================================================

def plot_scaling_panel(ax, cache_data: dict, model: str, dataset: str,
                       ylim=None, show_ylabel=False, show_xlabel=True, show_legend=False):
    """Plot a single scaling panel (trace count vs accuracy)."""
    from matplotlib.ticker import MaxNLocator

    data = get_scaling_data(cache_data, model, dataset)

    for method, values in data.items():
        style_key, label = METHOD_MAP[method]
        linewidth = 2.0 if method == 'cdg' else 1.5
        ax.plot(TRACE_COUNTS, values, label=label,
                color=SCALING_COLORS[style_key],
                marker=SCALING_MARKERS[style_key],
                linewidth=linewidth)

    ax.set_xscale('log', base=2)
    ax.set_xticks(TRACE_COUNTS)
    ax.set_xticklabels([str(t) for t in TRACE_COUNTS])

    if ylim:
        ax.set_ylim(ylim)

    # Use integer y-axis labels
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if show_xlabel:
        ax.set_xlabel(r'Number of Traces $L$')

    if show_ylabel:
        ax.set_ylabel('Accuracy (%)')

    if show_legend:
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
                  handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)


def plot_beta_panel(ax, sweep_cache: dict, model: str, alpha: float = 0.5,
                    position_pct: int = 10, show_ylabel=False, show_legend=False):
    """Plot beta ablation panel for one model (4 datasets)."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    sweep_results = sweep_cache.get('results', sweep_cache)

    # Filter sweep data for this model with alpha and position_pct
    model_sweep = [
        entry for entry in sweep_results
        if entry['model'] == model and entry['alpha'] == alpha and entry['position_pct'] == position_pct
    ]

    # Organize by dataset
    datasets_beta = {ds: [] for ds in DATASET_ORDER}

    for dataset in DATASET_ORDER:
        for beta in BETA_VALUES:
            entry = next((e for e in model_sweep if e['dataset'] == dataset and e['beta'] == beta), None)
            if entry:
                datasets_beta[dataset].append(entry['accuracy'] * 100)
            else:
                datasets_beta[dataset].append(None)

    # Add gradient band showing theoretical optimal beta range
    beta_min, beta_max = get_optimal_beta_range(model)
    beta_center = (beta_min + beta_max) / 2
    half_width = (beta_max - beta_min) / 2

    n_slices = 50
    max_alpha = 0.20
    for i in range(n_slices):
        left = beta_min + (i / n_slices) * (beta_max - beta_min)
        right = beta_min + ((i + 1) / n_slices) * (beta_max - beta_min)
        slice_center = (left + right) / 2
        dist_from_center = abs(slice_center - beta_center) / half_width
        alpha_val = max_alpha * (1 - dist_from_center ** 3)
        ax.axvspan(left, right, alpha=alpha_val, color='#2ca02c', linewidth=0)

    # Plot each dataset
    for dataset in DATASET_ORDER:
        accuracies = datasets_beta[dataset]
        ax.plot(BETA_VALUES, accuracies,
                color=DATASET_COLORS[dataset],
                marker='o', linewidth=1.5, markersize=5,
                label=DATASET_DISPLAY[dataset])

    ax.set_xlabel(r'CDG weight $\beta$')
    ax.set_xticks(BETA_VALUES)
    ax.set_xticklabels([str(b) for b in BETA_VALUES])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Use integer y-axis labels
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if show_ylabel:
        ax.set_ylabel('Accuracy (%)')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        optimal_patch = Patch(facecolor='#2ca02c', alpha=0.20, label='Optimal range')
        handles.append(optimal_patch)
        labels.append('Optimal range')
        ax.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.9, edgecolor='gray',
                  handlelength=1.2, handletextpad=0.3, borderpad=0.3, labelspacing=0.2, columnspacing=0.8)


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def generate_scaling_appendix_full(output_path: Path = None):
    """Generate 4x4 grid: rows=models, cols=datasets."""
    plt = setup_matplotlib()

    if output_path is None:
        APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
        output_path = APPENDIX_DIR / 'scaling_4x4.pdf'

    scaling_cache = load_scaling_cache()

    fig, axes = plt.subplots(4, 4, figsize=(14, 12), dpi=300)

    for row, model in enumerate(MODEL_ORDER):
        for col, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]

            # Set ylim for first panel to make room for legend
            ylim = [87, 96] if (row == 0 and col == 0) else None

            plot_scaling_panel(
                ax, scaling_cache, model, dataset,
                ylim=ylim,
                show_ylabel=(col == 0),
                show_xlabel=(row == 3),
                show_legend=(row == 0 and col == 0)
            )

            # Column header (dataset name) on top row
            if row == 0:
                ax.set_title(DATASET_DISPLAY[dataset], fontweight='bold')

            # Row label (model name) on leftmost column
            if col == 0:
                ax.set_ylabel(f'{MODEL_DISPLAY[model]}\nAccuracy (%)', fontweight='bold')

    plt.tight_layout()

    # Save PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")

    # # Save PNG
    # png_path = output_path.with_suffix('.png')
    # plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    # print(f"PNG preview saved to: {png_path}")

    plt.close(fig)
    return output_path


def generate_beta_appendix_full(output_path: Path = None):
    """Generate 1x4 beta ablation: one panel per model, 4 datasets each."""
    plt = setup_matplotlib()

    if output_path is None:
        APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
        output_path = APPENDIX_DIR / 'beta_ablation_4models.pdf'

    sweep_cache = load_sweep_cache()

    # Wider figure, less spacing between subplots
    fig, axes = plt.subplots(1, 4, figsize=(17.64, 3.5), dpi=300)
    plt.subplots_adjust(wspace=0.13)

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]

        plot_beta_panel(
            ax, sweep_cache, model,
            alpha=0.5, position_pct=10,
            show_ylabel=(idx == 0),
            show_legend=(idx == 0)
        )

        ax.set_title(MODEL_DISPLAY[model], fontweight='bold')

        # Set y-axis to 60-100 for first panel to make room for legend
        if idx == 0:
            ax.set_ylim([60, 100])

    # Save PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")

    # # Save PNG
    # png_path = output_path.with_suffix('.png')
    # plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    # print(f"PNG preview saved to: {png_path}")

    plt.close(fig)
    return output_path


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate appendix figures for scaling analysis')
    parser.add_argument('--type', choices=['scaling', 'beta', 'all'], default='all',
                        help='Type of figure to generate (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/figures/appendix)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else APPENDIX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ['scaling', 'all']:
        generate_scaling_appendix_full(output_dir / 'scaling_4x4.pdf')

    if args.type in ['beta', 'all']:
        generate_beta_appendix_full(output_dir / 'beta_ablation_4models.pdf')
