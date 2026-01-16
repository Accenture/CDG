#!/usr/bin/env python3
"""
Subsample/Scaling Analysis Figures (Exp 2).

Generates figures showing how accuracy scales with number of traces (8 -> 512).
Multiple design options for paper presentation:

Design Options:
1. **Grid Layout** (Appendix): 4x4 grid (models x datasets) with all methods
2. **Single Row** (Main): One model across 4 datasets, key methods only
3. **Method Comparison**: One panel per method, models as different lines
4. **Improvement Plot**: Show improvement over majority baseline
5. **Area Plot**: Stack areas showing method contributions

Usage:
    # Generate all figure variants
    python scripts/eval/fig_subsample.py

    # Main paper figure (one model, key methods)
    python scripts/eval/fig_subsample.py --main-paper --model qwen32b

    # Method comparison view
    python scripts/eval/fig_subsample.py --method-comparison

    # Show improvement over baseline
    python scripts/eval/fig_subsample.py --improvement
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import PathConfig
from eval_methods import (
    load_run_results, evaluate_with_method, get_available_methods,
    EVAL_METHODS, DEFAULT_PARAMS
)
from eval_cache import EvalCache
from hyperparam_config import HyperparamConfig

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.lines import Line2D
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Error: matplotlib and numpy required")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE
OUTPUT_BASE = Path(__file__).parent.parent.parent / "results" / "fig_subsample"
SUBSET_SUBDIR = "subset_trace"

# Trace counts
TRACE_COUNTS = [8, 16, 32, 64, 128, 256, 512]

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

# Key methods for main paper (subset)
KEY_METHODS = ['majority', 'deepconf_mean', 'deepconf_tail', 'cdg']

# Method styles
METHOD_STYLES = {
    'oracle': {'color': '#7f7f7f', 'marker': 'x', 'linestyle': ':', 'label': 'Oracle (Upper Bound)'},
    'majority': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'Majority Vote'},
    'deepconf_mean': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': 'DeepConf (Mean)'},
    'deepconf_tail': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'label': 'DeepConf (Tail)'},
    'deepconf_bottom': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'label': 'DeepConf (Bottom)'},
    'gradient': {'color': '#8c564b', 'marker': 'D', 'linestyle': '-', 'label': 'Gradient Only'},
    'cdg': {'color': '#d62728', 'marker': '*', 'linestyle': '-', 'label': 'CDG (Ours)', 'linewidth': 2.5},
}

# Model colors (for method comparison view)
MODEL_COLORS = {
    'qwen32b': '#1f77b4',
    'deepseek8b': '#ff7f0e',
    'gemma3_27b': '#2ca02c',
    'gptoss20b': '#d62728',
}


# =============================================================================
# Data Loading
# =============================================================================

def discover_runs(results_dir: str) -> dict:
    """
    Discover all runs including subsampled versions.

    Returns:
        {(model, dataset, trace_count): [run_paths]}
    """
    runs = defaultdict(list)
    base = Path(results_dir)

    if not base.exists():
        return runs

    # Full runs (512)
    for item in base.iterdir():
        if item.is_dir() and item.name not in ['subset_trace', '__pycache__', '.eval_cache']:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                for model in MODELS:
                    for dataset in DATASETS:
                        if f'{model}_{dataset}_' in item.name:
                            runs[(model, dataset, 512)].append(str(item))
                            break

    # Subset runs
    subset_base = base / SUBSET_SUBDIR
    if subset_base.exists():
        import re
        for item in subset_base.iterdir():
            if item.is_dir():
                pkl_files = list(item.glob("*.pkl"))
                if pkl_files:
                    # Parse subset size
                    match = re.search(r'_subset(\d+)v\d+$', item.name)
                    if match:
                        trace_count = int(match.group(1))
                        for model in MODELS:
                            for dataset in DATASETS:
                                if f'{model}_{dataset}_' in item.name:
                                    runs[(model, dataset, trace_count)].append(str(item))
                                    break

    return dict(runs)


def evaluate_all_runs(results_dir: str, methods: list, params: dict,
                      cache: EvalCache = None, model_filter: str = None,
                      dataset_filter: str = None, cache_only: bool = False) -> dict:
    """
    Evaluate all runs with specified methods.

    Args:
        cache_only: If True, only use cached results (skip evaluation on cache miss).

    Returns:
        {(model, dataset): {trace_count: {method: {'mean': x, 'std': x}}}}
    """
    runs = discover_runs(results_dir)

    if not runs:
        print(f"No runs found in {results_dir}")
        return {}

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total = 0
    cache_hits = 0
    cache_misses = 0

    for (model, dataset, trace_count), run_paths in runs.items():
        if model_filter and model != model_filter:
            continue
        if dataset_filter and dataset != dataset_filter:
            continue

        for run_path in run_paths:
            run_id = Path(run_path).name

            for method in methods:
                # Check cache
                if cache:
                    cached = cache.load_result(run_id, method, params)
                    if cached:
                        results[(model, dataset)][trace_count][method].append(cached['accuracy'])
                        cache_hits += 1
                        continue

                # Skip evaluation if cache_only mode
                if cache_only:
                    cache_misses += 1
                    continue

                # Evaluate
                results_by_qid = load_run_results(run_path, use_threads=False)
                if results_by_qid:
                    result = evaluate_with_method(results_by_qid, method, params)
                    results[(model, dataset)][trace_count][method].append(result['accuracy'])

                    if cache:
                        cache.save_result(run_id, method, params, result)

                    total += 1

    if cache_only:
        print(f"Cache-only mode: {cache_hits} hits, {cache_misses} misses")
    else:
        print(f"Evaluated {total} runs (cache hits: {cache_hits})")

    # Aggregate to mean/std
    aggregated = {}
    for (model, dataset), trace_data in results.items():
        aggregated[(model, dataset)] = {}
        for trace_count, method_data in trace_data.items():
            aggregated[(model, dataset)][trace_count] = {}
            for method, values in method_data.items():
                aggregated[(model, dataset)][trace_count][method] = {
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0,
                    'n': len(values),
                }

    return aggregated


# =============================================================================
# Figure Generation - Design Options
# =============================================================================

def create_single_scaling_plot(ax, trace_data: dict, methods: list,
                               title: str = None, show_legend: bool = True,
                               ylabel: str = 'Accuracy (%)', show_ylabel: bool = True):
    """Create a single scaling plot (accuracy vs trace count)."""

    for method in methods:
        style = METHOD_STYLES.get(method, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})

        x_vals = []
        y_vals = []
        y_errs = []

        for trace_count in TRACE_COUNTS:
            if trace_count in trace_data and method in trace_data[trace_count]:
                data = trace_data[trace_count][method]
                x_vals.append(trace_count)
                y_vals.append(100 * data['mean'])
                y_errs.append(100 * data['std'])

        if x_vals:
            ax.errorbar(
                x_vals, y_vals,
                yerr=y_errs if any(y_errs) else None,
                label=style.get('label', EVAL_METHODS.get(method, {}).get('name', method)),
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=style.get('linewidth', 2),
                markersize=8,
                capsize=3,
            )

    ax.set_xscale('log', base=2)
    ax.set_xticks(TRACE_COUNTS)
    ax.set_xticklabels([str(t) for t in TRACE_COUNTS])
    ax.set_xlabel('Number of Traces', fontsize=11)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    if title:
        ax.set_title(title, fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    if show_legend:
        ax.legend(loc='lower right', fontsize=9)


def design_1_grid_layout(results: dict, methods: list, output_dir: Path):
    """
    Design 1: Full grid layout (4x4 for appendix).

    Rows: Models
    Columns: Datasets
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)

    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            key = (model, dataset)

            if key in results:
                create_single_scaling_plot(
                    ax, results[key], methods,
                    show_legend=False,
                    show_ylabel=(j == 0)
                )

            # Row/column labels
            if j == 0:
                ax.set_ylabel(f'{MODEL_NAMES.get(model, model)}\nAccuracy (%)', fontsize=10)
            if i == 0:
                ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11, fontweight='bold')

    # Shared legend at bottom
    handles = []
    labels = []
    for method in methods:
        style = METHOD_STYLES.get(method, {})
        handle = Line2D([0], [0],
                       color=style.get('color', 'gray'),
                       marker=style.get('marker', 'o'),
                       linestyle=style.get('linestyle', '-'),
                       linewidth=style.get('linewidth', 2),
                       markersize=8)
        handles.append(handle)
        labels.append(style.get('label', method))

    fig.legend(handles, labels, loc='lower center', ncol=len(methods),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Accuracy vs Number of Traces: All Models and Datasets',
                fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    output_path = output_dir / 'design1_grid_appendix.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def design_2_single_row(results: dict, methods: list, output_dir: Path,
                        model: str = None, dataset: str = None):
    """
    Design 2: Single row (main paper).

    Either: One model across 4 datasets
    Or: One dataset across 4 models
    """
    if model:
        # One model, 4 datasets
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

        for idx, ds in enumerate(DATASETS):
            key = (model, ds)
            if key in results:
                create_single_scaling_plot(
                    axes[idx], results[key], methods,
                    title=DATASET_NAMES.get(ds, ds),
                    show_legend=(idx == 0),
                    show_ylabel=(idx == 0)
                )
            else:
                axes[idx].set_title(DATASET_NAMES.get(ds, ds))
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)

        fig.suptitle(f'Scaling Analysis: {MODEL_NAMES.get(model, model)}',
                    fontsize=14, fontweight='bold')
        filename = f'design2_main_model_{model}'

    elif dataset:
        # One dataset, 4 models
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

        for idx, m in enumerate(MODELS):
            key = (m, dataset)
            if key in results:
                create_single_scaling_plot(
                    axes[idx], results[key], methods,
                    title=MODEL_NAMES.get(m, m),
                    show_legend=(idx == 0),
                    show_ylabel=(idx == 0)
                )
            else:
                axes[idx].set_title(MODEL_NAMES.get(m, m))
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)

        fig.suptitle(f'Scaling Analysis: {DATASET_NAMES.get(dataset, dataset)}',
                    fontsize=14, fontweight='bold')
        filename = f'design2_main_dataset_{dataset}'

    else:
        raise ValueError("Must specify model or dataset")

    plt.tight_layout()

    output_path = output_dir / f'{filename}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def design_3_method_comparison(results: dict, methods: list, output_dir: Path,
                               dataset: str = 'aime2025'):
    """
    Design 3: Method comparison view.

    One subplot per method, models as different colored lines.
    Good for comparing how each method scales across different models.
    """
    n_methods = len(methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        style = METHOD_STYLES.get(method, {})

        for model in MODELS:
            key = (model, dataset)
            if key not in results:
                continue

            trace_data = results[key]
            x_vals = []
            y_vals = []
            y_errs = []

            for trace_count in TRACE_COUNTS:
                if trace_count in trace_data and method in trace_data[trace_count]:
                    data = trace_data[trace_count][method]
                    x_vals.append(trace_count)
                    y_vals.append(100 * data['mean'])
                    y_errs.append(100 * data['std'])

            if x_vals:
                ax.errorbar(
                    x_vals, y_vals,
                    yerr=y_errs if any(y_errs) else None,
                    label=MODEL_NAMES.get(model, model),
                    color=MODEL_COLORS.get(model, 'gray'),
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    capsize=3,
                )

        ax.set_xscale('log', base=2)
        ax.set_xticks(TRACE_COUNTS)
        ax.set_xticklabels([str(t) for t in TRACE_COUNTS])
        ax.set_xlabel('Number of Traces', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(style.get('label', method), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=8)

    # Hide unused
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Method Comparison on {DATASET_NAMES.get(dataset, dataset)}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'design3_method_comparison_{dataset}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def design_4_improvement_plot(results: dict, methods: list, output_dir: Path,
                              model: str = 'qwen32b'):
    """
    Design 4: Improvement over majority baseline.

    Shows relative improvement (%) over majority voting.
    Highlights where advanced methods provide the most value.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        key = (model, dataset)

        if key not in results:
            ax.set_title(DATASET_NAMES.get(dataset, dataset))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        trace_data = results[key]

        # Get majority baseline
        baseline = {}
        for tc in TRACE_COUNTS:
            if tc in trace_data and 'majority' in trace_data[tc]:
                baseline[tc] = trace_data[tc]['majority']['mean']

        # Plot improvement for each method
        for method in methods:
            if method == 'majority':
                continue

            style = METHOD_STYLES.get(method, {})
            x_vals = []
            y_vals = []

            for tc in TRACE_COUNTS:
                if tc in trace_data and method in trace_data[tc] and tc in baseline:
                    method_acc = trace_data[tc][method]['mean']
                    base_acc = baseline[tc]
                    if base_acc > 0:
                        improvement = (method_acc - base_acc) / base_acc * 100
                        x_vals.append(tc)
                        y_vals.append(improvement)

            if x_vals:
                ax.plot(
                    x_vals, y_vals,
                    label=style.get('label', method),
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    markersize=8,
                )

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks(TRACE_COUNTS)
        ax.set_xticklabels([str(t) for t in TRACE_COUNTS])
        ax.set_xlabel('Number of Traces', fontsize=10)

        if idx == 0:
            ax.set_ylabel('Improvement over Majority (%)', fontsize=10)

        ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Improvement over Majority Baseline: {MODEL_NAMES.get(model, model)}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'design4_improvement_{model}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def design_5_compact_summary(results: dict, methods: list, output_dir: Path):
    """
    Design 5: Compact summary heatmap.

    Shows accuracy at 512 traces as a heatmap (models x datasets).
    One heatmap per method, or difference from baseline.
    """
    # Get accuracy at 512 traces for all combinations
    methods_to_show = ['majority', 'cdg']

    fig, axes = plt.subplots(1, len(methods_to_show), figsize=(6 * len(methods_to_show), 5))
    if len(methods_to_show) == 1:
        axes = [axes]

    for idx, method in enumerate(methods_to_show):
        ax = axes[idx]

        # Build matrix
        matrix = np.zeros((len(MODELS), len(DATASETS)))
        matrix[:] = np.nan

        for i, model in enumerate(MODELS):
            for j, dataset in enumerate(DATASETS):
                key = (model, dataset)
                if key in results and 512 in results[key] and method in results[key][512]:
                    matrix[i, j] = 100 * results[key][512][method]['mean']

        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in DATASETS], rotation=45, ha='right')
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in MODELS])

        # Annotate values
        for i in range(len(MODELS)):
            for j in range(len(DATASETS)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                  ha='center', va='center', color='black', fontsize=10)

        ax.set_title(METHOD_STYLES.get(method, {}).get('label', method), fontsize=12)

        plt.colorbar(im, ax=ax, label='Accuracy (%)')

    fig.suptitle('Accuracy at 512 Traces', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'design5_compact_heatmap.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def save_numerical_data(results: dict, output_dir: Path):
    """Save numerical data to JSON."""
    data = {}
    for (model, dataset), trace_data in results.items():
        key = f"{model}_{dataset}"
        data[key] = {}
        for trace_count, method_data in trace_data.items():
            data[key][str(trace_count)] = {}
            for method, metrics in method_data.items():
                data[key][str(trace_count)][method] = {
                    'mean': float(metrics['mean']),
                    'std': float(metrics['std']),
                    'n': int(metrics['n']),
                }

    output_path = output_dir / 'subsample_data.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'trace_counts': TRACE_COUNTS,
            'results': data,
        }, f, indent=2)

    print(f"Saved numerical data to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate subsample/scaling analysis figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Design Options:
  1. Grid Layout (--grid): 4x4 for appendix
  2. Single Row (--main-paper): One model/dataset, key methods
  3. Method Comparison (--method-comparison): Compare methods across models
  4. Improvement Plot (--improvement): Show improvement over majority
  5. Compact Summary (--compact): Heatmap at 512 traces

Recommendations:
  - Main paper: Use Design 2 (single row) with key methods
  - Appendix: Use Design 1 (full grid)
  - Analysis: Use Design 3 or 4 to highlight specific insights
        """
    )

    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Base directory for experiment results')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_BASE),
                        help='Output directory for figures')

    # Design options
    parser.add_argument('--grid', action='store_true',
                        help='Generate full grid layout (Design 1)')
    parser.add_argument('--main-paper', action='store_true',
                        help='Generate main paper figure (Design 2)')
    parser.add_argument('--method-comparison', action='store_true',
                        help='Generate method comparison view (Design 3)')
    parser.add_argument('--improvement', action='store_true',
                        help='Generate improvement plot (Design 4)')
    parser.add_argument('--compact', action='store_true',
                        help='Generate compact heatmap (Design 5)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all design variants')

    # Filters
    parser.add_argument('--model', type=str, choices=MODELS,
                        help='Model filter (for main-paper)')
    parser.add_argument('--dataset', type=str, choices=DATASETS,
                        help='Dataset filter (for main-paper)')

    # Method selection
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods (default: key methods for main, all for appendix)')
    parser.add_argument('--key-methods-only', action='store_true',
                        help='Use only key methods (majority, deepconf_mean, deepconf_tail, cdg)')

    # Parameters
    parser.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'])
    parser.add_argument('--beta', type=float, default=DEFAULT_PARAMS['beta'])

    # Cache
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only use cached results, skip evaluation on cache miss (fast mode)')
    parser.add_argument('--no-show', action='store_true')

    args = parser.parse_args()

    # Determine methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    elif args.key_methods_only or args.main_paper:
        methods = KEY_METHODS
    else:
        methods = get_available_methods()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build params
    params = {
        'alpha': args.alpha,
        'beta': args.beta,
        'position_pct': DEFAULT_PARAMS['position_pct'],
        'tail_window': DEFAULT_PARAMS['tail_window'],
        'bottom_window': DEFAULT_PARAMS['bottom_window'],
        'bottom_pct': DEFAULT_PARAMS['bottom_pct'],
    }

    # Initialize cache
    cache = EvalCache(args.results_dir, enabled=not args.no_cache)

    print("=" * 70)
    print("Subsample/Scaling Analysis Figures")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {methods}")
    print(f"Cache-only mode: {args.cache_only}")
    print()

    # Evaluate all runs
    results = evaluate_all_runs(
        args.results_dir, methods, params, cache,
        model_filter=args.model if not args.all else None,
        cache_only=args.cache_only,
        dataset_filter=args.dataset if not args.all else None
    )

    if not results:
        print("No results found!")
        return

    # Save numerical data
    save_numerical_data(results, output_dir)

    # Generate figures based on selected designs
    if args.all or args.grid:
        all_methods = get_available_methods() if args.all else methods
        # Re-evaluate with all methods if needed
        if args.all:
            results = evaluate_all_runs(args.results_dir, all_methods, params, cache,
                                        cache_only=args.cache_only)
        design_1_grid_layout(results, all_methods, output_dir)

    if args.all or args.main_paper:
        # Generate for each model and dataset
        if args.model:
            design_2_single_row(results, KEY_METHODS, output_dir, model=args.model)
        elif args.dataset:
            design_2_single_row(results, KEY_METHODS, output_dir, dataset=args.dataset)
        elif args.all:
            for model in MODELS:
                if any((model, d) in results for d in DATASETS):
                    design_2_single_row(results, KEY_METHODS, output_dir, model=model)
            for dataset in DATASETS:
                if any((m, dataset) in results for m in MODELS):
                    design_2_single_row(results, KEY_METHODS, output_dir, dataset=dataset)

    if args.all or args.method_comparison:
        for dataset in (DATASETS if args.all else [args.dataset or 'aime2025']):
            design_3_method_comparison(results, methods, output_dir, dataset=dataset)

    if args.all or args.improvement:
        for model in (MODELS if args.all else [args.model or 'qwen32b']):
            design_4_improvement_plot(results, methods, output_dir, model=model)

    if args.all or args.compact:
        design_5_compact_summary(results, methods, output_dir)

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
