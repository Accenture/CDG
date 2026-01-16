#!/usr/bin/env python3
"""
Alpha-Beta Parameter Sweep Figures for Paper.

Generates figures showing CDG accuracy vs beta values for different alpha values.
- Main paper: Single subplot (one model across 4 datasets OR one dataset across 4 models)
- Appendix: Full grid of 4x4 (models x datasets)

Saves individual figures and composed versions.

Usage:
    # Generate all figures (individual + composed)
    python scripts/eval/fig_hyperparam.py

    # Main paper figure: one model, 4 datasets
    python scripts/eval/fig_hyperparam.py --main-paper --model qwen32b

    # Main paper figure: one dataset, 4 models
    python scripts/eval/fig_hyperparam.py --main-paper --dataset aime2025

    # Custom output directory
    python scripts/eval/fig_hyperparam.py --output-dir results/fig_hyperparam

    # Include beta=0 in sweep
    python scripts/eval/fig_hyperparam.py --include-beta-zero
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
from eval_methods import load_run_results, evaluate_cdg, DEFAULT_PARAMS
from eval_cache import EvalCache

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Error: matplotlib and numpy required. Install with: pip install matplotlib numpy")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_DIR = PathConfig.OUTPUT_BASE
OUTPUT_BASE = Path(__file__).parent.parent.parent / "results" / "fig_hyperparam"

# Parameter sweep ranges
# NOTE: alpha=0.5 is sqrt dampening (1/2), alpha=0.25 is fourth-root (1/4)
ALPHA_VALUES = [0, 0.25, 0.5, 1.0]  # Added alpha=0 and alpha=0.25 (1/4)
BETA_VALUES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Added beta=0
POSITION_PCT = 20

# Models and datasets
MODELS = ['qwq32b', 'deepseek8b', 'gemma3_27b', 'gptoss20b']
DATASETS = ['aime2025', 'aime2024', 'hmmt2025', 'bruno2025']

# Pretty names for display
MODEL_NAMES = {
    'qwq32b': 'QwQ 32B',
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

# Style configuration
ALPHA_COLORS = {
    0: '#9467bd',      # Purple for alpha=0 (pure confidence)
    0.25: '#2ca02c',   # Green for alpha=0.25 (1/4)
    0.5: '#1f77b4',    # Blue for alpha=0.5 (1/2)
    1.0: '#d62728',    # Red for alpha=1.0 (linear)
}

ALPHA_MARKERS = {
    0: 'v',
    0.25: '^',
    0.5: 'o',
    1.0: 's',
}


# =============================================================================
# Data Loading and Evaluation
# =============================================================================

def discover_runs(results_dir: str) -> dict:
    """Discover available runs by model and dataset."""
    runs = {}
    base = Path(results_dir)

    if not base.exists():
        return runs

    for item in base.iterdir():
        if item.is_dir() and item.name not in ['subset_trace', '__pycache__', '.eval_cache']:
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                # Parse run_id
                for model in MODELS:
                    for dataset in DATASETS:
                        if f'{model}_{dataset}_' in item.name:
                            key = (model, dataset)
                            if key not in runs:
                                runs[key] = []
                            runs[key].append({
                                'run_id': item.name,
                                'path': str(item),
                            })
                            break

    return runs


def evaluate_with_params(run_path: str, alpha: float, beta: float,
                         position_pct: int, cache: EvalCache = None,
                         run_id: str = None, cache_only: bool = False) -> dict:
    """Evaluate a single run with specific parameters.

    Args:
        cache_only: If True, skip evaluation when cache misses (return None).
    """
    params = {
        'alpha': alpha,
        'beta': beta,
        'position_pct': position_pct,
        'tail_window': DEFAULT_PARAMS['tail_window'],
        'bottom_window': DEFAULT_PARAMS['bottom_window'],
        'bottom_pct': DEFAULT_PARAMS['bottom_pct'],
    }

    # Check cache
    if cache is not None and run_id:
        cached = cache.load_result(run_id, 'cdg', params)
        if cached is not None:
            return cached

    # Skip evaluation if cache_only mode
    if cache_only:
        return None

    # Load and evaluate
    results_by_qid = load_run_results(run_path, use_threads=False)
    if not results_by_qid:
        return None

    result = evaluate_cdg(results_by_qid, params)

    # Save to cache
    if cache is not None and run_id:
        cache.save_result(run_id, 'cdg', params, result)

    return result


def run_parameter_sweep(results_dir: str, alpha_values: list, beta_values: list,
                        cache: EvalCache = None, model_filter: str = None,
                        dataset_filter: str = None, cache_only: bool = False) -> dict:
    """
    Run parameter sweep across all models and datasets.

    Args:
        cache_only: If True, only use cached results (skip evaluation on cache miss).

    Returns:
        {(model, dataset): {alpha: {beta: {'accuracy': x, 'correct': n, 'total': m}}}}
    """
    runs = discover_runs(results_dir)

    if not runs:
        print(f"No runs found in {results_dir}")
        return {}

    results = defaultdict(lambda: defaultdict(dict))
    total_evals = 0
    cache_hits = 0
    cache_misses = 0

    for (model, dataset), run_list in runs.items():
        # Apply filters
        if model_filter and model != model_filter:
            continue
        if dataset_filter and dataset != dataset_filter:
            continue

        print(f"{'[cache-only] ' if cache_only else ''}Loading {model} - {dataset}...")

        for alpha in alpha_values:
            for beta in beta_values:
                # Evaluate each run and average
                accuracies = []

                for run_info in run_list:
                    result = evaluate_with_params(
                        run_info['path'], alpha, beta, POSITION_PCT,
                        cache=cache, run_id=run_info['run_id'],
                        cache_only=cache_only
                    )

                    if result:
                        accuracies.append(result['accuracy'])
                        total_evals += 1
                        cache_hits += 1
                    elif cache_only:
                        cache_misses += 1

                if accuracies:
                    results[(model, dataset)][alpha][beta] = {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies) if len(accuracies) > 1 else 0,
                        'n': len(accuracies),
                    }

    if cache_only:
        print(f"Cache-only mode: {cache_hits} hits, {cache_misses} misses")
    else:
        print(f"Completed {total_evals} evaluations")
    return dict(results)


# =============================================================================
# Figure Generation
# =============================================================================

def create_single_subplot(ax, data: dict, alpha_values: list, beta_values: list,
                          title: str = None, show_legend: bool = True,
                          ylabel: str = 'Accuracy (%)', xlabel: str = r'$\beta$ (gradient weight)'):
    """Create a single subplot with alpha-beta curves."""

    for alpha in alpha_values:
        if alpha not in data:
            continue

        x_vals = []
        y_vals = []
        y_errs = []

        for beta in beta_values:
            if beta in data[alpha]:
                x_vals.append(beta)
                y_vals.append(100 * data[alpha][beta]['mean'])
                y_errs.append(100 * data[alpha][beta]['std'])

        if x_vals:
            color = ALPHA_COLORS.get(alpha, 'gray')
            marker = ALPHA_MARKERS.get(alpha, 'o')

            # Label format
            if alpha == 0:
                label = r'$\alpha=0$ (no count)'
            elif alpha == 0.25:
                label = r'$\alpha=0.25$ ($\sqrt[4]{n}$)'
            elif alpha == 0.5:
                label = r'$\alpha=0.5$ ($\sqrt{n}$)'
            else:
                label = r'$\alpha=1.0$ (linear)'

            ax.errorbar(x_vals, y_vals, yerr=y_errs if any(y_errs) else None,
                       label=label, color=color, marker=marker,
                       capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    if show_legend:
        ax.legend(loc='best', fontsize=9)


def create_main_paper_figure(results: dict, alpha_values: list, beta_values: list,
                             output_dir: Path, by_model: str = None, by_dataset: str = None):
    """
    Create main paper figure: either one model across datasets or one dataset across models.
    """
    if by_model:
        # One model, 4 datasets
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

        for idx, dataset in enumerate(DATASETS):
            key = (by_model, dataset)
            if key in results:
                create_single_subplot(
                    axes[idx], results[key], alpha_values, beta_values,
                    title=DATASET_NAMES.get(dataset, dataset),
                    show_legend=(idx == 0),
                    ylabel='Accuracy (%)' if idx == 0 else ''
                )
            else:
                axes[idx].set_title(DATASET_NAMES.get(dataset, dataset))
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)

        fig.suptitle(f'CDG Parameter Sweep: {MODEL_NAMES.get(by_model, by_model)}',
                    fontsize=14, fontweight='bold')
        filename = f'main_model_{by_model}.pdf'

    elif by_dataset:
        # One dataset, 4 models
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

        for idx, model in enumerate(MODELS):
            key = (model, by_dataset)
            if key in results:
                create_single_subplot(
                    axes[idx], results[key], alpha_values, beta_values,
                    title=MODEL_NAMES.get(model, model),
                    show_legend=(idx == 0),
                    ylabel='Accuracy (%)' if idx == 0 else ''
                )
            else:
                axes[idx].set_title(MODEL_NAMES.get(model, model))
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)

        fig.suptitle(f'CDG Parameter Sweep: {DATASET_NAMES.get(by_dataset, by_dataset)}',
                    fontsize=14, fontweight='bold')
        filename = f'main_dataset_{by_dataset}.pdf'

    else:
        raise ValueError("Must specify either by_model or by_dataset")

    plt.tight_layout()

    # Save
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def create_full_grid_figure(results: dict, alpha_values: list, beta_values: list,
                            output_dir: Path):
    """Create full 4x4 grid figure for appendix."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            key = (model, dataset)

            if key in results:
                create_single_subplot(
                    ax, results[key], alpha_values, beta_values,
                    title=None,
                    show_legend=False,
                    ylabel=MODEL_NAMES.get(model, model) if j == 0 else '',
                    xlabel=DATASET_NAMES.get(dataset, dataset) if i == 3 else ''
                )
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

            # Row labels (models) on the left
            if j == 0:
                ax.set_ylabel(f'{MODEL_NAMES.get(model, model)}\nAccuracy (%)', fontsize=10)

            # Column labels (datasets) on top
            if i == 0:
                ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=11, fontweight='bold')

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('CDG Parameter Sweep: All Models and Datasets (Appendix)',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'appendix_full_grid.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def create_individual_figures(results: dict, alpha_values: list, beta_values: list,
                              output_dir: Path):
    """Create individual figures for each model-dataset combination."""
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), data in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        create_single_subplot(
            ax, data, alpha_values, beta_values,
            title=f'{MODEL_NAMES.get(model, model)} on {DATASET_NAMES.get(dataset, dataset)}',
            show_legend=True
        )

        plt.tight_layout()

        filename = f'{model}_{dataset}.pdf'
        output_path = individual_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(results)} individual figures to {individual_dir}")


def save_numerical_data(results: dict, output_dir: Path):
    """Save numerical results to JSON."""
    # Convert to serializable format
    data = {}
    for (model, dataset), alpha_data in results.items():
        key = f"{model}_{dataset}"
        data[key] = {}
        for alpha, beta_data in alpha_data.items():
            data[key][str(alpha)] = {}
            for beta, metrics in beta_data.items():
                data[key][str(alpha)][str(beta)] = {
                    'mean': float(metrics['mean']),
                    'std': float(metrics['std']),
                    'n': int(metrics['n']),
                }

    output_path = output_dir / 'hyperparam_data.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'alpha_values': ALPHA_VALUES,
            'beta_values': BETA_VALUES,
            'position_pct': POSITION_PCT,
            'results': data,
        }, f, indent=2)

    print(f"Saved numerical data to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate alpha-beta parameter sweep figures for paper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Base directory for experiment results')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_BASE),
                        help='Output directory for figures')

    # Main paper figure options
    parser.add_argument('--main-paper', action='store_true',
                        help='Generate main paper figure (requires --model or --dataset)')
    parser.add_argument('--model', type=str, choices=MODELS,
                        help='Model for main paper figure (one model, 4 datasets)')
    parser.add_argument('--dataset', type=str, choices=DATASETS,
                        help='Dataset for main paper figure (one dataset, 4 models)')

    # Parameter options
    parser.add_argument('--include-beta-zero', action='store_true',
                        help='Include beta=0 in sweep (default: start from 0)')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Comma-separated alpha values (default: 0,0.25,0.5,1.0)')
    parser.add_argument('--betas', type=str, default=None,
                        help='Comma-separated beta values')

    # Output options
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip individual figures')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip full grid figure')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures')

    # Cache
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only use cached results, skip evaluation on cache miss (fast mode)')

    args = parser.parse_args()

    # Parse alpha/beta values
    if args.alphas:
        alpha_values = [float(a.strip()) for a in args.alphas.split(',')]
    else:
        alpha_values = ALPHA_VALUES

    if args.betas:
        beta_values = [int(b.strip()) for b in args.betas.split(',')]
    else:
        beta_values = BETA_VALUES

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cache
    cache = EvalCache(args.results_dir, enabled=not args.no_cache) if not args.no_cache else None

    print("=" * 70)
    print("Alpha-Beta Parameter Sweep Figures")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Alpha values: {alpha_values}")
    print(f"Beta values: {beta_values}")
    print(f"Cache-only mode: {args.cache_only}")
    print()

    # Run parameter sweep
    results = run_parameter_sweep(
        args.results_dir,
        alpha_values=alpha_values,
        beta_values=beta_values,
        cache=cache,
        model_filter=args.model if args.main_paper else None,
        dataset_filter=args.dataset if args.main_paper else None,
        cache_only=args.cache_only
    )

    if not results:
        print("No results to plot!")
        return

    # Save numerical data
    save_numerical_data(results, output_dir)

    # Generate figures
    if args.main_paper:
        if args.model:
            create_main_paper_figure(results, alpha_values, beta_values, output_dir, by_model=args.model)
        elif args.dataset:
            create_main_paper_figure(results, alpha_values, beta_values, output_dir, by_dataset=args.dataset)
        else:
            print("Warning: --main-paper requires --model or --dataset")
    else:
        # Generate all figures
        if not args.no_individual:
            create_individual_figures(results, alpha_values, beta_values, output_dir)

        if not args.no_grid:
            create_full_grid_figure(results, alpha_values, beta_values, output_dir)

        # Generate main paper candidates
        for model in MODELS:
            if any((model, d) in results for d in DATASETS):
                create_main_paper_figure(results, alpha_values, beta_values, output_dir, by_model=model)

        for dataset in DATASETS:
            if any((m, dataset) in results for m in MODELS):
                create_main_paper_figure(results, alpha_values, beta_values, output_dir, by_dataset=dataset)

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
