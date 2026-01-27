#!/usr/bin/env python3
"""
Generate position_pct ablation figure for appendix.

Sweeps position_pct values [5, 10, 15, 20, 25, 30] with fixed alpha and beta.
Generates 1x4 figure: one panel per model, 4 datasets each.

Usage:
    python figure_position_appendix.py              # Run sweep and generate figure
    python figure_position_appendix.py --no-compute # Load cache, regenerate figure
    python figure_position_appendix.py --output fig.pdf  # Custom output path
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import DATASETS, APPENDIX_DIR, OUTPUT_DIRS, CDG_PARAMS

# Import shared functions from existing scripts
from exp_cdg_sweep import quick_parse, equal_func


# ============================================================================
# CONSTANTS
# ============================================================================

# Position values to sweep
POSITION_PCT_VALUES = [5, 10, 15, 20, 25, 30]

# Model/dataset order and display names
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

# Colors for datasets (matching figure_scaling_appendix.py)
DATASET_COLORS = {
    'aime2024': '#9467bd',
    'aime2025': '#e377c2',
    'brumo2025': '#17becf',
    'hmmt2025': '#bcbd22',
}

# Markers for datasets (for color-blind accessibility)
DATASET_MARKERS = {
    'aime2024': 'o',      # circle
    'aime2025': 's',      # square
    'brumo2025': '^',     # triangle up
    'hmmt2025': 'D',      # diamond
}

# Cache file
CACHE_DIR = OUTPUT_DIRS.get('exp_cdg_sweep', Path(__file__).parent.parent.parent / 'results' / 'cache' / 'cdg_sweep')
CACHE_FILE = CACHE_DIR / 'position_sweep_cache.json'


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
# HELPER FUNCTIONS
# ============================================================================

def compute_gradient(confs, percentile=10):
    """Compute gradient: mean(last percentile%) - mean(first percentile%)."""
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def load_raw_dataset(results_dir: str) -> list:
    """Load raw question data (without precomputing gradients)."""
    results_dir = Path(results_dir)

    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    if not pkl_files:
        return []

    questions = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        questions.append(data)

    return questions


def cdg_voting_with_position(questions: list, alpha: float, beta: float, position_pct: int) -> tuple:
    """
    CDG voting with specified position_pct for gradient computation.

    Returns: (correct_count, total_count)
    """
    correct = 0
    total = 0

    for q in questions:
        gt = q.get('ground_truth', '')
        traces = q.get('all_traces', [])

        if not gt or not traces:
            continue

        # Filter valid traces and compute scores
        answer_scores = defaultdict(list)
        for trace in traces:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is None or len(confs) < 10:
                continue

            mean_conf = np.mean(confs)
            gradient = compute_gradient(confs, percentile=position_pct)
            score = mean_conf + beta * gradient
            answer_scores[answer].append(score)

        if not answer_scores:
            continue

        total += 1

        # Compute final scores with count dampening
        final_scores = {
            ans: (len(scores) ** alpha) * np.mean(scores)
            for ans, scores in answer_scores.items()
        }
        winner = max(final_scores.items(), key=lambda x: x[1])[0]

        if equal_func(str(winner), str(gt)):
            correct += 1

    return correct, total


# ============================================================================
# SWEEP FUNCTIONS
# ============================================================================

def run_position_sweep() -> dict:
    """Run position_pct sweep for all models and datasets."""
    print('=' * 80)
    print('POSITION PCT SWEEP')
    print('=' * 80)
    print(f'Position values: {POSITION_PCT_VALUES}')
    print(f'Models: {MODEL_ORDER}')
    print()

    results = []

    for model in MODEL_ORDER:
        print(f'\n{model.upper()}')
        print('-' * 60)

        # Get CDG params for this model (use optimal alpha and beta)
        params = CDG_PARAMS.get(model, {'alpha': 0.5, 'beta': 10})
        alpha = params['alpha']
        beta = params['beta']
        print(f'Using alpha={alpha}, beta={beta}')

        # Header
        header = f"{'Dataset':<12}"
        for pos in POSITION_PCT_VALUES:
            header += f"P={pos:<5}"
        print(header)

        for dataset in DATASET_ORDER:
            dataset_path = DATASETS.get(model, {}).get(dataset)
            if not dataset_path:
                print(f"{dataset:<12} [NO PATH]")
                continue

            # Load raw data once per dataset
            questions = load_raw_dataset(dataset_path)
            if not questions:
                print(f"{dataset:<12} [NO DATA]")
                continue

            row = f"{dataset:<12}"

            for position_pct in POSITION_PCT_VALUES:
                correct, total = cdg_voting_with_position(questions, alpha, beta, position_pct)
                acc = correct / total if total > 0 else 0
                row += f"{100*acc:.1f}%  "

                results.append({
                    'model': model,
                    'dataset': dataset,
                    'position_pct': position_pct,
                    'alpha': alpha,
                    'beta': beta,
                    'correct': correct,
                    'total': total,
                    'accuracy': acc,
                })

            print(row)

    # Save cache
    cache_data = {
        'position_pct_values': POSITION_PCT_VALUES,
        'results': results,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f'\nCache saved to: {CACHE_FILE}')

    return cache_data


def load_cache() -> dict:
    """Load cached results."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_FILE}\nRun without --no-compute first.")
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_position_panel(ax, cache_data: dict, model: str, show_ylabel=False, show_legend=False):
    """Plot position ablation panel for one model (4 datasets)."""
    results = cache_data['results']
    position_values = cache_data['position_pct_values']

    # Filter results for this model
    model_results = [r for r in results if r['model'] == model]

    # Plot each dataset
    for dataset in DATASET_ORDER:
        accuracies = []
        for pos in position_values:
            entry = next((r for r in model_results
                         if r['dataset'] == dataset and r['position_pct'] == pos), None)
            if entry:
                accuracies.append(entry['accuracy'] * 100)
            else:
                accuracies.append(None)

        ax.plot(position_values, accuracies,
                color=DATASET_COLORS[dataset],
                marker=DATASET_MARKERS[dataset], linewidth=1.5, markersize=5,
                label=DATASET_DISPLAY[dataset])

    ax.set_xlabel(r'Position percentile $P$ (%)')
    ax.set_xticks(position_values)
    ax.set_xticklabels([str(p) for p in position_values])
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if show_ylabel:
        ax.set_ylabel('Accuracy (%)')
    else:
        ax.set_yticklabels([])

    if show_legend:
        ax.legend(loc='lower left', ncol=2, framealpha=0.9, edgecolor='gray',
                  handlelength=1.2, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.2, columnspacing=0.8)


def generate_position_appendix_figure(cache_data: dict, output_path: Path = None):
    """Generate 1x4 position ablation figure."""
    plt = setup_matplotlib()

    if output_path is None:
        APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
        output_path = APPENDIX_DIR / 'position_ablation_4models.pdf'

    # Create 1x4 figure (same dimensions as beta ablation)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=300)
    plt.subplots_adjust(wspace=0.05)

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]

        plot_position_panel(
            ax, cache_data, model,
            show_ylabel=(idx == 0),
            show_legend=(idx == 0)
        )

        ax.set_title(MODEL_DISPLAY[model], fontweight='bold')

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
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate position ablation figure for appendix')
    parser.add_argument('--no-compute', action='store_true',
                        help='Load from cache, skip computation')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for figure')
    args = parser.parse_args()

    if args.no_compute:
        print('Loading from cache...')
        cache_data = load_cache()
    else:
        cache_data = run_position_sweep()

    output_path = Path(args.output) if args.output else None
    generate_position_appendix_figure(cache_data, output_path)
