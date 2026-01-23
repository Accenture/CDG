#!/usr/bin/env python3
"""
Generate 1x4 figure showing confidence curves with 10 bins for different models.

Figure 2: Confidence curve with 10 bins for the reasoning trajectories
for different models on AIME 2025 dataset.

Usage:
    python figure_confidence_curve.py                    # Generate figure (AIME 2025)
    python figure_confidence_curve.py --dataset aime2024 # Different dataset
    python figure_confidence_curve.py --output fig.pdf   # Custom output path
    python figure_confidence_curve.py --no-cache         # Force recompute (skip cache)
"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import config for paths
from config import DATASETS, FIGURES_DIR, MODEL_NAMES, DATASET_NAMES, OUTPUT_DIRS

# dynasor for math evaluation
# from dynasor.core.evaluator import math_equal


# =============================================================================
# Constants
# =============================================================================

# Model order for figure: deepseek, gemma, qwq, gpt
MODEL_ORDER = ['deepseek8b', 'gemma3_27b', 'qwq32b', 'gptoss20b']
MODEL_DISPLAY = {
    'deepseek8b': 'DeepSeek-R1-8B',
    'gemma3_27b': 'GEMMA-3-27B',
    'qwq32b': 'QWQ-32B',
    'gptoss20b': 'gpt-oss-20B',
}

NUM_BINS = 10
COLOR_CORRECT = '#2ca02c'  # Green
COLOR_WRONG = '#d62728'    # Red

# Cache directory
CACHE_DIR = OUTPUT_DIRS.get('util_histogram_cache', FIGURES_DIR.parent / 'cache' / 'histogram') / 'confidence_curve_cache'


# =============================================================================
# Cache Functions
# =============================================================================

def get_cache_path(model: str, dataset: str) -> Path:
    """Get cache file path for a model-dataset pair."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f'{model}_{dataset}_confidence_curve.json'


def load_cache(model: str, dataset: str) -> dict:
    """Load cached stats if available."""
    cache_path = get_cache_path(model, dataset)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None


def save_cache(model: str, dataset: str, stats: dict):
    """Save stats to cache."""
    cache_path = get_cache_path(model, dataset)
    with open(cache_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Cached to: {cache_path}")


# =============================================================================
# Setup
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    return plt


# =============================================================================
# Helper Functions
# =============================================================================

def quick_parse(text: str) -> str:
    """Parse LaTeX text commands."""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    try:
        return math_equal(answer, ground_truth)
    except:
        return str(answer).strip() == str(ground_truth).strip()


def load_all_questions(dataset_path: str) -> list:
    """Load all questions from a dataset."""
    results_dir = Path(dataset_path)
    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    questions = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        questions.append(data)

    return questions


def divide_trace_into_bins(confs: list, num_bins: int = 10) -> list:
    """Divide a trace into N bins and compute mean confidence per bin."""
    if not confs:
        return [np.nan] * num_bins

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


def analyze_confidence_by_position(questions: list, num_bins: int = 10) -> dict:
    """Analyze confidence vs position for all traces in questions."""
    correct_by_position = defaultdict(list)
    wrong_by_position = defaultdict(list)

    for q in questions:
        ground_truth = q.get('ground_truth', '')
        traces = q.get('all_traces', [])

        if not ground_truth:
            continue

        for trace in traces:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is None or len(confs) < 10:
                continue

            try:
                is_correct = equal_func(str(answer), str(ground_truth))
            except:
                is_correct = str(answer) == str(ground_truth)

            bin_means = divide_trace_into_bins(confs, num_bins)

            target = correct_by_position if is_correct else wrong_by_position
            for pos, mean_conf in enumerate(bin_means, 1):
                if not np.isnan(mean_conf):
                    target[pos].append(mean_conf)

    # Compute statistics
    positions = list(range(1, num_bins + 1))
    stats = {
        'positions': positions,
        'correct': {'means': [], 'stds': []},
        'wrong': {'means': [], 'stds': []},
    }

    for pos in positions:
        for category, data in [('correct', correct_by_position), ('wrong', wrong_by_position)]:
            confs = data[pos]
            stats[category]['means'].append(np.mean(confs) if confs else np.nan)
            stats[category]['stds'].append(np.std(confs) if confs else np.nan)

    return stats


# =============================================================================
# Figure Generation
# =============================================================================

def generate_confidence_curve_figure(dataset: str = 'aime2025', output_path: Path = None, use_cache: bool = True):
    """Generate 1x4 confidence curve figure for all models on a dataset."""
    plt = setup_matplotlib()

    # Default output path
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / 'confidence_curve_figure.pdf'

    # Create 1x4 figure
    fig, axes = plt.subplots(1, 4, figsize=(12.92, 2.78), dpi=300)

    positions = list(range(1, NUM_BINS + 1))

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]

        # Try to load from cache first
        stats = None
        if use_cache:
            stats = load_cache(model, dataset)
            if stats:
                print(f"Loaded {model} / {dataset} from cache")

        # If no cache, compute from scratch
        if stats is None:
            dataset_path = DATASETS.get(model, {}).get(dataset)
            if not dataset_path:
                print(f"Warning: No path for {model}/{dataset}")
                continue

            print(f"Loading {model} / {dataset}...")
            questions = load_all_questions(dataset_path)
            print(f"  Loaded {len(questions)} questions")

            # Analyze confidence by position
            stats = analyze_confidence_by_position(questions, NUM_BINS)

            # Save to cache
            save_cache(model, dataset, stats)

        # Plot correct traces
        ax.errorbar(positions, stats['correct']['means'], yerr=stats['correct']['stds'],
                    marker='s', markersize=5, linewidth=1.5, capsize=3,
                    label='Correct', color=COLOR_CORRECT, alpha=0.9)

        # Plot wrong traces
        ax.errorbar(positions, stats['wrong']['means'], yerr=stats['wrong']['stds'],
                    marker='s', markersize=5, linewidth=1.5, capsize=3,
                    label='Wrong', color=COLOR_WRONG, alpha=0.9)

        # Formatting
        ax.set_title(MODEL_DISPLAY[model], fontweight='bold')
        ax.set_xlabel('Position (1=start, 10=end)', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Mean Confidence', fontweight='bold')
            ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray',
                      handlelength=1.2, handletextpad=0.4, borderpad=0.3)
        ax.set_xticks([1, 5, 10])
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.22, wspace=0.11)

    # Save figure
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nFigure saved to: {output_path}")

    # png_path = output_path.with_suffix('.png')
    # plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    # print(f"PNG preview saved to: {png_path}")

    plt.close(fig)
    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate confidence curve figure for paper')
    parser.add_argument('--dataset', '-d', type=str, default='aime2025',
                        choices=['aime2024', 'aime2025', 'brumo2025', 'hmmt2025'],
                        help='Dataset to analyze (default: aime2025)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path (default: results/figures/confidence_curve_figure.pdf)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force recompute, skip cache')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_confidence_curve_figure(args.dataset, output_path, use_cache=not args.no_cache)
