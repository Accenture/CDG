#!/usr/bin/env python3
"""
Histogram Figures: Correct vs Wrong Answer Distributions

Generates histograms comparing confidence metrics for correct vs wrong answers.
Reads directly from Yu's pickle files.

Metrics:
1. Vote Count (per answer) - majority voting
2. Mean Confidence (per trace)
3. Tail Confidence (per trace)
4. CDG Score (per answer)
5. Confidence Gradient (per trace)

Usage:
    python exp_histogram.py                            # Run all model/dataset combinations
    python exp_histogram.py --model deepseek8b --dataset aime2024  # Single combination
    python exp_histogram.py --no-compute               # Load cache, regenerate figures only

Cache files:
    results/exp_histogram/cache/{model}_{dataset}_metrics.json
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys
import json
import argparse
from datetime import datetime

# Import centralized config
from config import (
    DATASETS, CDG_PARAMS, MODEL_NAMES, DATASET_NAMES
)

# dynasor for math evaluation: pip install git+https://github.com/hao-ai-lab/Dynasor.git
from dynasor.core.evaluator import math_equal

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'exp_histogram'

# Cache directory for storing extracted metrics
CACHE_DIR = OUTPUT_DIR / 'cache'


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def get_cache_path(model: str, dataset: str) -> Path:
    """Get cache file path for a model-dataset pair."""
    return CACHE_DIR / f'{model}_{dataset}_metrics.json'


def save_cache(model: str, dataset: str, metrics: dict):
    """Save metrics to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(model, dataset)

    # Convert numpy arrays to lists for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(cache_path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f'  Cache saved: {cache_path}')


def load_cache(model: str, dataset: str) -> dict:
    """Load metrics from cache if available."""
    cache_path = get_cache_path(model, dataset)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            print(f'  Loading from cache: {cache_path}')
            return json.load(f)
    return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_parse(text: str) -> str:
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
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    try:
        return math_equal(answer, ground_truth)
    except:
        return str(answer).strip() == str(ground_truth).strip()


def compute_gradient(confs, percentile=10):
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def compute_tail_conf(confs, window=2048):
    if not confs:
        return 0.0
    tail = confs[-window:] if len(confs) > window else confs
    return float(np.mean(tail))


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_metrics(dataset_path: str, cdg_params: dict) -> dict:
    """Extract metrics for correct vs wrong answers."""
    results_dir = Path(dataset_path)

    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    if not pkl_files:
        return None

    alpha = cdg_params['alpha']
    beta = cdg_params['beta']
    position_pct = cdg_params['position_pct']

    metrics = {
        'correct': {
            'mean_conf': [],
            'tail_conf': [],
            'gradient': [],
            'cdg_score': [],
            'vote_count': [],
        },
        'wrong': {
            'mean_conf': [],
            'tail_conf': [],
            'gradient': [],
            'cdg_score': [],
            'vote_count': [],
        },
        'meta': {
            'n_questions': 0,
            'n_correct_answers': 0,
            'n_wrong_answers': 0,
        }
    }

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        gt = data['ground_truth']
        all_traces = data['all_traces']
        metrics['meta']['n_questions'] += 1

        # Group traces by answer
        answer_groups = defaultdict(list)
        for trace in all_traces:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is not None and len(confs) >= 10:
                mean_conf = float(np.mean(confs))
                gradient = compute_gradient(confs, position_pct)
                tail_conf = compute_tail_conf(confs)

                answer_groups[answer].append({
                    'mean_conf': mean_conf,
                    'gradient': gradient,
                    'tail_conf': tail_conf,
                })

        # Process each answer group
        for answer, traces_list in answer_groups.items():
            count = len(traces_list)

            try:
                is_correct = equal_func(str(answer), str(gt))
            except:
                is_correct = str(answer) == str(gt)

            key = 'correct' if is_correct else 'wrong'

            # Trace-level metrics
            for t in traces_list:
                metrics[key]['mean_conf'].append(t['mean_conf'])
                metrics[key]['tail_conf'].append(t['tail_conf'])
                metrics[key]['gradient'].append(t['gradient'])

            # Answer-level metrics
            trace_scores = [t['mean_conf'] + beta * t['gradient'] for t in traces_list]
            cdg_score = (count ** alpha) * np.mean(trace_scores)
            metrics[key]['cdg_score'].append(cdg_score)
            metrics[key]['vote_count'].append(count)

            if is_correct:
                metrics['meta']['n_correct_answers'] += 1
            else:
                metrics['meta']['n_wrong_answers'] += 1

    return metrics


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def create_histogram(ax, correct_data, wrong_data, title, xlabel, n_bins=50, show_stats=True):
    """Create a histogram subplot."""
    all_data = correct_data + wrong_data
    if not all_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    vmin, vmax = np.percentile(all_data, [1, 99])
    bins = np.linspace(vmin, vmax, n_bins + 1)

    ax.hist(correct_data, bins=bins, alpha=0.6, label='Correct', color='#2ca02c',
            density=True, edgecolor='darkgreen', linewidth=0.5)
    ax.hist(wrong_data, bins=bins, alpha=0.6, label='Wrong', color='#d62728',
            density=True, edgecolor='darkred', linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    if show_stats and len(correct_data) > 1 and len(wrong_data) > 1:
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(correct_data, wrong_data)
            correct_mean = np.mean(correct_data)
            wrong_mean = np.mean(wrong_data)

            stats_text = f'Correct: {correct_mean:.3f}\nWrong: {wrong_mean:.3f}\np={p_val:.2e}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except ImportError:
            pass


def create_count_histogram(ax, correct_counts, wrong_counts, title='Vote Count Distribution'):
    """Create histogram for vote counts."""
    if not correct_counts and not wrong_counts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

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

    ax.set_xlabel('Answer Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax.set_xticklabels([str(x_vals[i]) for i in range(0, len(x_vals), max(1, len(x_vals)//10))])
    ax.legend(loc='upper right', fontsize=9)


def generate_figure(model: str, dataset: str, metrics: dict, output_dir: Path):
    """Generate combined figure for one model-dataset pair."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    model_name = MODEL_NAMES.get(model, model)
    dataset_name = DATASET_NAMES.get(dataset, dataset)

    # 1. Vote count
    create_count_histogram(axes[0], metrics['correct']['vote_count'], metrics['wrong']['vote_count'],
                          title='(a) Vote Count (per answer)')

    # 2. Mean confidence
    create_histogram(axes[1], metrics['correct']['mean_conf'], metrics['wrong']['mean_conf'],
                    title='(b) Mean Confidence (per trace)', xlabel='Mean Token Confidence')

    # 3. Tail confidence
    create_histogram(axes[2], metrics['correct']['tail_conf'], metrics['wrong']['tail_conf'],
                    title='(c) Tail Confidence (per trace)', xlabel='Tail Token Confidence')

    # 4. CDG score
    create_histogram(axes[3], metrics['correct']['cdg_score'], metrics['wrong']['cdg_score'],
                    title='(d) CDG Score (per answer)', xlabel='CDG Score')

    # 5. Gradient
    create_histogram(axes[4], metrics['correct']['gradient'], metrics['wrong']['gradient'],
                    title='(e) Confidence Gradient (per trace)', xlabel='Gradient (Tail - Head)')

    # 6. Summary
    axes[5].axis('off')
    summary = f"""Summary

Model: {model_name}
Dataset: {dataset_name}

Questions: {metrics['meta']['n_questions']}
Correct answers: {metrics['meta']['n_correct_answers']}
Wrong answers: {metrics['meta']['n_wrong_answers']}
"""
    axes[5].text(0.1, 0.9, summary, transform=axes[5].transAxes,
                fontsize=11, verticalalignment='top', family='monospace')

    fig.suptitle(f'Correct vs Wrong: {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / f'histogram_{model}_{dataset}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate histogram figures')
    parser.add_argument('--model', type=str, default=None, help='Specific model')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset')
    parser.add_argument('--all', action='store_true', help='Generate for all combinations')
    parser.add_argument('--no-compute', action='store_true',
                        help='Load from cache and regenerate figures only (no computation)')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Skip cache and always extract fresh metrics')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('HISTOGRAM FIGURES: Correct vs Wrong Distributions')
    print('=' * 70)
    if args.no_compute:
        print('MODE: no-compute (loading from cache)')
    elif args.force_recompute:
        print('MODE: force-recompute (extracting fresh metrics)')

    def process_pair(model, dataset):
        """Process a single model-dataset pair."""
        # Try cache first (unless --force-recompute)
        if not args.force_recompute:
            metrics = load_cache(model, dataset)
            if metrics:
                generate_figure(model, dataset, metrics, OUTPUT_DIR)
                return True

        # If no-compute mode and no cache, skip
        if args.no_compute:
            print(f'  WARNING: No cache for {model}-{dataset}, skipping')
            return False

        # Extract fresh metrics
        dataset_path = DATASETS[model][dataset]
        cdg_params = CDG_PARAMS[model]
        metrics = extract_metrics(dataset_path, cdg_params)
        if metrics:
            save_cache(model, dataset, metrics)
            generate_figure(model, dataset, metrics, OUTPUT_DIR)
            return True
        return False

    if args.all or (args.model is None and args.dataset is None):
        # Generate for all
        for model in DATASETS.keys():
            for dataset in DATASETS[model].keys():
                print(f'\nProcessing {model} - {dataset}...')
                process_pair(model, dataset)
    else:
        # Single model-dataset
        model = args.model or 'qwq32b'
        dataset = args.dataset or 'aime2025'
        print(f'\nProcessing {model} - {dataset}...')
        process_pair(model, dataset)

    print('\n*** Cache is stored in: {} ***'.format(CACHE_DIR))
    print('Use --no-compute to regenerate figures from cache')
    print('\nDone!')


if __name__ == '__main__':
    main()
