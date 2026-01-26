#!/usr/bin/env python3
"""
Statistical significance test for Figure 2 (confidence curves).

Tests whether correct traces (green) have significantly higher confidence
than incorrect traces (red) at each position bin.

Statistical tests used:
- Mann-Whitney U test (non-parametric, independent samples)
- Welch's t-test (parametric, unequal variances)
- Cohen's d effect size

Usage:
    python significance_test_confidence_curve.py                    # Run on AIME 2025
    python significance_test_confidence_curve.py --dataset aime2024 # Different dataset
    python significance_test_confidence_curve.py --no-compute       # Use cached data
    python significance_test_confidence_curve.py --all-datasets     # All 4 datasets
"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

from config import DATASETS, OUTPUT_DIRS

# =============================================================================
# Constants
# =============================================================================

MODEL_ORDER = ['deepseek8b', 'gemma3_27b', 'qwq32b', 'gptoss20b']
MODEL_DISPLAY = {
    'deepseek8b': 'DeepSeek-R1-8B',
    'gemma3_27b': 'GEMMA-3-27B',
    'qwq32b': 'QWQ-32B',
    'gptoss20b': 'gpt-oss-20B',
}

DATASET_ORDER = ['aime2024', 'aime2025', 'brumo2025', 'hmmt2025']
NUM_BINS = 10

# Cache directory
CACHE_DIR = OUTPUT_DIRS.get('util_histogram_cache', Path(__file__).parent.parent.parent / 'results' / 'cache' / 'histogram') / 'significance_test_cache'


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
        from dynasor.core.evaluator import math_equal
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


def collect_confidence_by_position(questions: list, num_bins: int = 10) -> dict:
    """Collect all confidence values by position for correct/wrong traces."""
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

    return {
        'correct': {pos: correct_by_position[pos] for pos in range(1, num_bins + 1)},
        'wrong': {pos: wrong_by_position[pos] for pos in range(1, num_bins + 1)},
    }


# =============================================================================
# Statistical Tests
# =============================================================================

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def significance_stars(p):
    """Return significance stars for p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def run_significance_tests(data: dict) -> dict:
    """Run statistical tests at each position bin."""
    results = {}

    for pos in range(1, NUM_BINS + 1):
        correct_vals = np.array(data['correct'].get(pos, []))
        wrong_vals = np.array(data['wrong'].get(pos, []))

        if len(correct_vals) < 5 or len(wrong_vals) < 5:
            results[pos] = {
                'n_correct': len(correct_vals),
                'n_wrong': len(wrong_vals),
                'error': 'Insufficient samples'
            }
            continue

        # Basic statistics
        mean_correct = np.mean(correct_vals)
        mean_wrong = np.mean(wrong_vals)
        std_correct = np.std(correct_vals, ddof=1)
        std_wrong = np.std(wrong_vals, ddof=1)

        # Mann-Whitney U test (non-parametric)
        # Alternative: 'greater' tests if correct > wrong
        u_stat, mw_pvalue = stats.mannwhitneyu(correct_vals, wrong_vals, alternative='greater')

        # Welch's t-test (unequal variances)
        t_stat, t_pvalue = stats.ttest_ind(correct_vals, wrong_vals, equal_var=False, alternative='greater')

        # Effect size (Cohen's d)
        d = cohens_d(correct_vals, wrong_vals)

        results[pos] = {
            'n_correct': len(correct_vals),
            'n_wrong': len(wrong_vals),
            'mean_correct': mean_correct,
            'mean_wrong': mean_wrong,
            'std_correct': std_correct,
            'std_wrong': std_wrong,
            'diff': mean_correct - mean_wrong,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': mw_pvalue,
            'welch_t': t_stat,
            'welch_p': t_pvalue,
            'cohens_d': d,
            'effect_interpretation': interpret_effect_size(d),
        }

    return results


# =============================================================================
# Cache Functions
# =============================================================================

def get_cache_path(model: str, dataset: str) -> Path:
    """Get cache file path for raw data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f'{model}_{dataset}_raw_data.json'


def save_cache(model: str, dataset: str, data: dict):
    """Save raw data to cache."""
    cache_path = get_cache_path(model, dataset)
    # Convert to JSON-serializable format
    serializable = {
        'correct': {str(k): v for k, v in data['correct'].items()},
        'wrong': {str(k): v for k, v in data['wrong'].items()},
    }
    with open(cache_path, 'w') as f:
        json.dump(serializable, f)
    print(f"  Cached raw data to: {cache_path}")


def load_cache(model: str, dataset: str) -> dict:
    """Load cached raw data."""
    cache_path = get_cache_path(model, dataset)
    if not cache_path.exists():
        return None
    with open(cache_path, 'r') as f:
        loaded = json.load(f)
    # Convert keys back to integers
    return {
        'correct': {int(k): v for k, v in loaded['correct'].items()},
        'wrong': {int(k): v for k, v in loaded['wrong'].items()},
    }


# =============================================================================
# Reporting
# =============================================================================

def print_results_table(model: str, dataset: str, results: dict):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"Model: {MODEL_DISPLAY.get(model, model)} | Dataset: {dataset.upper()}")
    print(f"{'='*90}")

    # Header
    print(f"{'Pos':<5} {'N_corr':<8} {'N_wrong':<8} {'Mean_C':<8} {'Mean_W':<8} {'Diff':<8} {'M-W p':<12} {'t-test p':<12} {'Cohen d':<8} {'Effect':<10}")
    print("-" * 90)

    for pos in range(1, NUM_BINS + 1):
        r = results.get(pos, {})
        if 'error' in r:
            print(f"{pos:<5} {r.get('n_correct', 0):<8} {r.get('n_wrong', 0):<8} {'---':<8} {'---':<8} {'---':<8} {r['error']}")
            continue

        mw_p_str = f"{r['mann_whitney_p']:.2e}" if r['mann_whitney_p'] < 0.01 else f"{r['mann_whitney_p']:.4f}"
        t_p_str = f"{r['welch_p']:.2e}" if r['welch_p'] < 0.01 else f"{r['welch_p']:.4f}"

        print(f"{pos:<5} {r['n_correct']:<8} {r['n_wrong']:<8} {r['mean_correct']:<8.4f} {r['mean_wrong']:<8.4f} "
              f"{r['diff']:<8.4f} {mw_p_str:<12} {t_p_str:<12} {r['cohens_d']:<8.3f} {r['effect_interpretation']:<10}")

    # Summary
    print("-" * 90)

    # Check if all positions are significant
    all_significant = all(
        results.get(pos, {}).get('mann_whitney_p', 1) < 0.05
        for pos in range(1, NUM_BINS + 1)
        if 'error' not in results.get(pos, {})
    )

    # Average effect size
    effect_sizes = [results[pos]['cohens_d'] for pos in range(1, NUM_BINS + 1) if 'cohens_d' in results.get(pos, {})]
    avg_effect = np.mean(effect_sizes) if effect_sizes else 0

    print(f"Summary: All positions significant (p<0.05): {'YES' if all_significant else 'NO'}")
    print(f"         Average Cohen's d: {avg_effect:.3f} ({interpret_effect_size(avg_effect)})")


def print_latex_table(all_results: dict):
    """Print LaTeX-formatted table for paper."""
    print("\n" + "=" * 80)
    print("LATEX TABLE (for paper appendix)")
    print("=" * 80)

    print(r"""
\begin{table}[h]
\centering
\caption{Statistical significance of confidence difference between correct and incorrect traces.
Mann-Whitney U test (one-sided, correct $>$ incorrect). $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}
\label{tab:significance}
\begin{tabular}{l|cccccccccc}
\toprule
Model & P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9 & P10 \\
\midrule""")

    for model in MODEL_ORDER:
        for dataset, results in all_results.get(model, {}).items():
            row = f"{MODEL_DISPLAY[model]} ({dataset}) "
            for pos in range(1, NUM_BINS + 1):
                r = results.get(pos, {})
                if 'mann_whitney_p' in r:
                    stars = significance_stars(r['mann_whitney_p'])
                    row += f"& {stars} "
                else:
                    row += "& -- "
            row += r"\\"
            print(row)

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# =============================================================================
# Main
# =============================================================================

def analyze_model_dataset(model: str, dataset: str, use_cache: bool = True) -> dict:
    """Analyze a single model-dataset pair."""
    # Try cache first
    if use_cache:
        data = load_cache(model, dataset)
        if data:
            print(f"Loaded {model}/{dataset} from cache")
            return run_significance_tests(data)

    # Load raw data
    dataset_path = DATASETS.get(model, {}).get(dataset)
    if not dataset_path:
        print(f"Warning: No path for {model}/{dataset}")
        return {}

    print(f"Loading {model}/{dataset}...")
    questions = load_all_questions(dataset_path)
    print(f"  Loaded {len(questions)} questions")

    if not questions:
        return {}

    # Collect confidence values
    data = collect_confidence_by_position(questions, NUM_BINS)

    # Save to cache
    save_cache(model, dataset, data)

    # Run tests
    return run_significance_tests(data)


def main():
    parser = argparse.ArgumentParser(description='Significance test for confidence curves')
    parser.add_argument('--dataset', '-d', type=str, default='aime2025',
                        choices=DATASET_ORDER,
                        help='Dataset to analyze (default: aime2025)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        choices=MODEL_ORDER,
                        help='Single model to analyze (default: all)')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Analyze all datasets')
    parser.add_argument('--no-compute', action='store_true',
                        help='Only use cached data, do not compute')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    args = parser.parse_args()

    print("=" * 80)
    print("SIGNIFICANCE TEST: Correct vs Incorrect Confidence Curves")
    print("Testing H0: confidence(correct) <= confidence(incorrect)")
    print("Testing H1: confidence(correct) > confidence(incorrect)")
    print("=" * 80)

    datasets = DATASET_ORDER if args.all_datasets else [args.dataset]
    models = [args.model] if args.model else MODEL_ORDER

    all_results = defaultdict(dict)

    for model in models:
        for dataset in datasets:
            results = analyze_model_dataset(model, dataset, use_cache=args.no_compute)
            if results:
                all_results[model][dataset] = results
                print_results_table(model, dataset, results)

    if args.latex:
        print_latex_table(all_results)

    # Final summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_tests = 0
    significant_tests = 0

    for model in all_results:
        for dataset in all_results[model]:
            for pos in range(1, NUM_BINS + 1):
                r = all_results[model][dataset].get(pos, {})
                if 'mann_whitney_p' in r:
                    total_tests += 1
                    if r['mann_whitney_p'] < 0.05:
                        significant_tests += 1

    print(f"Total position tests: {total_tests}")
    print(f"Significant (p<0.05): {significant_tests} ({100*significant_tests/total_tests:.1f}%)" if total_tests > 0 else "No tests run")
    print("\nConclusion: " + ("Green line (correct) is SIGNIFICANTLY above red line (incorrect)"
                             if significant_tests == total_tests and total_tests > 0
                             else f"{significant_tests}/{total_tests} positions show significant difference"))


if __name__ == '__main__':
    main()
