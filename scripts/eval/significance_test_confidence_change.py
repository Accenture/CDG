#!/usr/bin/env python3
"""
Statistical significance test for confidence change (end - start).

Tests whether correct traces (green) have significantly larger confidence
increase (end - start) than incorrect traces (red).

Statistical tests used:
- Mann-Whitney U test (non-parametric, independent samples)
- Welch's t-test (parametric, unequal variances)
- Cohen's d effect size

Usage:
    python significance_test_confidence_change.py                    # Run on AIME 2025
    python significance_test_confidence_change.py --dataset aime2024 # Different dataset
    python significance_test_confidence_change.py --no-compute       # Use cached data
    python significance_test_confidence_change.py --all-datasets     # All 4 datasets
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


def compute_confidence_change(questions: list, num_bins: int = 10) -> dict:
    """Compute confidence change (end - start) for correct/wrong traces."""
    correct_changes = []
    wrong_changes = []

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

            # Compute change: end (last bin) - start (first bin)
            start_conf = bin_means[0]
            end_conf = bin_means[-1]

            if np.isnan(start_conf) or np.isnan(end_conf):
                continue

            conf_change = end_conf - start_conf

            if is_correct:
                correct_changes.append(conf_change)
            else:
                wrong_changes.append(conf_change)

    return {
        'correct': correct_changes,
        'wrong': wrong_changes,
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
    """Run statistical tests on confidence change data."""
    correct_vals = np.array(data['correct'])
    wrong_vals = np.array(data['wrong'])

    if len(correct_vals) < 5 or len(wrong_vals) < 5:
        return {
            'n_correct': len(correct_vals),
            'n_wrong': len(wrong_vals),
            'error': 'Insufficient samples'
        }

    # Basic statistics
    mean_correct = np.mean(correct_vals)
    mean_wrong = np.mean(wrong_vals)
    std_correct = np.std(correct_vals, ddof=1)
    std_wrong = np.std(wrong_vals, ddof=1)
    median_correct = np.median(correct_vals)
    median_wrong = np.median(wrong_vals)

    # Mann-Whitney U test (non-parametric)
    # Alternative: 'greater' tests if correct > wrong
    u_stat, mw_pvalue = stats.mannwhitneyu(correct_vals, wrong_vals, alternative='greater')

    # Welch's t-test (unequal variances)
    t_stat, t_pvalue = stats.ttest_ind(correct_vals, wrong_vals, equal_var=False, alternative='greater')

    # Effect size (Cohen's d)
    d = cohens_d(correct_vals, wrong_vals)

    return {
        'n_correct': len(correct_vals),
        'n_wrong': len(wrong_vals),
        'mean_correct': mean_correct,
        'mean_wrong': mean_wrong,
        'std_correct': std_correct,
        'std_wrong': std_wrong,
        'median_correct': median_correct,
        'median_wrong': median_wrong,
        'diff': mean_correct - mean_wrong,
        'mann_whitney_u': u_stat,
        'mann_whitney_p': mw_pvalue,
        'welch_t': t_stat,
        'welch_p': t_pvalue,
        'cohens_d': d,
        'effect_interpretation': interpret_effect_size(d),
    }


# =============================================================================
# Cache Functions
# =============================================================================

def get_cache_path(model: str, dataset: str) -> Path:
    """Get cache file path for raw data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f'{model}_{dataset}_conf_change_data.json'


def save_cache(model: str, dataset: str, data: dict):
    """Save raw data to cache."""
    cache_path = get_cache_path(model, dataset)
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    print(f"  Cached raw data to: {cache_path}")


def load_cache(model: str, dataset: str) -> dict:
    """Load cached raw data."""
    cache_path = get_cache_path(model, dataset)
    if not cache_path.exists():
        return None
    with open(cache_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Reporting
# =============================================================================

def print_results(model: str, dataset: str, results: dict):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"Model: {MODEL_DISPLAY.get(model, model)} | Dataset: {dataset.upper()}")
    print(f"{'='*80}")

    if 'error' in results:
        print(f"Error: {results['error']}")
        print(f"N_correct: {results.get('n_correct', 0)}, N_wrong: {results.get('n_wrong', 0)}")
        return

    print(f"\nSample Sizes:")
    print(f"  Correct traces (green): {results['n_correct']}")
    print(f"  Wrong traces (red):     {results['n_wrong']}")

    print(f"\nConfidence Change (end - start):")
    print(f"  Correct: mean={results['mean_correct']:.4f}, std={results['std_correct']:.4f}, median={results['median_correct']:.4f}")
    print(f"  Wrong:   mean={results['mean_wrong']:.4f}, std={results['std_wrong']:.4f}, median={results['median_wrong']:.4f}")
    print(f"  Difference (correct - wrong): {results['diff']:.4f}")

    print(f"\nStatistical Tests (H1: correct change > wrong change):")
    mw_stars = significance_stars(results['mann_whitney_p'])
    t_stars = significance_stars(results['welch_p'])
    print(f"  Mann-Whitney U: U={results['mann_whitney_u']:.1f}, p={results['mann_whitney_p']:.2e} {mw_stars}")
    print(f"  Welch's t-test: t={results['welch_t']:.3f}, p={results['welch_p']:.2e} {t_stars}")

    print(f"\nEffect Size:")
    print(f"  Cohen's d: {results['cohens_d']:.3f} ({results['effect_interpretation']})")

    # Interpretation
    print(f"\nInterpretation:")
    if results['mann_whitney_p'] < 0.05:
        print(f"  Correct traces show SIGNIFICANTLY larger confidence increase than wrong traces.")
    else:
        print(f"  No significant difference in confidence change between correct and wrong traces.")


def print_summary_table(all_results: dict):
    """Print summary table for all model-dataset combinations."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Confidence Change (end - start)")
    print("=" * 100)

    # Header
    print(f"{'Model':<20} {'Dataset':<12} {'N_corr':<8} {'N_wrong':<8} {'Mean_C':<10} {'Mean_W':<10} {'Diff':<10} {'M-W p':<12} {'Cohen d':<10} {'Sig':<6}")
    print("-" * 100)

    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            r = all_results.get(model, {}).get(dataset, {})
            if not r or 'error' in r:
                continue

            mw_p_str = f"{r['mann_whitney_p']:.2e}" if r['mann_whitney_p'] < 0.01 else f"{r['mann_whitney_p']:.4f}"
            sig = significance_stars(r['mann_whitney_p'])

            print(f"{MODEL_DISPLAY[model]:<20} {dataset:<12} {r['n_correct']:<8} {r['n_wrong']:<8} "
                  f"{r['mean_correct']:<10.4f} {r['mean_wrong']:<10.4f} {r['diff']:<10.4f} "
                  f"{mw_p_str:<12} {r['cohens_d']:<10.3f} {sig:<6}")


def print_latex_table(all_results: dict):
    """Print LaTeX-formatted table for paper."""
    print("\n" + "=" * 80)
    print("LATEX TABLE (for paper appendix)")
    print("=" * 80)

    print(r"""
\begin{table}[h]
\centering
\caption{Statistical significance of confidence change (end - start) difference between correct and incorrect traces.
Mann-Whitney U test (one-sided, correct $>$ incorrect). $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}
\label{tab:significance_conf_change}
\begin{tabular}{l|l|cccc|c}
\toprule
Model & Dataset & $\bar{\Delta}_{correct}$ & $\bar{\Delta}_{wrong}$ & Cohen's $d$ & $p$-value & Sig. \\
\midrule""")

    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            r = all_results.get(model, {}).get(dataset, {})
            if not r or 'error' in r:
                continue

            stars = significance_stars(r['mann_whitney_p'])
            p_str = f"{r['mann_whitney_p']:.2e}" if r['mann_whitney_p'] < 0.01 else f"{r['mann_whitney_p']:.3f}"

            print(f"{MODEL_DISPLAY[model]} & {dataset} & {r['mean_correct']:.3f} & {r['mean_wrong']:.3f} & "
                  f"{r['cohens_d']:.3f} & {p_str} & {stars} \\\\")

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

    # Compute confidence changes
    data = compute_confidence_change(questions, NUM_BINS)
    print(f"  Correct traces: {len(data['correct'])}, Wrong traces: {len(data['wrong'])}")

    # Save to cache
    save_cache(model, dataset, data)

    # Run tests
    return run_significance_tests(data)


def main():
    parser = argparse.ArgumentParser(description='Significance test for confidence change (end - start)')
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
    print("SIGNIFICANCE TEST: Confidence Change (End - Start)")
    print("Testing H0: change(correct) <= change(incorrect)")
    print("Testing H1: change(correct) > change(incorrect)")
    print("=" * 80)
    print("\nThis test compares whether correct traces show larger confidence")
    print("increases over time compared to incorrect traces.\n")

    datasets = DATASET_ORDER if args.all_datasets else [args.dataset]
    models = [args.model] if args.model else MODEL_ORDER

    all_results = defaultdict(dict)

    for model in models:
        for dataset in datasets:
            results = analyze_model_dataset(model, dataset, use_cache=args.no_compute)
            if results:
                all_results[model][dataset] = results
                print_results(model, dataset, results)

    # Print summary table if multiple datasets
    if args.all_datasets or len(models) > 1:
        print_summary_table(all_results)

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
            r = all_results[model][dataset]
            if 'mann_whitney_p' in r:
                total_tests += 1
                if r['mann_whitney_p'] < 0.05:
                    significant_tests += 1

    print(f"Total model-dataset pairs tested: {total_tests}")
    print(f"Significant (p<0.05): {significant_tests} ({100*significant_tests/total_tests:.1f}%)" if total_tests > 0 else "No tests run")

    if total_tests > 0:
        print("\nConclusion: " + (
            "Correct traces show SIGNIFICANTLY larger confidence increase (end - start) than incorrect traces across all tested combinations."
            if significant_tests == total_tests
            else f"{significant_tests}/{total_tests} model-dataset pairs show significant difference in confidence change."
        ))


if __name__ == '__main__':
    main()
