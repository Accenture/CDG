#!/usr/bin/env python3
"""
Statistical significance test for CDG (ours) vs baseline methods.

Tests whether CDG (red) significantly outperforms other methods across
the 4x4 model-dataset grid from figure_scaling_appendix.py.

Comparisons:
1. CDG vs DeepConf-Tail (Green) - head-to-head comparison
2. CDG vs All Others (Majority, DeepConf-Mean, DeepConf-Tail)

Statistical tests used:
- Wilcoxon signed-rank test (paired, non-parametric)
- Paired t-test (parametric)
- Win/Tie/Loss counts
- Effect size (Cohen's d)

Usage:
    python significance_test_scaling.py              # Full analysis
    python significance_test_scaling.py --latex      # Include LaTeX tables
    python significance_test_scaling.py --trace 512  # Specific trace count only
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

from config import OUTPUT_DIRS, TRACE_COUNTS

# =============================================================================
# Constants
# =============================================================================

MODEL_ORDER = ['deepseek8b', 'gemma3_27b', 'qwq32b', 'gptoss20b']
DATASET_ORDER = ['aime2024', 'aime2025', 'brumo2025', 'hmmt2025']

MODEL_DISPLAY = {
    'deepseek8b': 'DeepSeek-R1-8B',
    'gemma3_27b': 'Gemma-3-27B',
    'qwq32b': 'QWQ-32B',
    'gptoss20b': 'gpt-oss-20B',
}

DATASET_DISPLAY = {
    'aime2024': 'AIME 2024',
    'aime2025': 'AIME 2025',
    'brumo2025': 'BRUMO 2025',
    'hmmt2025': 'HMMT 2025',
}

# Method names matching the cache
METHODS = ['majority', 'mean_weighted', 'top10_tail', 'cdg']

METHOD_DISPLAY = {
    'majority': 'Majority (Blue)',
    'mean_weighted': 'DeepConf-Mean (Orange)',
    'top10_tail': 'DeepConf-Tail (Green)',
    'cdg': 'CDG (Red, Ours)',
}

CACHE_FILE = OUTPUT_DIRS['exp_voting_methods'] / 'cache.json'


# =============================================================================
# Data Loading
# =============================================================================

def load_cache() -> dict:
    """Load cached scaling results."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_FILE}\nRun exp_voting_methods.py first.")
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)


def get_accuracy_by_trace(results: list, model: str, dataset: str, method: str) -> dict:
    """Get accuracy values keyed by trace count for a specific (model, dataset, method)."""
    return {
        r['trace_count']: r['accuracy'] * 100
        for r in results
        if r['model'] == model and r['dataset'] == dataset and r['method'] == method
    }


# =============================================================================
# Statistical Tests
# =============================================================================

def cohens_d_paired(diff):
    """Compute Cohen's d for paired samples (differences)."""
    if len(diff) == 0 or np.std(diff, ddof=1) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff, ddof=1)


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


def run_paired_test(cdg_accs: list, other_accs: list) -> dict:
    """
    Run paired statistical tests comparing CDG vs another method.

    Args:
        cdg_accs: List of CDG accuracies (one per trace count)
        other_accs: List of other method accuracies (same order)

    Returns:
        Dict with test statistics
    """
    cdg = np.array(cdg_accs)
    other = np.array(other_accs)
    diff = cdg - other  # positive means CDG is better

    n = len(diff)
    if n < 3:
        return {'error': 'Insufficient samples', 'n': n}

    # Basic stats
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Win/Tie/Loss
    wins = np.sum(diff > 0)
    ties = np.sum(diff == 0)
    losses = np.sum(diff < 0)

    # Wilcoxon signed-rank test (one-sided: CDG > other)
    # Note: scipy's wilcoxon uses two-sided by default, divide p by 2 for one-sided
    try:
        wilcox_stat, wilcox_p_two = stats.wilcoxon(diff, alternative='greater')
        wilcox_p = wilcox_p_two
    except ValueError:
        # All differences are zero
        wilcox_stat, wilcox_p = 0, 1.0

    # Paired t-test (one-sided: CDG > other)
    t_stat, t_p_two = stats.ttest_rel(cdg, other)
    t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2

    # Effect size
    d = cohens_d_paired(diff)

    return {
        'n': n,
        'mean_cdg': np.mean(cdg),
        'mean_other': np.mean(other),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'wins': int(wins),
        'ties': int(ties),
        'losses': int(losses),
        'wilcoxon_stat': wilcox_stat,
        'wilcoxon_p': wilcox_p,
        't_stat': t_stat,
        't_p': t_p,
        'cohens_d': d,
        'effect_interpretation': interpret_effect_size(d),
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_model_dataset_pair(results: list, model: str, dataset: str,
                                trace_counts: list = None) -> dict:
    """
    Analyze CDG vs all other methods for a specific (model, dataset) pair.

    Returns dict with comparisons against each baseline method.
    """
    if trace_counts is None:
        trace_counts = TRACE_COUNTS

    comparisons = {}

    # Get CDG accuracies
    cdg_by_trace = get_accuracy_by_trace(results, model, dataset, 'cdg')
    cdg_accs = [cdg_by_trace.get(tc) for tc in trace_counts if tc in cdg_by_trace]

    if not cdg_accs:
        return {'error': 'No CDG data'}

    # Compare against each baseline
    for baseline in ['majority', 'mean_weighted', 'top10_tail']:
        baseline_by_trace = get_accuracy_by_trace(results, model, dataset, baseline)

        # Align trace counts
        common_traces = [tc for tc in trace_counts
                         if tc in cdg_by_trace and tc in baseline_by_trace]

        if not common_traces:
            comparisons[baseline] = {'error': 'No common trace counts'}
            continue

        cdg_vals = [cdg_by_trace[tc] for tc in common_traces]
        baseline_vals = [baseline_by_trace[tc] for tc in common_traces]

        comparisons[baseline] = run_paired_test(cdg_vals, baseline_vals)
        comparisons[baseline]['trace_counts'] = common_traces

    return comparisons


def aggregate_across_grid(results: list, trace_counts: list = None) -> dict:
    """
    Aggregate CDG vs baseline comparisons across all 4x4 model-dataset pairs.

    For each baseline method, collects all accuracy differences across:
    - All models (4)
    - All datasets (4)
    - All trace counts (7 by default)

    This gives us 4*4*7 = 112 paired observations for each comparison.
    """
    if trace_counts is None:
        trace_counts = TRACE_COUNTS

    aggregated = {baseline: {'cdg': [], 'other': []} for baseline in ['majority', 'mean_weighted', 'top10_tail']}

    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            cdg_by_trace = get_accuracy_by_trace(results, model, dataset, 'cdg')

            for baseline in aggregated.keys():
                baseline_by_trace = get_accuracy_by_trace(results, model, dataset, baseline)

                for tc in trace_counts:
                    if tc in cdg_by_trace and tc in baseline_by_trace:
                        aggregated[baseline]['cdg'].append(cdg_by_trace[tc])
                        aggregated[baseline]['other'].append(baseline_by_trace[tc])

    # Run tests on aggregated data
    aggregate_results = {}
    for baseline, data in aggregated.items():
        if data['cdg'] and data['other']:
            aggregate_results[baseline] = run_paired_test(data['cdg'], data['other'])

    return aggregate_results


def aggregate_per_model(results: list, trace_counts: list = None) -> dict:
    """Aggregate comparisons per model (across all datasets and trace counts)."""
    if trace_counts is None:
        trace_counts = TRACE_COUNTS

    per_model = {}

    for model in MODEL_ORDER:
        model_data = {baseline: {'cdg': [], 'other': []}
                      for baseline in ['majority', 'mean_weighted', 'top10_tail']}

        for dataset in DATASET_ORDER:
            cdg_by_trace = get_accuracy_by_trace(results, model, dataset, 'cdg')

            for baseline in model_data.keys():
                baseline_by_trace = get_accuracy_by_trace(results, model, dataset, baseline)

                for tc in trace_counts:
                    if tc in cdg_by_trace and tc in baseline_by_trace:
                        model_data[baseline]['cdg'].append(cdg_by_trace[tc])
                        model_data[baseline]['other'].append(baseline_by_trace[tc])

        per_model[model] = {}
        for baseline, data in model_data.items():
            if data['cdg'] and data['other']:
                per_model[model][baseline] = run_paired_test(data['cdg'], data['other'])

    return per_model


# =============================================================================
# Reporting
# =============================================================================

def print_comparison(baseline: str, result: dict, indent: str = ""):
    """Print a single comparison result."""
    if 'error' in result:
        print(f"{indent}{METHOD_DISPLAY[baseline]}: {result['error']}")
        return

    wilcox_stars = significance_stars(result['wilcoxon_p'])
    t_stars = significance_stars(result['t_p'])

    print(f"{indent}CDG vs {METHOD_DISPLAY[baseline]}:")
    print(f"{indent}  N pairs: {result['n']}")
    print(f"{indent}  Mean CDG: {result['mean_cdg']:.2f}%, Mean Other: {result['mean_other']:.2f}%")
    print(f"{indent}  Mean diff: {result['mean_diff']:+.2f}% (CDG - other)")
    print(f"{indent}  Win/Tie/Loss: {result['wins']}/{result['ties']}/{result['losses']}")
    print(f"{indent}  Wilcoxon signed-rank: W={result['wilcoxon_stat']:.1f}, p={result['wilcoxon_p']:.4f} {wilcox_stars}")
    print(f"{indent}  Paired t-test: t={result['t_stat']:.3f}, p={result['t_p']:.4f} {t_stars}")
    print(f"{indent}  Cohen's d: {result['cohens_d']:.3f} ({result['effect_interpretation']})")


def print_aggregate_summary(aggregate_results: dict):
    """Print aggregate summary across all model-dataset pairs."""
    print("\n" + "=" * 100)
    print("AGGREGATE ANALYSIS: CDG vs Baselines (All 4x4 Model-Dataset Pairs)")
    print("=" * 100)
    print("\nPooling all (model, dataset, trace_count) combinations:")

    for baseline in ['top10_tail', 'mean_weighted', 'majority']:
        if baseline in aggregate_results:
            print()
            print_comparison(baseline, aggregate_results[baseline])


def print_per_model_summary(per_model: dict):
    """Print per-model summary."""
    print("\n" + "=" * 100)
    print("PER-MODEL ANALYSIS: CDG vs Baselines")
    print("=" * 100)

    for model in MODEL_ORDER:
        print(f"\n{MODEL_DISPLAY[model]}")
        print("-" * 60)

        if model not in per_model:
            print("  No data")
            continue

        for baseline in ['top10_tail', 'mean_weighted', 'majority']:
            if baseline in per_model[model]:
                print_comparison(baseline, per_model[model][baseline], indent="  ")
                print()


def print_summary_table(aggregate_results: dict, per_model: dict):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: CDG vs Baselines")
    print("=" * 100)

    header = f"{'Comparison':<35} {'N':<6} {'Mean Δ':<10} {'W/T/L':<12} {'Wilcoxon p':<14} {'Cohen d':<10} {'Sig':<6}"
    print(header)
    print("-" * 100)

    # Aggregate results
    print("AGGREGATE (all model-dataset pairs):")
    for baseline in ['top10_tail', 'mean_weighted', 'majority']:
        r = aggregate_results.get(baseline, {})
        if 'error' in r or not r:
            continue

        name = f"CDG vs {baseline}"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        p_str = f"{r['wilcoxon_p']:.2e}" if r['wilcoxon_p'] < 0.01 else f"{r['wilcoxon_p']:.4f}"
        sig = significance_stars(r['wilcoxon_p'])

        print(f"  {name:<33} {r['n']:<6} {r['mean_diff']:+.2f}%     {wtl:<12} {p_str:<14} {r['cohens_d']:.3f}      {sig:<6}")

    # Per-model results (CDG vs DeepConf-Tail only, as primary comparison)
    print("\nPER-MODEL (CDG vs DeepConf-Tail):")
    for model in MODEL_ORDER:
        r = per_model.get(model, {}).get('top10_tail', {})
        if 'error' in r or not r:
            continue

        name = f"  {MODEL_DISPLAY[model]}"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        p_str = f"{r['wilcoxon_p']:.2e}" if r['wilcoxon_p'] < 0.01 else f"{r['wilcoxon_p']:.4f}"
        sig = significance_stars(r['wilcoxon_p'])

        print(f"{name:<35} {r['n']:<6} {r['mean_diff']:+.2f}%     {wtl:<12} {p_str:<14} {r['cohens_d']:.3f}      {sig:<6}")


def print_latex_table(aggregate_results: dict, per_model: dict):
    """Print LaTeX-formatted table."""
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    print(r"""
\begin{table}[h]
\centering
\caption{Statistical significance of CDG improvement over baseline methods.
Wilcoxon signed-rank test (one-sided, CDG $>$ baseline). W/T/L = Win/Tie/Loss counts.
$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}
\label{tab:significance_scaling}
\begin{tabular}{l|ccccc}
\toprule
Comparison & $N$ & Mean $\Delta$ (\%) & W/T/L & $p$-value & Cohen's $d$ \\
\midrule""")

    # Aggregate
    print(r"\multicolumn{6}{l}{\textit{Aggregate (all model-dataset pairs)}} \\")
    for baseline in ['top10_tail', 'mean_weighted', 'majority']:
        r = aggregate_results.get(baseline, {})
        if 'error' in r or not r:
            continue

        baseline_name = {'top10_tail': 'DeepConf-Tail', 'mean_weighted': 'DeepConf-Mean', 'majority': 'Majority'}[baseline]
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        p_str = f"{r['wilcoxon_p']:.2e}" if r['wilcoxon_p'] < 0.01 else f"{r['wilcoxon_p']:.3f}"
        stars = significance_stars(r['wilcoxon_p']).replace('n.s.', '')

        print(f"CDG vs {baseline_name} & {r['n']} & {r['mean_diff']:+.2f} & {wtl} & {p_str}{stars} & {r['cohens_d']:.3f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{6}{l}{\textit{Per-model (CDG vs DeepConf-Tail)}} \\")

    for model in MODEL_ORDER:
        r = per_model.get(model, {}).get('top10_tail', {})
        if 'error' in r or not r:
            continue

        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        p_str = f"{r['wilcoxon_p']:.2e}" if r['wilcoxon_p'] < 0.01 else f"{r['wilcoxon_p']:.3f}"
        stars = significance_stars(r['wilcoxon_p']).replace('n.s.', '')

        print(f"{MODEL_DISPLAY[model]} & {r['n']} & {r['mean_diff']:+.2f} & {wtl} & {p_str}{stars} & {r['cohens_d']:.3f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def print_final_conclusion(aggregate_results: dict):
    """Print final conclusion."""
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    # Check CDG vs DeepConf-Tail (the key comparison: Red vs Green)
    tail_result = aggregate_results.get('top10_tail', {})

    if 'error' not in tail_result and tail_result:
        p = tail_result['wilcoxon_p']
        d = tail_result['cohens_d']
        wtl = f"{tail_result['wins']}/{tail_result['ties']}/{tail_result['losses']}"

        print(f"\n1. CDG (Red) vs DeepConf-Tail (Green):")
        print(f"   - Mean improvement: {tail_result['mean_diff']:+.2f}%")
        print(f"   - Win/Tie/Loss: {wtl}")
        print(f"   - Wilcoxon p-value: {p:.4f} ({significance_stars(p)})")
        print(f"   - Effect size: {d:.3f} ({tail_result['effect_interpretation']})")

        if p < 0.05:
            print(f"   => CDG SIGNIFICANTLY outperforms DeepConf-Tail")
        else:
            print(f"   => No significant difference")

    # Check CDG vs all others
    all_significant = True
    print(f"\n2. CDG (Red) vs All Baselines:")
    for baseline in ['top10_tail', 'mean_weighted', 'majority']:
        r = aggregate_results.get(baseline, {})
        if 'error' not in r and r:
            stars = significance_stars(r['wilcoxon_p'])
            print(f"   - vs {baseline}: p={r['wilcoxon_p']:.4f} {stars}, Δ={r['mean_diff']:+.2f}%")
            if r['wilcoxon_p'] >= 0.05:
                all_significant = False

    if all_significant:
        print(f"\n   => CDG SIGNIFICANTLY outperforms ALL baselines")
    else:
        print(f"\n   => CDG does not significantly outperform all baselines")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Significance test for CDG vs baselines')
    parser.add_argument('--trace', type=int, default=None,
                        help='Analyze specific trace count only (e.g., 512)')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    args = parser.parse_args()

    print("=" * 100)
    print("SIGNIFICANCE TEST: CDG (Red/Ours) vs Baseline Methods")
    print("=" * 100)
    print("\nThis test evaluates whether CDG significantly outperforms baselines")
    print("across the 4x4 model-dataset grid (4 models × 4 datasets × 7 trace counts).\n")

    # Load data
    cache_data = load_cache()
    results = cache_data.get('results', cache_data)
    print(f"Loaded {len(results)} result entries from cache.\n")

    # Determine trace counts to analyze
    if args.trace:
        trace_counts = [args.trace]
        print(f"Analyzing trace count: {args.trace} only\n")
    else:
        trace_counts = TRACE_COUNTS
        print(f"Analyzing all trace counts: {trace_counts}\n")

    # Run aggregate analysis
    aggregate_results = aggregate_across_grid(results, trace_counts)

    # Run per-model analysis
    per_model = aggregate_per_model(results, trace_counts)

    # Print results
    print_aggregate_summary(aggregate_results)
    print_per_model_summary(per_model)
    print_summary_table(aggregate_results, per_model)

    if args.latex:
        print_latex_table(aggregate_results, per_model)

    print_final_conclusion(aggregate_results)


if __name__ == '__main__':
    main()
