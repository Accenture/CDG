#!/usr/bin/env python3
"""
Pass@1 Baseline Computation

Computes Pass@1 accuracy: the model's raw accuracy on a single randomly sampled trace.
This is Baseline 1 for the appendix table.

For each question, we sample a single trace multiple times and compute the
expected accuracy (equivalent to the fraction of correct traces).

Usage:
    python pass_at_1.py                    # Run full analysis
    python pass_at_1.py --no-compute       # Load from cache
    python pass_at_1.py --latex            # Output LaTeX table

Output format matches Tablebench.tex structure.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import random

from config import (
    DATASETS, RANDOM_SEED, MODEL_NAMES, DATASET_NAMES,
    RESULTS_DIR, CACHE_DIR
)

# dynasor for math evaluation
from dynasor.core.evaluator import math_equal

OUTPUT_DIR = CACHE_DIR / 'pass_at_1'
CACHE_FILE = OUTPUT_DIR / 'cache.json'

# Model order matching Tablebench.tex
MODEL_ORDER = ['deepseek8b', 'gemma3_27b', 'gptoss20b', 'qwq32b']
DATASET_ORDER = ['aime2024', 'aime2025', 'brumo2025', 'hmmt2025']

# Number of subsamples for Pass@1 estimation
NUM_SUBSAMPLES = 5


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


# ============================================================================
# PASS@1 COMPUTATION
# ============================================================================

def compute_pass_at_1(questions: list, num_subsamples: int = 100) -> dict:
    """
    Compute Pass@1 accuracy.

    Pass@1 = probability that a randomly sampled trace is correct.
    This is equivalent to: (# correct traces) / (# total traces with valid answers)

    We also compute it via Monte Carlo sampling for variance estimation.
    """
    # Method 1: Exact computation (fraction of correct traces)
    total_correct = 0
    total_valid = 0

    per_question_correct_rate = []

    for q in questions:
        gt = q.get('ground_truth', '')
        traces = q.get('all_traces', [])

        if not gt:
            continue

        # Count correct traces for this question
        valid_traces = [t for t in traces if t.get('extracted_answer') is not None]
        if not valid_traces:
            continue

        correct_count = 0
        for t in valid_traces:
            answer = t['extracted_answer']
            try:
                if equal_func(str(answer), str(gt)):
                    correct_count += 1
            except:
                pass

        total_correct += correct_count
        total_valid += len(valid_traces)
        per_question_correct_rate.append(correct_count / len(valid_traces))

    # Exact Pass@1
    exact_pass_at_1 = total_correct / total_valid if total_valid > 0 else 0.0

    # Method 2: Monte Carlo sampling (for variance estimation)
    mc_results = []
    for _ in range(num_subsamples):
        subsample_correct = 0
        subsample_total = 0

        for q in questions:
            gt = q.get('ground_truth', '')
            traces = q.get('all_traces', [])

            if not gt:
                continue

            valid_traces = [t for t in traces if t.get('extracted_answer') is not None]
            if not valid_traces:
                continue

            # Sample one trace randomly
            sampled_trace = random.choice(valid_traces)
            answer = sampled_trace['extracted_answer']

            try:
                if equal_func(str(answer), str(gt)):
                    subsample_correct += 1
            except:
                pass
            subsample_total += 1

        if subsample_total > 0:
            mc_results.append(subsample_correct / subsample_total)

    mc_mean = np.mean(mc_results) if mc_results else 0.0
    mc_std = np.std(mc_results) if mc_results else 0.0

    return {
        'pass_at_1': exact_pass_at_1,
        'pass_at_1_mc_mean': mc_mean,
        'pass_at_1_mc_std': mc_std,
        'n_questions': len(questions),
        'n_valid_traces': total_valid,
        'n_correct_traces': total_correct,
        'per_question_mean': np.mean(per_question_correct_rate) if per_question_correct_rate else 0.0,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis():
    """Run Pass@1 analysis for all models and datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print('=' * 80)
    print('PASS@1 BASELINE COMPUTATION')
    print('=' * 80)
    print()

    all_results = {}

    for model in MODEL_ORDER:
        print(f'\n{"-"*60}')
        print(f'{MODEL_NAMES.get(model, model)}')
        print(f'{"-"*60}')

        all_results[model] = {}

        for dataset in DATASET_ORDER:
            dataset_path = DATASETS.get(model, {}).get(dataset)
            if not dataset_path:
                print(f'  {dataset}: [NO PATH]')
                continue

            questions = load_all_questions(dataset_path)
            if not questions:
                print(f'  {dataset}: [NO DATA]')
                continue

            result = compute_pass_at_1(questions, NUM_SUBSAMPLES)
            all_results[model][dataset] = result

            print(f'  {dataset}: Pass@1 = {100*result["pass_at_1"]:.1f}% '
                  f'(MC: {100*result["pass_at_1_mc_mean"]:.1f}% +/- {100*result["pass_at_1_mc_std"]:.1f}%) '
                  f'[{result["n_correct_traces"]}/{result["n_valid_traces"]} traces]')

    # Save cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nCache saved to: {CACHE_FILE}')

    return all_results


def load_cache():
    """Load results from cache."""
    if not CACHE_FILE.exists():
        print(f'ERROR: Cache file not found: {CACHE_FILE}')
        print('Run without --no-compute first.')
        return None

    with open(CACHE_FILE, 'r') as f:
        return json.load(f)


# ============================================================================
# OUTPUT TABLES
# ============================================================================

def print_table(all_results: dict):
    """Print results table matching Tablebench.tex format."""
    print()
    print('=' * 80)
    print('PASS@1 RESULTS TABLE')
    print('=' * 80)

    # Header
    print(f'{"Model":<20} {"AIME 2024":<12} {"AIME 2025":<12} {"BRUMO 2025":<12} {"HMMT 2025":<12} {"Average":<10}')
    print('-' * 80)

    overall_total = []

    for model in MODEL_ORDER:
        model_results = all_results.get(model, {})
        row = f'{MODEL_NAMES.get(model, model):<20}'

        model_accs = []
        for dataset in DATASET_ORDER:
            r = model_results.get(dataset, {})
            if r:
                acc = r['pass_at_1']
                model_accs.append(acc)
                row += f'{100*acc:<12.1f}'
            else:
                row += f'{"N/A":<12}'

        if model_accs:
            avg = np.mean(model_accs)
            row += f'{100*avg:<10.1f}'
            overall_total.extend(model_accs)
        else:
            row += f'{"N/A":<10}'

        print(row)

    # Overall average
    print('-' * 80)
    if overall_total:
        print(f'{"Overall Average":<20} {"":<12} {"":<12} {"":<12} {"":<12} {100*np.mean(overall_total):<10.1f}')


def print_latex_table(all_results: dict):
    """Print LaTeX table for appendix."""
    print()
    print('=' * 80)
    print('LATEX TABLE (for appendix)')
    print('=' * 80)

    print(r"""
\begin{table}[h]
\centering
\caption{Pass@1 accuracy (\%): Single-trace baseline where one trace is randomly sampled per question.}
\label{tab:pass_at_1}
\begin{tabular}{lcccc|c}
\toprule
\textbf{Model} & \textbf{AIME 2024} & \textbf{AIME 2025} & \textbf{BRUMO 2025} & \textbf{HMMT 2025} & \textbf{Average} \\
\midrule""")

    overall_total = []

    for model in MODEL_ORDER:
        model_results = all_results.get(model, {})

        # Model display name matching Tablebench.tex
        model_display = {
            'deepseek8b': 'DeepSeek-R1-8B',
            'gemma3_27b': 'Gemma-3-27B',
            'gptoss20b': 'gpt-oss-20B',
            'qwq32b': 'QwQ-32B',
        }.get(model, model)

        row = f'{model_display}'

        model_accs = []
        for dataset in DATASET_ORDER:
            r = model_results.get(dataset, {})
            if r:
                acc = r['pass_at_1']
                model_accs.append(acc)
                row += f' & {100*acc:.1f}'
            else:
                row += ' & --'

        if model_accs:
            avg = np.mean(model_accs)
            row += f' & {100*avg:.1f}'
            overall_total.extend(model_accs)
        else:
            row += ' & --'

        row += r' \\'
        print(row)

    print(r'\midrule')
    if overall_total:
        print(f'\\textbf{{Overall}} & & & & & {100*np.mean(overall_total):.1f} \\\\')

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def print_comparison_with_main_table(all_results: dict):
    """Print Pass@1 vs voting methods comparison."""
    print()
    print('=' * 80)
    print('COMPARISON: Pass@1 vs Voting Methods (from main table)')
    print('=' * 80)
    print()
    print('This shows how much voting methods improve over the Pass@1 baseline.')
    print('Values from Tablebench.tex for reference:')
    print()

    # Main table values (from Tablebench.tex)
    main_table = {
        'deepseek8b': {'aime2024': 93.3, 'aime2025': 93.3, 'brumo2025': 93.3, 'hmmt2025': 83.3},
        'gemma3_27b': {'aime2024': 56.7, 'aime2025': 40.0, 'brumo2025': 46.7, 'hmmt2025': 23.3},
        'gptoss20b': {'aime2024': 93.3, 'aime2025': 93.3, 'brumo2025': 83.3, 'hmmt2025': 73.3},
        'qwq32b': {'aime2024': 90.0, 'aime2025': 76.7, 'brumo2025': 90.0, 'hmmt2025': 63.3},
    }

    print(f'{"Model":<20} {"Dataset":<12} {"Pass@1":<10} {"CDG":<10} {"Gain":<10}')
    print('-' * 65)

    for model in MODEL_ORDER:
        model_results = all_results.get(model, {})
        for dataset in DATASET_ORDER:
            r = model_results.get(dataset, {})
            cdg = main_table.get(model, {}).get(dataset, 0)

            if r:
                pass1 = 100 * r['pass_at_1']
                gain = cdg - pass1
                print(f'{MODEL_NAMES.get(model, model):<20} {dataset:<12} {pass1:<10.1f} {cdg:<10.1f} {gain:+.1f}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Pass@1 baseline accuracy')
    parser.add_argument('--no-compute', action='store_true',
                        help='Load from cache instead of computing')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_compute:
        all_results = load_cache()
        if all_results is None:
            exit(1)
    else:
        all_results = run_analysis()

    print_table(all_results)

    if args.latex:
        print_latex_table(all_results)

    print_comparison_with_main_table(all_results)

    print()
    print('=' * 80)
    print('DONE')
    print('=' * 80)
