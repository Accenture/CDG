"""
Unified Evaluation Script for Voting Methods

Compares multiple voting methods on LLM trace results:
1. Majority Voting (baseline)
2. DeepConf Mean Weighted
3. DeepConf Tail Weighted
4. DeepConf Bottom Window Weighted
5. Simple Gradient Weighted
6. Count-Dampened Gradient (CDG)

Usage:
    python evaluate_voting_methods.py --data_dirs DIR1 DIR2 ... [options]

Arguments:
    --data_dirs: One or more directories containing pickle files
    --alpha: CDG count dampening exponent (default: 0.5)
    --beta: CDG gradient weight (default: 10)
    --position_pct: Percentile for gradient computation (default: 20)
    --tail_window: Window size for tail confidence (default: 2048)
    --bottom_window: Window size for bottom window method (default: 2048)
    --bottom_pct: Bottom percentile for bottom window (default: 0.1)
"""
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))
from dynasor.core.evaluator import math_equal


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_parse(text: str) -> str:
    """Parse LaTeX text command."""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        try:
            return math_equal(answer, ground_truth)
        except:
            return str(answer).strip() == str(ground_truth).strip()


def compute_gradient(confs: List[float], percentile: int = 20) -> float:
    """Compute gradient: mean(last percentile%) - mean(first percentile%)"""
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def compute_tail_conf(confs: List[float], window: int = 2048) -> float:
    """Compute tail confidence (last window tokens)"""
    if not confs:
        return 0.0
    tail = confs[-window:] if len(confs) > window else confs
    return np.mean(tail)


def compute_bottom_window_conf(confs: List[float], window_size: int = 2048, bottom_pct: float = 0.1) -> float:
    """Compute bottom window confidence using sliding window."""
    if not confs:
        return 0.0
    if len(confs) < window_size:
        return np.mean(confs)

    window_means = []
    current_sum = sum(confs[:window_size])
    window_means.append(current_sum / window_size)

    for i in range(1, len(confs) - window_size + 1):
        current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
        window_means.append(current_sum / window_size)

    if not window_means:
        return 0.0

    if bottom_pct == -1:  # Min window
        return min(window_means)

    num_bottom = max(1, int(len(window_means) * bottom_pct))
    if num_bottom == 1:
        return min(window_means)
    else:
        bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
        return np.mean(bottom_means)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(results_dir: str, position_pct: int = 20, tail_window: int = 2048,
                 bottom_window: int = 2048, bottom_pct: float = 0.1) -> Dict:
    """Load dataset with all trace info and precompute features."""
    results_dir = Path(results_dir)
    pkl_files = sorted(results_dir.glob('qid*.pkl'))

    # If no pkl files found, try subdirectories
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    if not pkl_files:
        return {}

    question_data = {}
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {pkl_file}: {e}")
            continue

        qid = data.get('qid', pkl_file.stem.split('_')[0])
        gt = data['ground_truth']

        traces_list = []
        for trace in data['all_traces']:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is not None and len(confs) >= 10:
                traces_list.append({
                    'answer': answer,
                    'confs': confs,
                    'mean_conf': np.mean(confs),
                    'gradient': compute_gradient(confs, position_pct),
                    'tail_conf': compute_tail_conf(confs, tail_window),
                    'bottom_window_conf': compute_bottom_window_conf(confs, bottom_window, bottom_pct),
                })

        question_data[qid] = {'gt': gt, 'traces': traces_list}

    return question_data


# ============================================================================
# VOTING METHODS
# ============================================================================

def majority_voting(question_data: Dict) -> tuple:
    """Simple majority voting by count."""
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        if traces:
            answers = [t['answer'] for t in traces]
            winner = Counter(answers).most_common(1)[0][0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


def deepconf_mean_voting(question_data: Dict) -> tuple:
    """DeepConf mean confidence weighted voting."""
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            answer_scores[t['answer']] += t['mean_conf']

        if answer_scores:
            winner = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


def deepconf_tail_voting(question_data: Dict) -> tuple:
    """DeepConf tail confidence weighted voting."""
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            answer_scores[t['answer']] += t['tail_conf']

        if answer_scores:
            winner = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


def deepconf_bottom_window_voting(question_data: Dict) -> tuple:
    """DeepConf bottom window confidence weighted voting."""
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            answer_scores[t['answer']] += t['bottom_window_conf']

        if answer_scores:
            winner = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


def simple_gradient_voting(question_data: Dict, beta: float = 10) -> tuple:
    """Simple gradient weighted voting (sum of scores)."""
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            score = t['mean_conf'] + beta * t['gradient']
            answer_scores[t['answer']] += score

        if answer_scores:
            winner = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


def count_dampened_gradient_voting(question_data: Dict, alpha: float = 0.5, beta: float = 10) -> tuple:
    """Count-Dampened Gradient Voting.

    score = count^alpha * mean(mean_conf + beta * gradient)
    """
    correct = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(list)
        for t in traces:
            score = t['mean_conf'] + beta * t['gradient']
            answer_scores[t['answer']].append(score)

        if answer_scores:
            final_scores = {
                ans: (len(scores) ** alpha) * np.mean(scores)
                for ans, scores in answer_scores.items()
            }
            winner = max(final_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1
        results[qid] = {'correct': is_correct, 'winner': winner}

    return correct, results


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_all_methods(question_data: Dict, alpha: float, beta: float) -> Dict[str, tuple]:
    """Evaluate all voting methods on the dataset."""
    methods = {}

    methods['Majority'] = majority_voting(question_data)
    methods['DeepConf-Mean'] = deepconf_mean_voting(question_data)
    methods['DeepConf-Tail'] = deepconf_tail_voting(question_data)
    methods['DeepConf-BottomWin'] = deepconf_bottom_window_voting(question_data)
    methods['Gradient'] = simple_gradient_voting(question_data, beta=beta)
    methods['CDG'] = count_dampened_gradient_voting(question_data, alpha=alpha, beta=beta)

    return methods


def format_results_table(all_results: Dict[str, Dict[str, tuple]], method_names: List[str]) -> str:
    """Format results as a table with datasets as rows and methods as columns."""
    # Calculate column widths
    dataset_width = max(len(name) for name in all_results.keys()) + 2
    method_width = 12

    # Header
    header = f"{'Dataset':<{dataset_width}}"
    for method in method_names:
        header += f" | {method:^{method_width}}"

    separator = '-' * len(header)

    lines = [separator, header, separator]

    # Data rows
    for dataset_name, methods in all_results.items():
        row = f"{dataset_name:<{dataset_width}}"
        for method in method_names:
            if method in methods:
                correct, results = methods[method]
                total = len(results)
                pct = 100 * correct / total if total > 0 else 0
                cell = f"{correct}/{total} ({pct:.1f}%)"
            else:
                cell = "N/A"
            row += f" | {cell:^{method_width}}"
        lines.append(row)

    lines.append(separator)

    return '\n'.join(lines)


def print_detailed_results(dataset_name: str, methods: Dict[str, tuple]):
    """Print detailed results for a single dataset."""
    print(f"\n{'='*90}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*90}\n")

    for method_name, (correct, results) in methods.items():
        total = len(results)
        pct = 100 * correct / total if total > 0 else 0
        wrong_qids = sorted([qid for qid, r in results.items() if not r['correct']])

        print(f"{method_name}:")
        print(f"  Accuracy: {correct}/{total} ({pct:.1f}%)")
        if wrong_qids:
            print(f"  Wrong QIDs: {wrong_qids}")
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate voting methods on LLM trace results')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                        help='Directories containing pickle files')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='CDG count dampening exponent (default: 0.5)')
    parser.add_argument('--beta', type=float, default=10,
                        help='CDG gradient weight (default: 10)')
    parser.add_argument('--position_pct', type=int, default=20,
                        help='Percentile for gradient computation (default: 20)')
    parser.add_argument('--tail_window', type=int, default=2048,
                        help='Window size for tail confidence (default: 2048)')
    parser.add_argument('--bottom_window', type=int, default=2048,
                        help='Window size for bottom window method (default: 2048)')
    parser.add_argument('--bottom_pct', type=float, default=0.1,
                        help='Bottom percentile for bottom window (default: 0.1)')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed results per dataset')

    args = parser.parse_args()

    print('='*90)
    print('VOTING METHODS EVALUATION')
    print('='*90)
    print(f'\nParameters:')
    print(f'  alpha (CDG count exponent): {args.alpha}')
    print(f'  beta (gradient weight): {args.beta}')
    print(f'  position_pct (gradient percentile): {args.position_pct}%')
    print(f'  tail_window: {args.tail_window}')
    print(f'  bottom_window: {args.bottom_window}')
    print(f'  bottom_pct: {args.bottom_pct}')
    print()

    method_names = ['Majority', 'DeepConf-Mean', 'DeepConf-Tail', 'DeepConf-BottomWin', 'Gradient', 'CDG']
    all_results = {}

    for data_dir in args.data_dirs:
        # Extract dataset name from path
        dataset_name = Path(data_dir).name

        print(f'Loading {dataset_name}...')
        question_data = load_dataset(
            data_dir,
            position_pct=args.position_pct,
            tail_window=args.tail_window,
            bottom_window=args.bottom_window,
            bottom_pct=args.bottom_pct
        )

        if not question_data:
            print(f'  Warning: No data found in {data_dir}')
            continue

        print(f'  Loaded {len(question_data)} questions')

        # Evaluate all methods
        methods = evaluate_all_methods(question_data, args.alpha, args.beta)
        all_results[dataset_name] = methods

        if args.detailed:
            print_detailed_results(dataset_name, methods)

    # Print summary table
    print('\n')
    print('='*90)
    print('SUMMARY TABLE')
    print('='*90)
    print()
    print(format_results_table(all_results, method_names))
    print()

    # Print best method per dataset
    print('\nBest method per dataset:')
    for dataset_name, methods in all_results.items():
        best_method = max(methods.items(), key=lambda x: x[1][0])
        correct, results = best_method[1]
        total = len(results)
        pct = 100 * correct / total if total > 0 else 0
        print(f"  {dataset_name}: {best_method[0]} ({pct:.1f}%)")

    print()


if __name__ == '__main__':
    main()
