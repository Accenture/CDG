"""
Experiment with custom weighting strategies for voting.

This implements your research ideas:
- High attention → Forking → low confidence → weight = 1/x
- Low attention → No forking → high confidence → weight = x

Current focus: confidence-based strategies (attention requires VLLM modification)

Usage:
    python custom_weighting.py --results_dir offline_results/
"""
import os
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

from dynasor.core.evaluator import math_equal


def quick_parse(text: str) -> str:
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
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def weighted_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Weighted majority vote"""
    if not answers:
        return None
    answer_weights = defaultdict(float)
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_weights[str(answer)] += weight
    if not answer_weights:
        return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


# ============= WEIGHTING STRATEGIES =============

def uniform_weight(trace: Dict) -> float:
    """Baseline: uniform weight = 1"""
    return 1.0


def mean_conf_weight(trace: Dict) -> float:
    """Weight = mean confidence (lower uncertainty = higher weight)"""
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    # Note: conf values are negative log probs, so LOWER = more confident
    # Invert so higher confidence = higher weight
    mean_conf = np.mean(confs)
    return 1.0 / (mean_conf + 0.1)  # +0.1 to avoid division by zero


def inverse_conf_weight(trace: Dict) -> float:
    """Weight = 1/mean_conf (your proposed: high conf -> high weight)"""
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    mean_conf = np.mean(confs)
    return 1.0 / (mean_conf + 0.1)


def min_window_weight(trace: Dict, window_size: int = 2048) -> float:
    """Weight based on minimum sliding window confidence"""
    confs = trace.get('confs', [])
    if len(confs) < window_size:
        return 1.0 / (np.mean(confs) + 0.1) if confs else 1.0

    min_window_mean = float('inf')
    current_sum = sum(confs[:window_size])
    min_window_mean = min(min_window_mean, current_sum / window_size)

    for i in range(1, len(confs) - window_size + 1):
        current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
        min_window_mean = min(min_window_mean, current_sum / window_size)

    return 1.0 / (min_window_mean + 0.1)


def entropy_based_weight(trace: Dict) -> float:
    """
    Your proposed forking token idea approximation:
    - Look at variance/entropy of confidence across the trace
    - High variance might indicate "forking points"
    - Weight inversely to variance (stable = more trustworthy)
    """
    confs = trace.get('confs', [])
    if len(confs) < 10:
        return 1.0

    variance = np.var(confs)
    # Higher variance = more forking = less trustworthy
    return 1.0 / (1.0 + variance)


def tail_focused_weight(trace: Dict, tail_tokens: int = 2048) -> float:
    """Weight based on confidence in final tokens (near the answer)"""
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    mean_tail_conf = np.mean(tail_confs)
    return 1.0 / (mean_tail_conf + 0.1)


def combined_weight(trace: Dict) -> float:
    """Combine multiple signals"""
    # Get components
    mean_w = inverse_conf_weight(trace)
    tail_w = tail_focused_weight(trace)
    entropy_w = entropy_based_weight(trace)

    # Geometric mean to combine
    return (mean_w * tail_w * entropy_w) ** (1/3)


def softmax_conf_weight(trace: Dict, temperature: float = 1.0) -> float:
    """Apply softmax-style normalization to confidences"""
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    # Convert to "confidence" (invert uncertainty)
    inv_confs = [1.0 / (c + 0.1) for c in confs]
    mean_inv_conf = np.mean(inv_confs)

    # Apply temperature scaling
    return np.exp(mean_inv_conf / temperature)


# Registry of all strategies
WEIGHTING_STRATEGIES = {
    'uniform': uniform_weight,
    'mean_conf': mean_conf_weight,
    'inverse_conf': inverse_conf_weight,
    'min_window': min_window_weight,
    'entropy_based': entropy_based_weight,
    'tail_focused': tail_focused_weight,
    'combined': combined_weight,
    'softmax_t1': lambda t: softmax_conf_weight(t, 1.0),
    'softmax_t05': lambda t: softmax_conf_weight(t, 0.5),
    'softmax_t2': lambda t: softmax_conf_weight(t, 2.0),
}


def evaluate_strategy(results_by_qid: dict, weight_fn: Callable) -> dict:
    """Evaluate a weighting strategy across all questions"""
    correct = 0
    total = 0

    per_question = {}

    for qid, results_list in results_by_qid.items():
        # Combine traces
        all_traces = []
        ground_truth = None
        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        total += 1

        # Get answers and weights
        answers = []
        weights = []
        for trace in all_traces:
            answer = trace.get('extracted_answer')
            if answer:
                answers.append(answer)
                weights.append(weight_fn(trace))

        if not answers:
            continue

        # Vote
        predicted = weighted_vote(answers, weights)

        try:
            is_correct = equal_func(predicted, ground_truth) if predicted else False
        except:
            is_correct = str(predicted) == str(ground_truth)

        if is_correct:
            correct += 1

        per_question[qid] = {
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': is_correct,
        }

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


def load_results(results_dir: str) -> dict:
    results_by_qid = defaultdict(list)
    for filename in Path(results_dir).glob("*.pkl"):
        if 'stats' in filename.name:
            continue
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        qid = data.get('qid')
        if qid is not None:
            results_by_qid[qid].append(data)
    return dict(results_by_qid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results')
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results_by_qid = load_results(args.results_dir)
    print(f"Found {len(results_by_qid)} questions\n")

    print("=" * 70)
    print("WEIGHTING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)

    all_results = {}

    for name, weight_fn in WEIGHTING_STRATEGIES.items():
        result = evaluate_strategy(results_by_qid, weight_fn)
        all_results[name] = result
        print(f"{name:<20} {result['correct']:<10} {result['total']:<10} {100*result['accuracy']:.1f}%")

    # Find best strategy
    best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 70)
    print(f"Best strategy: {best[0]} ({100*best[1]['accuracy']:.1f}%)")

    # Save detailed results
    output_file = os.path.join(args.results_dir, 'weighting_comparison.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nDetailed results saved to {output_file}")

    # Show per-question comparison for top strategies
    print("\n" + "=" * 70)
    print("PER-QUESTION COMPARISON (Uniform vs Best)")
    print("=" * 70)

    uniform_res = all_results['uniform']['per_question']
    best_res = best[1]['per_question']

    print(f"{'QID':<5} {'GT':<10} {'Uniform':<15} {'Best':<15}")
    print("-" * 70)

    for qid in sorted(uniform_res.keys()):
        gt = str(uniform_res[qid]['ground_truth'])[:8]
        uni_pred = str(uniform_res[qid]['predicted'])[:12]
        uni_mark = "✓" if uniform_res[qid]['correct'] else "✗"
        best_pred = str(best_res[qid]['predicted'])[:12] if qid in best_res else "N/A"
        best_mark = "✓" if best_res.get(qid, {}).get('correct', False) else "✗"

        print(f"{qid:<5} {gt:<10} {uni_pred:<12}{uni_mark}  {best_pred:<12}{best_mark}")


if __name__ == "__main__":
    main()
