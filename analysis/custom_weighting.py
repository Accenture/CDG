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
#
# BACKGROUND:
# - conf values are NEGATIVE LOG PROBS: lower value = higher probability = more confident
# - Typical range: ~1.5 (very confident) to ~10+ (uncertain)
# - Goal: weight traces for majority voting to improve accuracy
#
# HYPOTHESIS (forking tokens):
# - High attention tokens → "forking points" → model made a choice → LOW confidence
# - Low attention tokens → straightforward continuation → HIGH confidence
# - We want: high conf → weight=x (reward), low conf → weight=1/x (penalize)
# - Without attention data, we approximate using confidence statistics
#
# =============================================================================

def uniform_weight(trace: Dict) -> float:
    """
    BASELINE: All traces weighted equally.

    Intuition: Simple majority vote. Every sample counts the same regardless
    of how confident the model was. This is what most papers use by default.

    Expected behavior: Should work reasonably but leaves accuracy on the table
    if some traces are clearly more reliable than others.
    """
    return 1.0


def mean_conf_weight(trace: Dict) -> float:
    """
    Weight = 1 / mean(confidence)

    Intuition: If a trace has low average confidence (high uncertainty throughout),
    it was "guessing" more and should count less. Traces where the model was
    consistently confident should dominate the vote.

    This is the approach used in the original deepconf paper.

    Potential issue: Penalizes traces that explored difficult reasoning paths
    (which might have necessary low-confidence "forking" steps).
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    # Note: conf values are negative log probs, so LOWER = more confident
    # Invert so higher confidence = higher weight
    mean_conf = np.mean(confs)
    return 1.0 / (mean_conf + 0.1)  # +0.1 to avoid division by zero


def inverse_conf_weight(trace: Dict) -> float:
    """
    Weight = 1 / mean(confidence)  [same as mean_conf_weight]

    Intuition: Same as mean_conf_weight. Kept as separate function for
    clarity in experiments. High confidence → high weight.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    mean_conf = np.mean(confs)
    return 1.0 / (mean_conf + 0.1)


def min_window_weight(trace: Dict, window_size: int = 2048) -> float:
    """
    Weight based on the WORST (least confident) sliding window.

    Intuition: A chain is only as strong as its weakest link. If there's
    any segment of 2048 tokens where the model was very uncertain, the
    whole trace might be unreliable. This finds that worst segment.

    Why 2048? Roughly the size of a reasoning "step" or paragraph.

    Expected behavior: More conservative than mean - will downweight traces
    that have even one shaky reasoning step, even if the rest is confident.
    """
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
    Weight = 1 / (1 + variance(confidence))

    Intuition: Approximates the "forking token" hypothesis WITHOUT attention.

    - High variance in confidence → trace had some very confident parts and
      some very uncertain parts → likely had "forking points" where model
      made risky decisions → less trustworthy overall
    - Low variance → consistent confidence throughout → stable reasoning path

    This is a PROXY for attention-based forking detection. A trace with many
    forking points will show high variance (low conf at forks, high elsewhere).

    Limitation: Can't distinguish between "good forking" (necessary creative
    leaps) and "bad forking" (random guessing).
    """
    confs = trace.get('confs', [])
    if len(confs) < 10:
        return 1.0

    variance = np.var(confs)
    # Higher variance = more forking = less trustworthy
    return 1.0 / (1.0 + variance)


def tail_focused_weight(trace: Dict, tail_tokens: int = 2048) -> float:
    """
    Weight based on confidence in the FINAL tokens only.

    Intuition: The end of the trace (near the answer) matters most. A trace
    might struggle through difficult reasoning (low confidence) but then
    converge to a confident answer. That final confidence is what we care about.

    Alternatively: If the model is uncertain even at the end, it's probably
    not sure about its answer.

    Why 2048? Captures roughly the "answer extraction" phase after </think>.

    Potential issue: Ignores the quality of reasoning in the middle.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    mean_tail_conf = np.mean(tail_confs)
    return 1.0 / (mean_tail_conf + 0.1)


def combined_weight(trace: Dict) -> float:
    """
    Geometric mean of multiple signals.

    Intuition: No single metric captures everything. Combine:
    - Overall confidence (inverse_conf)
    - Final answer confidence (tail_focused)
    - Stability of reasoning (entropy_based)

    Geometric mean ensures all factors matter - one very bad score pulls
    down the whole weight, even if others are good.

    This is exploratory - trying to get benefits of multiple heuristics.
    """
    # Get components
    mean_w = inverse_conf_weight(trace)
    tail_w = tail_focused_weight(trace)
    entropy_w = entropy_based_weight(trace)

    # Geometric mean to combine
    return (mean_w * tail_w * entropy_w) ** (1/3)


def softmax_conf_weight(trace: Dict, temperature: float = 1.0) -> float:
    """
    Softmax-style weighting with temperature control.

    Intuition: Instead of simple 1/x, use exp() to create sharper distinctions.
    Temperature controls how much we differentiate:
    - T < 1: More extreme (confident traces dominate more)
    - T = 1: Moderate
    - T > 1: Softer (closer to uniform)

    This mirrors how softmax temperature works in sampling - we're essentially
    "sampling" which trace to trust based on confidence.

    Mathematical note: exp(1/conf) grows very fast for high-confidence traces,
    potentially causing numerical issues or over-concentration.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    # Convert to "confidence" (invert uncertainty)
    inv_confs = [1.0 / (c + 0.1) for c in confs]
    mean_inv_conf = np.mean(inv_confs)

    # Apply temperature scaling
    return np.exp(mean_inv_conf / temperature)


# ============= TEAMMATE'S PROPOSED FUNCTIONS =============
#
# These implement the smooth transition idea: y = f(x) where
# - Small x (high conf) → behavior A
# - Large x (low conf) → behavior B
#
# The key question: what is x? We test both interpretations.
# =============================================================================

def teammate_formula_raw(trace: Dict) -> float:
    """
    Teammate's formula: y = x^{20*((-tanh(0.8x))+1.05)}
    where x = mean_conf (raw negative log prob)

    Intuition: Creates a smooth, tunable transition function.
    - The tanh squashes the exponent between ~1 and ~21
    - Small x → large exponent → x^big (but x is small, so moderate result)
    - Large x → exponent≈1 → y≈x

    Empirical behavior (see explore_weight_functions.py):
    - x=2: weight≈5.9
    - x=6: weight≈6.0
    - x=10: weight≈10.0

    This gives relatively FLAT weights in the confident range (1.5-4),
    then linear for uncertain traces. Might work if we want to NOT
    over-reward confident traces (they're all "good enough").

    EXPERIMENT: Compare with teammate_formula_inv to see which x works better.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    x = np.mean(confs)
    if x <= 0:
        return 1.0
    exp = 20 * ((-np.tanh(0.8 * x)) + 1.05)
    return x ** exp


def teammate_formula_inv(trace: Dict) -> float:
    """
    Teammate's formula with INVERTED input: x = 1/mean_conf

    Intuition: If teammate meant x to be "confidence" (not uncertainty),
    then high x = high confidence. This interpretation:
    - x=10 (high conf, mean_conf=0.1) → weight≈10
    - x=2 (low conf, mean_conf=0.5) → weight≈5.9

    This would give higher weights to confident traces, matching the
    standard intuition.

    EXPERIMENT: Run both raw and inv versions to see which performs better.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    mean_conf = np.mean(confs)
    x = 1.0 / (mean_conf + 0.1)  # Invert: high conf → high x
    exp = 20 * ((-np.tanh(0.8 * x)) + 1.05)
    return x ** exp


def smooth_blend_weight(trace: Dict, midpoint: float = 3.0, steepness: float = 2.0) -> float:
    """
    Smooth sigmoid blend between y=x and y=1/x behaviors.

    Intuition: Directly implements the forking hypothesis:
    - High confidence (x < midpoint): weight ≈ x (reward)
    - Low confidence (x > midpoint): weight ≈ 1/x (penalize)
    - Smooth transition around the midpoint

    Parameters:
    - midpoint: confidence value where behavior switches (default 3.0)
    - steepness: how sharp the transition is (default 2.0)

    The sigmoid blend factor:
    - blend = σ(steepness * (x - midpoint))
    - weight = (1-blend)*x + blend*(1/x)

    EXPERIMENT: Tune midpoint based on actual confidence distribution in data.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0
    x = np.mean(confs)
    if x <= 0:
        return 1.0
    blend = 1 / (1 + np.exp(-steepness * (x - midpoint)))
    return (1 - blend) * x + blend * (1/x)


# ============= TOKEN-LEVEL WEIGHTING (Use Case 1) =============
#
# Instead of using mean(conf) directly, weight individual tokens first,
# then aggregate. This lets us implement the forking hypothesis per-token.
# =============================================================================

def token_weighted_score(trace: Dict, token_weight_fn: callable = None) -> float:
    """
    Token-level weighting to compute trace score.

    Instead of: trace_weight = f(mean(confs))
    We do:      trace_weight = mean(token_weight(conf_i) for each token)

    This allows per-token weighting based on the forking hypothesis:
    - Each token gets weighted by some function of its confidence
    - The trace score is the aggregate

    Default token_weight_fn: 1/x (penalize uncertain tokens)
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    if token_weight_fn is None:
        token_weight_fn = lambda c: 1.0 / (c + 0.1)

    token_weights = [token_weight_fn(c) for c in confs]
    return np.mean(token_weights)


def token_level_teammate(trace: Dict) -> float:
    """
    Apply teammate's formula at TOKEN level, then average.

    Intuition: Instead of computing mean(conf) then applying formula,
    apply the formula to each token's confidence, then average.

    This treats each token independently - a forking token (low conf)
    contributes less to the trace score than a stable token (high conf).

    y_token = x^{20*((-tanh(0.8x))+1.05)} for each token
    trace_score = mean(y_token)
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    def teammate_per_token(x):
        if x <= 0:
            return 1.0
        exp = 20 * ((-np.tanh(0.8 * x)) + 1.05)
        return x ** exp

    token_scores = [teammate_per_token(c) for c in confs]
    return np.mean(token_scores)


def token_level_smooth_blend(trace: Dict, midpoint: float = 3.0, steepness: float = 2.0) -> float:
    """
    Apply smooth blend (y=x vs y=1/x) at TOKEN level, then average.

    Intuition: Each token is classified as "forking" or "stable" based
    on its confidence, and weighted accordingly:
    - Stable tokens (conf < midpoint): contribute more (weight≈conf)
    - Forking tokens (conf > midpoint): contribute less (weight≈1/conf)

    This is the cleanest implementation of the forking hypothesis
    without attention data.
    """
    confs = trace.get('confs', [])
    if not confs:
        return 1.0

    def blend_per_token(x):
        if x <= 0:
            return 1.0
        blend = 1 / (1 + np.exp(-steepness * (x - midpoint)))
        return (1 - blend) * x + blend * (1/x)

    token_scores = [blend_per_token(c) for c in confs]
    return np.mean(token_scores)


# Registry of all strategies
WEIGHTING_STRATEGIES = {
    # Baselines
    'uniform': uniform_weight,
    'mean_conf': mean_conf_weight,
    'inverse_conf': inverse_conf_weight,

    # Window/segment based
    'min_window': min_window_weight,
    'tail_focused': tail_focused_weight,

    # Variance/stability based
    'entropy_based': entropy_based_weight,

    # Combined
    'combined': combined_weight,

    # Temperature variations
    'softmax_t1': lambda t: softmax_conf_weight(t, 1.0),
    'softmax_t05': lambda t: softmax_conf_weight(t, 0.5),
    'softmax_t2': lambda t: softmax_conf_weight(t, 2.0),

    # Teammate's formula (test both x interpretations)
    'teammate_raw': teammate_formula_raw,
    'teammate_inv': teammate_formula_inv,

    # Smooth blend
    'smooth_blend': smooth_blend_weight,
    'smooth_blend_m2': lambda t: smooth_blend_weight(t, midpoint=2.0),
    'smooth_blend_m4': lambda t: smooth_blend_weight(t, midpoint=4.0),

    # Token-level weighting
    'token_teammate': token_level_teammate,
    'token_blend': token_level_smooth_blend,
    'token_blend_m2': lambda t: token_level_smooth_blend(t, midpoint=2.0),
    'token_blend_m4': lambda t: token_level_smooth_blend(t, midpoint=4.0),
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
