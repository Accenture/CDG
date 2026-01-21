"""
Modular evaluation methods for inference results.

This module provides reusable evaluation functions that can be called by
different evaluation scripts.

Usage:
    from eval_methods import load_run_results, evaluate_with_method, get_available_methods

Available Methods:
    - majority: Simple majority voting by count
    - deepconf_mean: Sum of mean confidence per answer
    - deepconf_tail: Sum of tail (last N tokens) confidence
    - deepconf_bottom: Sum of sliding window min confidence
    - gradient: Sum of (mean_conf + β × gradient)
    - cdg: Count-Dampened Gradient - count^α × mean(conf + β × gradient)
"""

import os
import sys
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepconf'))

from dynasor.core.evaluator import math_equal


# ============================================================
# Answer Comparison
# ============================================================

def quick_parse(text: str) -> str:
    """Remove \\text{} wrappers from LaTeX."""
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
    """Check if answer equals ground truth using math_equal."""
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        try:
            return math_equal(answer, ground_truth)
        except:
            return str(answer).strip() == str(ground_truth).strip()


# ============================================================
# Confidence Feature Computation
# ============================================================

def compute_gradient(confs: list, percentile: int = 20) -> float:
    """
    Compute confidence gradient: mean(last percentile%) - mean(first percentile%).

    Positive gradient indicates increasing confidence towards the end,
    which correlates with correct answers.
    """
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return float(np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff]))


def compute_tail_conf(confs: list, window: int = 2048) -> float:
    """Compute mean confidence of last `window` tokens."""
    if not confs:
        return 0.0
    tail = confs[-window:] if len(confs) > window else confs
    return float(np.mean(tail))


def compute_bottom_window_conf(confs: list, window_size: int = 2048, bottom_pct: float = 0.1) -> float:
    """
    Compute bottom window confidence using sliding window.

    Returns mean of the bottom `bottom_pct` window means.
    Use bottom_pct=-1 for minimum window only.
    """
    if not confs:
        return 0.0
    if len(confs) < window_size:
        return float(np.mean(confs))

    # Efficient sliding window mean computation
    window_means = []
    current_sum = sum(confs[:window_size])
    window_means.append(current_sum / window_size)

    for i in range(1, len(confs) - window_size + 1):
        current_sum = current_sum - confs[i - 1] + confs[i + window_size - 1]
        window_means.append(current_sum / window_size)

    if not window_means:
        return 0.0

    if bottom_pct == -1:  # Min window only
        return min(window_means)

    num_bottom = max(1, int(len(window_means) * bottom_pct))
    if num_bottom == 1:
        return min(window_means)
    else:
        bottom_means = np.partition(window_means, num_bottom - 1)[:num_bottom]
        return float(np.mean(bottom_means))


# ============================================================
# Data Loading
# ============================================================

def load_single_pickle(filename: Path) -> tuple:
    """Load a single pickle file."""
    try:
        if 'stats' in filename.name or 'comparison' in filename.name:
            return None, None
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        qid = data.get('qid')
        return qid, data
    except Exception as e:
        return None, None


def load_run_results(results_dir: str, num_workers: int = None, use_threads: bool = True) -> dict:
    """
    Load all result pickle files from a run directory.

    Args:
        results_dir: Path to run directory containing .pkl files
        num_workers: Number of parallel workers (None = auto)
        use_threads: If False, load sequentially (use when called from within threads)

    Returns:
        dict: {qid: [result_data, ...]} mapping question IDs to result data
    """
    pkl_files = list(Path(results_dir).glob("*.pkl"))

    if not pkl_files:
        return {}

    results_by_qid = defaultdict(list)

    if use_threads and len(pkl_files) > 4:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(pkl_files), 8)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(load_single_pickle, pkl_files)
            for qid, data in results:
                if qid is not None:
                    results_by_qid[qid].append(data)
    else:
        # Sequential loading - safe when called from within thread pool
        for pkl_file in pkl_files:
            qid, data = load_single_pickle(pkl_file)
            if qid is not None:
                results_by_qid[qid].append(data)

    return dict(results_by_qid)


# ============================================================
# Evaluation Methods
# ============================================================

def majority_vote(answers: list) -> str:
    """
    Simple majority vote - return most common answer.

    Args:
        answers: List of answer strings

    Returns:
        Most common answer or None if no valid answers
    """
    if not answers:
        return None
    counter = Counter(str(a) for a in answers if a)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def evaluate_majority_vote(results_by_qid: dict) -> dict:
    """
    Evaluate using majority vote method.

    Args:
        results_by_qid: {qid: [result_data, ...]} from load_run_results

    Returns:
        dict with 'correct', 'total', 'accuracy', 'per_question' keys
    """
    correct = 0
    total = 0
    per_question = []

    for qid in sorted(results_by_qid.keys()):
        results_list = results_by_qid[qid]

        # Collect all traces
        all_traces = []
        ground_truth = None
        question = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')
            if question is None:
                question = result.get('question', '')[:100]

        if not all_traces or not ground_truth:
            continue

        # Extract answers
        answers = [t.get('extracted_answer') for t in all_traces if t.get('extracted_answer')]

        if not answers:
            continue

        total += 1

        # Majority vote
        predicted = majority_vote(answers)

        # Check correctness
        try:
            is_correct = equal_func(predicted, ground_truth) if predicted else False
        except:
            is_correct = str(predicted) == str(ground_truth)

        if is_correct:
            correct += 1

        # Answer distribution
        counter = Counter(str(a) for a in answers)
        top_answers = counter.most_common(3)

        per_question.append({
            'qid': qid,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct,
            'num_traces': len(all_traces),
            'num_valid_answers': len(answers),
            'top_answers': top_answers,
            'question_preview': question,
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


def _extract_traces_with_features(results_by_qid: dict, params: dict = None) -> dict:
    """
    Extract traces with precomputed confidence features.

    Args:
        results_by_qid: {qid: [result_data, ...]} from load_run_results
        params: Dict with 'position_pct', 'tail_window', 'bottom_window', 'bottom_pct'

    Returns:
        {qid: {'gt': ground_truth, 'traces': [trace_dict, ...]}}
    """
    params = params or {}
    position_pct = params.get('position_pct', 20)
    tail_window = params.get('tail_window', 2048)
    bottom_window = params.get('bottom_window', 2048)
    bottom_pct = params.get('bottom_pct', 0.1)

    question_data = {}

    for qid in sorted(results_by_qid.keys()):
        results_list = results_by_qid[qid]

        all_traces = []
        ground_truth = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        traces_list = []
        for trace in all_traces:
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

        if traces_list:
            question_data[qid] = {'gt': ground_truth, 'traces': traces_list}

    return question_data


def _weighted_voting_core(question_data: dict, weight_key: str) -> dict:
    """Core weighted voting logic - sums weights per answer."""
    correct = 0
    total = 0
    per_question = []

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            answer_scores[t['answer']] += t[weight_key]

        if answer_scores:
            predicted = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            predicted = None

        try:
            is_correct = equal_func(str(predicted), str(gt)) if predicted else False
        except:
            is_correct = str(predicted) == str(gt)

        total += 1
        if is_correct:
            correct += 1

        per_question.append({
            'qid': qid,
            'ground_truth': gt,
            'predicted': predicted,
            'correct': is_correct,
            'num_traces': len(traces),
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


def evaluate_deepconf_mean(results_by_qid: dict, params: dict = None) -> dict:
    """DeepConf mean confidence weighted voting."""
    question_data = _extract_traces_with_features(results_by_qid, params)
    return _weighted_voting_core(question_data, 'mean_conf')


def evaluate_deepconf_tail(results_by_qid: dict, params: dict = None) -> dict:
    """DeepConf tail confidence weighted voting."""
    question_data = _extract_traces_with_features(results_by_qid, params)
    return _weighted_voting_core(question_data, 'tail_conf')


def evaluate_deepconf_bottom(results_by_qid: dict, params: dict = None) -> dict:
    """DeepConf bottom window confidence weighted voting."""
    question_data = _extract_traces_with_features(results_by_qid, params)
    return _weighted_voting_core(question_data, 'bottom_window_conf')


def evaluate_gradient(results_by_qid: dict, params: dict = None) -> dict:
    """Simple gradient weighted voting: sum(mean_conf + beta * gradient)."""
    params = params or {}
    beta = params.get('beta', 10)
    question_data = _extract_traces_with_features(results_by_qid, params)

    correct = 0
    total = 0
    per_question = []

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        answer_scores = defaultdict(float)
        for t in traces:
            score = t['mean_conf'] + beta * t['gradient']
            answer_scores[t['answer']] += score

        if answer_scores:
            predicted = max(answer_scores.items(), key=lambda x: x[1])[0]
        else:
            predicted = None

        try:
            is_correct = equal_func(str(predicted), str(gt)) if predicted else False
        except:
            is_correct = str(predicted) == str(gt)

        total += 1
        if is_correct:
            correct += 1

        per_question.append({
            'qid': qid,
            'ground_truth': gt,
            'predicted': predicted,
            'correct': is_correct,
            'num_traces': len(traces),
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


def evaluate_oracle(results_by_qid: dict, params: dict = None) -> dict:
    """
    Oracle upper bound - correct if ANY trace has the right answer.

    This represents the theoretical maximum accuracy achievable with perfect
    answer selection. Useful as an upper bound to compare voting methods against.
    """
    correct = 0
    total = 0
    per_question = []

    for qid in sorted(results_by_qid.keys()):
        results_list = results_by_qid[qid]

        all_traces = []
        ground_truth = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        # Extract all answers
        answers = [t.get('extracted_answer') for t in all_traces if t.get('extracted_answer')]

        if not answers:
            continue

        total += 1

        # Oracle: correct if ANY answer matches ground truth
        is_correct = False
        matching_answer = None
        for ans in answers:
            try:
                if equal_func(str(ans), str(ground_truth)):
                    is_correct = True
                    matching_answer = ans
                    break
            except:
                if str(ans) == str(ground_truth):
                    is_correct = True
                    matching_answer = ans
                    break

        if is_correct:
            correct += 1

        per_question.append({
            'qid': qid,
            'ground_truth': ground_truth,
            'predicted': matching_answer if is_correct else answers[0] if answers else None,
            'correct': is_correct,
            'num_traces': len(all_traces),
            'num_valid_answers': len(answers),
            'has_correct_trace': is_correct,
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


def evaluate_cdg(results_by_qid: dict, params: dict = None) -> dict:
    """
    Count-Dampened Gradient (CDG) voting.

    score = count^alpha * mean(mean_conf + beta * gradient)

    Novel method that combines:
    - Count dampening: sqrt(count) instead of linear count
    - Confidence gradient: rewards traces with increasing confidence
    """
    params = params or {}
    alpha = params.get('alpha', 0.5)
    beta = params.get('beta', 10)
    question_data = _extract_traces_with_features(results_by_qid, params)

    correct = 0
    total = 0
    per_question = []

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        # Collect scores per answer
        answer_scores = defaultdict(list)
        for t in traces:
            score = t['mean_conf'] + beta * t['gradient']
            answer_scores[t['answer']].append(score)

        if answer_scores:
            # CDG formula: count^alpha * mean(scores)
            final_scores = {
                ans: (len(scores) ** alpha) * np.mean(scores)
                for ans, scores in answer_scores.items()
            }
            predicted = max(final_scores.items(), key=lambda x: x[1])[0]
        else:
            predicted = None

        try:
            is_correct = equal_func(str(predicted), str(gt)) if predicted else False
        except:
            is_correct = str(predicted) == str(gt)

        total += 1
        if is_correct:
            correct += 1

        per_question.append({
            'qid': qid,
            'ground_truth': gt,
            'predicted': predicted,
            'correct': is_correct,
            'num_traces': len(traces),
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'per_question': per_question,
    }


# ============================================================
# Method Registry
# ============================================================

EVAL_METHODS = {
    'oracle': {
        'name': 'Oracle',
        'func': evaluate_oracle,
        'description': 'Upper bound - correct if ANY trace has right answer',
        'has_params': False,
    },
    'majority': {
        'name': 'Majority',
        'func': evaluate_majority_vote,
        'description': 'Simple majority voting by count',
        'has_params': False,
    },
    'deepconf_mean': {
        'name': 'DeepConf-Mean',
        'func': evaluate_deepconf_mean,
        'description': 'Sum of mean confidence per answer',
        'has_params': True,
    },
    'deepconf_tail': {
        'name': 'DeepConf-Tail',
        'func': evaluate_deepconf_tail,
        'description': 'Sum of tail (last N tokens) confidence',
        'has_params': True,
    },
    'deepconf_bottom': {
        'name': 'DeepConf-Bottom',
        'func': evaluate_deepconf_bottom,
        'description': 'Sum of sliding window min confidence',
        'has_params': True,
    },
    'gradient': {
        'name': 'Gradient',
        'func': evaluate_gradient,
        'description': 'Sum of (mean_conf + beta * gradient)',
        'has_params': True,
    },
    'cdg': {
        'name': 'CDG',
        'func': evaluate_cdg,
        'description': 'Count-Dampened Gradient: count^alpha * mean(conf + beta * gradient)',
        'has_params': True,
    },
}

# Default parameters for voting methods
DEFAULT_PARAMS = {
    'alpha': 0.5,       # CDG count dampening exponent
    'beta': 10,         # Gradient weight
    'position_pct': 20, # Percentile for gradient computation
    'tail_window': 2048,
    'bottom_window': 2048,
    'bottom_pct': 0.1,
}


def get_available_methods() -> list:
    """Return list of available evaluation method names."""
    return list(EVAL_METHODS.keys())


def get_method_info(method: str) -> dict:
    """Get info about a specific method."""
    if method not in EVAL_METHODS:
        raise ValueError(f"Unknown method: {method}")
    return EVAL_METHODS[method]


def evaluate_with_method(results_by_qid: dict, method: str = 'majority', params: dict = None) -> dict:
    """
    Evaluate results using specified method.

    Args:
        results_by_qid: Results from load_run_results
        method: Method name from EVAL_METHODS
        params: Optional dict of parameters (alpha, beta, tail_window, etc.)

    Returns:
        Evaluation results dict
    """
    if method not in EVAL_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")

    method_info = EVAL_METHODS[method]
    func = method_info['func']

    if method_info.get('has_params', False):
        # Merge default params with provided params
        merged_params = {**DEFAULT_PARAMS, **(params or {})}
        return func(results_by_qid, merged_params)
    else:
        return func(results_by_qid)
