"""
Modular evaluation methods for inference results.

This module provides reusable evaluation functions that can be called by
different evaluation scripts. Add new evaluation methods here.

Usage:
    from eval_methods import majority_vote_accuracy, load_run_results

Available Methods:
    - majority_vote: Simple majority voting across all traces
    - (Future: weighted_vote, confidence_based, etc.)
"""

import os
import sys
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


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


def load_run_results(results_dir: str, num_workers: int = None) -> dict:
    """
    Load all result pickle files from a run directory.

    Returns:
        dict: {qid: [result_data, ...]} mapping question IDs to result data
    """
    pkl_files = list(Path(results_dir).glob("*.pkl"))

    if not pkl_files:
        return {}

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(pkl_files), 8)

    results_by_qid = defaultdict(list)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(load_single_pickle, pkl_files)
        for qid, data in results:
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


# ============================================================
# Method Registry
# ============================================================

EVAL_METHODS = {
    'majority': {
        'name': 'Majority Vote',
        'func': evaluate_majority_vote,
        'description': 'Simple majority voting across all traces',
    },
    # Future methods:
    # 'weighted': {
    #     'name': 'Weighted Vote',
    #     'func': evaluate_weighted_vote,
    #     'description': 'Confidence-weighted voting',
    # },
}


def get_available_methods() -> list:
    """Return list of available evaluation method names."""
    return list(EVAL_METHODS.keys())


def evaluate_with_method(results_by_qid: dict, method: str = 'majority') -> dict:
    """
    Evaluate results using specified method.

    Args:
        results_by_qid: Results from load_run_results
        method: Method name from EVAL_METHODS

    Returns:
        Evaluation results dict
    """
    if method not in EVAL_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {get_available_methods()}")

    return EVAL_METHODS[method]['func'](results_by_qid)
