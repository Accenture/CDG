"""
Compute upper bound (oracle accuracy) from offline results.

The upper bound is: for each question, if ANY trace has the correct answer,
count it as correct. This shows the maximum achievable accuracy with
perfect weighting/selection.

Usage:
    python compute_upper_bound.py --results_dir offline_results/
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

from dynasor.core.evaluator import math_equal


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
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
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def load_results(results_dir: str) -> dict:
    """Load all result pickle files, grouped by qid"""
    results_by_qid = defaultdict(list)

    for filename in Path(results_dir).glob("*.pkl"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        qid = data.get('qid')
        if qid is not None:
            results_by_qid[qid].append(data)

    return dict(results_by_qid)


def compute_upper_bound(results_by_qid: dict) -> dict:
    """Compute upper bound and per-trace accuracy statistics"""
    stats = {
        'total_questions': 0,
        'oracle_correct': 0,
        'per_question': {},
        'answer_distribution': defaultdict(int),
        'total_traces': 0,
        'correct_traces': 0,
    }

    for qid, results_list in sorted(results_by_qid.items()):
        # Combine all traces for this question (might have multiple runs)
        all_traces = []
        ground_truth = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        stats['total_questions'] += 1
        stats['total_traces'] += len(all_traces)

        # Check each trace
        has_correct = False
        correct_count = 0
        answers = defaultdict(int)

        for trace in all_traces:
            answer = trace.get('extracted_answer')
            if answer:
                answers[answer] += 1
                try:
                    is_correct = equal_func(answer, ground_truth)
                    if is_correct:
                        has_correct = True
                        correct_count += 1
                except:
                    pass

        stats['correct_traces'] += correct_count

        if has_correct:
            stats['oracle_correct'] += 1

        stats['per_question'][qid] = {
            'ground_truth': ground_truth,
            'num_traces': len(all_traces),
            'num_correct': correct_count,
            'trace_accuracy': correct_count / len(all_traces) if all_traces else 0,
            'has_correct': has_correct,
            'num_unique_answers': len(answers),
            'top_answers': sorted(answers.items(), key=lambda x: -x[1])[:5],
        }

        # Track answer distribution
        for answer, count in answers.items():
            stats['answer_distribution'][answer] += count

    return stats


def print_report(stats: dict):
    """Print detailed report"""
    print("\n" + "=" * 70)
    print("UPPER BOUND ANALYSIS")
    print("=" * 70)

    total = stats['total_questions']
    oracle = stats['oracle_correct']

    print(f"\nOracle Accuracy (Upper Bound): {oracle}/{total} = {100*oracle/total:.1f}%")
    print(f"Total traces analyzed: {stats['total_traces']}")
    print(f"Correct traces: {stats['correct_traces']} ({100*stats['correct_traces']/stats['total_traces']:.1f}%)")

    print("\n" + "-" * 70)
    print("Per-Question Details")
    print("-" * 70)
    print(f"{'QID':<5} {'GT':<10} {'Traces':<8} {'Correct':<8} {'Acc%':<8} {'Oracle'}")
    print("-" * 70)

    for qid, q_stats in sorted(stats['per_question'].items()):
        gt = str(q_stats['ground_truth'])[:8]
        oracle_mark = "✓" if q_stats['has_correct'] else "✗"
        print(f"{qid:<5} {gt:<10} {q_stats['num_traces']:<8} {q_stats['num_correct']:<8} "
              f"{100*q_stats['trace_accuracy']:.1f}%    {oracle_mark}")

    # Questions where oracle fails (no trace got correct answer)
    failed = [qid for qid, q in stats['per_question'].items() if not q['has_correct']]
    if failed:
        print(f"\nQuestions with NO correct trace: {failed}")
        print("(These questions are impossible to get right with current sampling)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='offline_results')
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results_by_qid = load_results(args.results_dir)
    print(f"Found {len(results_by_qid)} questions")

    stats = compute_upper_bound(results_by_qid)
    print_report(stats)

    # Save stats
    output_file = os.path.join(args.results_dir, 'upper_bound_stats.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\nStats saved to {output_file}")


if __name__ == "__main__":
    main()
