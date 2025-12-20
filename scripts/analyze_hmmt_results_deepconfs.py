"""
Analyze HMMT offline batch inference results
Computes accuracy for all DeepConf voting methods

Usage:
    python scripts/analyze_hmmt_results_deepconfs.py --results_dir /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b
"""
import os
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List
from tqdm import tqdm
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
    """Check if answer equals ground truth using mathematical equivalence"""
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))

    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        try:
            return math_equal(answer, ground_truth)
        except:
            return str(answer).strip() == str(ground_truth).strip()


def find_pickle_files(results_dir: str) -> List[Path]:
    """Find all pickle files in results directory"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return []

    # Find all .pkl files
    pkl_files = list(results_path.glob("qid*.pkl"))

    # Sort by qid
    def get_qid(filepath):
        import re
        match = re.search(r'qid(\d+)', filepath.name)
        return int(match.group(1)) if match else float('inf')

    return sorted(pkl_files, key=get_qid)


def load_result(filepath: Path) -> Dict:
    """Load a single result pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def evaluate_result(result: Dict) -> Dict:
    """Evaluate all voting methods for a single result"""
    if not result or not result.get('voting_results'):
        return None

    ground_truth = str(result.get('ground_truth', '')).strip()
    voting_results = result['voting_results']

    evaluation = {}

    for method, result_data in voting_results.items():
        if result_data and result_data.get('answer'):
            try:
                is_correct = equal_func(result_data['answer'], ground_truth)
            except Exception as e:
                print(f"Error comparing answers: {e}")
                is_correct = False

            evaluation[method] = {
                'answer': result_data['answer'],
                'is_correct': is_correct,
                'confidence': result_data.get('confidence'),
                'num_votes': result_data.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None,
                'is_correct': False,
                'confidence': None,
                'num_votes': 0
            }

    return evaluation


def analyze_voting_methods(results: List[Dict]) -> Dict:
    """Analyze accuracy of different voting methods across all results"""
    method_stats = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'answers': [],
        'num_votes': [],
        'qids_correct': [],
        'qids_wrong': []
    })

    for idx, result in enumerate(results):
        if not result:
            continue

        evaluation = evaluate_result(result)
        if not evaluation:
            continue

        qid = result.get('qid', idx)

        for method, eval_data in evaluation.items():
            if eval_data.get('answer') is not None:
                method_stats[method]['total'] += 1
                if eval_data.get('is_correct'):
                    method_stats[method]['correct'] += 1
                    method_stats[method]['qids_correct'].append(qid)
                else:
                    method_stats[method]['qids_wrong'].append(qid)

                method_stats[method]['answers'].append({
                    'qid': qid,
                    'answer': eval_data['answer'],
                    'confidence': eval_data.get('confidence'),
                    'is_correct': eval_data.get('is_correct')
                })
                method_stats[method]['num_votes'].append(eval_data['num_votes'])

    # Calculate accuracy for each method
    method_accuracy = {}
    for method, stats in method_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            method_accuracy[method] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'qids_correct': stats['qids_correct'],
                'qids_wrong': stats['qids_wrong'],
                'avg_confidence': np.mean([a['confidence'] for a in stats['answers']
                                          if a['confidence'] is not None])
                                 if any(a['confidence'] for a in stats['answers']) else None,
                'avg_num_votes': np.mean(stats['num_votes']) if stats['num_votes'] else 0,
            }

    return method_accuracy


def analyze_token_usage(results: List[Dict]) -> Dict:
    """Analyze token usage across all results"""
    total_tokens = []
    tokens_per_question = []
    traces_per_question = []

    for result in results:
        if result:
            total_tokens.append(result.get('total_tokens', 0))

            num_traces = result.get('total_traces_count', 0)
            if num_traces > 0:
                tokens_per_question.append(result['total_tokens'])
                traces_per_question.append(num_traces)

    return {
        'total_tokens': {
            'sum': sum(total_tokens),
            'mean': np.mean(total_tokens) if total_tokens else 0,
            'std': np.std(total_tokens) if total_tokens else 0,
            'min': min(total_tokens) if total_tokens else 0,
            'max': max(total_tokens) if total_tokens else 0,
        },
        'avg_traces_per_question': np.mean(traces_per_question) if traces_per_question else 0,
    }


def print_statistics(method_accuracy: Dict, token_stats: Dict, results: List[Dict]):
    """Print comprehensive statistics"""
    print("\n" + "="*80)
    print("HMMT FEBRUARY 2025 RESULTS ANALYSIS")
    print("="*80)

    # Overall statistics
    print(f"\n📊 Overall Statistics")
    print("-"*80)
    print(f"Total questions analyzed: {len(results)}")
    valid_results = sum(1 for r in results if r is not None)
    print(f"Valid results: {valid_results}")

    # Token usage
    if token_stats:
        print(f"\n💰 Token Usage")
        print("-"*80)
        print(f"Total tokens: {token_stats['total_tokens']['sum']:,}")
        print(f"Average tokens per question: {token_stats['total_tokens']['mean']:,.0f} ± {token_stats['total_tokens']['std']:,.0f}")
        print(f"Average traces per question: {token_stats['avg_traces_per_question']:.0f}")

    # Voting methods accuracy
    if method_accuracy:
        print(f"\n🗳️  VOTING METHODS ACCURACY (DeepConf)")
        print("="*80)
        print(f"{'Method':<35} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Conf':<12}")
        print("-"*80)

        # Sort by accuracy
        sorted_methods = sorted(method_accuracy.items(),
                              key=lambda x: x[1]['accuracy'], reverse=True)

        for method, stats in sorted_methods:
            acc_str = f"{stats['accuracy']:.2%}"
            ratio_str = f"{stats['correct']}/{stats['total']}"
            conf_str = f"{stats['avg_confidence']:.4f}" if stats['avg_confidence'] else "N/A"

            # Highlight the main deepconf method
            marker = " ⭐" if method == "bottom_window_weighted" else ""

            print(f"{method:<35} {acc_str:<12} {ratio_str:<15} {conf_str:<12}{marker}")

        print("\n⭐ = Main DeepConf method (bottom window weighted)")

        # Print detailed wrong questions for best method
        best_method = sorted_methods[0][0]
        best_stats = sorted_methods[0][1]

        if best_stats['qids_wrong']:
            print(f"\n❌ Questions wrong by best method ({best_method}):")
            print(f"   QIDs: {best_stats['qids_wrong']}")

    print("\n" + "="*80)


def print_detailed_comparison(results: List[Dict], method_accuracy: Dict):
    """Print detailed per-question comparison"""
    print("\n📋 DETAILED PER-QUESTION RESULTS")
    print("="*80)

    print(f"\n{'QID':<5} {'Ground Truth':<20} ", end="")

    # Get method names
    methods = list(method_accuracy.keys())
    for method in methods[:4]:  # Show first 4 methods
        print(f"{method[:15]:<17}", end="")
    print()
    print("-"*80)

    for idx, result in enumerate(results):
        if not result:
            continue

        qid = result.get('qid', idx)
        gt = str(result.get('ground_truth', 'N/A'))[:18]

        print(f"{qid:<5} {gt:<20} ", end="")

        evaluation = evaluate_result(result)
        if evaluation:
            for method in methods[:4]:
                if method in evaluation:
                    is_correct = evaluation[method]['is_correct']
                    marker = "✓" if is_correct else "✗"
                    answer = str(evaluation[method].get('answer', 'N/A'))[:13]
                    print(f"{marker} {answer:<15}", end="")
                else:
                    print(f"  {'N/A':<15}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description='Analyze HMMT inference results with DeepConf voting methods')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result pickle files')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed per-question results')

    args = parser.parse_args()

    print(f"\n🔍 Analyzing results from: {args.results_dir}")

    # Find pickle files
    print("Finding pickle files...")
    pickle_files = find_pickle_files(args.results_dir)

    if not pickle_files:
        print("❌ No pickle files found!")
        return

    print(f"Found {len(pickle_files)} pickle files")

    # Load results
    print("Loading results...")
    results = []
    for filepath in tqdm(pickle_files):
        result = load_result(filepath)
        if result:
            results.append(result)

    print(f"Successfully loaded {len(results)} results")

    if not results:
        print("❌ No valid results to analyze!")
        return

    # Analyze
    print("\nAnalyzing voting methods...")
    method_accuracy = analyze_voting_methods(results)

    print("Analyzing token usage...")
    token_stats = analyze_token_usage(results)

    # Print statistics
    print_statistics(method_accuracy, token_stats, results)

    # Print detailed comparison if requested
    if args.detailed:
        print_detailed_comparison(results, method_accuracy)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
