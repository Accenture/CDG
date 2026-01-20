#!/usr/bin/env python3
"""
Exp 2: Baseline Methods vs CDG Comparison

Compares three baseline methods against CDG:
1. Majority Vote - simple count-based voting
2. Mean Weighted - weighted vote using mean confidence
3. Top10 Tail Filtered - DeepConf's best method (filter top 10% by tail conf)
4. CDG - Count-Dampened Gradient (our method)

CDG Parameters (from Yu's optimal settings):
- DeepSeek-R1-8B: alpha=0.5, beta=10, position_pct=10
- GPT-OSS-20B: alpha=0.5, beta=10, position_pct=10
- GEMMA-3-27B: alpha=0.5, beta=3, position_pct=10
- QWQ-32B: alpha=0.5, beta=3, position_pct=10

Baseline implementation matches Yu's test_other_models.py exactly.

Usage:
    python scripts/eval/exp2_baseline_vs_cdg.py
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys
import json
from datetime import datetime

# Use dynasor from conda
sys.path.insert(0, '/mnt/yuenvs/deepconf/lib/python3.10/site-packages')
from dynasor.core.evaluator import math_equal


# ============================================================================
# HARDCODED DATA PATHS (Yu's directories - DO NOT MODIFY PICKLE FILES)
# ============================================================================

DATASETS = {
    'deepseek8b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2024_run_512_results_deepseek8b',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2025_run_512_results_newpara',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/bruno_2025_run_512_results_deepseek8b',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/hmmt_feb_2025_run_512_results',
    },
    'gptoss20b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason',
    },
    'gemma3_27b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512',
    },
    'qwq32b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512',
    },
}

# CDG optimal parameters per model (from Yu's CDG_ablation_beta_paper.md)
CDG_PARAMS = {
    'deepseek8b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gptoss20b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gemma3_27b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
    'qwq32b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
}

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'exp2_baseline_vs_cdg'


# ============================================================================
# HELPER FUNCTIONS (matching Yu's test_other_models.py exactly)
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
            text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    try:
        return math_equal(answer, ground_truth)
    except:
        return str(answer).strip() == str(ground_truth).strip()


# ============================================================================
# BASELINE METHODS (exactly matching Yu's test_other_models.py)
# ============================================================================

def calculate_mean_confidence(trace):
    """Calculate mean confidence of all tokens in trace."""
    try:
        if 'confs' in trace and trace['confs']:
            return np.mean(trace['confs'])
        return 0.0
    except:
        return 0.0


def calculate_tail_confidence(trace, tail_tokens=2048):
    """Calculate mean confidence of last tail_tokens."""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except:
        return 0.0


def simple_majority_vote(answers):
    """Simple majority vote - return most common answer."""
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def weighted_majority_vote(answers, weights):
    """Weighted majority vote - sum weights per answer."""
    if not answers:
        return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights:
        return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def filter_top_confidence(traces, top_percent=0.1):
    """Filter to top percent traces by tail confidence."""
    if not traces:
        return []
    confidences = [calculate_tail_confidence(t) for t in traces]
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    return [t for t, c in zip(traces, confidences) if c >= threshold]


# ============================================================================
# CDG METHOD (matching Yu's CDG_ablation_beta_paper.py)
# ============================================================================

def compute_gradient(confs, percentile=10):
    """Compute gradient: mean(last percentile%) - mean(first percentile%)"""
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def cdg_vote(traces, alpha=0.5, beta=10, position_pct=10):
    """
    CDG voting on a list of traces.

    Formula: score = count^alpha * mean(mean_conf + beta * gradient)
    """
    if not traces:
        return None

    answer_scores = defaultdict(list)
    for trace in traces:
        answer = trace.get('extracted_answer')
        confs = trace.get('confs', [])

        if answer is not None and len(confs) >= 10:
            mean_conf = np.mean(confs)
            gradient = compute_gradient(confs, percentile=position_pct)
            score = mean_conf + beta * gradient
            answer_scores[answer].append(score)

    if not answer_scores:
        return None

    final_scores = {
        ans: (len(scores) ** alpha) * np.mean(scores)
        for ans, scores in answer_scores.items()
    }
    return max(final_scores.items(), key=lambda x: x[1])[0]


# ============================================================================
# EVALUATION
# ============================================================================

def load_pickle_files(results_dir: str) -> list:
    """Load all pickle files from directory."""
    results_dir = Path(results_dir)

    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    return pkl_files


def evaluate_dataset(dataset_path: str, cdg_params: dict) -> dict:
    """
    Evaluate all 4 methods on a single dataset.

    Returns dict with counts for each method.
    """
    pkl_files = load_pickle_files(dataset_path)

    if not pkl_files:
        return None

    results = {
        'majority': {'correct': 0, 'total': 0},
        'mean_weighted': {'correct': 0, 'total': 0},
        'top10_tail': {'correct': 0, 'total': 0},
        'cdg': {'correct': 0, 'total': 0},
    }

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        gt = data['ground_truth']
        valid_traces = [t for t in data['all_traces'] if t.get('extracted_answer')]

        if not valid_traces:
            continue

        answers = [t['extracted_answer'] for t in valid_traces]

        # 1. Majority Vote
        results['majority']['total'] += 1
        winner = simple_majority_vote(answers)
        if winner and equal_func(str(winner), str(gt)):
            results['majority']['correct'] += 1

        # 2. Mean Confidence Weighted
        results['mean_weighted']['total'] += 1
        mean_confs = [calculate_mean_confidence(t) for t in valid_traces]
        if any(c > 0 for c in mean_confs):
            winner = weighted_majority_vote(answers, mean_confs)
            if winner and equal_func(str(winner), str(gt)):
                results['mean_weighted']['correct'] += 1

        # 3. Top 10% Tail Filtered
        results['top10_tail']['total'] += 1
        top_traces = filter_top_confidence(valid_traces, 0.1)
        if top_traces:
            top_answers = [t['extracted_answer'] for t in top_traces]
            top_confs = [calculate_tail_confidence(t) for t in top_traces]
            if any(c > 0 for c in top_confs):
                winner = weighted_majority_vote(top_answers, top_confs)
                if winner and equal_func(str(winner), str(gt)):
                    results['top10_tail']['correct'] += 1

        # 4. CDG
        results['cdg']['total'] += 1
        winner = cdg_vote(
            valid_traces,
            alpha=cdg_params['alpha'],
            beta=cdg_params['beta'],
            position_pct=cdg_params['position_pct']
        )
        if winner and equal_func(str(winner), str(gt)):
            results['cdg']['correct'] += 1

    return results


def run_comparison():
    """Run full comparison of baselines vs CDG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 120)
    print('EXP 2: BASELINE METHODS VS CDG COMPARISON')
    print('=' * 120)
    print()
    print('Methods:')
    print('  1. Majority Vote - simple count-based voting')
    print('  2. Mean Weighted - weighted vote using mean confidence')
    print('  3. Top10 Tail - filter top 10% by tail conf, then weighted vote (DeepConf baseline)')
    print('  4. CDG - Count-Dampened Gradient (our method)')
    print()
    print('CDG Parameters:')
    for model, params in CDG_PARAMS.items():
        print(f"  {model}: alpha={params['alpha']}, beta={params['beta']}, position_pct={params['position_pct']}")
    print()

    all_results = {}

    for model_name, datasets in DATASETS.items():
        print()
        print('=' * 120)
        print(f'{model_name.upper()} RESULTS')
        print('=' * 120)

        cdg_params = CDG_PARAMS[model_name]
        print(f"CDG params: alpha={cdg_params['alpha']}, beta={cdg_params['beta']}, position_pct={cdg_params['position_pct']}")
        print()

        # Header
        print(f"{'Dataset':<15} {'Majority':<18} {'Mean Weighted':<18} {'Top10 Tail':<18} {'CDG':<18}")
        print('-' * 120)

        model_totals = {
            'majority': {'correct': 0, 'total': 0},
            'mean_weighted': {'correct': 0, 'total': 0},
            'top10_tail': {'correct': 0, 'total': 0},
            'cdg': {'correct': 0, 'total': 0},
        }

        model_results = {}

        for dataset_name, dataset_path in datasets.items():
            results = evaluate_dataset(dataset_path, cdg_params)

            if results is None:
                print(f"{dataset_name:<15} [NO DATA]")
                continue

            model_results[dataset_name] = results

            # Print row
            row = f"{dataset_name:<15}"
            for method in ['majority', 'mean_weighted', 'top10_tail', 'cdg']:
                c, t = results[method]['correct'], results[method]['total']
                pct = 100 * c / t if t > 0 else 0
                row += f"{c}/{t} ({pct:.1f}%)     "
                model_totals[method]['correct'] += c
                model_totals[method]['total'] += t
            print(row)

        # Print totals
        print('-' * 120)
        row = f"{'TOTAL':<15}"
        for method in ['majority', 'mean_weighted', 'top10_tail', 'cdg']:
            c, t = model_totals[method]['correct'], model_totals[method]['total']
            pct = 100 * c / t if t > 0 else 0
            row += f"{c}/{t} ({pct:.1f}%)     "
        print(row)

        all_results[model_name] = {
            'datasets': model_results,
            'totals': model_totals,
            'cdg_params': cdg_params,
        }

    # Grand summary
    print()
    print('=' * 120)
    print('GRAND SUMMARY')
    print('=' * 120)
    print()
    print(f"{'Model':<15} {'Majority':<15} {'Mean Weighted':<15} {'Top10 Tail':<15} {'CDG':<15} {'CDG vs Maj':<12} {'CDG vs Top10':<12}")
    print('-' * 120)

    for model_name in DATASETS.keys():
        totals = all_results[model_name]['totals']

        maj_acc = totals['majority']['correct'] / totals['majority']['total'] if totals['majority']['total'] > 0 else 0
        mean_acc = totals['mean_weighted']['correct'] / totals['mean_weighted']['total'] if totals['mean_weighted']['total'] > 0 else 0
        top10_acc = totals['top10_tail']['correct'] / totals['top10_tail']['total'] if totals['top10_tail']['total'] > 0 else 0
        cdg_acc = totals['cdg']['correct'] / totals['cdg']['total'] if totals['cdg']['total'] > 0 else 0

        cdg_vs_maj = cdg_acc - maj_acc
        cdg_vs_top10 = cdg_acc - top10_acc

        print(f"{model_name:<15} {100*maj_acc:.1f}%          {100*mean_acc:.1f}%          {100*top10_acc:.1f}%          {100*cdg_acc:.1f}%          {100*cdg_vs_maj:+.1f}%        {100*cdg_vs_top10:+.1f}%")

    # Save results
    output_file = OUTPUT_DIR / f'baseline_vs_cdg_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'cdg_params': CDG_PARAMS,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f'\nResults saved to: {output_file}')

    return all_results


if __name__ == '__main__':
    run_comparison()
