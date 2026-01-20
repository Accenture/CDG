#!/usr/bin/env python3
"""
Exp 3: CDG Hyperparameter Sweep

Standalone script that reads directly from Yu's pickle directories.
Sweeps alpha, beta, and position_pct to find optimal CDG parameters.

Beta values: 1, 2, 3, 4, 5, 10, 15, 20, 25, 30
Alpha values: 0.5, 1.0
Position pct: 10, 20 (prioritize alpha=0.5, position=10 first)

Formula: score = count^alpha * mean(mean_conf + beta * gradient)
Where gradient = mean(last P%) - mean(first P%)

Usage:
    python scripts/eval/exp3_cdg_hyperparam_sweep.py
"""
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
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

# Hyperparameter sweep values
BETA_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
ALPHA_VALUES = [0.5, 1.0]
POSITION_PCT_VALUES = [10, 20]

# Output directory for results (local - small files only)
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'exp3_cdg_sweep'


# ============================================================================
# HELPER FUNCTIONS (matching Yu's implementation exactly)
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


def compute_gradient(confs, percentile=10):
    """
    Compute gradient: mean(last percentile%) - mean(first percentile%)
    Exactly matching Yu's implementation.
    """
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


# ============================================================================
# DATA LOADING (deterministic - matching Yu's implementation)
# ============================================================================

def load_dataset(results_dir: str, position_pct: int = 10) -> dict:
    """
    Load dataset and precompute metrics for all traces.
    Exactly matching Yu's load_dataset function.
    """
    results_dir = Path(results_dir)

    # Find pickle files - try direct glob first, then subdirectories
    pkl_files = sorted(results_dir.glob('qid*.pkl'))
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
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        qid = data['qid']
        gt = data['ground_truth']

        traces_list = []
        for trace in data['all_traces']:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is not None and len(confs) >= 10:
                mean_conf = np.mean(confs)
                gradient = compute_gradient(confs, percentile=position_pct)
                traces_list.append({
                    'answer': answer,
                    'mean_conf': mean_conf,
                    'gradient': gradient,
                })

        question_data[qid] = {'gt': gt, 'traces': traces_list}

    return question_data


# ============================================================================
# CDG VOTING (exactly matching Yu's implementation)
# ============================================================================

def cdg_voting(question_data: dict, alpha: float = 0.5, beta: float = 10) -> tuple:
    """
    CDG (Count-Dampened Gradient) Voting.

    Formula: score = count^alpha * mean(mean_conf + beta * gradient)

    Returns:
        (correct_count, total_count, per_question_results)
    """
    correct = 0
    total = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        if not traces:
            continue

        total += 1

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

        results[qid] = {
            'correct': is_correct,
            'winner': winner,
            'gt': gt,
        }

    return correct, total, results


# ============================================================================
# MAIN SWEEP
# ============================================================================

def run_sweep():
    """Run full hyperparameter sweep."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 100)
    print('EXP 3: CDG HYPERPARAMETER SWEEP')
    print('=' * 100)
    print()
    print(f'Beta values: {BETA_VALUES}')
    print(f'Alpha values: {ALPHA_VALUES}')
    print(f'Position pct values: {POSITION_PCT_VALUES}')
    print(f'Models: {list(DATASETS.keys())}')
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    all_results = []

    # Prioritize alpha=0.5, position=10 first
    position_order = [10, 20]
    alpha_order = [0.5, 1.0]

    for position_pct in position_order:
        print()
        print('#' * 100)
        print(f'# POSITION PCT = {position_pct}')
        print('#' * 100)

        for alpha in alpha_order:
            print()
            print(f'--- Alpha = {alpha}, Position = {position_pct} ---')
            print()

            for model_name, datasets in DATASETS.items():
                print(f'\n{model_name.upper()}')
                print('-' * 80)

                # Header
                header = f"{'Dataset':<15}"
                for beta in BETA_VALUES:
                    header += f"β={beta:<6}"
                print(header)

                model_data = {}

                for dataset_name, dataset_path in datasets.items():
                    # Load data once per dataset/position_pct combination
                    question_data = load_dataset(dataset_path, position_pct=position_pct)

                    if not question_data:
                        print(f"{dataset_name:<15} [NO DATA]")
                        continue

                    row = f"{dataset_name:<15}"

                    for beta in BETA_VALUES:
                        correct, total, _ = cdg_voting(question_data, alpha=alpha, beta=beta)
                        acc = correct / total if total > 0 else 0
                        row += f"{100*acc:.1f}%   "

                        # Store result
                        result = {
                            'model': model_name,
                            'dataset': dataset_name,
                            'alpha': alpha,
                            'beta': beta,
                            'position_pct': position_pct,
                            'correct': correct,
                            'total': total,
                            'accuracy': acc,
                        }
                        all_results.append(result)

                    print(row)

    # Save all results to JSON
    output_file = OUTPUT_DIR / f'cdg_sweep_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'beta_values': BETA_VALUES,
            'alpha_values': ALPHA_VALUES,
            'position_pct_values': POSITION_PCT_VALUES,
            'results': all_results,
        }, f, indent=2)
    print(f'\nResults saved to: {output_file}')

    # Find and print best parameters per model
    print()
    print('=' * 100)
    print('BEST PARAMETERS PER MODEL')
    print('=' * 100)

    best_params = {}
    for model_name in DATASETS.keys():
        model_results = [r for r in all_results if r['model'] == model_name]

        # Aggregate across datasets for each param combination
        param_scores = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in model_results:
            key = (r['alpha'], r['beta'], r['position_pct'])
            param_scores[key]['correct'] += r['correct']
            param_scores[key]['total'] += r['total']

        # Find best
        best_key = None
        best_acc = -1
        for key, scores in param_scores.items():
            acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_key = key

        if best_key:
            alpha, beta, position_pct = best_key
            scores = param_scores[best_key]
            best_params[model_name] = {
                'alpha': alpha,
                'beta': beta,
                'position_pct': position_pct,
                'correct': scores['correct'],
                'total': scores['total'],
                'accuracy': best_acc,
            }
            print(f"{model_name:<15} alpha={alpha}, beta={beta}, position_pct={position_pct} "
                  f"=> {scores['correct']}/{scores['total']} ({100*best_acc:.1f}%)")

    # Save best params
    best_params_file = OUTPUT_DIR / f'best_params_{timestamp}.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'\nBest params saved to: {best_params_file}')

    return all_results, best_params


if __name__ == '__main__':
    run_sweep()
