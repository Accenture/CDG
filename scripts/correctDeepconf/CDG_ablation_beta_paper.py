"""
CDG Paper Results: Testing with optimal beta values per model

- DeepSeek-R1-8B: α=0.5, β=10, P=10%
- GPT-OSS-20B: α=0.5, β=10, P=10%
- GEMMA-3-27B: α=0.5, β=3, P=10%
- QWQ-32B: α=0.5, β=3, P=10%

Formula: score = count^α × mean(mean_conf + β × gradient)
Where gradient = mean(last P%) - mean(first P%)
"""
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os

sys.path.insert(0, '/mnt/yuenvs/deepconf/lib/python3.10/site-packages')
from dynasor.core.evaluator import math_equal


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
    """
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def cdg_voting(question_data, alpha=0.5, beta=10):
    """
    CDG (Count-Dampened Gradient) Voting.

    Formula: score = count^α × mean(mean_conf + β × gradient)

    Args:
        question_data: dict of {qid: {'gt': ground_truth, 'traces': [...]}}
        alpha: Count dampening exponent (0.5 = sqrt)
        beta: Gradient weight

    Returns:
        correct_count, results dict
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

        results[qid] = {
            'correct': is_correct,
            'winner': winner,
        }

    return correct, results


def load_dataset(results_dir, position_pct=10):
    """Load dataset and precompute metrics for all traces."""
    results_dir = Path(results_dir)

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


def run_evaluation(datasets, model_name, alpha, beta, position_pct):
    """Run evaluation on all datasets for a model."""
    print('='*100)
    print(f'{model_name} RESULTS')
    print('='*100)
    print(f'CDG Parameters: α={alpha}, β={beta}, P={position_pct}%')
    print()

    # Load all datasets
    print('Loading datasets...')
    all_data = {}
    for name, path in datasets.items():
        data = load_dataset(path, position_pct=position_pct)
        if data:
            all_data[name] = data
            print(f'  {name}: {len(data)} questions')
    print()

    # Main Results
    header = f'{"Dataset":<30} {"CDG":<20} {"Wrong QIDs":<50}'
    print(header)
    print('-'*100)

    total_correct = 0
    total_questions = 0
    all_wrong_qids = {}
    all_results = {}

    for dataset_name, question_data in all_data.items():
        n = len(question_data)
        total_questions += n

        cdg_correct, cdg_results = cdg_voting(question_data, alpha=alpha, beta=beta)
        total_correct += cdg_correct

        # Track wrong QIDs
        wrong = sorted([qid for qid, res in cdg_results.items() if not res['correct']])
        all_wrong_qids[dataset_name] = wrong

        wrong_str = str(wrong[:5]) + ('...' if len(wrong) > 5 else '')
        print(f'{dataset_name:<30} {cdg_correct}/{n} ({100*cdg_correct/n:.1f}%){" "*10} {wrong_str:<50}')

        all_results[dataset_name] = {
            'correct': cdg_correct,
            'total': n,
            'accuracy': 100 * cdg_correct / n,
            'wrong_qids': wrong
        }

    print('-'*100)
    print(f'{"TOTAL":<30} {total_correct}/{total_questions} ({100*total_correct/total_questions:.1f}%)')
    print()

    print('Detailed Wrong QIDs per dataset:')
    for name, qids in all_wrong_qids.items():
        print(f'  {name}: {qids}')
    print()

    return total_correct, total_questions, all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # CDG Parameters - different beta for different models
    ALPHA = 0.5
    POSITION_PCT = 10

    # Beta values optimized per model
    BETA_HIGH = 10  # For DeepSeek and GPT-OSS
    BETA_LOW = 3    # For GEMMA and QWQ

    # All 16 datasets from paper_results_deepconfscore.md

    # DeepSeek-R1-8B Datasets (beta=10)
    DEEPSEEK_DATASETS = {
        'AIME 2024 (DeepSeek)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2024_run_512_results_deepseek8b',
        'AIME 2025 (DeepSeek)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2025_run_512_results_newpara',
        'BRUNO 2025 (DeepSeek)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/bruno_2025_run_512_results_deepseek8b',
        'HMMT 2025 (DeepSeek)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/hmmt_feb_2025_run_512_results',
    }

    # GPT-OSS-20B Datasets (beta=10)
    GPTOSS_DATASETS = {
        'AIME 2024 (GPT-OSS)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason',
        'AIME 2025 (GPT-OSS)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason',
        'BRUNO 2025 (GPT-OSS)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason',
        'HMMT 2025 (GPT-OSS)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason',
    }

    # GEMMA-3-27B Datasets (beta=3)
    GEMMA_DATASETS = {
        'AIME 2024 (GEMMA)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512',
        'AIME 2025 (GEMMA)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512',
        'BRUNO 2025 (GEMMA)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512',
        'HMMT 2025 (GEMMA)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512',
    }

    # QWQ-32B Datasets (beta=3)
    QWQ_DATASETS = {
        'AIME 2024 (QWQ)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512',
        'AIME 2025 (QWQ)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512',
        'BRUNO 2025 (QWQ)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512',
        'HMMT 2025 (QWQ)': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512',
    }

    print('='*100)
    print('CDG PAPER RESULTS: OPTIMIZED BETA VALUES PER MODEL')
    print('='*100)
    print()
    print('Configuration:')
    print('  DeepSeek-R1-8B: α=0.5, β=10, P=10%')
    print('  GPT-OSS-20B:    α=0.5, β=10, P=10%')
    print('  GEMMA-3-27B:    α=0.5, β=3,  P=10%')
    print('  QWQ-32B:        α=0.5, β=3,  P=10%')
    print()

    all_model_results = {}

    # Run evaluation for each model with appropriate beta
    deepseek_correct, deepseek_total, deepseek_results = run_evaluation(
        DEEPSEEK_DATASETS, 'DeepSeek-R1-8B', ALPHA, BETA_HIGH, POSITION_PCT
    )
    all_model_results['DeepSeek-R1-8B'] = deepseek_results
    print('\n' + '='*100 + '\n')

    gptoss_correct, gptoss_total, gptoss_results = run_evaluation(
        GPTOSS_DATASETS, 'GPT-OSS-20B', ALPHA, BETA_HIGH, POSITION_PCT
    )
    all_model_results['GPT-OSS-20B'] = gptoss_results
    print('\n' + '='*100 + '\n')

    gemma_correct, gemma_total, gemma_results = run_evaluation(
        GEMMA_DATASETS, 'GEMMA-3-27B', ALPHA, BETA_LOW, POSITION_PCT
    )
    all_model_results['GEMMA-3-27B'] = gemma_results
    print('\n' + '='*100 + '\n')

    qwq_correct, qwq_total, qwq_results = run_evaluation(
        QWQ_DATASETS, 'QWQ-32B', ALPHA, BETA_LOW, POSITION_PCT
    )
    all_model_results['QWQ-32B'] = qwq_results

    # Final Summary
    grand_total_correct = deepseek_correct + gptoss_correct + gemma_correct + qwq_correct
    grand_total = deepseek_total + gptoss_total + gemma_total + qwq_total

    print()
    print('='*100)
    print('FINAL SUMMARY')
    print('='*100)
    print()
    print(f'{"Model":<20} {"Correct/Total":<20} {"Accuracy":<15} {"Beta":<10}')
    print('-'*100)
    print(f'{"DeepSeek-R1-8B":<20} {deepseek_correct}/{deepseek_total:<15} {100*deepseek_correct/deepseek_total:.1f}%{" "*10} β={BETA_HIGH}')
    print(f'{"GPT-OSS-20B":<20} {gptoss_correct}/{gptoss_total:<15} {100*gptoss_correct/gptoss_total:.1f}%{" "*10} β={BETA_HIGH}')
    print(f'{"GEMMA-3-27B":<20} {gemma_correct}/{gemma_total:<15} {100*gemma_correct/gemma_total:.1f}%{" "*10} β={BETA_LOW}')
    print(f'{"QWQ-32B":<20} {qwq_correct}/{qwq_total:<15} {100*qwq_correct/qwq_total:.1f}%{" "*10} β={BETA_LOW}')
    print('-'*100)
    print(f'{"GRAND TOTAL":<20} {grand_total_correct}/{grand_total:<15} {100*grand_total_correct/grand_total:.1f}%')
    print()
