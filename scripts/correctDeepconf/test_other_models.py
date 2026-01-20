"""
Test 3 methods on GPT-OSS, GEMMA, QWQ: majority, mean_confidence_weighted, top10_tail_filtered
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, '/eph/nvme0/env/deepconf/lib/python3.10/site-packages')
from dynasor.core.evaluator import math_equal

def quick_parse(text):
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1: break
            end = text.find('}', start)
            if end == -1: break
            text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text

def equal_func(answer, ground_truth):
    answer, ground_truth = quick_parse(str(answer)), quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    try: return math_equal(answer, ground_truth)
    except: return str(answer).strip() == str(ground_truth).strip()

def calculate_mean_confidence(trace):
    try:
        if 'confs' in trace and trace['confs']:
            return np.mean(trace['confs'])
        return 0.0
    except: return 0.0

def calculate_tail_confidence(trace, tail_tokens=2048):
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except: return 0.0

def simple_majority_vote(answers):
    if not answers: return None
    return Counter(answers).most_common(1)[0][0]

def weighted_majority_vote(answers, weights):
    if not answers: return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights: return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def filter_top_confidence(traces, top_percent=0.1):
    if not traces: return []
    confidences = [calculate_tail_confidence(t) for t in traces]
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    return [t for t, c in zip(traces, confidences) if c >= threshold]

def test_dataset(path, name):
    results_dir = Path(path)
    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files: break
    if not pkl_files:
        return 0, 0, 0, 0

    maj_c, mean_c, top10_c = 0, 0, 0
    n = len(pkl_files)

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        gt = data['ground_truth']
        valid_traces = [t for t in data['all_traces'] if t.get('extracted_answer')]
        if not valid_traces: continue

        answers = [t['extracted_answer'] for t in valid_traces]

        # 1. Majority
        w = simple_majority_vote(answers)
        if w and equal_func(str(w), str(gt)): maj_c += 1

        # 2. Mean confidence weighted
        mean_confs = [calculate_mean_confidence(t) for t in valid_traces]
        if any(c > 0 for c in mean_confs):
            w = weighted_majority_vote(answers, mean_confs)
            if w and equal_func(str(w), str(gt)): mean_c += 1

        # 3. Top 10% tail filtered
        top_traces = filter_top_confidence(valid_traces, 0.1)
        if top_traces:
            top_answers = [t['extracted_answer'] for t in top_traces]
            top_confs = [calculate_tail_confidence(t) for t in top_traces]
            if any(c > 0 for c in top_confs):
                w = weighted_majority_vote(top_answers, top_confs)
                if w and equal_func(str(w), str(gt)): top10_c += 1

    return n, maj_c, mean_c, top10_c

def run_model(model_name, datasets):
    print("="*70)
    print(f"{model_name} Results")
    print("="*70)
    print(f"{'Dataset':<15} {'Majority':<15} {'Mean Weighted':<15} {'Top10 Tail':<15}")
    print("-"*70)

    tot_n, tot_maj, tot_mean, tot_top10 = 0, 0, 0, 0
    for name, path in datasets:
        n, maj, mean, top10 = test_dataset(path, name)
        tot_n += n; tot_maj += maj; tot_mean += mean; tot_top10 += top10
        if n > 0:
            print(f"{name:<15} {maj}/{n} ({100*maj/n:.1f}%)      {mean}/{n} ({100*mean/n:.1f}%)      {top10}/{n} ({100*top10/n:.1f}%)")

    print("-"*70)
    if tot_n > 0:
        print(f"{'TOTAL':<15} {tot_maj}/{tot_n} ({100*tot_maj/tot_n:.1f}%)      {tot_mean}/{tot_n} ({100*tot_mean/tot_n:.1f}%)      {tot_top10}/{tot_n} ({100*tot_top10/tot_n:.1f}%)")
    print("="*70)
    print()

if __name__ == '__main__':
    # GPT-OSS
    gptoss_datasets = [
        ('AIME 2024', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason'),
        ('AIME 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason'),
        ('BRUNO 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason'),
        ('HMMT 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason'),
    ]
    run_model('GPT-OSS-20B', gptoss_datasets)

    # GEMMA
    gemma_datasets = [
        ('AIME 2024', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512'),
        ('AIME 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512'),
        ('BRUNO 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512'),
        ('HMMT 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512'),
    ]
    run_model('GEMMA-3-27B', gemma_datasets)

    # QWQ
    qwq_datasets = [
        ('AIME 2024', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512'),
        ('AIME 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512'),
        ('BRUNO 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512'),
        ('HMMT 2025', '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512'),
    ]
    run_model('QWQ-32B', qwq_datasets)
