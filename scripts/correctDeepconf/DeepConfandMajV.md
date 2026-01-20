# DeepConf and Majority Voting Methods

This document explains how to use the scripts to evaluate three voting methods on LLM reasoning traces.

## Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Majority Vote** | Simple count-based voting | `Counter(answers).most_common(1)[0][0]` |
| **Mean Confidence Weighted** | Weighted vote using mean of all token confidences | `weighted_majority_vote(answers, mean_confs)` |
| **Top10 Tail Filtered** | Filter to top 10% by tail confidence, then weighted vote | Filter → `weighted_majority_vote(answers, tail_confs)` |

## Code Location in Script

The script `test_other_models.py` implements all three methods. Here's where each method is computed:

### Method 1: Majority Vote (Lines 46-48, 89-91)

```python
# Function definition (line 46-48):
def simple_majority_vote(answers):
    if not answers: return None
    return Counter(answers).most_common(1)[0][0]

# Called at line 89-91:
# 1. Majority
w = simple_majority_vote(answers)
if w and equal_func(str(w), str(gt)): maj_c += 1
```

### Method 2: Mean Confidence Weighted (Lines 30-35, 93-97)

```python
# Function definition (line 30-35):
def calculate_mean_confidence(trace):
    if 'confs' in trace and trace['confs']:
        return np.mean(trace['confs'])
    return 0.0

# Called at line 93-97:
# 2. Mean confidence weighted
mean_confs = [calculate_mean_confidence(t) for t in valid_traces]
if any(c > 0 for c in mean_confs):
    w = weighted_majority_vote(answers, mean_confs)
    if w and equal_func(str(w), str(gt)): mean_c += 1
```

### Method 3: Top10 Tail Filtered (Lines 37-44, 60-64, 99-106)

```python
# Function definition (line 37-44):
def calculate_tail_confidence(trace, tail_tokens=2048):
    if 'confs' in trace and trace['confs']:
        confs = trace['confs']
        tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
        return np.mean(tail_confs) if tail_confs else 0.0
    return 0.0

# Filter function (line 60-64):
def filter_top_confidence(traces, top_percent=0.1):
    confidences = [calculate_tail_confidence(t) for t in traces]
    threshold = np.percentile(confidences, (1 - top_percent) * 100)  # 90th percentile
    return [t for t, c in zip(traces, confidences) if c >= threshold]

# Called at line 99-106:
# 3. Top 10% tail filtered
top_traces = filter_top_confidence(valid_traces, 0.1)
if top_traces:
    top_answers = [t['extracted_answer'] for t in top_traces]
    top_confs = [calculate_tail_confidence(t) for t in top_traces]
    if any(c > 0 for c in top_confs):
        w = weighted_majority_vote(top_answers, top_confs)
        if w and equal_func(str(w), str(gt)): top10_c += 1
```

## How to Run Individual Methods for Ablation

To test only one method, modify the `test_dataset()` function (lines 66-108). Comment out the methods you don't need:

```python
def test_dataset(path, name):
    # ... setup code ...

    # METHOD 1: Majority - comment out lines 89-91 to disable
    w = simple_majority_vote(answers)
    if w and equal_func(str(w), str(gt)): maj_c += 1

    # METHOD 2: Mean Weighted - comment out lines 93-97 to disable
    mean_confs = [calculate_mean_confidence(t) for t in valid_traces]
    if any(c > 0 for c in mean_confs):
        w = weighted_majority_vote(answers, mean_confs)
        if w and equal_func(str(w), str(gt)): mean_c += 1

    # METHOD 3: Top10 Tail - comment out lines 99-106 to disable
    top_traces = filter_top_confidence(valid_traces, 0.1)
    if top_traces:
        top_answers = [t['extracted_answer'] for t in top_traces]
        top_confs = [calculate_tail_confidence(t) for t in top_traces]
        if any(c > 0 for c in top_confs):
            w = weighted_majority_vote(top_answers, top_confs)
            if w and equal_func(str(w), str(gt)): top10_c += 1
```

## Key Functions (from DeepConf library)

```python
# Tail confidence: mean of last 2048 tokens
def calculate_tail_confidence(trace, tail_tokens=2048):
    confs = trace['confs']
    tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    return np.mean(tail_confs)

# Filter to top 10% by tail confidence
def filter_top_confidence(traces, top_percent=0.1):
    confidences = [calculate_tail_confidence(t) for t in traces]
    threshold = np.percentile(confidences, (1 - top_percent) * 100)  # 90th percentile
    return [t for t, c in zip(traces, confidences) if c >= threshold]

# Weighted majority vote
def weighted_majority_vote(answers, weights):
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        answer_weights[str(answer)] = answer_weights.get(str(answer), 0.0) + float(weight)
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])
```

## Scripts

### 1. Test GPT-OSS, GEMMA, QWQ models

**Script:** `/mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_other_models.py`

**Run:**
```bash
/eph/nvme0/env/deepconf/bin/python /mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_other_models.py
```

### 2. Test DeepSeek model

**Script:** `/mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_deepseek_three_methods.py`

**Run:**
```bash
/eph/nvme0/env/deepconf/bin/python /mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_deepseek_three_methods.py
```

## Dataset Paths

### DeepSeek-R1-8B
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2024_run_512_results_deepseek8b`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2025_run_512_results_newpara`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/bruno_2025_run_512_results_deepseek8b`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/hmmt_feb_2025_run_512_results`

### GPT-OSS-20B
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason`

### GEMMA-3-27B
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512`

### QWQ-32B
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512`

## Output

Results are printed in table format:

```
======================================================================
GPT-OSS-20B Results
======================================================================
Dataset         Majority        Mean Weighted   Top10 Tail
----------------------------------------------------------------------
AIME 2024       28/30 (93.3%)   28/30 (93.3%)   28/30 (93.3%)
AIME 2025       27/30 (90.0%)   27/30 (90.0%)   27/30 (90.0%)
BRUNO 2025      24/30 (80.0%)   25/30 (83.3%)   25/30 (83.3%)
HMMT 2025       20/30 (66.7%)   21/30 (70.0%)   22/30 (73.3%)
----------------------------------------------------------------------
TOTAL           99/120 (82.5%)  101/120 (84.2%) 102/120 (85.0%)
======================================================================
```

## Reference Implementation

The methods are implemented identically to the DeepConf library:
`/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/deepconf/deepconf/utils.py`

## Results File

Full results saved at:
`/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/analysis/paper_results_deepconfscore.md`
