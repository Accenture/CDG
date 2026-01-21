# DeepConf Voting Methods Results

## Methods Tested
- **Majority**: Simple count-based majority voting (`Counter(answers).most_common(1)[0][0]`)
- **Mean Weighted**: Weighted majority vote using mean confidence as weights (sum of mean_conf per answer)
- **Top10 Tail Filtered**: Filter to top 10% traces by tail confidence (last 2048 tokens), then weighted majority vote

## Scripts Used
- DeepSeek: `/mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_deepseek_three_methods.py`
- GPT-OSS, GEMMA, QWQ: `/mnt/batch/tasks/shared/LS_root/mounts/clusters/yuexp/code/Users/yu.bu.wang/sampling_credit/analysis/expbig/test_other_models.py`

Implementation matches `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/deepconf/deepconf/utils.py` exactly.

---

## DeepSeek-R1-8B Results

| Dataset | Majority | Mean Weighted | Top10 Tail Filtered |
|---------|----------|---------------|---------------------|
| AIME 2024 | 27/30 (90.0%) | 27/30 (90.0%) | 28/30 (93.3%) |
| AIME 2025 | 25/30 (83.3%) | 25/30 (83.3%) | 25/30 (83.3%) |
| BRUNO 2025 | 28/30 (93.3%) | 28/30 (93.3%) | 28/30 (93.3%) |
| HMMT 2025 | 21/30 (70.0%) | 21/30 (70.0%) | 25/30 (83.3%) |
| **TOTAL** | **101/120 (84.2%)** | **101/120 (84.2%)** | **106/120 (88.3%)** |

---

## GPT-OSS-20B Results

| Dataset | Majority | Mean Weighted | Top10 Tail Filtered |
|---------|----------|---------------|---------------------|
| AIME 2024 | 28/30 (93.3%) | 28/30 (93.3%) | 28/30 (93.3%) |
| AIME 2025 | 27/30 (90.0%) | 27/30 (90.0%) | 27/30 (90.0%) |
| BRUNO 2025 | 24/30 (80.0%) | 25/30 (83.3%) | 25/30 (83.3%) |
| HMMT 2025 | 20/30 (66.7%) | 21/30 (70.0%) | 22/30 (73.3%) |
| **TOTAL** | **99/120 (82.5%)** | **101/120 (84.2%)** | **102/120 (85.0%)** |

---

## GEMMA-3-27B Results

| Dataset | Majority | Mean Weighted | Top10 Tail Filtered |
|---------|----------|---------------|---------------------|
| AIME 2024 | 15/30 (50.0%) | 15/30 (50.0%) | 16/30 (53.3%) |
| AIME 2025 | 9/30 (30.0%) | 9/30 (30.0%) | 14/30 (46.7%) |
| BRUNO 2025 | 12/30 (40.0%) | 12/30 (40.0%) | 14/30 (46.7%) |
| HMMT 2025 | 6/30 (20.0%) | 6/30 (20.0%) | 4/30 (13.3%) |
| **TOTAL** | **42/120 (35.0%)** | **42/120 (35.0%)** | **48/120 (40.0%)** |

---

## QWQ-32B Results

| Dataset | Majority | Mean Weighted | Top10 Tail Filtered |
|---------|----------|---------------|---------------------|
| AIME 2024 | 26/30 (86.7%) | 27/30 (90.0%) | 26/30 (86.7%) |
| AIME 2025 | 23/30 (76.7%) | 23/30 (76.7%) | 23/30 (76.7%) |
| BRUNO 2025 | 24/30 (80.0%) | 24/30 (80.0%) | 26/30 (86.7%) |
| HMMT 2025 | 17/30 (56.7%) | 17/30 (56.7%) | 19/30 (63.3%) |
| **TOTAL** | **90/120 (75.0%)** | **91/120 (75.8%)** | **94/120 (78.3%)** |

---

## Summary Across All Models

| Model | Majority | Mean Weighted | Top10 Tail Filtered | Improvement |
|-------|----------|---------------|---------------------|-------------|
| DeepSeek-R1-8B | 84.2% | 84.2% | **88.3%** | +4.1% |
| GPT-OSS-20B | 82.5% | 84.2% | **85.0%** | +2.5% |
| GEMMA-3-27B | 35.0% | 35.0% | **40.0%** | +5.0% |
| QWQ-32B | 75.0% | 75.8% | **78.3%** | +3.3% |

---

## Dataset Paths

**DeepSeek-R1-8B:**
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2024_run_512_results_deepseek8b`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2025_run_512_results_newpara`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/bruno_2025_run_512_results_deepseek8b`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/hmmt_feb_2025_run_512_results`

**GPT-OSS-20B:**
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason`

**GEMMA-3-27B:**
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512`

**QWQ-32B:**
- AIME 2024: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512`
- AIME 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512`
- BRUNO 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512`
- HMMT 2025: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512`
