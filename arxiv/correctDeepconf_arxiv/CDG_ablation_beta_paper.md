# CDG Paper Results: Optimized Beta Values Per Model

## Overview

This document reports the results of running the **CDG (Count-Dampened Gradient)** voting method with optimized beta values for each model:
- **DeepSeek-R1-8B**: β=10
- **GPT-OSS-20B**: β=10
- **GEMMA-3-27B**: β=3
- **QWQ-32B**: β=3

### Method: CDG (Count-Dampened Gradient)

**Formula:**
```
score_answer = count^α × mean(mean_conf + β × gradient)
```

**Where:**
- `count`: number of traces for this answer
- `α`: dampening exponent (0.5 = square root)
- `β`: gradient weight (10 for DeepSeek/GPT-OSS, 3 for GEMMA/QWQ)
- `gradient`: mean(last P%) - mean(first P%) of confidence values
- `mean_conf`: mean confidence across all tokens in a trace
- `P`: position percentage (10%)

### Parameters Tested

| Model | α (alpha) | β (beta) | P (position %) |
|-------|-----------|----------|----------------|
| DeepSeek-R1-8B | 0.5 | **10** | 10% |
| GPT-OSS-20B | 0.5 | **10** | 10% |
| GEMMA-3-27B | 0.5 | **3** | 10% |
| QWQ-32B | 0.5 | **3** | 10% |

---

## Complete Results: All 16 Datasets

### DeepSeek-R1-8B Results (β=10)

| Dataset | Correct/Total | Accuracy | Wrong QIDs |
|---------|---------------|----------|------------|
| AIME 2024 (DeepSeek) | 28/30 | **93.3%** | [11, 29] |
| AIME 2025 (DeepSeek) | 28/30 | **93.3%** | [12, 13] |
| BRUNO 2025 (DeepSeek) | 28/30 | **93.3%** | [12, 22] |
| HMMT 2025 (DeepSeek) | 25/30 | **83.3%** | [9, 12, 16, 17, 18] |
| **TOTAL** | **109/120** | **90.8%** | |

---

### GPT-OSS-20B Results (β=10)

| Dataset | Correct/Total | Accuracy | Wrong QIDs |
|---------|---------------|----------|------------|
| AIME 2024 (GPT-OSS) | 28/30 | **93.3%** | [11, 29] |
| AIME 2025 (GPT-OSS) | 28/30 | **93.3%** | [13, 14] |
| BRUNO 2025 (GPT-OSS) | 25/30 | **83.3%** | [11, 12, 16, 22, 26] |
| HMMT 2025 (GPT-OSS) | 22/30 | **73.3%** | [2, 9, 16, 17, 18, 19, 24, 29] |
| **TOTAL** | **103/120** | **85.8%** | |

---

### GEMMA-3-27B Results (β=3)

| Dataset | Correct/Total | Accuracy | Wrong QIDs |
|---------|---------------|----------|------------|
| AIME 2024 (GEMMA) | 17/30 | **56.7%** | [8, 9, 10, 11, 12, 19, 21, 22, 23, 25, 26, 28, 29] |
| AIME 2025 (GEMMA) | 12/30 | **40.0%** | [1, 6, 9, 10, 11, 12, 13, 14, 17, 19, 21, 22, 23, 24, 25, 27, 28, 29] |
| BRUNO 2025 (GEMMA) | 14/30 | **46.7%** | [2, 4, 9, 10, 11, 12, 14, 16, 20, 21, 23, 24, 26, 27, 28, 29] |
| HMMT 2025 (GEMMA) | 7/30 | **23.3%** | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29] |
| **TOTAL** | **50/120** | **41.7%** | |

---

### QWQ-32B Results (β=3)

| Dataset | Correct/Total | Accuracy | Wrong QIDs |
|---------|---------------|----------|------------|
| AIME 2024 (QWQ) | 27/30 | **90.0%** | [11, 23, 29] |
| AIME 2025 (QWQ) | 23/30 | **76.7%** | [1, 9, 12, 13, 14, 27, 29] |
| BRUNO 2025 (QWQ) | 27/30 | **90.0%** | [22, 28, 29] |
| HMMT 2025 (QWQ) | 19/30 | **63.3%** | [9, 12, 13, 14, 16, 17, 18, 19, 23, 24, 29] |
| **TOTAL** | **96/120** | **80.0%** | |

---

## Summary Statistics

### Grand Total Across All Models and Datasets
- **Total Correct**: 358/480
- **Overall Accuracy**: **74.6%**

### Per-Model Performance

| Model | Total Correct | Accuracy | Beta | Rank |
|-------|---------------|----------|------|------|
| DeepSeek-R1-8B | 109/120 | **90.8%** | β=10 | 🥇 1st |
| GPT-OSS-20B | 103/120 | **85.8%** | β=10 | 🥈 2nd |
| QWQ-32B | 96/120 | **80.0%** | β=3 | 🥉 3rd |
| GEMMA-3-27B | 50/120 | **41.7%** | β=3 | 4th |

### Per-Dataset Performance (Across All Models)

| Dataset Category | Avg Accuracy | Best Model | Worst Model |
|------------------|--------------|------------|-------------|
| AIME 2024 | 83.3% | DeepSeek/GPT-OSS (93.3%) | GEMMA (56.7%) |
| AIME 2025 | 75.8% | DeepSeek/GPT-OSS (93.3%) | GEMMA (40.0%) |
| BRUNO 2025 | 78.3% | DeepSeek (93.3%) | GEMMA (46.7%) |
| HMMT 2025 | 60.8% | DeepSeek (83.3%) | GEMMA (23.3%) |

### Individual Dataset Rankings (Best to Worst)

1. 🥇 AIME 2024 (DeepSeek, β=10): **93.3%**
2. 🥇 AIME 2025 (DeepSeek, β=10): **93.3%**
3. 🥇 BRUNO 2025 (DeepSeek, β=10): **93.3%**
4. 🥇 AIME 2024 (GPT-OSS, β=10): **93.3%**
5. 🥇 AIME 2025 (GPT-OSS, β=10): **93.3%**
6. AIME 2024 (QWQ, β=3): **90.0%**
7. BRUNO 2025 (QWQ, β=3): **90.0%**
8. HMMT 2025 (DeepSeek, β=10): **83.3%**
9. BRUNO 2025 (GPT-OSS, β=10): **83.3%**
10. AIME 2025 (QWQ, β=3): **76.7%**
11. HMMT 2025 (GPT-OSS, β=10): **73.3%**
12. HMMT 2025 (QWQ, β=3): **63.3%**
13. AIME 2024 (GEMMA, β=3): **56.7%**
14. BRUNO 2025 (GEMMA, β=3): **46.7%**
15. AIME 2025 (GEMMA, β=3): **40.0%**
16. 🔴 HMMT 2025 (GEMMA, β=3): **23.3%**

---

## 16 Individual Results Summary

| # | Dataset | Model | β | Accuracy | Correct |
|---|---------|-------|---|----------|---------|
| 1 | AIME 2024 | DeepSeek | 10 | 93.3% | 28/30 |
| 2 | AIME 2025 | DeepSeek | 10 | 93.3% | 28/30 |
| 3 | BRUNO 2025 | DeepSeek | 10 | 93.3% | 28/30 |
| 4 | HMMT 2025 | DeepSeek | 10 | 83.3% | 25/30 |
| 5 | AIME 2024 | GPT-OSS | 10 | 93.3% | 28/30 |
| 6 | AIME 2025 | GPT-OSS | 10 | 93.3% | 28/30 |
| 7 | BRUNO 2025 | GPT-OSS | 10 | 83.3% | 25/30 |
| 8 | HMMT 2025 | GPT-OSS | 10 | 73.3% | 22/30 |
| 9 | AIME 2024 | GEMMA | 3 | 56.7% | 17/30 |
| 10 | AIME 2025 | GEMMA | 3 | 40.0% | 12/30 |
| 11 | BRUNO 2025 | GEMMA | 3 | 46.7% | 14/30 |
| 12 | HMMT 2025 | GEMMA | 3 | 23.3% | 7/30 |
| 13 | AIME 2024 | QWQ | 3 | 90.0% | 27/30 |
| 14 | AIME 2025 | QWQ | 3 | 76.7% | 23/30 |
| 15 | BRUNO 2025 | QWQ | 3 | 90.0% | 27/30 |
| 16 | HMMT 2025 | QWQ | 3 | 63.3% | 19/30 |

---

## Key Findings

### 1. Beta Value Impact
- **Higher β (β=10)** works better for stronger models (DeepSeek, GPT-OSS)
  - DeepSeek: 90.8% accuracy
  - GPT-OSS: 85.8% accuracy
- **Lower β (β=3)** is more suitable for weaker models (GEMMA, QWQ)
  - QWQ: 80.0% accuracy
  - GEMMA: 41.7% accuracy

### 2. Model Performance Patterns
- **DeepSeek-R1-8B** achieved the highest accuracy (90.8%) with β=10
- **GPT-OSS-20B** performed very well (85.8%) with β=10
- **QWQ-32B** showed solid performance (80.0%) with β=3
- **GEMMA-3-27B** struggled significantly (41.7%) even with optimized β=3

### 3. Dataset Difficulty
- **HMMT 2025** remains the hardest dataset across all models (avg 60.8%)
- **AIME 2024** is generally the easiest (avg 83.3%)
- **DeepSeek** achieved 93.3% on AIME 2024, AIME 2025, and BRUNO 2025

### 4. Common Hard Questions
- **QID 29** appears frequently in wrong lists across multiple models
- **QID 11, 12, 13** are consistently challenging
- **HMMT 2025** has the most difficult questions overall

### 5. Comparison to β=0 Baseline
Comparing to the β=0 (no gradient) results:

| Model | β=0 Accuracy | Optimized β | Improvement |
|-------|--------------|-------------|-------------|
| DeepSeek-R1-8B | 84.2% | 90.8% (β=10) | **+6.6%** |
| GPT-OSS-20B | 84.2% | 85.8% (β=10) | **+1.6%** |
| GEMMA-3-27B | 35.8% | 41.7% (β=3) | **+5.9%** |
| QWQ-32B | 75.8% | 80.0% (β=3) | **+4.2%** |

**The gradient component (β > 0) consistently improves performance across all models!**

---

## Comparison Table: β=0 vs Optimized β

### DeepSeek-R1-8B

| Dataset | β=0 | β=10 | Improvement |
|---------|-----|------|-------------|
| AIME 2024 | 27/30 (90.0%) | 28/30 (93.3%) | +3.3% |
| AIME 2025 | 25/30 (83.3%) | 28/30 (93.3%) | +10.0% |
| BRUNO 2025 | 28/30 (93.3%) | 28/30 (93.3%) | 0% |
| HMMT 2025 | 21/30 (70.0%) | 25/30 (83.3%) | +13.3% |
| **TOTAL** | **101/120 (84.2%)** | **109/120 (90.8%)** | **+6.6%** |

### GPT-OSS-20B

| Dataset | β=0 | β=10 | Improvement |
|---------|-----|------|-------------|
| AIME 2024 | 28/30 (93.3%) | 28/30 (93.3%) | 0% |
| AIME 2025 | 27/30 (90.0%) | 28/30 (93.3%) | +3.3% |
| BRUNO 2025 | 25/30 (83.3%) | 25/30 (83.3%) | 0% |
| HMMT 2025 | 21/30 (70.0%) | 22/30 (73.3%) | +3.3% |
| **TOTAL** | **101/120 (84.2%)** | **103/120 (85.8%)** | **+1.6%** |

### GEMMA-3-27B

| Dataset | β=0 | β=3 | Improvement |
|---------|-----|-----|-------------|
| AIME 2024 | 15/30 (50.0%) | 17/30 (56.7%) | +6.7% |
| AIME 2025 | 9/30 (30.0%) | 12/30 (40.0%) | +10.0% |
| BRUNO 2025 | 13/30 (43.3%) | 14/30 (46.7%) | +3.4% |
| HMMT 2025 | 6/30 (20.0%) | 7/30 (23.3%) | +3.3% |
| **TOTAL** | **43/120 (35.8%)** | **50/120 (41.7%)** | **+5.9%** |

### QWQ-32B

| Dataset | β=0 | β=3 | Improvement |
|---------|-----|-----|-------------|
| AIME 2024 | 27/30 (90.0%) | 27/30 (90.0%) | 0% |
| AIME 2025 | 23/30 (76.7%) | 23/30 (76.7%) | 0% |
| BRUNO 2025 | 24/30 (80.0%) | 27/30 (90.0%) | +10.0% |
| HMMT 2025 | 17/30 (56.7%) | 19/30 (63.3%) | +6.6% |
| **TOTAL** | **91/120 (75.8%)** | **96/120 (80.0%)** | **+4.2%** |

---

## Reproducibility

### Script Location
```
/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/CDG_ablation_beta_paper.py
```

### Running the Script
```bash
/mnt/yuenvs/deepconf/bin/python /home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/CDG_ablation_beta_paper.py
```

### Dataset Paths
All dataset paths are hardcoded in the script and come from:
```
/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/
```

### Dependencies
- Python environment: `/mnt/yuenvs/deepconf`
- Required package: `dynasor` (for `math_equal` function)
- Standard packages: `pickle`, `numpy`, `pathlib`, `collections`

---

## Conclusions

1. **The gradient component is valuable**: All models show improvement when using optimized β > 0 compared to β=0
2. **Different models need different β values**: Stronger models (DeepSeek, GPT-OSS) benefit from higher β=10, while weaker models work better with β=3
3. **DeepSeek-R1-8B is the strongest**: Achieving 90.8% overall accuracy with consistent 93.3% on three datasets
4. **HMMT 2025 is consistently hard**: All models struggle with this dataset
5. **Gradient helps most on hard datasets**: The biggest improvements from adding gradient are seen on HMMT and challenging AIME 2025 questions

---

*Generated on 2026-01-18*
