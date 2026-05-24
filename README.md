# Sampling Credit: CDG Voting for Math Reasoning

This repository contains code for our ICML 2026 paper *Inference Time Optimization with Confidence Dynamics*. It implements Confidence Dynamic Gain (CDG) based voting — a training-free inference-time answer-selection method that exploits how model confidence evolves along reasoning trajectories — and evaluates it on competition-level math reasoning benchmarks (AIME 2024, AIME 2025, HMMT 2025, BRUMO 2025) across four open-source reasoning LLMs (DeepSeek-R1-8B, gpt-oss-20B, Gemma-3-27B, QwQ-32B).

---

## Setup

### From Zip File

```bash
unzip submission.zip -d sampling_credit
cd sampling_credit
git init
git submodule update --init
```

### From Git Repository

```bash
git clone --recurse-submodules <repo-url>
cd sampling_credit
```

### Submodules

| Submodule | Source |
|-----------|--------|
| `deepconf` | [facebookresearch/deepconf](https://github.com/facebookresearch/deepconf) |
| `gpt-oss` | [openai/gpt-oss](https://github.com/openai/gpt-oss) |

---

## Project Structure

```
sampling_credit/
├── scripts/
│   ├── prepare_data/       # Dataset preparation scripts
│   ├── inference/          # Model inference scripts
│   └── eval/               # Evaluation and figure generation
├── results/
│   ├── cache/              # Cached experiment results
│   │   ├── voting_methods/ # exp_voting_methods.py cache
│   │   ├── cdg_sweep/      # exp_cdg_sweep.py cache
│   │   ├── pass_at_1/      # exp_pass_at_1.py cache
│   │   └── histogram/      # util_histogram_cache.py cache
│   │       └── significance_test_cache/  # significance_test_confidence_change.py cache
│   └── figures/            # Generated figures
│       └── appendix/       # Appendix figures
├── deepconf/               # [submodule] DeepConf baseline
├── gpt-oss/                # [submodule] GPT-OSS model support
└── README.md
```

---

## Quick Start

### 1. Data Preparation

```bash
cd scripts/prepare_data
./prepare_all_datasets.sh
```

This prepares all datasets (AIME 2024, AIME 2025, BRUMO 2025, HMMT 2025) in JSONL format.

### 2. Run Inference

```bash
cd scripts/inference
./run.sh --model <model> --dataset <dataset>
```

See `scripts/inference/README.md` for detailed inference instructions.

### 3. Run Evaluation

```bash
cd scripts/eval
./run.sh --task all
```

Available tasks:
- `cache` - Generate all caches (voting, histogram, sweep)
- `figures` - Generate all figures (requires caches)
- `sweep` - Run CDG hyperparameter sweep only
- `position` - Run position ablation experiment
- `all` - Run everything (cache + figures)

---

## Evaluation Scripts

### Experiment Scripts (generate caches)

| Script | Description |
|--------|-------------|
| `exp_voting_methods.py` | Compare voting methods (Majority, Mean, Top10 Tail, CDG) |
| `exp_cdg_sweep.py` | CDG hyperparameter sweep (alpha, beta, position_pct) |
| `exp_pass_at_1.py` | Pass@1 baseline accuracy (single randomly sampled trace) |
| `util_histogram_cache.py` | Extract histogram metrics for correct/wrong distributions |

### Statistical Analysis Scripts

| Script | Description |
|--------|-------------|
| `significance_test_confidence_change.py` | Mann-Whitney U / Welch's t-test for confidence change (correct vs wrong traces) |

### Figure Scripts (generate figures from caches)

| Script | Description |
|--------|-------------|
| `figure_scaling.py` | Main scaling figure (accuracy vs trace count) |
| `figure_scaling_appendix.py` | Per-model scaling figures for appendix |
| `figure_histogram.py` | Confidence distribution histograms |
| `figure_confidence_curve.py` | Confidence calibration curves |
| `figure_position_appendix.py` | Position_pct ablation figure |

---

## Datasets

| Dataset | Description |
|---------|-------------|
| `aime2024` | AIME 2024 (30 questions) |
| `aime2025` | AIME 2025 (30 questions) |
| `brumo2025` | BRUMO 2025 |
| `hmmt2025` | HMMT February 2025 (30 questions) |

---

## Models

| Model Key | Model Name |
|-----------|------------|
| `deepseek8b` | DeepSeek-R1-8B |
| `gemma3_27b` | Gemma-3-27B |
| `qwq32b` | QWQ-32B |
| `gptoss20b` | gpt-oss-20B |

---

## Configuration

All paths and parameters are centralized in config files:

- `scripts/eval/config.py` - Evaluation configuration
- `scripts/inference/config.py` - Inference configuration
- `scripts/inference/config.sh` - Shell configuration for inference

---

## CDG Voting Method

CDG (Confidence Dynamic Gain) augments majority voting with two signals: (1) the mean per-trace confidence, and (2) the *dynamic gain* — i.e., how confidence evolves from the head to the tail of each reasoning trace. The final score for each candidate answer is:

```
score = count^alpha * mean(mean_conf + beta * gradient)
```

Where:
- `count` = number of traces voting for this answer
- `alpha` = count dampening factor (default: 0.5)
- `beta` = gradient weight (model-specific)
- `gradient` = mean(last P%) - mean(first P%) of confidence trajectory

---

## Output

### Caches
- `results/cache/voting_methods/cache.json` - Voting method comparison results
- `results/cache/cdg_sweep/sweep_cache.json` - Hyperparameter sweep results
- `results/cache/pass_at_1/cache.json` - Pass@1 baseline results
- `results/cache/histogram/*.json` - Per model-dataset histogram metrics
- `results/cache/histogram/significance_test_cache/*.json` - Significance test raw data

### Figures
- `results/figures/*.pdf` - Main paper figures
- `results/figures/appendix/*.pdf` - Appendix figures

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- dynasor (`pip install git+https://github.com/hao-ai-lab/Dynasor.git`)

For inference:
- vLLM
- PyTorch
- Transformers

---

## Citation

If you find this work useful, please cite our ICML 2026 paper:

```bibtex
@inproceedings{wang2026cdg,
  title     = {Inference Time Optimization with Confidence Dynamics},
  author    = {Wang, Yu and Liu, Minghao and Wang, Jiayun and Huang, Jinrui and Shah, Ankit and Wei, Wei},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
