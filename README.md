# Sampling Credit: CDG Voting for Math Reasoning

This repository contains code for evaluating Count-Dampened Gradient (CDG) voting methods on math reasoning benchmarks.

---

## Setup

### Clone with Submodules

```bash
git clone --recurse-submodules <repo-url>
cd sampling_credit
```

### If Already Cloned (pull submodules)

```bash
git submodule update --init --recursive
```

### Update Submodules to Latest

```bash
git submodule update --remote --merge
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
│   │   └── histogram/      # util_histogram_cache.py cache
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
| `util_histogram_cache.py` | Extract histogram metrics for correct/wrong distributions |

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

CDG (Count-Dampened Gradient) combines vote count with confidence gradient:

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
- `results/cache/histogram/*.json` - Per model-dataset histogram metrics

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
