# Evaluation Scripts

This directory contains scripts for evaluating sampling credit methods on math reasoning benchmarks.

## Overview

| Script | Purpose |
|--------|---------|
| `eval_voting.py` | Main evaluation - compare voting methods across models |
| `eval_subsample.py` | Scaling analysis - accuracy vs trace count |
| `eval_hyperparam.py` | Hyperparameter tuning for CDG method |
| `eval_visualization.py` | Confidence position analysis |
| `eval_methods.py` | Core evaluation functions (library) |
| `eval_cache.py` | Disk-based caching for evaluation results |
| `run_all_experiments.sh` | Master script to run all experiments |

## Caching

All evaluation scripts support disk-based caching to speed up repeated runs:

```bash
# First run - computes everything, saves to cache
python scripts/eval/eval_voting.py --all --all-methods

# Second run - loads from cache instantly
python scripts/eval/eval_voting.py --all --all-methods

# Force recompute (disable cache)
python scripts/eval/eval_voting.py --all --all-methods --no-cache

# Clear cache and recompute
python scripts/eval/eval_voting.py --all --all-methods --clear-cache

# Check cache statistics
python scripts/eval/eval_voting.py --cache-stats
```

Cache is stored in `{results_dir}/.eval_cache/` and automatically invalidates when source pickle files are modified.

---

## Experiments

### Exp 1: Main Evaluation

**Goal**: Compare all voting methods across all models and datasets.

**Methods**:
- `oracle` - Upper bound (correct if ANY trace has right answer)
- `majority` - Simple majority vote by count
- `deepconf_mean` - Sum of mean confidence per answer
- `deepconf_tail` - Sum of tail token confidence
- `deepconf_bottom` - Sum of sliding window min confidence
- `gradient` - Sum of (mean_conf + β × gradient)
- `cdg` - Count-Dampened Gradient: count^α × mean(conf + β × gradient)

**Run**:
```bash
# All models, all methods
python scripts/eval/eval_voting.py --all --all-methods

# Single model
python scripts/eval/eval_voting.py --pattern "*qwen32b*" --all-methods

# With tuned parameters
python scripts/eval/eval_voting.py --all --all-methods --alpha 0.5 --beta 10 --position_pct 20
```

**Output**: Table showing accuracy (correct/total) for each model-dataset-method combination.

---

### Exp 2: Subsample Scaling Analysis

**Goal**: Analyze how accuracy scales with number of traces (8 → 512).

**Figure**:
- X-axis: Trace count (8, 16, 32, 64, 128, 256, 512)
- Y-axis: Accuracy (%)
- Color/Shape: Different methods

**Run**:
```bash
python scripts/eval/eval_subsample.py --output scaling.png --no-show
```

**Prerequisites**: Subset runs must exist in `subset_trace/` directory with naming pattern `{model}_{dataset}_512_subset{N}v{version}`.

---

### Exp 3: Hyperparameter Tuning

**Goal**: Find optimal α and β parameters for CDG method.

**Parameters**:
- `α` (alpha): Count dampening exponent (0.5 or 1.0)
- `β` (beta): Gradient weight (10 to 50, step 5)
- `position_pct`: Percentile for gradient calculation (10 or 20)

**Figure**:
- X-axis: β values [10, 15, 20, 25, 30, 35, 40, 45, 50]
- Y-axis: Accuracy (%)
- Color: α values (blue=0.5, red=1.0)

**Run**:
```bash
# Sweep position_pct first
python scripts/eval/eval_hyperparam.py --sweep position_pct --trace_count 256

# Sweep beta with α=0.5, position_pct=10
python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256 \
  --alphas 0.5 --position_pct 10 \
  --figure beta_a0.5_pct10.png --no-show

# Sweep beta with α=0.5, position_pct=20
python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256 \
  --alphas 0.5 --position_pct 20 \
  --figure beta_a0.5_pct20.png --no-show

# Sweep beta with α=1.0, position_pct=10
python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256 \
  --alphas 1.0 --position_pct 10 \
  --figure beta_a1.0_pct10.png --no-show

# Sweep beta with α=1.0, position_pct=20
python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256 \
  --alphas 1.0 --position_pct 20 \
  --figure beta_a1.0_pct20.png --no-show

# Combined figure (both alphas)
python scripts/eval/eval_hyperparam.py --sweep beta --trace_count 256 \
  --alphas 0.5,1.0 --position_pct 20 \
  --figure beta_combined.png --no-show
```

**Output**: JSON results + PNG figures saved to `results/eval_outputs/`.

---

## Recommended Workflow

Run experiments in this order:

```
Exp 3 (Hyperparameter Tuning)
         ↓
    Find best α, β, position_pct per model
         ↓
Exp 2 (Scaling Analysis)
         ↓
Exp 1 (Main Evaluation with tuned params)
```

### Using the Master Script

```bash
# Make executable
chmod +x scripts/eval/run_all_experiments.sh

# Dry run (show commands without executing)
./scripts/eval/run_all_experiments.sh --dry-run

# Run only hyperparameter tuning
./scripts/eval/run_all_experiments.sh --exp 3

# Run only scaling analysis
./scripts/eval/run_all_experiments.sh --exp 2

# Run only main evaluation
./scripts/eval/run_all_experiments.sh --exp 1

# Run all experiments
./scripts/eval/run_all_experiments.sh
```

---

## Directory Structure

```
results/
├── eval_outputs/                    # Experiment outputs
│   ├── hyperparam_*.json           # Hyperparameter sweep results
│   ├── hyperparam_*.png            # Beta vs accuracy figures
│   ├── subsample_scaling_*.png     # Scaling analysis figure
│   └── main_eval_*.txt             # Main evaluation results
│
└── {model}_{dataset}_{budget}/     # Raw inference results
    ├── q_*.pkl                     # Per-question results
    └── subset_trace/               # Subsampled runs
        └── {model}_{dataset}_512_subset{N}v{V}/
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.5 | CDG count dampening exponent |
| `--beta` | 10 | Gradient weight |
| `--position_pct` | 20 | Percentile for gradient calculation |
| `--tail_window` | 2048 | Window size for tail confidence |
| `--bottom_window` | 2048 | Window size for bottom confidence |

---

## Models & Datasets

**Models**: qwen32b, deepseek8b, gemma3_27b, gptoss20b, gptoss20b_nohighreason

**Datasets**: aime2024, aime2025, bruno2025, hmmt2025

**Budget**: 512 traces per question
