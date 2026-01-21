# Voting Methods Evaluation

This folder contains scripts to evaluate and compare different voting methods for LLM reasoning trace aggregation.

## Overview

When multiple reasoning traces are generated for a single problem, we need a method to aggregate them into a final answer. This evaluation framework compares several approaches:

| Method | Description | Formula |
|--------|-------------|---------|
| **Majority** | Simple count-based voting | `argmax(count(answer))` |
| **DeepConf-Mean** | Weighted by mean confidence | `argmax(sum(mean_conf))` |
| **DeepConf-Tail** | Weighted by tail confidence (last N tokens) | `argmax(sum(tail_conf))` |
| **DeepConf-BottomWin** | Weighted by bottom percentile of sliding window | `argmax(sum(bottom_window_conf))` |
| **Gradient** | Weighted by confidence + gradient | `argmax(sum(mean_conf + beta * gradient))` |
| **CDG** | Count-Dampened Gradient | `argmax(count^alpha * mean(mean_conf + beta * gradient))` |

### Count-Dampened Gradient (CDG)

CDG is our proposed method that combines:
- **Confidence gradient**: `gradient = mean(conf_last_P%) - mean(conf_first_P%)`
- **Count dampening**: Prevents low-count answers from dominating due to high gradient scores

The key insight is that correct traces tend to show increasing confidence (positive gradient) while incorrect traces show decreasing or flat confidence patterns.

## Files

- `evaluate_voting_methods.py` - Main evaluation script for comparing voting methods
- `run_evaluation.sh` - Shell wrapper for voting methods evaluation
- `draw_confidence_position.py` - Script to visualize confidence vs position patterns
- `rundrawfigures.sh` - Shell wrapper for drawing confidence figures

## Requirements

- Python environment with `numpy` and the `dynasor` module
- Pickle files containing trace data with the following structure:
  ```python
  {
      'qid': int,
      'ground_truth': str,
      'all_traces': [
          {
              'extracted_answer': str,
              'confs': List[float],  # Token-level confidence values
              ...
          },
          ...
      ]
  }
  ```

## Usage

### Using the Shell Script

```bash
# Run on all default datasets
./run_evaluation.sh

# Custom alpha and beta parameters
./run_evaluation.sh --alpha 0.3 --beta 15

# Show detailed results (wrong QIDs per method)
./run_evaluation.sh --detailed

# Run on custom data directory
./run_evaluation.sh --custom /path/to/your/data

# Multiple custom directories
./run_evaluation.sh --custom /path/to/data1 --custom /path/to/data2

# Combine options
./run_evaluation.sh --alpha 0.5 --beta 10 --position_pct 25 --detailed
```

### Using Python Directly

```bash
/eph/nvme0/env/deepconf/bin/python evaluate_voting_methods.py \
    --data_dirs /path/to/data1 /path/to/data2 \
    --alpha 0.5 \
    --beta 10 \
    --position_pct 20 \
    --detailed
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dirs` | (required) | One or more directories containing pickle files |
| `--alpha` | 0.5 | CDG count dampening exponent (0=ignore count, 1=linear count) |
| `--beta` | 10 | Gradient weight in score computation |
| `--position_pct` | 20 | Percentile for gradient computation (first/last P% of trace) |
| `--tail_window` | 2048 | Window size for tail confidence (tokens) |
| `--bottom_window` | 2048 | Window size for bottom window method (tokens) |
| `--bottom_pct` | 0.1 | Bottom percentile for bottom window method |
| `--detailed` | False | Print detailed results showing wrong QIDs per method |

## Output

### Summary Table

```
--------------------------------------------------------------------------------------------------------------------------
Dataset                  |   Majority   | DeepConf-Mean | DeepConf-Tail | DeepConf-BottomWin |   Gradient   |     CDG
--------------------------------------------------------------------------------------------------------------------------
qwq32b_aime2025_512      | 22/27 (81.5%) | 22/27 (81.5%) | 22/27 (81.5%) | 22/27 (81.5%) | 21/27 (77.8%) | 21/27 (77.8%)
aime2025run3qdeepseek    | 25/30 (83.3%) | 25/30 (83.3%) | 25/30 (83.3%) | 25/30 (83.3%) | 26/30 (86.7%) | 27/30 (90.0%)
aime2025run3qgptoss20b   | 27/30 (90.0%) | 27/30 (90.0%) | 27/30 (90.0%) | 27/30 (90.0%) | 28/30 (93.3%) | 28/30 (93.3%)
--------------------------------------------------------------------------------------------------------------------------
```

### Detailed Output (with `--detailed`)

```
DATASET: aime2025run3qdeepseek

Majority:
  Accuracy: 25/30 (83.3%)
  Wrong QIDs: [12, 13, 14, 27, 29]

CDG:
  Accuracy: 27/30 (90.0%)
  Wrong QIDs: [13, 14, 27]
```

## When to Use Each Method

Based on empirical results:

| Model | Recommended Method | Notes |
|-------|-------------------|-------|
| **DeepSeek** | CDG | +6.7% over majority (83.3% -> 90.0%) |
| **GPT-OSS** | CDG/Gradient | +3.3% over majority (90.0% -> 93.3%) |
| **QwQ-32B** | Majority | Gradient methods show slight degradation |
| **Qwen3-32B** | Majority | Gradient methods hurt performance significantly |

**Key insight**: CDG works best on models trained with RL + verifiable rewards, which exhibit divergent confidence gradients between correct and incorrect traces.

## Confidence vs Position Visualization

The `draw_confidence_position.py` script analyzes how confidence changes across the trace for correct vs wrong answers.

### Usage

```bash
# Using the shell script
./rundrawfigures.sh --data_dir /path/to/data

# With custom output directory
./rundrawfigures.sh --data_dir /path/to/data --output_dir ./figures

# With custom number of bins
./rundrawfigures.sh --data_dir /path/to/data --num_bins 20

# Using Python directly
/eph/nvme0/env/deepconf/bin/python draw_confidence_position.py \
    --data_dir /path/to/data \
    --output_dir ./figures \
    --num_bins 10
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | (required) | Directory containing pickle files with trace data |
| `--output_dir` | data_dir | Directory to save output figure |
| `--num_bins` | 10 | Number of bins to divide traces into |

### Output

The script generates:
1. A PNG figure with two plots:
   - Left: Confidence vs Position for correct and wrong traces (with error bars)
   - Right: Confidence difference (Correct - Wrong) by position
2. Console output with:
   - Mean confidence table by position
   - Key observations (overall levels, trends, max difference position)
   - Interpretation of the patterns

### Example Output

```
CONFIDENCE VS POSITION ANALYSIS - qwq32b_aime2025_512
==========================================================================================
Position     Correct Traces            Wrong Traces              Difference
1            14.70 +/- 2.65            12.37 +/- 1.56            +2.32 [+]
...
10           17.50 +/- 4.62            12.28 +/- 2.31            +5.22 [+]

KEY OBSERVATIONS:
1. Overall confidence levels:
   Correct traces: 14.20 (averaged across all positions)
   Wrong traces:   11.70
   Difference:     +2.50

2. Confidence trends from beginning to end:
   Correct traces: 14.39 -> 15.60 (change: +1.20)
   Wrong traces:   12.36 -> 11.40 (change: -0.96)
```

## Confidence Definition

Confidence is computed as:
```
C_i = -1/K * sum_{j in top-K} log(p_i(j))
```

This is the negative mean log probability over top-K tokens at position i, computed by the deepconf library.
