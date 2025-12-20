# HMMT Inference and Analysis Instructions

This guide explains how to run HMMT inference and analyze results using DeepConf voting methods.

---

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Overview](#overview)
3. [Running Inference](#running-inference)
4. [Analyzing Results](#analyzing-results)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## Data Preparation (Please run these 4 datasets and inference in priority order : aime2025 -> hmmt2025 -> aime2024 -> bruno2025 )

### Overview

Before running inference, you need to prepare your dataset in JSONL format. The `prepare_aime2025.py` script provides an example of how to convert datasets from HuggingFace to the required format.

### Required JSONL Format

Each line in the JSONL file should be a JSON object with two fields:
```json
{"question": "What is 2+2?", "answer": "4"}
{"question": "Find the value of x if 3x = 9", "answer": "3"}
```

**Fields**:
- `question`: The problem statement (string)
- `answer`: The ground truth answer (string)

### Example: Using prepare_aime2025.py

**Script location**: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/prepare_aime2025.py`

#### Step 1: Review the Script

```python
"""
Prepare AIME 2025 dataset in JSONL format for deepconf experiments.
"""
import json
from datasets import load_dataset

def main():
    # Load dataset from HuggingFace
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset("MathArena/aime_2025", split="train")

    # Convert to JSONL
    output_file = "aime_2025.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            entry = {
                "question": example["problem"],
                "answer": str(example["answer"])
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Converted {len(dataset)} examples to {output_file}")
```

#### Step 2: Run the Data Preparation Script

```bash
cd /home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit

# Run the preparation script
python scripts/prepare_aime2025.py
```

**Output**:
```
Loading AIME 2025 dataset...
Converted 30 examples to aime_2025.jsonl

First example:
Question: In rectangle ABCD, AB = 20 and BC = 10...
Answer: 100
```

#### Step 3: Verify the Output

```bash
# Check the first few lines
head -3 aime_2025.jsonl

# Count total questions
wc -l aime_2025.jsonl
```

#### Step 4: Move to Storage Location

```bash
# Create directory if needed
mkdir -p /eph/nvme0/aime_2025/

# Move the JSONL file
mv aime_2025.jsonl /eph/nvme0/aime_2025/
```

### Preparing Your Own Dataset

To prepare a custom dataset, modify the script to match your data source:

**Example for HMMT**:
```python
import json

def prepare_hmmt():
    # Your custom data loading logic
    problems = [
        {"problem": "Question 1 text", "solution": "42"},
        {"problem": "Question 2 text", "solution": "\\frac{1}{2}"},
        # ... more problems
    ]

    output_file = "hmmt_feb_2025.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in problems:
            entry = {
                "question": item["problem"],
                "answer": item["solution"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Prepared {len(problems)} questions")

if __name__ == "__main__":
    prepare_hmmt()
```

### Important Notes

1. **Answer Format**: Keep answers as strings to preserve LaTeX formatting (e.g., `"\frac{1}{2}"`, `"3.14"`)
2. **Encoding**: Always use UTF-8 encoding to handle mathematical symbols
3. **Validation**: Check first few examples after preparation to ensure correct format
4. **Storage**: Place datasets in `/eph/nvme0/` for faster I/O during inference

### Dataset Locations

Standard dataset paths used in this project:
```
/eph/nvme0/aime_2025/aime_2025.jsonl              # AIME 2025
/eph/nvme0/hmmt_feb_2025/hmmt_feb_2025.jsonl      # HMMT February 2025
```

---

## Overview

### What These Scripts Do

**Inference Script** (`run_hmmt_3q.sh`):
- Runs offline batch inference on HMMT February 2025 dataset
- Uses DeepSeek-R1-0528-Qwen3-8B model with vLLM
- Generates 512 traces per question
- Processes questions in chunks of 3 to avoid memory issues
- Outputs pickle files with traces and confidence scores

**Analysis Script** (`analyze_hmmt_results_deepconfs.py`):
- Loads generated pickle files
- Evaluates all DeepConf voting methods
- Compares accuracy across different voting strategies
- Reports token usage statistics

---

## Running Inference

### Prerequisites

1. **Dataset**: HMMT February 2025 dataset at `/eph/nvme0/hmmt_feb_2025/hmmt_feb_2025.jsonl`
2. **Environment**: DeepConf conda environment at `/eph/nvme0/env/deepconf`
3. **GPU**: Requires 8 GPUs (tensor_parallel_size=8)
4. **Disk**: ~100GB free space in `/eph/nvme0/` for output

### Step 1: Start a Screen Session

**Important**: Inference takes several hours, so run it in a screen session to prevent interruption.

```bash
# Create a new screen session
screen -S hmmt_inference

# Or reattach to existing session
screen -r hmmt_inference
```

### Step 2: Navigate to Project Directory

```bash
cd /home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit
```

### Step 3: Run the Inference using scripts/run_offline_batch_3_q.py via run_hmmt_3q.sh

```bash
# Make sure the script is executable
chmod +x scripts/run_hmmt_3q.sh

# Run the script
./scripts/run_hmmt_3q.sh
```

### Important!!!!!!!! Script Parameters and system prompt

The inference script uses these parameters:
- `--dataset`: Path to HMMT JSONL file
- `--budget`: Number of traces per question (512)
- `--rid`: Run identifier (hmmt2025run3qdeepseek8b)
- `--max_tokens`: Maximum tokens per trace (64000)
- `--temperature`: Sampling temperature (0.6)
- `--model`: Model name (deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- `--tensor_parallel_size`: Number of GPUs (8)
- `--chunk_size`: Questions per batch (3)

Read Table 11 in paper, use those paramters, Make sure to add the top-k parameters to the Qwen model and GPT-OSS-20B/120B models. and change --max_tokens for each model.

For system prompt: read below from paper under Table 11:
Prompt templates: For Qwen3 and GPT-OSS, we append the same instruction to every problem
prompt: “Please reason step by step, and put your final answer within \boxed{}.” For GPT-OSS, we additionally keep the provider’s official system prompt and enable the reasoning effort = high setting. For DeepSeek-8B, we use the official system prompt and put the problem in the user message.

### Output Location

Results are saved to:
```
/eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b/
```

Each question generates a pickle file:
```
qid<N>_<timestamp>.pkl
```

### Monitoring Progress

Check the inference log:
```bash
tail -f /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run/inference.log
```

### Detaching from Screen

Once inference starts:
1. Press `Ctrl+A`, then `D` to detach
2. Inference continues running in background
3. Reattach anytime with `screen -r hmmt_inference`

---

## Analyzing Results

### Prerequisites

1. **Completed inference**: All pickle files generated
2. **DeepConf environment**: Contains required dependencies (dynasor)

### Step 1: Activate Environment

```bash
# Use the DeepConf conda environment
/eph/nvme0/env/deepconf/bin/python
```

### Step 2: Run Analysis Script

**Basic analysis** (summary statistics only):
```bash
/eph/nvme0/env/deepconf/bin/python scripts/analyze_hmmt_results_deepconfs.py \
    --results_dir /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b
```

**Detailed analysis** (includes per-question breakdown):
```bash
/eph/nvme0/env/deepconf/bin/python scripts/analyze_hmmt_results_deepconfs.py \
    --results_dir /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b \
    --detailed
```

### Analysis Parameters

- `--results_dir`: Directory containing pickle files (required)
- `--detailed`: Flag to print per-question results (optional)

---

## Understanding the Output

### 1. Overall Statistics

```
📊 Overall Statistics
--------------------------------------------------------------------------------
Total questions analyzed: 30
Valid results: 30
```

Shows how many questions were successfully processed.

### 2. Token Usage

```
💰 Token Usage
--------------------------------------------------------------------------------
Total tokens: 15,234,567
Average tokens per question: 507,819 ± 12,345
Average traces per question: 512
```

- **Total tokens**: Cumulative tokens across all questions
- **Average per question**: Mean and standard deviation
- **Average traces**: Should be ~512 (your budget)

### 3. Voting Methods Accuracy

```
🗳️  VOTING METHODS ACCURACY (DeepConf)
================================================================================
Method                              Accuracy     Correct/Total   Avg Conf
--------------------------------------------------------------------------------
top10_tail_filtered                 83.33%       25/30           0.1234
bottom_window_weighted              80.00%       24/30           0.1156      ⭐
majority_vote                       76.67%       23/30           0.1089
uniform_avg                         73.33%       22/30           0.1045

⭐ = Main DeepConf method (bottom window weighted)
```

**Voting Methods Explained**:

1. **top10_tail_filtered**: Filter top 10% by tail confidence, then vote
2. **bottom_window_weighted**: Weight by bottom 10% sliding window confidence
3. **majority_vote**: Simple majority voting
4. **uniform_avg**: Uniform confidence averaging

**Best Method**: Highest accuracy is the winner (top10_tail_filtered in example)

### 4. Failed Questions

```
❌ Questions wrong by best method (top10_tail_filtered):
   QIDs: [9, 12, 16, 18, 19]
```

Lists question IDs that the best method failed on.

### 5. Detailed Per-Question Results (with --detailed)

```
📋 DETAILED PER-QUESTION RESULTS
================================================================================

QID   Ground Truth         top10_tail_fi    bottom_window    majority_vote
--------------------------------------------------------------------------------
1     42                   ✓ 42             ✓ 42             ✓ 42
2     \frac{1}{2}          ✓ \frac{1}{2}    ✓ \frac{1}{2}    ✗ \frac{2}{4}
9     3                    ✗ 5              ✗ 5              ✗ 5
```

- ✓ = Correct answer
- ✗ = Wrong answer

---

## Troubleshooting

### Issue: Inference Script Fails Immediately

**Check**:
1. GPU availability: `nvidia-smi`
2. Dataset exists: `ls -lh /eph/nvme0/hmmt_feb_2025/hmmt_feb_2025.jsonl`
3. Output directory writable: `touch /eph/nvme0/hmmt_feb_2025_run_512_results/test.txt`

**Solution**: Ensure all prerequisites are met

### Issue: Out of Memory (OOM) Errors

**Symptoms**: CUDA out of memory errors during inference

**Solutions**:
1. Reduce `--chunk_size` from 3 to 2 or 1
2. Reduce `--budget` from 512 to 256
3. Check GPU memory: `nvidia-smi`

### Issue: Analysis Script Can't Find Files

**Error**: `❌ No pickle files found!`

**Check**:
1. Results directory exists: `ls /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b/`
2. Pickle files present: `ls /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b/*.pkl | wc -l`

**Solution**: Verify inference completed successfully

### Issue: Import Errors in Analysis Script

**Error**: `ModuleNotFoundError: No module named 'dynasor'`

**Solution**: Use the correct conda environment:
```bash
/eph/nvme0/env/deepconf/bin/python scripts/analyze_hmmt_results_deepconfs.py ...
```

### Issue: Mathematical Comparison Errors

**Symptoms**: Many answers marked wrong despite looking correct

**Check**: LaTeX formatting differences (e.g., `\frac{1}{2}` vs `0.5`)

**Note**: The script uses `math_equal()` from dynasor which handles mathematical equivalence

---

## Quick Reference

### Run Full Pipeline

```bash
# 1. Start screen session
screen -S hmmt_inference

# 2. Run inference (takes several hours)
cd /home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit
./scripts/run_hmmt_3q.sh

# 3. Detach from screen (Ctrl+A, then D)

# 4. Later: Analyze results
/eph/nvme0/env/deepconf/bin/python scripts/analyze_hmmt_results_deepconfs.py \
    --results_dir /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b \
    --detailed
```

### Directory Structure

```
/eph/nvme0/
├── hmmt_feb_2025/
│   └── hmmt_feb_2025.jsonl                    # Input dataset
├── hmmt_feb_2025_run_512_results/
│   └── hmmt2025run3qdeepseek8b/               # Output directory
│       ├── qid1_<timestamp>.pkl               # Question 1 results
│       ├── qid2_<timestamp>.pkl               # Question 2 results
│       └── ...                                # More results
└── env/
    └── deepconf/                              # Conda environment
```

### Key Files

- **Inference script**: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/run_hmmt_3q.sh`
- **Analysis script**: `/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/analyze_hmmt_results_deepconfs.py`
- **Results directory**: `/eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b/`

---

## Expected Results

For HMMT February 2025 with 512 traces:
- **Total questions**: 30
- **Expected accuracy**: 75-85% (depending on voting method)
- **Runtime**: 4-8 hours (depending on GPU availability)
- **Storage**: ~50-100GB for all pickle files
- **Token usage**: ~15M tokens total

---

## Additional Notes

### Why Chunk Size = 3?

Processing 3 questions at a time balances:
- **Memory efficiency**: Prevents OOM with 512 traces/question
- **Speed**: Parallelizes within each chunk
- **Reliability**: Smaller chunks = easier to resume if interrupted

### Understanding Confidence Scores

Confidence scores are computed using:
- **KL divergence** from top-20 logprobs at each token
- **Lower scores** = higher confidence
- **Typical range**: 10-20 for math problems

### Voting Method Selection

**For research**:
- Use `top10_tail_filtered` - best performance on HMMT
- Compare with `bottom_window_weighted` (original DeepConf)

**For production**:
- Use the method with highest accuracy on your validation set

---

## Contact

For questions or issues:
- Check script comments for inline documentation
- Review error logs in `/eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run/inference.log`
- Verify environment setup matches prerequisites

---

**Last Updated**: December 2025
**Dataset**: HMMT February 2025
**Model**: DeepSeek-R1-0528-Qwen3-8B
