## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd sampling_credit

# Initialize submodules (pulls deepconf dependency)
git submodule update --init --recursive
```

This repo uses [deepconf](https://github.com/facebookresearch/deepconf) as a git submodule for the core LLM wrapper and confidence computation utilities.

## Dependencies

**Important**: The `dynasor` package for math evaluation must be installed from GitHub (NOT PyPI):

```bash
# The PyPI 'dynasor' is a different package (molecular dynamics)!
# Install the correct one from hao-ai-lab:
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

## Paths (Azure VM)

- **Model cache**: `/mnt/dev/model_ckpt/hf_cache`
- **Results output**: `/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results`

## Setup

```bash
conda env create -f environment.yml
conda activate sampling_credit
```

## Run Experiments

```bash
# Prepare AIME 2025 dataset
python scripts/prepare_aime2025.py

# Run offline inference - RECOMMENDED: batch mode (uses vLLM native batching)
# --rid auto-generates as run001, run002, etc. if not specified
python scripts/run_offline_batch.py --budget 256 --tensor_parallel_size 8

# Alternative: single question mode (useful for debugging)
python scripts/run_offline_single.py --qid 0 --budget 256

# Analyze results
python analysis/compute_upper_bound.py --results_dir offline_results/run001/
python analysis/custom_weighting.py --results_dir offline_results/run001/ --no-parallel
```

## Output Structure

```
{output_dir}/
└── {rid}/
    ├── qid0_{timestamp}.pkl
    ├── qid1_{timestamp}.pkl
    └── ...
```

### Data Structure (per-question pickle file)

```python
{
    # Metadata
    'question': str,           # The math problem
    'ground_truth': str,       # Correct answer
    'qid': int,                # Question ID
    'run_id': str,             # Run identifier

    # Results
    'all_traces': list[256],   # All 256 generated traces
    'total_tokens': int,       # Total tokens across all traces
    'total_traces_count': int, # 256

    # Voting
    'voting_results': {
        'majority': ...,
        'mean_confidence_weighted': ...,
        'tail_confidence_weighted': ...,
        'bottom_window_weighted': ...,
        'min_window_weighted': ...,
    },
    'voted_answer': str,
    'final_answer': str,

    # Config
    'config': {'model', 'mode', 'budget', 'window_size', 'temperature'},
}
```

### Trace Structure (`all_traces[i]`)

```python
{
    'stop_reason': str,        # 'stop' or 'length'
    'text': str,               # Full reasoning text (can be very long)
    'token_ids': list[N],      # Token IDs (N = num_tokens)
    'num_tokens': int,         # e.g., 7587
    'confs': list[N],          # Per-token confidence (same length as token_ids)
    'extracted_answer': str,   # Parsed answer from \boxed{}

    # Future: attention metrics (when attention capture is enabled)
    # 'attention_received': list[N],   # Forking signal
    # 'self_attention': list[N],       # Self-reliance
    # 'attention_entropy': list[N],    # Decision spread
}
```

## Analyze Results

### Upper Bound (Oracle Accuracy)

Compute the maximum achievable accuracy if we had perfect trace selection:

```bash
python analysis/compute_upper_bound.py --results_dir offline_results/run001/
```

### Weighting Strategy Comparison

Compare all weighting strategies for majority voting:

```bash
# Auto-detect CPU count for parallel evaluation
python analysis/custom_weighting.py --results_dir offline_results/run001/

# Specify number of workers
python analysis/custom_weighting.py --results_dir offline_results/run001/ --parallel 32

# Sequential (for debugging)
python analysis/custom_weighting.py --results_dir offline_results/run001/ --no-parallel
```

### Weighting Strategies

| Strategy | Description |
|----------|-------------|
| `uniform` | Baseline: all traces equal weight |
| `inverse_conf` | Weight = 1/mean(conf). High confidence → high weight |
| `min_window` | Weight by worst 2048-token window (weakest link) |
| `tail_focused` | Weight by final 2048 tokens (near the answer) |
| `entropy_based` | Weight = 1/(1+variance). Stable traces → high weight |
| `teammate_raw` | `x^{20*((-tanh(0.8x))+1.05)}` where x=mean_conf |
| `teammate_inv` | Same formula, x=1/mean_conf |
| `smooth_blend` | Sigmoid blend between y=x and y=1/x |
| `token_blend` | Per-token smooth blend, then average |

**Research hypothesis**: Forking tokens (high attention from others) have low confidence. We want to weight:
- High confidence tokens → y = x (reward)
- Low confidence tokens → y = 1/x (penalize)

### Forking Token Analysis

Analyze the relationship between "forking tokens" (uncertainty markers like "but", "however", "perhaps") and model confidence. Reference: https://arxiv.org/pdf/2506.01939

```bash
# Run all analyses (convert to JSON + Exp 1 + Exp 2)
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --all

# Individual experiments:

# Convert pickle files to human-readable JSON
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --convert_to_json

# Exp 1: Analyze entropy of forking vs non-forking tokens
# Outputs: histograms (token-level + sample-level) and t-tests
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --exp1

# Exp 1 with custom window size for sample-level analysis
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --exp1 --window_size 100

# Exp 2: Brute force search for optimal sampling threshold (0-15, step 0.1)
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --exp2

# Exp 2 with custom range
python analysis/forking_token_analysis.py --results_dir offline_results/run001/ --exp2 --threshold_start 0 --threshold_end 20 --threshold_step 0.05
```

**Exp 1 Output**:
- Token-level: Mean confidence of forking tokens vs non-forking tokens
- Sample-level: Per-trace average confidence of forking vs non-forking tokens
- Window-level: Confidence of windows around forking tokens vs other regions
- Statistical t-tests at all three levels
- Histograms: `exp1_histograms.png`

**Exp 2 Output**:
- Accuracy at each threshold value
- Best threshold and accuracy
- Plot: `exp2_threshold_plot.png`

**Forking Tokens** (from paper word cloud, by importance tier):
- Tier 1: since, thus, suppose, perhaps, actually, define, which, assume, also, given, wait, maybe, let, what, however
- Tier 2: using, express, specific, denote, unless, consider, earlier, just, the, that, we, it
- Tier 3: going, solving, recall, calculate, re, note, now, how, but, because, yes
- Tier 4: if, when, therefore, might, hence, alternatively, or, etc.

### Save Results as JSON

You can also save inference results as human-readable JSON files:

```bash
# Save both pickle and JSON during inference
python scripts/run_offline_batch.py --budget 256 --save_json

# Save only JSON (no pickle files)
python scripts/run_offline_batch.py --budget 256 --json_only
```

## Attention Capture (Experimental) Minghao doesn't understand this part yet !!

To get real attention weights from the last transformer layer:

```bash
# First, test if attention capture works with your setup
CUDA_VISIBLE_DEVICES=0 python scripts/test_attention_capture.py --enforce-eager

# If hooks work, attention can be captured during inference
# See scripts/attention_capture.py for implementation
```

**Requirements**:
- `enforce_eager=True` in VLLM (disables CUDA graphs, ~20-30% slower)
- PyTorch hooks on the last attention layer

**What we capture**:
- `attention_received[i]` = sum of attention that later tokens paid to token i
- High value → token is a "forking point" (other tokens looked back at it)
- Storage: O(seq_len) per trace, not O(seq_len²)

**Fallback**: If hooks don't work, confidence variance serves as a proxy for attention (already implemented in `entropy_based_weight`).
