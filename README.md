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
python analysis/custom_weighting.py --results_dir offline_results/run001/
```

## Output Structure

```
{output_dir}/
└── {rid}/
    ├── qid0_{timestamp}.pkl
    ├── qid1_{timestamp}.pkl
    └── ...
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
