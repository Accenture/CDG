  git submodule update --init --recursive

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
