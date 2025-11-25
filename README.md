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

# Run offline inference (8 GPUs, 256 samples per question)
seq 0 29 | parallel -j8 'CUDA_VISIBLE_DEVICES=$(({} % 8)) python scripts/run_offline_single.py --qid {} --budget 256'

# Analyze results
python analysis/compute_upper_bound.py --results_dir offline_results/
python analysis/custom_weighting.py --results_dir offline_results/
```
