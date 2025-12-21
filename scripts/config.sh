#!/bin/bash
# ============================================================
# Centralized configuration for all shell scripts
# Source this file at the beginning of your shell scripts:
#   source "$(dirname "$0")/../config.sh"
# ============================================================

# ============================================================
# BASE DIRECTORIES
# ============================================================

# Base directory for datasets (JSONL files)
export DATASETS_BASE="/mnt/dev/datasets"

# Base directory for model checkpoints and HuggingFace cache
export MODEL_CACHE_BASE="/mnt/dev/model_ckpt"

# Base directory for inference results output
export OUTPUT_BASE="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"

# ============================================================
# HUGGINGFACE CACHE PATHS
# ============================================================

export HF_HOME="${MODEL_CACHE_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ============================================================
# DATASET PATHS
# ============================================================

export DATASET_AIME_2025="${DATASETS_BASE}/aime_2025/aime_2025.jsonl"
export DATASET_AIME_2024="${DATASETS_BASE}/aime_2024/aime_2024.jsonl"
export DATASET_HMMT_2025="${DATASETS_BASE}/hmmt_feb_2025/hmmt_feb_2025.jsonl"
export DATASET_BRUNO_2025="${DATASETS_BASE}/bruno_2025/bruno_2025.jsonl"

# ============================================================
# MODEL CONFIGURATIONS (per DeepConf paper Table 11)
# ============================================================

# Qwen3-32B settings
export QWEN32B_MODEL="Qwen/Qwen3-32B"
export QWEN32B_MODEL_TYPE="qwen"
export QWEN32B_TEMPERATURE="0.6"
export QWEN32B_TOP_P="0.95"
export QWEN32B_TOP_K="20"
export QWEN32B_MAX_TOKENS="32000"

# DeepSeek-8B settings
export DEEPSEEK8B_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
export DEEPSEEK8B_MODEL_TYPE="deepseek"
export DEEPSEEK8B_TEMPERATURE="0.6"
export DEEPSEEK8B_TOP_P="0.95"
export DEEPSEEK8B_TOP_K="-1"
export DEEPSEEK8B_MAX_TOKENS="64000"

# ============================================================
# DEFAULT INFERENCE SETTINGS
# ============================================================

export DEFAULT_BUDGET="512"
export DEFAULT_CHUNK_SIZE="3"
export DEFAULT_TENSOR_PARALLEL_SIZE="8"

# ============================================================
# HELPER FUNCTION
# ============================================================

print_config() {
    echo "========================================"
    echo "Configuration Settings"
    echo "========================================"
    echo "DATASETS_BASE: ${DATASETS_BASE}"
    echo "MODEL_CACHE_BASE: ${MODEL_CACHE_BASE}"
    echo "OUTPUT_BASE: ${OUTPUT_BASE}"
    echo "HF_HOME: ${HF_HOME}"
    echo "========================================"
}

# Print config when sourced with -v flag
if [[ "$1" == "-v" ]]; then
    print_config
fi
