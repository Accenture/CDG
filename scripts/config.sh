#!/bin/bash
# ============================================================
# Centralized configuration for all shell scripts
# Source this file at the beginning of your shell scripts:
#   source "$(dirname "$0")/../config.sh"
# ============================================================

# ============================================================
# BASE DIRECTORIES
# ============================================================

# Base directory for datasets (JSONL files) - use fast NVMe
export DATASETS_BASE="/eph/nvme0/datasets"

# Base directory for inference results output
export OUTPUT_BASE="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"

# ============================================================
# HUGGINGFACE CACHE PATHS
# ============================================================

# HuggingFace cache - use NVMe (27TB free, faster)
export HF_HOME="/eph/nvme0/hf_cache"
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
export QWEN32B_ENABLE_THINKING="true"

# DeepSeek-8B settings
export DEEPSEEK8B_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
export DEEPSEEK8B_MODEL_TYPE="deepseek"
export DEEPSEEK8B_TEMPERATURE="0.6"
export DEEPSEEK8B_TOP_P="0.95"
export DEEPSEEK8B_TOP_K="-1"
export DEEPSEEK8B_MAX_TOKENS="64000"

# GPT-OSS-20B settings (per DeepConf paper Table 11)
export GPTOSS20B_MODEL="openai/gpt-oss-20b"
export GPTOSS20B_MODEL_TYPE="gpt-oss"
export GPTOSS20B_TEMPERATURE="1.0"
export GPTOSS20B_TOP_P="1.0"
export GPTOSS20B_TOP_K="40"
export GPTOSS20B_MAX_TOKENS="130000"
export GPTOSS20B_REASONING_EFFORT="high"

# GPT-OSS-120B settings (per DeepConf paper Table 11)
export GPTOSS120B_MODEL="openai/gpt-oss-120b"
export GPTOSS120B_MODEL_TYPE="gpt-oss"
export GPTOSS120B_TEMPERATURE="1.0"
export GPTOSS120B_TOP_P="1.0"
export GPTOSS120B_TOP_K="40"
export GPTOSS120B_MAX_TOKENS="130000"
export GPTOSS120B_REASONING_EFFORT="high"

# Gemma 3-27B settings (non-reasoning baseline)
export GEMMA3_27B_MODEL="google/gemma-3-27b-it"
export GEMMA3_27B_MODEL_TYPE="gemma"
export GEMMA3_27B_TEMPERATURE="0.6"
export GEMMA3_27B_TOP_P="0.95"
export GEMMA3_27B_TOP_K="40"
export GEMMA3_27B_MAX_TOKENS="8192"  # Gemma 3 output limit

# QwQ-32B settings (reasoning model based on Qwen2.5)
export QWQ32B_MODEL="Qwen/QwQ-32B"
export QWQ32B_MODEL_TYPE="qwq"
export QWQ32B_TEMPERATURE="0.6"
export QWQ32B_TOP_P="0.95"
export QWQ32B_TOP_K="20"
export QWQ32B_MAX_TOKENS="32768"

# Qwen2.5-32B settings (base model)
export QWEN25_32B_MODEL="Qwen/Qwen2.5-32B"
export QWEN25_32B_MODEL_TYPE="qwen"
export QWEN25_32B_TEMPERATURE="0.6"
export QWEN25_32B_TOP_P="0.95"
export QWEN25_32B_TOP_K="20"
export QWEN25_32B_MAX_TOKENS="8192"

# DeepSeek-R1-Distill-Llama-70B settings (reasoning model distilled from R1)
export DEEPSEEK_R1_LLAMA_70B_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
export DEEPSEEK_R1_LLAMA_70B_MODEL_TYPE="deepseek"
export DEEPSEEK_R1_LLAMA_70B_TEMPERATURE="0.6"
export DEEPSEEK_R1_LLAMA_70B_TOP_P="0.95"
export DEEPSEEK_R1_LLAMA_70B_TOP_K="-1"
export DEEPSEEK_R1_LLAMA_70B_MAX_TOKENS="32768"

# ============================================================
# DEFAULT INFERENCE SETTINGS
# ============================================================

export DEFAULT_BUDGET="512"
export DEFAULT_CHUNK_SIZE="1"
export DEFAULT_TENSOR_PARALLEL_SIZE="8"
export DEFAULT_LOGPROBS="20"  # Number of top logprobs to return per token

# ============================================================
# HELPER FUNCTION
# ============================================================

print_config() {
    echo "========================================"
    echo "Configuration Settings"
    echo "========================================"
    echo "DATASETS_BASE: ${DATASETS_BASE}"
    echo "OUTPUT_BASE: ${OUTPUT_BASE}"
    echo "HF_HOME: ${HF_HOME}"
    echo "========================================"
}

# Print config when sourced with -v flag
if [[ "$1" == "-v" ]]; then
    print_config
fi
