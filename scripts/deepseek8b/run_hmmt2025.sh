#!/bin/bash
# HMMT February 2025 Inference - DeepSeek-R1-0528-Qwen3-8B (512 traces)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "DeepSeek-R1-8B - HMMT 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${DEEPSEEK8B_TEMPERATURE}"
echo "Top-p: ${DEEPSEEK8B_TOP_P}"
echo "Top-k: ${DEEPSEEK8B_TOP_K}"
echo "Max tokens: ${DEEPSEEK8B_MAX_TOKENS}"
echo "Dataset: ${DATASET_HMMT_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_HMMT_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid deepseek8b_hmmt2025_512 \
    --max_tokens "${DEEPSEEK8B_MAX_TOKENS}" \
    --temperature "${DEEPSEEK8B_TEMPERATURE}" \
    --top_p "${DEEPSEEK8B_TOP_P}" \
    --top_k "${DEEPSEEK8B_TOP_K}" \
    --model "${DEEPSEEK8B_MODEL}" \
    --model_type "${DEEPSEEK8B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "HMMT 2025 completed!"
echo "========================================="
