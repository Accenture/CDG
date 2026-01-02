#!/bin/bash
# AIME 2024 Inference - Qwen2.5-32B (512 traces)
# Base model (non-instruction-tuned)

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "Qwen2.5-32B - AIME 2024 (512 traces)"
echo "========================================="
echo "Temperature: ${QWEN25_32B_TEMPERATURE}"
echo "Top-p: ${QWEN25_32B_TOP_P}"
echo "Top-k: ${QWEN25_32B_TOP_K}"
echo "Max tokens: ${QWEN25_32B_MAX_TOKENS}"
echo "Dataset: ${DATASET_AIME_2024}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_AIME_2024}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwen25_32b_aime2024_512 \
    --max_tokens "${QWEN25_32B_MAX_TOKENS}" \
    --temperature "${QWEN25_32B_TEMPERATURE}" \
    --top_k "${QWEN25_32B_TOP_K}" \
    --model "${QWEN25_32B_MODEL}" \
    --model_type "${QWEN25_32B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --logprobs "${DEFAULT_LOGPROBS}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "AIME 2024 completed!"
echo "========================================="
