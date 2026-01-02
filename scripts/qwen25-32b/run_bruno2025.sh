#!/bin/bash
# Bruno 2025 Inference - Qwen2.5-32B (512 traces)
# Base model (non-instruction-tuned)

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "Qwen2.5-32B - Bruno 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${QWEN25_32B_TEMPERATURE}"
echo "Top-p: ${QWEN25_32B_TOP_P}"
echo "Top-k: ${QWEN25_32B_TOP_K}"
echo "Max tokens: ${QWEN25_32B_MAX_TOKENS}"
echo "Dataset: ${DATASET_BRUNO_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_BRUNO_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwen25_32b_bruno2025_512 \
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
echo "Bruno 2025 completed!"
echo "========================================="
