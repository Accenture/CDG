#!/bin/bash
# HMMT 2025 Inference - QwQ-32B (512 traces)
# Reasoning model based on Qwen2.5

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "QwQ-32B - HMMT 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${QWQ32B_TEMPERATURE}"
echo "Top-p: ${QWQ32B_TOP_P}"
echo "Top-k: ${QWQ32B_TOP_K}"
echo "Max tokens: ${QWQ32B_MAX_TOKENS}"
echo "Dataset: ${DATASET_HMMT_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_HMMT_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwq32b_hmmt2025_512 \
    --max_tokens "${QWQ32B_MAX_TOKENS}" \
    --temperature "${QWQ32B_TEMPERATURE}" \
    --top_k "${QWQ32B_TOP_K}" \
    --model "${QWQ32B_MODEL}" \
    --model_type "${QWQ32B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --logprobs "${DEFAULT_LOGPROBS}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "HMMT 2025 completed!"
echo "========================================="
