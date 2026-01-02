#!/bin/bash
# HMMT February 2025 Inference - Qwen3-32B (512 traces, top-k=20)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "Qwen3-32B - HMMT 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${QWEN32B_TEMPERATURE}"
echo "Top-p: ${QWEN32B_TOP_P}"
echo "Top-k: ${QWEN32B_TOP_K}"
echo "Max tokens: ${QWEN32B_MAX_TOKENS}"
echo "Enable thinking: ${QWEN32B_ENABLE_THINKING}"
echo "Dataset: ${DATASET_HMMT_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

# Build enable_thinking flag
THINKING_FLAG=""
if [[ "${QWEN32B_ENABLE_THINKING}" == "true" ]]; then
    THINKING_FLAG="--enable_thinking"
fi

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_HMMT_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwen32b_hmmt2025_512 \
    --max_tokens "${QWEN32B_MAX_TOKENS}" \
    --temperature "${QWEN32B_TEMPERATURE}" \
    --top_k "${QWEN32B_TOP_K}" \
    --model "${QWEN32B_MODEL}" \
    --model_type "${QWEN32B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --logprobs "${DEFAULT_LOGPROBS}" \
    --output_dir "${OUTPUT_BASE}" \
    ${THINKING_FLAG}

echo "========================================="
echo "HMMT 2025 completed!"
echo "========================================="
