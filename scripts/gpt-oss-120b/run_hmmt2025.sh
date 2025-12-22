#!/bin/bash
# HMMT Feb 2025 Inference - GPT-OSS-120B (512 traces)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "GPT-OSS-120B - HMMT 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${GPTOSS120B_TEMPERATURE}"
echo "Top-p: ${GPTOSS120B_TOP_P}"
echo "Top-k: ${GPTOSS120B_TOP_K}"
echo "Max tokens: ${GPTOSS120B_MAX_TOKENS}"
echo "Reasoning effort: ${GPTOSS120B_REASONING_EFFORT}"
echo "Dataset: ${DATASET_HMMT_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_gpt_oss_batch.py \
    --dataset "${DATASET_HMMT_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid gptoss120b_hmmt2025_512 \
    --max_tokens "${GPTOSS120B_MAX_TOKENS}" \
    --temperature "${GPTOSS120B_TEMPERATURE}" \
    --top_p "${GPTOSS120B_TOP_P}" \
    --top_k "${GPTOSS120B_TOP_K}" \
    --reasoning_effort "${GPTOSS120B_REASONING_EFFORT}" \
    --model "${GPTOSS120B_MODEL}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "HMMT 2025 completed!"
echo "========================================="
