#!/bin/bash
# AIME 2024 Inference - QwQ-32B (512 traces)
# Reasoning model based on Qwen2.5

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "QwQ-32B - AIME 2024 (512 traces)"
echo "========================================="
echo "Temperature: ${QWQ32B_TEMPERATURE}"
echo "Top-p: ${QWQ32B_TOP_P}"
echo "Top-k: ${QWQ32B_TOP_K}"
echo "Max tokens: ${QWQ32B_MAX_TOKENS}"
echo "Dataset: ${DATASET_AIME_2024}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_AIME_2024}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwq32b_aime2024_512 \
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
echo "AIME 2024 completed!"
echo "========================================="
