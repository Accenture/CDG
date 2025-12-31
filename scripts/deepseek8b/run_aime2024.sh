#!/bin/bash
# AIME 2024 Inference - DeepSeek-R1-0528-Qwen3-8B (512 traces)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "DeepSeek-R1-8B - AIME 2024 (512 traces)"
echo "========================================="
echo "Temperature: ${DEEPSEEK8B_TEMPERATURE}"
echo "Top-p: ${DEEPSEEK8B_TOP_P}"
echo "Top-k: ${DEEPSEEK8B_TOP_K}"
echo "Max tokens: ${DEEPSEEK8B_MAX_TOKENS}"
echo "Dataset: ${DATASET_AIME_2024}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_AIME_2024}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid deepseek8b_aime2024_512 \
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
echo "AIME 2024 completed!"
echo "========================================="
