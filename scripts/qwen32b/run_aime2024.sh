#!/bin/bash
# AIME 2024 Inference - Qwen3-32B (512 traces, top-k=20)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "Qwen3-32B - AIME 2024 (512 traces)"
echo "========================================="
echo "Temperature: ${QWEN32B_TEMPERATURE}"
echo "Top-p: ${QWEN32B_TOP_P}"
echo "Top-k: ${QWEN32B_TOP_K}"
echo "Max tokens: ${QWEN32B_MAX_TOKENS}"
echo "Dataset: ${DATASET_AIME_2024}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_offline_batch_3_q.py \
    --dataset "${DATASET_AIME_2024}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid qwen32b_aime2024_512 \
    --max_tokens "${QWEN32B_MAX_TOKENS}" \
    --temperature "${QWEN32B_TEMPERATURE}" \
    --top_k "${QWEN32B_TOP_K}" \
    --model "${QWEN32B_MODEL}" \
    --model_type "${QWEN32B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "AIME 2024 completed!"
echo "========================================="
