#!/bin/bash
# Bruno 2025 Inference - Gemma 3-27B (512 traces)
# Non-reasoning baseline for confidence pattern comparison

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "Gemma 3-27B - Bruno 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${GEMMA3_27B_TEMPERATURE}"
echo "Top-p: ${GEMMA3_27B_TOP_P}"
echo "Top-k: ${GEMMA3_27B_TOP_K}"
echo "Max tokens: ${GEMMA3_27B_MAX_TOKENS}"
echo "Dataset: ${DATASET_BRUNO_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_BRUNO_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid gemma3_27b_bruno2025_512 \
    --max_tokens "${GEMMA3_27B_MAX_TOKENS}" \
    --temperature "${GEMMA3_27B_TEMPERATURE}" \
    --top_p "${GEMMA3_27B_TOP_P}" \
    --top_k "${GEMMA3_27B_TOP_K}" \
    --model "${GEMMA3_27B_MODEL}" \
    --model_type "${GEMMA3_27B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --logprobs "${DEFAULT_LOGPROBS}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "Bruno 2025 completed!"
echo "========================================="
