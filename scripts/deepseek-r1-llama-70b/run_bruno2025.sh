#!/bin/bash
# Bruno 2025 Inference - DeepSeek-R1-Distill-Llama-70B (512 traces)
# Reasoning model distilled from DeepSeek-R1

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "DeepSeek-R1-Distill-Llama-70B - Bruno 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${DEEPSEEK_R1_LLAMA_70B_TEMPERATURE}"
echo "Top-p: ${DEEPSEEK_R1_LLAMA_70B_TOP_P}"
echo "Top-k: ${DEEPSEEK_R1_LLAMA_70B_TOP_K}"
echo "Max tokens: ${DEEPSEEK_R1_LLAMA_70B_MAX_TOKENS}"
echo "Dataset: ${DATASET_BRUNO_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_vllm_batch.py \
    --dataset "${DATASET_BRUNO_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid deepseek_r1_llama70b_bruno2025_512 \
    --max_tokens "${DEEPSEEK_R1_LLAMA_70B_MAX_TOKENS}" \
    --temperature "${DEEPSEEK_R1_LLAMA_70B_TEMPERATURE}" \
    --top_k "${DEEPSEEK_R1_LLAMA_70B_TOP_K}" \
    --model "${DEEPSEEK_R1_LLAMA_70B_MODEL}" \
    --model_type "${DEEPSEEK_R1_LLAMA_70B_MODEL_TYPE}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --logprobs "${DEFAULT_LOGPROBS}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "Bruno 2025 completed!"
echo "========================================="
