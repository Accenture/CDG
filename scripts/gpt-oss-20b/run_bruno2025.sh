#!/bin/bash
# Bruno 2025 Inference - GPT-OSS-20B (512 traces)
# Settings per DeepConf paper Table 11

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================="
echo "GPT-OSS-20B - Bruno 2025 (512 traces)"
echo "========================================="
echo "Temperature: ${GPTOSS20B_TEMPERATURE}"
echo "Top-p: ${GPTOSS20B_TOP_P}"
echo "Top-k: ${GPTOSS20B_TOP_K}"
echo "Max tokens: ${GPTOSS20B_MAX_TOKENS}"
echo "Reasoning effort: ${GPTOSS20B_REASONING_EFFORT}"
echo "Dataset: ${DATASET_BRUNO_2025}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================="

python scripts/run_gpt_oss_batch.py \
    --dataset "${DATASET_BRUNO_2025}" \
    --budget "${DEFAULT_BUDGET}" \
    --rid gptoss20b_bruno2025_512 \
    --max_tokens "${GPTOSS20B_MAX_TOKENS}" \
    --temperature "${GPTOSS20B_TEMPERATURE}" \
    --top_p "${GPTOSS20B_TOP_P}" \
    --top_k "${GPTOSS20B_TOP_K}" \
    --reasoning_effort "${GPTOSS20B_REASONING_EFFORT}" \
    --model "${GPTOSS20B_MODEL}" \
    --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
    --chunk_size "${DEFAULT_CHUNK_SIZE}" \
    --output_dir "${OUTPUT_BASE}"

echo "========================================="
echo "Bruno 2025 completed!"
echo "========================================="
