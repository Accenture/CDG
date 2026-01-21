#!/bin/bash
# ============================================================
# Unified Inference Runner
# ============================================================
#
# Usage:
#   ./scripts/inference/run.sh --model <model> --dataset <dataset>
#
# Examples:
#   ./scripts/inference/run.sh --model deepseek8b --dataset aime2025
#   ./scripts/inference/run.sh --model qwen32b --dataset all
#   ./scripts/inference/run.sh --model gpt-oss-20b --dataset bruno2025
#
# Available models:
#   deepseek8b, deepseek-r1-llama-70b, qwen32b, qwen25-32b,
#   qwq32b, gemma3-27b, gpt-oss-20b, gpt-oss-120b
#
# Available datasets:
#   aime2025, aime2024, hmmt2025, bruno2025, all
#
# ============================================================

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ============================================================
# Parse arguments
# ============================================================
MODEL=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help|-h)
            head -25 "$0" | tail -23
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]] || [[ -z "$DATASET" ]]; then
    echo "Error: --model and --dataset are required"
    echo "Usage: $0 --model <model> --dataset <dataset>"
    echo "Run with --help for more info"
    exit 1
fi

# ============================================================
# Model configuration lookup
# ============================================================
get_model_config() {
    local model=$1
    case $model in
        deepseek8b)
            MODEL_HF="${DEEPSEEK8B_MODEL}"
            MODEL_TYPE="${DEEPSEEK8B_MODEL_TYPE}"
            TEMPERATURE="${DEEPSEEK8B_TEMPERATURE}"
            TOP_P="${DEEPSEEK8B_TOP_P}"
            TOP_K="${DEEPSEEK8B_TOP_K}"
            MAX_TOKENS="${DEEPSEEK8B_MAX_TOKENS}"
            RID_PREFIX="deepseek8b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            EXTRA_FLAGS=""
            ;;
        deepseek-r1-llama-70b)
            MODEL_HF="${DEEPSEEK_R1_LLAMA_70B_MODEL}"
            MODEL_TYPE="${DEEPSEEK_R1_LLAMA_70B_MODEL_TYPE}"
            TEMPERATURE="${DEEPSEEK_R1_LLAMA_70B_TEMPERATURE}"
            TOP_P="${DEEPSEEK_R1_LLAMA_70B_TOP_P}"
            TOP_K="${DEEPSEEK_R1_LLAMA_70B_TOP_K}"
            MAX_TOKENS="${DEEPSEEK_R1_LLAMA_70B_MAX_TOKENS}"
            RID_PREFIX="deepseek_r1_llama70b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            EXTRA_FLAGS=""
            ;;
        qwen32b)
            MODEL_HF="${QWEN32B_MODEL}"
            MODEL_TYPE="${QWEN32B_MODEL_TYPE}"
            TEMPERATURE="${QWEN32B_TEMPERATURE}"
            TOP_P="${QWEN32B_TOP_P}"
            TOP_K="${QWEN32B_TOP_K}"
            MAX_TOKENS="${QWEN32B_MAX_TOKENS}"
            RID_PREFIX="qwen32b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            if [[ "${QWEN32B_ENABLE_THINKING}" == "true" ]]; then
                EXTRA_FLAGS="--enable_thinking"
            else
                EXTRA_FLAGS=""
            fi
            ;;
        qwen25-32b)
            MODEL_HF="${QWEN25_32B_MODEL}"
            MODEL_TYPE="${QWEN25_32B_MODEL_TYPE}"
            TEMPERATURE="${QWEN25_32B_TEMPERATURE}"
            TOP_P="${QWEN25_32B_TOP_P}"
            TOP_K="${QWEN25_32B_TOP_K}"
            MAX_TOKENS="${QWEN25_32B_MAX_TOKENS}"
            RID_PREFIX="qwen25_32b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            EXTRA_FLAGS=""
            ;;
        qwq32b)
            MODEL_HF="${QWQ32B_MODEL}"
            MODEL_TYPE="${QWQ32B_MODEL_TYPE}"
            TEMPERATURE="${QWQ32B_TEMPERATURE}"
            TOP_P="${QWQ32B_TOP_P}"
            TOP_K="${QWQ32B_TOP_K}"
            MAX_TOKENS="${QWQ32B_MAX_TOKENS}"
            RID_PREFIX="qwq32b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            EXTRA_FLAGS=""
            ;;
        gemma3-27b)
            MODEL_HF="${GEMMA3_27B_MODEL}"
            MODEL_TYPE="${GEMMA3_27B_MODEL_TYPE}"
            TEMPERATURE="${GEMMA3_27B_TEMPERATURE}"
            TOP_P="${GEMMA3_27B_TOP_P}"
            TOP_K="${GEMMA3_27B_TOP_K}"
            MAX_TOKENS="${GEMMA3_27B_MAX_TOKENS}"
            RID_PREFIX="gemma3_27b"
            PYTHON_SCRIPT="run_vllm_batch.py"
            EXTRA_FLAGS=""
            ;;
        gpt-oss-20b)
            MODEL_HF="${GPTOSS20B_MODEL}"
            MODEL_TYPE="${GPTOSS20B_MODEL_TYPE}"
            TEMPERATURE="${GPTOSS20B_TEMPERATURE}"
            TOP_P="${GPTOSS20B_TOP_P}"
            TOP_K="${GPTOSS20B_TOP_K}"
            MAX_TOKENS="${GPTOSS20B_MAX_TOKENS}"
            REASONING_EFFORT="${GPTOSS20B_REASONING_EFFORT}"
            RID_PREFIX="gptoss20b"
            PYTHON_SCRIPT="run_gpt_oss_batch.py"
            EXTRA_FLAGS="--reasoning_effort ${REASONING_EFFORT} --enforce_eager"
            ;;
        gpt-oss-120b)
            MODEL_HF="${GPTOSS120B_MODEL}"
            MODEL_TYPE="${GPTOSS120B_MODEL_TYPE}"
            TEMPERATURE="${GPTOSS120B_TEMPERATURE}"
            TOP_P="${GPTOSS120B_TOP_P}"
            TOP_K="${GPTOSS120B_TOP_K}"
            MAX_TOKENS="${GPTOSS120B_MAX_TOKENS}"
            REASONING_EFFORT="${GPTOSS120B_REASONING_EFFORT}"
            RID_PREFIX="gptoss120b"
            PYTHON_SCRIPT="run_gpt_oss_batch.py"
            EXTRA_FLAGS="--reasoning_effort ${REASONING_EFFORT} --enforce_eager"
            ;;
        *)
            echo "Error: Unknown model '$model'"
            echo "Available models: deepseek8b, deepseek-r1-llama-70b, qwen32b, qwen25-32b, qwq32b, gemma3-27b, gpt-oss-20b, gpt-oss-120b"
            exit 1
            ;;
    esac
}

# ============================================================
# Dataset configuration lookup
# ============================================================
get_dataset_config() {
    local dataset=$1
    case $dataset in
        aime2025)
            DATASET_PATH="${DATASET_AIME_2025}"
            DATASET_SUFFIX="aime2025"
            ;;
        aime2024)
            DATASET_PATH="${DATASET_AIME_2024}"
            DATASET_SUFFIX="aime2024"
            ;;
        hmmt2025)
            DATASET_PATH="${DATASET_HMMT_2025}"
            DATASET_SUFFIX="hmmt2025"
            ;;
        bruno2025)
            DATASET_PATH="${DATASET_BRUNO_2025}"
            DATASET_SUFFIX="bruno2025"
            ;;
        *)
            echo "Error: Unknown dataset '$dataset'"
            echo "Available datasets: aime2025, aime2024, hmmt2025, bruno2025, all"
            exit 1
            ;;
    esac
}

# ============================================================
# Run inference for a single dataset
# ============================================================
run_single() {
    local dataset=$1

    get_dataset_config "$dataset"

    local RID="${RID_PREFIX}_${DATASET_SUFFIX}_${DEFAULT_BUDGET}"

    echo "========================================="
    echo "Model: ${MODEL_HF}"
    echo "Dataset: ${DATASET_SUFFIX}"
    echo "========================================="
    echo "Temperature: ${TEMPERATURE}"
    echo "Top-p: ${TOP_P}"
    echo "Top-k: ${TOP_K}"
    echo "Max tokens: ${MAX_TOKENS}"
    echo "Budget: ${DEFAULT_BUDGET}"
    echo "Output: ${OUTPUT_BASE}"
    echo "========================================="

    python "scripts/inference/${PYTHON_SCRIPT}" \
        --dataset "${DATASET_PATH}" \
        --budget "${DEFAULT_BUDGET}" \
        --rid "${RID}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --model "${MODEL_HF}" \
        --model_type "${MODEL_TYPE}" \
        --tensor_parallel_size "${DEFAULT_TENSOR_PARALLEL_SIZE}" \
        --chunk_size "${DEFAULT_CHUNK_SIZE}" \
        --logprobs "${DEFAULT_LOGPROBS}" \
        --output_dir "${OUTPUT_BASE}" \
        ${EXTRA_FLAGS}

    echo "========================================="
    echo "${DATASET_SUFFIX} completed!"
    echo "========================================="
    echo ""
}

# ============================================================
# Main execution
# ============================================================

# Load model config
get_model_config "$MODEL"

echo "========================================================"
echo "Inference: ${MODEL} (${DEFAULT_BUDGET} traces)"
echo "========================================================"
echo ""

if [[ "$DATASET" == "all" ]]; then
    # Run all datasets in priority order
    echo "Running all datasets..."
    echo ""

    run_single "aime2025"
    run_single "bruno2025"
    run_single "hmmt2025"
    run_single "aime2024"

    echo "========================================================"
    echo "ALL DATASETS COMPLETED!"
    echo "========================================================"
else
    run_single "$DATASET"
fi

echo "Results saved to: ${OUTPUT_BASE}"
