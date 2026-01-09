#!/bin/bash
# ============================================================
# Gemini API Inference Runner
# ============================================================
#
# Usage:
#   ./scripts/inference/run_gemini.sh --model <model> --dataset <dataset>
#   ./scripts/inference/run_gemini.sh --model <model> --dataset debug
#
# Examples:
#   ./scripts/inference/run_gemini.sh --model gemini-2.0-flash --dataset aime2025
#   ./scripts/inference/run_gemini.sh --model gemini-2.0-flash --dataset all
#   ./scripts/inference/run_gemini.sh --model gemini-2.0-flash --dataset debug
#   ./scripts/inference/run_gemini.sh --model gemini-2.0-flash --dataset debug --budget 4
#
# Available models:
#   gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
#
# Available datasets:
#   aime2025, aime2024, hmmt2025, bruno2025, all, debug
#
# Environment:
#   GOOGLE_API_KEY    Required. Your Google API key
#
# ============================================================

set -e

# Source centralized config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# ============================================================
# Parse arguments
# ============================================================
MODEL=""
DATASET=""
BUDGET=""

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
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --help|-h)
            head -28 "$0" | tail -26
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Validation
# ============================================================
if [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "ERROR: GOOGLE_API_KEY not set"
    echo "  export GOOGLE_API_KEY='AIzaSy...'"
    exit 1
fi

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
        gemini-2.0-flash)
            MODEL_ID="${GEMINI_2_0_FLASH_MODEL}"
            TEMPERATURE="${GEMINI_2_0_FLASH_TEMPERATURE}"
            TOP_P="${GEMINI_2_0_FLASH_TOP_P}"
            TOP_K="${GEMINI_2_0_FLASH_TOP_K}"
            MAX_TOKENS="${GEMINI_2_0_FLASH_MAX_TOKENS}"
            RID_PREFIX="gemini_2_0_flash"
            ;;
        gemini-2.5-flash)
            MODEL_ID="${GEMINI_2_5_FLASH_MODEL}"
            TEMPERATURE="${GEMINI_2_5_FLASH_TEMPERATURE}"
            TOP_P="${GEMINI_2_5_FLASH_TOP_P}"
            TOP_K="${GEMINI_2_5_FLASH_TOP_K}"
            MAX_TOKENS="${GEMINI_2_5_FLASH_MAX_TOKENS}"
            RID_PREFIX="gemini_2_5_flash"
            ;;
        gemini-2.5-pro)
            MODEL_ID="${GEMINI_2_5_PRO_MODEL}"
            TEMPERATURE="${GEMINI_2_5_PRO_TEMPERATURE}"
            TOP_P="${GEMINI_2_5_PRO_TOP_P}"
            TOP_K="${GEMINI_2_5_PRO_TOP_K}"
            MAX_TOKENS="${GEMINI_2_5_PRO_MAX_TOKENS}"
            RID_PREFIX="gemini_2_5_pro"
            ;;
        *)
            echo "Error: Unknown model '$model'"
            echo "Available models: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro"
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
            echo "Available datasets: aime2025, aime2024, hmmt2025, bruno2025, all, debug"
            exit 1
            ;;
    esac
}

# ============================================================
# Run inference for a single dataset
# ============================================================
run_single() {
    local dataset=$1
    local budget=$2

    get_dataset_config "$dataset"

    local RID="${RID_PREFIX}_${DATASET_SUFFIX}_${budget}"

    echo "========================================="
    echo "Model: ${MODEL_ID}"
    echo "Dataset: ${DATASET_SUFFIX}"
    echo "========================================="
    echo "Temperature: ${TEMPERATURE}"
    echo "Top-p: ${TOP_P}"
    echo "Top-k: ${TOP_K}"
    echo "Max tokens: ${MAX_TOKENS}"
    echo "Budget: ${budget}"
    echo "Output: ${OUTPUT_BASE}/${RID}"
    echo "========================================="

    python "scripts/inference/run_gemini_batch.py" \
        --model "${MODEL_ID}" \
        --dataset "${DATASET_PATH}" \
        --budget "${budget}" \
        --rid "${RID}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --logprobs "${DEFAULT_LOGPROBS}" \
        --output_dir "${OUTPUT_BASE}" \
        --save_json

    echo "========================================="
    echo "${DATASET_SUFFIX} completed!"
    echo "========================================="
    echo ""
}

# ============================================================
# Debug mode - quick test with temp file
# ============================================================
run_debug() {
    local budget=${1:-2}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local RID="${RID_PREFIX}_debug_${timestamp}"

    # Create temp test file
    local TEST_FILE=$(mktemp /tmp/gemini_debug_XXXXXX.jsonl)
    echo '{"question": "Find all positive integers n such that n^2 - 19n + 99 is a perfect square. What is the sum of all such n?", "answer": "38"}' > "$TEST_FILE"

    echo "========================================="
    echo "DEBUG MODE"
    echo "========================================="
    echo "Model: ${MODEL_ID}"
    echo "Budget: ${budget} samples"
    echo "Output: ${OUTPUT_BASE}/${RID}"
    echo "========================================="

    python "scripts/inference/run_gemini_batch.py" \
        --model "${MODEL_ID}" \
        --dataset "${TEST_FILE}" \
        --budget "${budget}" \
        --rid "${RID}" \
        --max_tokens 8192 \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --logprobs "${DEFAULT_LOGPROBS}" \
        --output_dir "${OUTPUT_BASE}" \
        --save_json \
        --qid_end 1

    rm "$TEST_FILE"

    echo ""
    echo "========================================="
    echo "DEBUG COMPLETE"
    echo "========================================="
    echo "Output: ${OUTPUT_BASE}/${RID}"
    echo ""
    echo "Verify logprobs:"
    echo "  cat ${OUTPUT_BASE}/${RID}/qid*.json | grep -o '\"mean_confidence\":[^,]*' | head -3"
    echo ""
    echo "If mean_confidence is null, confidence-based voting won't work."
    echo "========================================="
}

# ============================================================
# Main execution
# ============================================================

# Load model config
get_model_config "$MODEL"

# Set budget (default 512 for full run, 2 for debug)
if [[ "$DATASET" == "debug" ]]; then
    BUDGET=${BUDGET:-2}
else
    BUDGET=${BUDGET:-${DEFAULT_BUDGET}}
fi

echo "========================================================"
echo "Gemini Inference: ${MODEL} (${BUDGET} traces)"
echo "========================================================"
echo ""

if [[ "$DATASET" == "debug" ]]; then
    run_debug "$BUDGET"
elif [[ "$DATASET" == "all" ]]; then
    echo "Running all datasets..."
    echo ""

    run_single "aime2025" "$BUDGET"
    run_single "bruno2025" "$BUDGET"
    run_single "hmmt2025" "$BUDGET"
    run_single "aime2024" "$BUDGET"

    echo "========================================================"
    echo "ALL DATASETS COMPLETED!"
    echo "========================================================"
else
    run_single "$DATASET" "$BUDGET"
fi

echo "Results saved to: ${OUTPUT_BASE}"
