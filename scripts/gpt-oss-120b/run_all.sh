#!/bin/bash
# Run all 4 datasets with GPT-OSS-120B (512 traces)
# Priority order: aime2025 -> bruno2025 -> hmmt2025 -> aime2024
#
# Requirements:
#   # Install special vLLM version for GPT-OSS
#   uv pip install --pre vllm==0.10.1+gptoss \
#       --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
#       --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
#       --index-strategy unsafe-best-match
#
#   # Install harmony package
#   uv pip install openai-harmony
#
# Usage:
#   screen -S gptoss120b_inference
#   cd /path/to/sampling_credit
#   ./scripts/gpt-oss-120b/run_all.sh
#   # Detach: Ctrl+A, then D

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source centralized config
source "${SCRIPT_DIR}/../config.sh"

echo "========================================================"
echo "GPT-OSS-120B Inference - All Datasets (512 traces)"
echo "========================================================"
echo "Settings (per DeepConf paper Table 11):"
echo "  Model: ${GPTOSS120B_MODEL}"
echo "  Temperature: ${GPTOSS120B_TEMPERATURE}"
echo "  Top-p: ${GPTOSS120B_TOP_P}"
echo "  Top-k: ${GPTOSS120B_TOP_K}"
echo "  Max tokens: ${GPTOSS120B_MAX_TOKENS}"
echo "  Reasoning effort: ${GPTOSS120B_REASONING_EFFORT}"
echo "========================================================"
echo ""
echo "Datasets (in priority order):"
echo "  1. AIME 2025: ${DATASET_AIME_2025}"
echo "  2. Bruno 2025: ${DATASET_BRUNO_2025}"
echo "  3. HMMT 2025: ${DATASET_HMMT_2025}"
echo "  4. AIME 2024: ${DATASET_AIME_2024}"
echo ""
echo "Output: ${OUTPUT_BASE}"
echo "========================================================"
echo ""

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/*.sh

# Run datasets in priority order
echo "[1/4] Starting AIME 2025..."
"$SCRIPT_DIR/run_aime2025.sh"
echo ""

echo "[2/4] Starting Bruno 2025..."
"$SCRIPT_DIR/run_bruno2025.sh"
echo ""

echo "[3/4] Starting HMMT 2025..."
"$SCRIPT_DIR/run_hmmt2025.sh"
echo ""

echo "[4/4] Starting AIME 2024..."
"$SCRIPT_DIR/run_aime2024.sh"
echo ""

echo "========================================================"
echo "ALL DATASETS COMPLETED!"
echo "========================================================"
echo "Results saved to: ${OUTPUT_BASE}"
echo "========================================================"
