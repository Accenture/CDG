#!/bin/bash
# Run all 4 datasets with Qwen3-32B (512 traces, top-k=20)
# Priority order: aime2025 -> bruno2025 -> hmmt2025 -> aime2024
#
# Usage:
#   screen -S qwen32b_inference
#   cd /path/to/sampling_credit
#   ./scripts/qwen32b/run_all.sh
#   # Detach: Ctrl+A, then D

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source centralized config
source "${SCRIPT_DIR}/../config.sh"

echo "========================================================"
echo "Qwen3-32B Inference - All Datasets (512 traces)"
echo "========================================================"
echo "Settings (per DeepConf paper Table 11):"
echo "  Model: ${QWEN32B_MODEL}"
echo "  Temperature: ${QWEN32B_TEMPERATURE}"
echo "  Top-p: ${QWEN32B_TOP_P}"
echo "  Top-k: ${QWEN32B_TOP_K}"
echo "  Max tokens: ${QWEN32B_MAX_TOKENS}"
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
