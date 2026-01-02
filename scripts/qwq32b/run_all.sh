#!/bin/bash
# Run all 4 datasets with QwQ-32B (512 traces)
# Priority order: aime2025 -> bruno2025 -> hmmt2025 -> aime2024
#
# Usage:
#   screen -S qwq32b_inference
#   cd /path/to/sampling_credit
#   ./scripts/qwq32b/run_all.sh
#   # Detach: Ctrl+A, then D

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source centralized config
source "${SCRIPT_DIR}/../config.sh"

echo "========================================================"
echo "QwQ-32B Inference - All Datasets (512 traces)"
echo "========================================================"
echo "Settings:"
echo "  Model: ${QWQ32B_MODEL}"
echo "  Temperature: ${QWQ32B_TEMPERATURE}"
echo "  Top-p: ${QWQ32B_TOP_P}"
echo "  Top-k: ${QWQ32B_TOP_K}"
echo "  Max tokens: ${QWQ32B_MAX_TOKENS}"
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
