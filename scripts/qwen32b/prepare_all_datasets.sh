#!/bin/bash
# Prepare all 4 datasets for Qwen3-32B experiments
# Run this BEFORE running inference
#
# Usage:
#   cd /path/to/sampling_credit
#   ./scripts/qwen32b/prepare_all_datasets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source centralized config
source "${SCRIPT_DIR}/../config.sh"

echo "========================================================"
echo "Preparing All Datasets"
echo "========================================================"
echo "Output directory: ${DATASETS_BASE}"
echo "========================================================"
echo ""

# Create main datasets directory
mkdir -p "${DATASETS_BASE}"

echo "[1/4] Preparing AIME 2025..."
python scripts/prepare_data/prepare_aime2025.py
echo ""

echo "[2/4] Preparing HMMT 2025..."
python scripts/prepare_data/prepare_hmmt2025.py
echo ""

echo "[3/4] Preparing AIME 2024..."
python scripts/prepare_data/prepare_aime2024.py
echo ""

echo "[4/4] Preparing Bruno 2025..."
python scripts/prepare_data/prepare_bruno2025.py
echo ""

echo "========================================================"
echo "ALL DATASETS PREPARED!"
echo "========================================================"
echo "Datasets saved to:"
echo "  ${DATASET_AIME_2025}"
echo "  ${DATASET_HMMT_2025}"
echo "  ${DATASET_AIME_2024}"
echo "  ${DATASET_BRUNO_2025}"
echo "========================================================"
