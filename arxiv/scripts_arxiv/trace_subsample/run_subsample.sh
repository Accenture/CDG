#!/bin/bash
# Run trace subsampling for all qwen32b experiments
#
# Usage:
#   cd /path/to/sampling_credit
#   ./scripts/trace_subsample/run_subsample.sh
#
#   # Dry run first to see what will be created
#   ./scripts/trace_subsample/run_subsample.sh --dry_run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

echo "========================================"
echo "Trace Subsampling"
echo "========================================"
echo "Working directory: $(pwd)"
echo "Arguments: $@"
echo "========================================"

python scripts/trace_subsample/subsample.py "$@"

echo "========================================"
echo "Subsampling completed!"
echo "========================================"
