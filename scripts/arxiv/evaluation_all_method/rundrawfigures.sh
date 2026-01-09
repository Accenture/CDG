#!/bin/bash
#
# Shell script to draw confidence vs position figures
#
# Usage examples:
#   ./rundrawfigures.sh --data_dir /path/to/data
#   ./rundrawfigures.sh --data_dir /path/to/data --output_dir /path/to/output
#   ./rundrawfigures.sh --data_dir /path/to/data --num_bins 20

# Default parameters
DATA_DIR=""
OUTPUT_DIR=""
NUM_BINS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_bins)
            NUM_BINS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --data_dir PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --data_dir PATH      Directory containing pickle files with trace data"
            echo ""
            echo "Options:"
            echo "  --output_dir PATH    Directory to save output figure (defaults to data_dir)"
            echo "  --num_bins INT       Number of bins to divide traces into (default: 10)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --data_dir /mnt/results/qwq32b_aime2025_512"
            echo "  $0 --data_dir /mnt/results/deepseek --output_dir ./figures"
            echo "  $0 --data_dir /mnt/results/gptoss --num_bins 20"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required argument
if [ -z "$DATA_DIR" ]; then
    echo "Error: --data_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Build command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/eph/nvme0/env/deepconf/bin/python"

# Fallback to system python if env not found
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

CMD="$PYTHON_BIN $SCRIPT_DIR/draw_confidence_position.py --data_dir $DATA_DIR --num_bins $NUM_BINS"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run the script
echo "Running: $CMD"
echo ""
$CMD
