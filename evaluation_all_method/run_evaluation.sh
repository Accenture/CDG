#!/bin/bash
#
# Shell script to run voting methods evaluation
#
# Usage examples:
#   ./run_evaluation.sh                    # Run on all default datasets
#   ./run_evaluation.sh --alpha 0.3        # Run with alpha=0.3
#   ./run_evaluation.sh --detailed         # Print detailed results
#   ./run_evaluation.sh --custom /path/to/data  # Run on custom dataset

# Default parameters
ALPHA=0.5
BETA=10
POSITION_PCT=20
TAIL_WINDOW=2048
BOTTOM_WINDOW=2048
BOTTOM_PCT=0.1
DETAILED=""

# Default data directories
DEFAULT_DATA_DIRS=(
    "/mnt/aime_2025_run_512_results_gptoss20b/aime2025run3qgptoss20b"
    "/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_backups/qwq32b_aime2025_512"
    "/mnt/aime_2025_run_512_results_deepseek_logprob10/aime2025run3qdeepseek"
    "/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_backups/tempt/resultsforqwen/qwen32b_aime2025_512"
)

CUSTOM_DIRS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --position_pct)
            POSITION_PCT="$2"
            shift 2
            ;;
        --tail_window)
            TAIL_WINDOW="$2"
            shift 2
            ;;
        --bottom_window)
            BOTTOM_WINDOW="$2"
            shift 2
            ;;
        --bottom_pct)
            BOTTOM_PCT="$2"
            shift 2
            ;;
        --detailed)
            DETAILED="--detailed"
            shift
            ;;
        --custom)
            CUSTOM_DIRS+=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --alpha FLOAT        CDG count dampening exponent (default: 0.5)"
            echo "  --beta FLOAT         CDG gradient weight (default: 10)"
            echo "  --position_pct INT   Percentile for gradient computation (default: 20)"
            echo "  --tail_window INT    Window size for tail confidence (default: 2048)"
            echo "  --bottom_window INT  Window size for bottom window method (default: 2048)"
            echo "  --bottom_pct FLOAT   Bottom percentile for bottom window (default: 0.1)"
            echo "  --detailed           Print detailed results per dataset"
            echo "  --custom PATH        Add custom data directory (can be used multiple times)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run on all default datasets"
            echo "  $0 --alpha 0.3 --beta 15    # Custom alpha and beta"
            echo "  $0 --custom /path/to/data   # Run on custom dataset"
            echo "  $0 --detailed               # Print detailed results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use custom dirs if provided, otherwise use defaults
if [ ${#CUSTOM_DIRS[@]} -gt 0 ]; then
    DATA_DIRS=("${CUSTOM_DIRS[@]}")
else
    DATA_DIRS=("${DEFAULT_DATA_DIRS[@]}")
fi

# Run the evaluation script using the deepconf environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/eph/nvme0/env/deepconf/bin/python"

# Fallback to system python if env not found
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_voting_methods.py" \
    --data_dirs "${DATA_DIRS[@]}" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --position_pct "$POSITION_PCT" \
    --tail_window "$TAIL_WINDOW" \
    --bottom_window "$BOTTOM_WINDOW" \
    --bottom_pct "$BOTTOM_PCT" \
    $DETAILED
