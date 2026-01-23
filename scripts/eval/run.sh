#!/bin/bash
# ============================================================
# Unified Eval Runner
# ============================================================
#
# Usage:
#   ./scripts/eval/run.sh --task <task>
#
# Examples:
#   ./scripts/eval/run.sh --task all           # Run everything
#   ./scripts/eval/run.sh --task cache         # Generate all caches
#   ./scripts/eval/run.sh --task figures       # Generate all figures
#   ./scripts/eval/run.sh --task sweep         # Run CDG hyperparam sweep
#
# Available tasks:
#   cache     - Generate caches (voting, histogram, sweep)
#   figures   - Generate all figures (requires caches)
#   sweep     - Run CDG hyperparameter sweep only
#   position  - Run position ablation experiment
#   all       - Run everything (cache + figures)
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Go to repo root

# ============================================================
# Parse arguments
# ============================================================
TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
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

if [[ -z "$TASK" ]]; then
    echo "Error: --task is required"
    echo "Usage: $0 --task <task>"
    echo "Run with --help for more info"
    exit 1
fi

# ============================================================
# Task functions
# ============================================================

run_voting_cache() {
    echo "========================================"
    echo "Running: compare_voting_methods.py"
    echo "========================================"
    python scripts/eval/compare_voting_methods.py
    echo ""
}

run_histogram_cache() {
    echo "========================================"
    echo "Running: plot_distributions.py (histogram cache)"
    echo "========================================"
    python scripts/eval/plot_distributions.py
    echo ""
}

run_sweep() {
    echo "========================================"
    echo "Running: cdg_hyperparam_sweep.py"
    echo "========================================"
    python scripts/eval/cdg_hyperparam_sweep.py
    echo ""
}

run_position_sweep() {
    echo "========================================"
    echo "Running: figure_position_appendix.py (position sweep)"
    echo "========================================"
    python scripts/eval/figure_position_appendix.py
    echo ""
}

run_figure_scaling() {
    echo "========================================"
    echo "Generating: figure_scaling.py"
    echo "========================================"
    python scripts/eval/figure_scaling.py
    echo ""
}

run_figure_scaling_appendix() {
    echo "========================================"
    echo "Generating: figure_scaling_appendix.py"
    echo "========================================"
    python scripts/eval/figure_scaling_appendix.py
    echo ""
}

run_figure_histogram() {
    echo "========================================"
    echo "Generating: figure_histogram.py"
    echo "========================================"
    python scripts/eval/figure_histogram.py
    echo ""
}

run_figure_confidence_curve() {
    echo "========================================"
    echo "Generating: figure_confidence_curve.py"
    echo "========================================"
    python scripts/eval/figure_confidence_curve.py
    echo ""
}

run_figure_position_appendix() {
    echo "========================================"
    echo "Generating: figure_position_appendix.py (from cache)"
    echo "========================================"
    python scripts/eval/figure_position_appendix.py --no-compute
    echo ""
}

# ============================================================
# Main execution
# ============================================================

echo "========================================================"
echo "Eval Runner: $TASK"
echo "========================================================"
echo ""

case $TASK in
    cache)
        echo "Generating all caches..."
        run_voting_cache
        run_histogram_cache
        run_sweep
        echo "========================================================"
        echo "ALL CACHES GENERATED!"
        echo "========================================================"
        ;;

    figures)
        echo "Generating all figures (from cache)..."
        run_figure_scaling
        run_figure_scaling_appendix
        run_figure_histogram
        run_figure_confidence_curve
        echo "========================================================"
        echo "ALL FIGURES GENERATED!"
        echo "========================================================"
        ;;

    sweep)
        run_sweep
        ;;

    position)
        run_position_sweep
        ;;

    voting)
        run_voting_cache
        ;;

    histogram)
        run_histogram_cache
        run_figure_histogram
        ;;

    all)
        echo "Running full pipeline..."
        echo ""

        # Step 1: Generate caches
        echo "[1/5] Generating voting methods cache..."
        run_voting_cache

        echo "[2/5] Generating histogram cache..."
        run_histogram_cache

        echo "[3/5] Running CDG hyperparameter sweep..."
        run_sweep

        echo "[4/5] Running position ablation..."
        run_position_sweep

        # Step 2: Generate figures
        echo "[5/5] Generating all figures..."
        run_figure_scaling
        run_figure_scaling_appendix
        run_figure_histogram
        run_figure_confidence_curve
        run_figure_position_appendix

        echo "========================================================"
        echo "FULL PIPELINE COMPLETED!"
        echo "========================================================"
        echo ""
        echo "Output directories:"
        echo "  results/exp2_baseline_vs_cdg/   (voting cache)"
        echo "  results/exp3_cdg_sweep/         (sweep cache)"
        echo "  results/exp_histogram/cache/    (histogram cache)"
        echo "  results/figures/                (main figures)"
        echo "  results/figures/appendix/       (appendix figures)"
        ;;

    *)
        echo "Error: Unknown task '$TASK'"
        echo "Available tasks: cache, figures, sweep, position, voting, histogram, all"
        exit 1
        ;;
esac

echo ""
echo "Done!"
