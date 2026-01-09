#!/bin/bash
# Master script to run all evaluation experiments
#
# Experiments:
#   1. Main: All models, all methods (after tuning)
#   2. Subsample: Accuracy vs trace count scaling
#   3. Hyperparam: Alpha/Beta tuning with position_pct sweep
#
# Usage:
#   ./run_all_experiments.sh                    # Run all experiments
#   ./run_all_experiments.sh --fast             # Fast debug mode (inaccurate results)
#   ./run_all_experiments.sh --exp 3            # Run only Exp 3 (hyperparam)
#   ./run_all_experiments.sh --exp 3 --fast     # Fast debug Exp 3 only
#   ./run_all_experiments.sh --exp 3 --dry-run  # Show commands without running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Output directory for results
OUTPUT_DIR="results/eval_outputs"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
DRY_RUN=false
FAST_MODE=false
RUN_EXP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --fast) FAST_MODE=true; shift ;;
        --exp) RUN_EXP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Fast mode flag for Python scripts
FAST_FLAG=""
if $FAST_MODE; then
    FAST_FLAG="--fast"
    echo ""
    echo "############################################"
    echo "#  FAST DEBUG MODE - Results inaccurate!  #"
    echo "############################################"
    echo ""
fi

run_cmd() {
    echo "=========================================="
    echo "Running: $@"
    echo "=========================================="
    if $DRY_RUN; then
        echo "[DRY-RUN] Would execute: $@"
    else
        "$@"
    fi
    echo ""
}

# ============================================================
# Experiment 3: Hyperparameter Tuning (RUN FIRST)
# ============================================================
run_exp3() {
    echo ""
    echo "########################################"
    echo "# EXP 3: HYPERPARAMETER TUNING"
    echo "########################################"
    echo ""

    # Full sweep: all alpha/beta combinations across all models
    # Using 512 traces (full runs) for more accurate tuning
    echo "--- Running comprehensive beta sweep ---"
    echo "Alpha values: 0.5, 1.0"
    echo "Beta range: 10-50 (step 5)"
    echo "Position PCT: 20"
    echo ""

    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 512 \
        --alphas 0.5,1.0 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 20 \
        --output "$OUTPUT_DIR/hyperparam_sweep_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_sweep_${TIMESTAMP}.png" \
        --save-config \
        --no-show \
        $FAST_FLAG

    echo ""
    echo "--- Saved optimal hyperparameters per model ---"
    run_cmd python scripts/eval/eval_hyperparam.py --show-config

    echo ""
    echo "Exp 3 complete. Optimal hyperparameters saved to cdg_hyperparams.json"
}

# ============================================================
# Experiment 2: Subsample Evaluation (Scaling Analysis)
# ============================================================
run_exp2() {
    echo ""
    echo "########################################"
    echo "# EXP 2: SUBSAMPLE SCALING ANALYSIS"
    echo "########################################"
    echo ""

    echo "Using model-specific tuned hyperparameters from Exp 3"
    echo ""

    run_cmd python scripts/eval/eval_subsample.py \
        --use-tuned-params \
        --output "$OUTPUT_DIR/subsample_scaling_${TIMESTAMP}.png" \
        --no-show \
        $FAST_FLAG

    echo "Exp 2 complete."
}

# ============================================================
# Experiment 1: Main Evaluation (All Models, All Methods)
# ============================================================
run_exp1() {
    echo ""
    echo "########################################"
    echo "# EXP 1: MAIN EVALUATION"
    echo "########################################"
    echo ""

    echo "Using model-specific tuned hyperparameters from Exp 3"
    echo ""

    run_cmd python scripts/eval/eval_voting.py \
        --all --all-methods \
        --use-tuned-params \
        $FAST_FLAG \
        2>&1 | tee "$OUTPUT_DIR/main_eval_${TIMESTAMP}.txt"

    echo "Exp 1 complete."
}

# ============================================================
# Main execution
# ============================================================

echo "========================================"
echo "EVALUATION EXPERIMENT RUNNER"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Output dir: $OUTPUT_DIR"
echo ""

if [[ -n "$RUN_EXP" ]]; then
    case $RUN_EXP in
        1) run_exp1 ;;
        2) run_exp2 ;;
        3) run_exp3 ;;
        *) echo "Unknown experiment: $RUN_EXP"; exit 1 ;;
    esac
else
    # Run in recommended order: Exp3 -> Exp2 -> Exp1
    echo "Running all experiments in order: Exp3 (tuning) -> Exp2 (scaling) -> Exp1 (main)"
    echo ""
    run_exp3
    run_exp2
    run_exp1
fi

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
