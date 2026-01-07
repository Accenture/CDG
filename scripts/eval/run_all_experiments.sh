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
#   ./run_all_experiments.sh --exp 3            # Run only Exp 3 (hyperparam)
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
RUN_EXP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --exp) RUN_EXP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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

    # 3a. Sweep position_pct (10 vs 20 vs 30) with 256 traces
    echo "--- 3a. Position PCT sweep ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep position_pct \
        --trace_count 256 \
        --output "$OUTPUT_DIR/hyperparam_position_pct_${TIMESTAMP}.json" \
        --no-show

    # 3b. Beta sweep with alpha=0.5, position_pct=10
    echo "--- 3b. Beta sweep (alpha=0.5, position_pct=10) ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 256 \
        --alphas 0.5 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 10 \
        --output "$OUTPUT_DIR/hyperparam_beta_a0.5_pct10_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_beta_a0.5_pct10_${TIMESTAMP}.png" \
        --no-show

    # 3c. Beta sweep with alpha=0.5, position_pct=20
    echo "--- 3c. Beta sweep (alpha=0.5, position_pct=20) ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 256 \
        --alphas 0.5 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 20 \
        --output "$OUTPUT_DIR/hyperparam_beta_a0.5_pct20_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_beta_a0.5_pct20_${TIMESTAMP}.png" \
        --no-show

    # 3d. Beta sweep with alpha=1.0, position_pct=10
    echo "--- 3d. Beta sweep (alpha=1.0, position_pct=10) ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 256 \
        --alphas 1.0 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 10 \
        --output "$OUTPUT_DIR/hyperparam_beta_a1.0_pct10_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_beta_a1.0_pct10_${TIMESTAMP}.png" \
        --no-show

    # 3e. Beta sweep with alpha=1.0, position_pct=20
    echo "--- 3e. Beta sweep (alpha=1.0, position_pct=20) ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 256 \
        --alphas 1.0 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 20 \
        --output "$OUTPUT_DIR/hyperparam_beta_a1.0_pct20_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_beta_a1.0_pct20_${TIMESTAMP}.png" \
        --no-show

    # 3f. Combined figure: both alphas
    echo "--- 3f. Combined beta sweep (both alphas) ---"
    run_cmd python scripts/eval/eval_hyperparam.py \
        --sweep beta \
        --trace_count 256 \
        --alphas 0.5,1.0 \
        --beta_min 10 --beta_max 50 --beta_step 5 \
        --position_pct 20 \
        --output "$OUTPUT_DIR/hyperparam_beta_combined_${TIMESTAMP}.json" \
        --figure "$OUTPUT_DIR/hyperparam_beta_combined_${TIMESTAMP}.png" \
        --no-show

    echo "Exp 3 complete. Check $OUTPUT_DIR for results."
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

    run_cmd python scripts/eval/eval_subsample.py \
        --output "$OUTPUT_DIR/subsample_scaling_${TIMESTAMP}.png" \
        --no-show

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

    # Note: Update alpha/beta based on Exp 3 results
    # Default params - modify after tuning
    ALPHA=0.5
    BETA=10
    POSITION_PCT=20

    echo "Using parameters: alpha=$ALPHA, beta=$BETA, position_pct=$POSITION_PCT"
    echo "(Update these based on Exp 3 hyperparameter tuning results)"
    echo ""

    run_cmd python scripts/eval/eval_voting.py \
        --all --all-methods \
        --alpha $ALPHA --beta $BETA --position_pct $POSITION_PCT \
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
