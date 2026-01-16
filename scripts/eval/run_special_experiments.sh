#!/bin/bash
# Special experiments with manually tuned parameters
#
# Exp 1-2: Model-specific tuned parameters (best found manually)
# Exp 1-2-2: Model-specific tuned parameters (1/15 2PM verified optimal)
# Exp 1-3-1: General params (alpha=0.5, beta=10, position_pct=20) for all models
# Exp 1-3-2: General params (alpha=0.5, beta=10, position_pct=10) for all models
# Exp 1-4: Beta=0 ablation (no gradient signal, expected worse results)
#
# Usage:
#   ./run_special_experiments.sh                    # Run all special experiments
#   ./run_special_experiments.sh --exp 1-2         # Run only Exp 1-2
#   ./run_special_experiments.sh --exp 1-2-2       # Run only Exp 1-2-2 (1/15 2PM verified)
#   ./run_special_experiments.sh --exp 1-3-1       # Run only Exp 1-3-1
#   ./run_special_experiments.sh --exp 1-3-2       # Run only Exp 1-3-2
#   ./run_special_experiments.sh --exp 1-4         # Run only Exp 1-4 (beta=0 ablation)
#   ./run_special_experiments.sh --dry-run         # Show commands without running

set -e

# Unbuffered Python output (so tee shows results in real-time)
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Output directory for results
OUTPUT_DIR="results/eval_outputs/special"
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
# Exp 1-2: Model-Specific Tuned Parameters
# ============================================================
# Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b
# Excluded: gemini_2_0_flash, gptoss20b_nohighreason, qwen32b
run_exp_1_2() {
    echo ""
    echo "########################################"
    echo "# EXP 1-2: MODEL-SPECIFIC TUNED PARAMS"
    echo "########################################"
    echo ""
    echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
    echo "Parameters per model:"
    echo "  DeepSeek:  alpha=0.5, beta=10, position_pct=20"
    echo "  Gptoss20b: alpha=0.5, beta=10, position_pct=10 (Bruno: beta=20, pos=20)"
    echo "  Gemma3:    alpha=0.5, beta=5,  position_pct=10 AND 20"
    echo "  QwQ:       alpha=0.5, beta=10, position_pct=20"
    echo ""

    OUTPUT_FILE="$OUTPUT_DIR/exp1-2_model_specific_${TIMESTAMP}.txt"

    {
        echo "=============================================="
        echo "EXP 1-2: MODEL-SPECIFIC TUNED PARAMETERS"
        echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
        echo "Timestamp: $TIMESTAMP"
        echo "=============================================="
        echo ""

        # DeepSeek: alpha=0.5, beta=10, position_pct=20
        echo "--- DeepSeek (alpha=0.5, beta=10, pos=20) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "deepseek8b*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 20

        # Gptoss20b (non-Bruno, excluding nohighreason): alpha=0.5, beta=10, position_pct=10
        echo "--- Gptoss20b non-Bruno (alpha=0.5, beta=10, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gptoss20b_aime*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 10

        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gptoss20b_hmmt*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 10

        # Gptoss20b Bruno: alpha=0.5, beta=20, position_pct=20
        echo "--- Gptoss20b Bruno (alpha=0.5, beta=20, pos=20) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gptoss20b_bruno*" \
            --method cdg \
            --alpha 0.5 --beta 20 --position_pct 20

        # Gemma3 with position_pct=10
        echo "--- Gemma3 (alpha=0.5, beta=5, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gemma3_27b*" \
            --method cdg \
            --alpha 0.5 --beta 5 --position_pct 10

        # Gemma3 with position_pct=20 (for comparison)
        echo "--- Gemma3 (alpha=0.5, beta=5, pos=20) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gemma3_27b*" \
            --method cdg \
            --alpha 0.5 --beta 5 --position_pct 20

        # QwQ: alpha=0.5, beta=10, position_pct=20
        echo "--- QwQ (alpha=0.5, beta=10, pos=20) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "qwq32b*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 20

    } 2>&1 | tee "$OUTPUT_FILE"

    echo "Exp 1-2 complete. Results saved to: $OUTPUT_FILE"
}

# ============================================================
# Exp 1-2-2: Model-Specific Tuned Parameters (1/15 2PM verified optimal)
# ============================================================
# All models use alpha=0.5, position_pct=10
# DeepSeek & Gptoss: beta=10
# Gemma3 & QwQ: beta=3
run_exp_1_2_2() {
    echo ""
    echo "########################################"
    echo "# EXP 1-2-2: VERIFIED OPTIMAL PARAMS"
    echo "# (1/15 2PM verified)"
    echo "########################################"
    echo ""
    echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
    echo "Parameters per model (all alpha=0.5, position_pct=10):"
    echo "  DeepSeek:  beta=10"
    echo "  Gptoss20b: beta=10"
    echo "  Gemma3:    beta=3"
    echo "  QwQ:       beta=3"
    echo ""

    OUTPUT_FILE="$OUTPUT_DIR/exp1-2-2_verified_optimal_${TIMESTAMP}.txt"

    {
        echo "=============================================="
        echo "EXP 1-2-2: VERIFIED OPTIMAL PARAMETERS"
        echo "(1/15 2PM verified)"
        echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
        echo "Timestamp: $TIMESTAMP"
        echo "=============================================="
        echo ""

        # DeepSeek: alpha=0.5, beta=10, position_pct=10
        echo "--- DeepSeek (alpha=0.5, beta=10, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "deepseek8b*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 10

        # Gptoss20b: alpha=0.5, beta=10, position_pct=10
        echo "--- Gptoss20b (alpha=0.5, beta=10, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --patterns "gptoss20b_aime*,gptoss20b_bruno*,gptoss20b_hmmt*" \
            --method cdg \
            --alpha 0.5 --beta 10 --position_pct 10

        # Gemma3: alpha=0.5, beta=3, position_pct=10
        echo "--- Gemma3 (alpha=0.5, beta=3, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "gemma3_27b*" \
            --method cdg \
            --alpha 0.5 --beta 3 --position_pct 10

        # QwQ: alpha=0.5, beta=3, position_pct=10
        echo "--- QwQ (alpha=0.5, beta=3, pos=10) ---"
        run_cmd python scripts/eval/eval_voting.py \
            --pattern "qwq32b*" \
            --method cdg \
            --alpha 0.5 --beta 3 --position_pct 10

    } 2>&1 | tee "$OUTPUT_FILE"

    echo "Exp 1-2-2 complete. Results saved to: $OUTPUT_FILE"
}

# ============================================================
# Exp 1-3-1: General Params (position_pct=20)
# ============================================================
# Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b
MODEL_PATTERNS="deepseek8b*,gemma3_27b*,qwq32b*,gptoss20b_aime*,gptoss20b_bruno*,gptoss20b_hmmt*"

run_exp_1_3_1() {
    echo ""
    echo "########################################"
    echo "# EXP 1-3-1: GENERAL PARAMS (pos=20)"
    echo "########################################"
    echo ""
    echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
    echo "All models: alpha=0.5, beta=10, position_pct=20"
    echo ""

    OUTPUT_FILE="$OUTPUT_DIR/exp1-3-1_general_pos20_${TIMESTAMP}.txt"

    {
        echo "=============================================="
        echo "EXP 1-3-1: GENERAL PARAMS (position_pct=20)"
        echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
        echo "All models: alpha=0.5, beta=10, position_pct=20"
        echo "Timestamp: $TIMESTAMP"
        echo "=============================================="
        echo ""

        run_cmd python scripts/eval/eval_voting.py \
            --patterns "$MODEL_PATTERNS" --all-methods \
            --alpha 0.5 --beta 10 --position_pct 20

    } 2>&1 | tee "$OUTPUT_FILE"

    echo "Exp 1-3-1 complete. Results saved to: $OUTPUT_FILE"
}

# ============================================================
# Exp 1-3-2: General Params (position_pct=10)
# ============================================================
run_exp_1_3_2() {
    echo ""
    echo "########################################"
    echo "# EXP 1-3-2: GENERAL PARAMS (pos=10)"
    echo "########################################"
    echo ""
    echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
    echo "All models: alpha=0.5, beta=10, position_pct=10"
    echo ""

    OUTPUT_FILE="$OUTPUT_DIR/exp1-3-2_general_pos10_${TIMESTAMP}.txt"

    {
        echo "=============================================="
        echo "EXP 1-3-2: GENERAL PARAMS (position_pct=10)"
        echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
        echo "All models: alpha=0.5, beta=10, position_pct=10"
        echo "Timestamp: $TIMESTAMP"
        echo "=============================================="
        echo ""

        run_cmd python scripts/eval/eval_voting.py \
            --patterns "$MODEL_PATTERNS" --all-methods \
            --alpha 0.5 --beta 10 --position_pct 10

    } 2>&1 | tee "$OUTPUT_FILE"

    echo "Exp 1-3-2 complete. Results saved to: $OUTPUT_FILE"
}

# ============================================================
# Exp 1-4: Beta=0 Ablation (no gradient signal)
# ============================================================
# This removes the gradient component, leaving only count-dampened mean confidence
# Expected: worse results than with beta > 0
run_exp_1_4() {
    echo ""
    echo "########################################"
    echo "# EXP 1-4: BETA=0 ABLATION"
    echo "########################################"
    echo ""
    echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
    echo "Testing: alpha=0.5, beta=0, position_pct=10 AND 20"
    echo "Expected: worse results (no gradient signal)"
    echo ""

    OUTPUT_FILE="$OUTPUT_DIR/exp1-4_beta0_ablation_${TIMESTAMP}.txt"

    {
        echo "=============================================="
        echo "EXP 1-4: BETA=0 ABLATION (no gradient signal)"
        echo "Models: deepseek8b, gemma3_27b, qwq32b, gptoss20b"
        echo "Expected: worse results without gradient component"
        echo "Timestamp: $TIMESTAMP"
        echo "=============================================="
        echo ""

        # Beta=0, position_pct=10
        echo "--- Beta=0, position_pct=10 ---"
        run_cmd python scripts/eval/eval_voting.py \
            --patterns "$MODEL_PATTERNS" --all-methods \
            --alpha 0.5 --beta 0 --position_pct 10

        # Beta=0, position_pct=20
        echo "--- Beta=0, position_pct=20 ---"
        run_cmd python scripts/eval/eval_voting.py \
            --patterns "$MODEL_PATTERNS" --all-methods \
            --alpha 0.5 --beta 0 --position_pct 20

    } 2>&1 | tee "$OUTPUT_FILE"

    echo "Exp 1-4 complete. Results saved to: $OUTPUT_FILE"
}

# ============================================================
# Main execution
# ============================================================

echo "========================================"
echo "SPECIAL EXPERIMENTS RUNNER"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Output dir: $OUTPUT_DIR"
echo ""

if [[ -n "$RUN_EXP" ]]; then
    case $RUN_EXP in
        1-2) run_exp_1_2 ;;
        1-2-2) run_exp_1_2_2 ;;
        1-3-1) run_exp_1_3_1 ;;
        1-3-2) run_exp_1_3_2 ;;
        1-4) run_exp_1_4 ;;
        *) echo "Unknown experiment: $RUN_EXP (valid: 1-2, 1-2-2, 1-3-1, 1-3-2, 1-4)"; exit 1 ;;
    esac
else
    # Run all special experiments
    echo "Running all special experiments: 1-2, 1-2-2, 1-3-1, 1-3-2, 1-4"
    echo ""
    run_exp_1_2
    run_exp_1_2_2
    run_exp_1_3_1
    run_exp_1_3_2
    run_exp_1_4
fi

echo ""
echo "========================================"
echo "ALL SPECIAL EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
