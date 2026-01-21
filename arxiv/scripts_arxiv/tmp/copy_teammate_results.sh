#!/bin/bash
# Copy teammate's results to main results directory with proper naming
#
# Sources:
#   - deepseek/         -> deepseek8b results
#   - gptoss_NO_high_reason_topk_40/ -> gptoss no high reason results
#   - resultsforGEMMA/  -> gemma3_27b results
#
# Usage:
#   ./copy_teammate_results.sh          # dry-run (show what would be copied)
#   ./copy_teammate_results.sh --run    # actually copy

set -e

# Source base directory
SRC_BASE="/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper"

# Target directory
TARGET_DIR="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"

# Mapping: source_subdir -> target_name
declare -A MAPPINGS=(
    # ===== DeepSeek 8B results =====
    ["deepseek/aime_2024_run_512_results_deepseek8b/aime2024run3qdeepseek8b"]="deepseek8b_aime2024_512"
    ["deepseek/aime_2025_run_512_results_deepseek8b/aime2025run3qdeepseek8b"]="deepseek8b_aime2025_512"
    ["deepseek/bruno_2025_run_512_results_deepseek8b/bruno2025run3qdeepseek8b"]="deepseek8b_bruno2025_512"
    ["deepseek/hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b"]="deepseek8b_hmmt2025_512"

    # ===== GPTOSS 20B (NO high reason) results =====
    ["gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason/aime2025run3qgptoss20b_topk_40"]="gptoss20b_nohighreason_aime2025_512"
    ["gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason/aime2024run3qgptoss20b_topk_40"]="gptoss20b_nohighreason_aime2024_512"
    ["gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason/bruno2025run3qgptoss20b_topk_40"]="gptoss20b_nohighreason_bruno2025_512"
    ["gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason/hmmt2025run3qgptoss20b_topk_40"]="gptoss20b_nohighreason_hmmt2025_512"

    # ===== Gemma 3 27B results =====
    ["resultsforGEMMA/gemma3_27b_aime2024_512"]="gemma3_27b_aime2024_512"
    ["resultsforGEMMA/gemma3_27b_aime2025_512"]="gemma3_27b_aime2025_512"
    ["resultsforGEMMA/gemma3_27b_bruno2025_512"]="gemma3_27b_bruno2025_512"
    ["resultsforGEMMA/gemma3_27b_hmmt2025_512"]="gemma3_27b_hmmt2025_512"
)

DRY_RUN=true
if [[ "$1" == "--run" ]]; then
    DRY_RUN=false
fi

echo "========================================"
echo "Copy Teammate's Results"
echo "========================================"
echo "Source base: $SRC_BASE"
echo "Target dir:  $TARGET_DIR"
echo "Mode: $(if $DRY_RUN; then echo 'DRY-RUN (use --run to execute)'; else echo 'EXECUTING'; fi)"
echo "========================================"
echo ""

# ===========================================
# PHASE 1: Pre-flight check - detect conflicts
# ===========================================
echo "Phase 1: Checking for conflicts..."
echo ""

CONFLICTS=()
VALID_COPIES=()
MISSING=()

for src_subdir in "${!MAPPINGS[@]}"; do
    target_name="${MAPPINGS[$src_subdir]}"
    src_path="$SRC_BASE/$src_subdir"
    target_path="$TARGET_DIR/$target_name"

    # Check if source exists
    if [[ ! -d "$src_path" ]]; then
        echo "  [SKIP] $target_name - source not found: $src_subdir"
        MISSING+=("$src_subdir")
        continue
    fi

    # Check if target already exists
    if [[ -d "$target_path" ]]; then
        existing_count=$(ls -1 "$target_path"/*.pkl 2>/dev/null | wc -l || echo "0")
        echo "  [CONFLICT] $target_name - target exists with $existing_count files"
        CONFLICTS+=("$target_name")
    else
        pkl_count=$(ls -1 "$src_path"/*.pkl 2>/dev/null | wc -l || echo "0")
        echo "  [OK] $target_name - $pkl_count files ready to copy"
        VALID_COPIES+=("$src_subdir")
    fi
done

echo ""

# Summary of skipped
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Skipped ${#MISSING[@]} missing source(s)"
fi

# Report conflicts (but continue)
if [[ ${#CONFLICTS[@]} -gt 0 ]]; then
    echo "Skipping ${#CONFLICTS[@]} conflict(s) - targets already exist"
fi

# If nothing to copy
if [[ ${#VALID_COPIES[@]} -eq 0 ]]; then
    echo "========================================"
    echo "Nothing to copy (no valid sources found)"
    echo "========================================"
    exit 0
fi

# ===========================================
# PHASE 2: Execute copies (no conflicts)
# ===========================================
echo "Phase 2: Copying ${#VALID_COPIES[@]} directories..."
echo ""

for src_subdir in "${VALID_COPIES[@]}"; do
    target_name="${MAPPINGS[$src_subdir]}"
    src_path="$SRC_BASE/$src_subdir"
    target_path="$TARGET_DIR/$target_name"
    pkl_count=$(ls -1 "$src_path"/*.pkl 2>/dev/null | wc -l || echo "0")

    echo "----------------------------------------"
    echo "Source:  $src_path"
    echo "Target:  $target_path"
    echo "Files:   $pkl_count"

    if $DRY_RUN; then
        echo "  [DRY-RUN] Would copy"
    else
        echo "  Copying..."
        cp -r "$src_path" "$target_path"
        echo "  [DONE]"
    fi
done

echo ""
echo "========================================"
if $DRY_RUN; then
    echo "DRY-RUN complete. Run with --run to execute."
else
    echo "Copy complete!"
    echo ""
    echo "Results in $TARGET_DIR:"
    ls -1d "$TARGET_DIR"/*_512 2>/dev/null | xargs -n1 basename | sort || echo "  (none)"
fi
echo "========================================"
