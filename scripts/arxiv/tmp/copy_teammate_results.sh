#!/bin/bash
# Copy teammate's DeepSeek results to main results directory with proper naming
#
# Source naming: aime2024run3qdeepseek8b, etc.
# Target naming: deepseek8b_aime2024_512, etc.
#
# Usage:
#   ./scripts/tmp/copy_teammate_results.sh          # dry-run (show what would be copied)
#   ./scripts/tmp/copy_teammate_results.sh --run    # actually copy

set -e

# Source base directory
SRC_BASE="/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_backups"

# Target directory
TARGET_DIR="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"

# Mapping: source_subdir -> target_name
declare -A MAPPINGS=(
    ["aime_2024_run_512_results_deepseek8b/aime2024run3qdeepseek8b"]="deepseek8b_aime2024_512"
    ["aime_2025_run_512_results_deepseek8b/aime2025run3qdeepseek8b"]="deepseek8b_aime2025_512"
    ["bruno_2025_run_512_results_deepseek8b/bruno2025run3qdeepseek8b"]="deepseek8b_bruno2025_512"
    ["hmmt_feb_2025_run_512_results/hmmt2025run3qdeepseek8b"]="deepseek8b_hmmt2025_512"
)

DRY_RUN=true
if [[ "$1" == "--run" ]]; then
    DRY_RUN=false
fi

echo "========================================"
echo "Copy Teammate's DeepSeek Results"
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

for src_subdir in "${!MAPPINGS[@]}"; do
    target_name="${MAPPINGS[$src_subdir]}"
    src_path="$SRC_BASE/$src_subdir"
    target_path="$TARGET_DIR/$target_name"

    # Check if source exists
    if [[ ! -d "$src_path" ]]; then
        echo "  [SKIP] $target_name - source not found"
        continue
    fi

    # Check if target already exists
    if [[ -d "$target_path" ]]; then
        existing_count=$(ls -1 "$target_path"/*.pkl 2>/dev/null | wc -l)
        echo "  [CONFLICT] $target_name - target exists with $existing_count files"
        CONFLICTS+=("$target_name")
    else
        pkl_count=$(ls -1 "$src_path"/*.pkl 2>/dev/null | wc -l)
        echo "  [OK] $target_name - $pkl_count files ready to copy"
        VALID_COPIES+=("$src_subdir")
    fi
done

echo ""

# If any conflicts, abort entirely
if [[ ${#CONFLICTS[@]} -gt 0 ]]; then
    echo "========================================"
    echo "ABORTING: ${#CONFLICTS[@]} conflict(s) detected!"
    echo "========================================"
    echo "The following targets already exist:"
    for conflict in "${CONFLICTS[@]}"; do
        echo "  - $TARGET_DIR/$conflict"
    done
    echo ""
    echo "No files were copied. Remove existing directories or rename them first."
    echo "========================================"
    exit 1
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
    pkl_count=$(ls -1 "$src_path"/*.pkl 2>/dev/null | wc -l)

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
    ls -d "$TARGET_DIR"/deepseek8b_* 2>/dev/null || echo "  (none)"
fi
echo "========================================"
