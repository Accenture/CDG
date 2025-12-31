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

for src_subdir in "${!MAPPINGS[@]}"; do
    target_name="${MAPPINGS[$src_subdir]}"
    src_path="$SRC_BASE/$src_subdir"
    target_path="$TARGET_DIR/$target_name"

    echo "----------------------------------------"
    echo "Source:  $src_path"
    echo "Target:  $target_path"

    # Check if source exists
    if [[ ! -d "$src_path" ]]; then
        echo "  [SKIP] Source directory not found"
        continue
    fi

    # Count files
    pkl_count=$(ls -1 "$src_path"/*.pkl 2>/dev/null | wc -l)
    echo "  Found $pkl_count pickle files"

    # Check if target already exists
    if [[ -d "$target_path" ]]; then
        existing_count=$(ls -1 "$target_path"/*.pkl 2>/dev/null | wc -l)
        echo "  [WARN] Target exists with $existing_count files"
        if $DRY_RUN; then
            echo "  [DRY-RUN] Would skip (target exists)"
        else
            echo "  [SKIP] Target already exists"
        fi
        continue
    fi

    # Copy
    if $DRY_RUN; then
        echo "  [DRY-RUN] Would copy $pkl_count files to $target_path"
    else
        echo "  Copying..."
        cp -r "$src_path" "$target_path"
        echo "  [DONE] Copied to $target_path"
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
