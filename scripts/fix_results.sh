#!/bin/bash
# Fix three experiments by replacing with Yu's corrected versions
#
# Issues:
# 1. DeepSeek AIME 2025 - ran on different result
# 2. GPTOSS AIME 2025 - needs replacement
# 3. GPTOSS HMMT 2025 - needs replacement
#
# Usage:
#   ./scripts/fix_results.sh --check    # Check if source paths exist
#   ./scripts/fix_results.sh --dry-run  # Show what would be done
#   ./scripts/fix_results.sh --execute  # Actually perform the fix

set -e

# Paths
RESULTS_BASE="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"
CACHE_DIR="$RESULTS_BASE/.eval_cache/results"
SUBSET_DIR="$RESULTS_BASE/subset_trace"

# Source paths from Yu's directory
YU_BASE="/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper"

# Mapping: [target_name]="source_path"
# NOTE: Yu's directories have nested subdirectories containing the actual pkl files
declare -A REPLACEMENTS=(
    ["deepseek8b_aime2025_512"]="$YU_BASE/deepseek/aime_2025_run_512_results_newpara/aime2025rundeepseek8b_newpara"
    ["gptoss20b_aime2025_512"]="$YU_BASE/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason/aime2025run3qgptoss20b_topk_40"
    ["gptoss20b_hmmt2025_512"]="$YU_BASE/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason/hmmt2025run3qgptoss20b_topk_40"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "FIX RESULTS SCRIPT"
echo "========================================"
echo "Results base: $RESULTS_BASE"
echo ""

check_sources() {
    echo "Checking source paths..."
    echo ""
    all_exist=true
    for target in "${!REPLACEMENTS[@]}"; do
        source="${REPLACEMENTS[$target]}"
        if [ -d "$source" ]; then
            count=$(ls "$source"/qid*.pkl 2>/dev/null | wc -l)
            echo -e "${GREEN}✓${NC} $target"
            echo "  Source: $source"
            echo "  Files: $count pkl files"
        else
            echo -e "${RED}✗${NC} $target"
            echo "  Source: $source"
            echo "  ERROR: Directory not found!"
            all_exist=false
        fi
        echo ""
    done

    if $all_exist; then
        echo -e "${GREEN}All source paths exist!${NC}"
    else
        echo -e "${RED}Some source paths are missing!${NC}"
        exit 1
    fi
}

show_cleanup() {
    echo "Items to clean up:"
    echo ""

    for target in "${!REPLACEMENTS[@]}"; do
        echo "=== $target ==="

        # Result directory
        if [ -d "$RESULTS_BASE/$target" ]; then
            count=$(ls "$RESULTS_BASE/$target"/qid*.pkl 2>/dev/null | wc -l)
            echo "  Result dir: $RESULTS_BASE/$target ($count files)"
        else
            echo "  Result dir: (not found)"
        fi

        # Cache entries
        cache_pattern="$CACHE_DIR/${target}*"
        cache_count=$(ls $cache_pattern 2>/dev/null | wc -l)
        echo "  Cache entries: $cache_count"

        # Subset entries (both 512 and subset versions)
        # Pattern: {experiment}_subset{N}v{V}
        subset_pattern="$SUBSET_DIR/${target}_subset*"
        subset_count=$(ls -d $subset_pattern 2>/dev/null | wc -l)
        echo "  Subset dirs: $subset_count"

        # Also check cache for subset versions
        subset_cache_pattern="$CACHE_DIR/${target}_subset*"
        subset_cache_count=$(ls $subset_cache_pattern 2>/dev/null | wc -l)
        echo "  Subset cache entries: $subset_cache_count"

        echo ""
    done
}

execute_fix() {
    echo "Executing fix..."
    echo ""

    for target in "${!REPLACEMENTS[@]}"; do
        source="${REPLACEMENTS[$target]}"
        echo "========================================"
        echo "Processing: $target"
        echo "========================================"

        # Step 1: Clean up cache entries for main experiment
        echo "Step 1: Cleaning cache entries..."
        cache_pattern="$CACHE_DIR/${target}*"
        cache_files=$(ls $cache_pattern 2>/dev/null || true)
        if [ -n "$cache_files" ]; then
            rm -f $cache_pattern
            echo "  Removed cache entries matching: ${target}*"
        else
            echo "  No cache entries found"
        fi

        # Step 2: Clean up subset cache entries
        echo "Step 2: Cleaning subset cache entries..."
        subset_cache_pattern="$CACHE_DIR/${target}_subset*"
        subset_cache_files=$(ls $subset_cache_pattern 2>/dev/null || true)
        if [ -n "$subset_cache_files" ]; then
            rm -f $subset_cache_pattern
            echo "  Removed subset cache entries"
        else
            echo "  No subset cache entries found"
        fi

        # Step 3: Clean up subset directories
        echo "Step 3: Cleaning subset directories..."
        subset_pattern="$SUBSET_DIR/${target}_subset*"
        subset_dirs=$(ls -d $subset_pattern 2>/dev/null || true)
        if [ -n "$subset_dirs" ]; then
            rm -rf $subset_pattern
            echo "  Removed subset directories"
        else
            echo "  No subset directories found"
        fi

        # Step 4: Backup and replace result directory
        echo "Step 4: Replacing result directory..."
        target_dir="$RESULTS_BASE/$target"

        if [ -d "$target_dir" ]; then
            # Move backup to separate directory to avoid matching MODEL_PATTERNS
            backup_base="$RESULTS_BASE/_backups"
            mkdir -p "$backup_base"
            backup_dir="$backup_base/${target}_$(date +%Y%m%d_%H%M%S)"
            mv "$target_dir" "$backup_dir"
            echo "  Backed up to: $backup_dir"
        fi

        # Copy new results
        cp -r "$source" "$target_dir"
        new_count=$(ls "$target_dir"/qid*.pkl 2>/dev/null | wc -l)
        echo "  Copied from: $source"
        echo "  New file count: $new_count"

        echo ""
    done

    echo "========================================"
    echo "FIX COMPLETE"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Run subsample creation for fixed experiments:"
    echo "   python scripts/trace_subsample/subsample.py --experiment deepseek8b_aime2025_512"
    echo "   python scripts/trace_subsample/subsample.py --experiment gptoss20b_aime2025_512"
    echo "   python scripts/trace_subsample/subsample.py --experiment gptoss20b_hmmt2025_512"
    echo ""
    echo "2. Run evaluation:"
    echo "   python scripts/eval/run_all_experiments.py"
    echo ""
}

dry_run() {
    echo "DRY RUN - showing what would be done"
    echo ""
    show_cleanup
    echo ""
    echo "Actions that would be taken:"
    for target in "${!REPLACEMENTS[@]}"; do
        source="${REPLACEMENTS[$target]}"
        echo "  - Remove cache: $CACHE_DIR/${target}*"
        echo "  - Remove subset cache: $CACHE_DIR/${target}_subset*"
        echo "  - Remove subsets: $SUBSET_DIR/${target}_subset*"
        echo "  - Backup: $RESULTS_BASE/$target → _backups/${target}_*"
        echo "  - Copy: $source → $RESULTS_BASE/$target"
        echo ""
    done
}

# Main
case "${1:-}" in
    --check)
        check_sources
        ;;
    --dry-run)
        check_sources
        echo ""
        dry_run
        ;;
    --execute)
        check_sources
        echo ""
        echo -e "${YELLOW}WARNING: This will modify files!${NC}"
        echo "Press Enter to continue or Ctrl+C to cancel..."
        read
        execute_fix
        ;;
    *)
        echo "Usage: $0 [--check|--dry-run|--execute]"
        echo ""
        echo "Options:"
        echo "  --check    Check if source paths exist"
        echo "  --dry-run  Show what would be done without making changes"
        echo "  --execute  Actually perform the fix"
        echo ""
        echo "Experiments to fix:"
        for target in "${!REPLACEMENTS[@]}"; do
            echo "  - $target"
        done
        ;;
esac
