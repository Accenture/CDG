#!/bin/bash
# Lock all result files to prevent accidental deletion
# Files become read-only; requires sudo to delete
#
# Usage:
#   ./scripts/tmp/lock_results.sh              # lock all results
#   ./scripts/tmp/lock_results.sh --unlock     # unlock (restore write permissions)
#   ./scripts/tmp/lock_results.sh --status     # check current status

set -e

# Target directory
RESULTS_DIR="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"

MODE="lock"
if [[ "$1" == "--unlock" ]]; then
    MODE="unlock"
elif [[ "$1" == "--status" ]]; then
    MODE="status"
fi

echo "========================================"
echo "Results Protection Script"
echo "========================================"
echo "Results dir: $RESULTS_DIR"
echo "Mode: $MODE"
echo "========================================"
echo ""

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: Results directory not found!"
    exit 1
fi

case $MODE in
    lock)
        echo "Locking all files (removing write permissions)..."
        echo ""

        # Make all files read-only (remove write for user, group, others)
        find "$RESULTS_DIR" -type f \( -name "*.pkl" -o -name "*.json" -o -name "*.log" \) -exec chmod a-w {} \;

        # Make directories read-only too (prevents file creation/deletion inside)
        find "$RESULTS_DIR" -mindepth 1 -type d -exec chmod a-w {} \;

        echo "Done! Files are now protected."
        echo ""
        echo "To delete files, you'll need: sudo rm <file>"
        echo "To unlock: ./scripts/tmp/lock_results.sh --unlock"
        ;;

    unlock)
        echo "Unlocking all files (restoring write permissions)..."
        echo ""

        # Restore write permissions for user
        find "$RESULTS_DIR" -type f \( -name "*.pkl" -o -name "*.json" -o -name "*.log" \) -exec chmod u+w {} \;
        find "$RESULTS_DIR" -mindepth 1 -type d -exec chmod u+w {} \;

        echo "Done! Files are now writable."
        ;;

    status)
        echo "Checking protection status..."
        echo ""

        # Count protected vs unprotected
        total_files=$(find "$RESULTS_DIR" -type f \( -name "*.pkl" -o -name "*.json" -o -name "*.log" \) | wc -l)
        protected_files=$(find "$RESULTS_DIR" -type f \( -name "*.pkl" -o -name "*.json" -o -name "*.log" \) ! -perm -u=w | wc -l)

        total_dirs=$(find "$RESULTS_DIR" -mindepth 1 -type d | wc -l)
        protected_dirs=$(find "$RESULTS_DIR" -mindepth 1 -type d ! -perm -u=w | wc -l)

        echo "Files: $protected_files / $total_files protected"
        echo "Dirs:  $protected_dirs / $total_dirs protected"
        echo ""

        # Show per-directory status
        echo "Per-directory status:"
        echo "----------------------------------------"
        for dir in "$RESULTS_DIR"/*/; do
            if [[ -d "$dir" ]]; then
                dirname=$(basename "$dir")
                file_count=$(find "$dir" -maxdepth 1 -type f -name "*.pkl" | wc -l)
                protected_count=$(find "$dir" -maxdepth 1 -type f -name "*.pkl" ! -perm -u=w | wc -l)

                if [[ $file_count -eq 0 ]]; then
                    status="(empty)"
                elif [[ $protected_count -eq $file_count ]]; then
                    status="LOCKED"
                elif [[ $protected_count -eq 0 ]]; then
                    status="unlocked"
                else
                    status="partial ($protected_count/$file_count)"
                fi

                printf "  %-35s %s\n" "$dirname" "$status"
            fi
        done
        ;;
esac

echo ""
echo "========================================"
