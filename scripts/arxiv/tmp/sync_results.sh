#!/bin/bash
# Sync sampling_credit_results between SMB drive and NVMe on butters-compute
# Bidirectional sync - copies changed files in both directions
#
# Usage: ./sync_results.sh [--dry-run|-n|--check|-c]

set -e

SMB_PATH="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results/"
NVME_PATH="/eph/nvme0/sampling_credit_results/sampling_credit_results/"

DRY_RUN=""
CHECK_MODE=false

if [[ "$1" == "--dry-run" || "$1" == "-n" ]]; then
    DRY_RUN="-n"
    echo "=== DRY RUN MODE - No files will be copied ==="
    echo ""
elif [[ "$1" == "--check" || "$1" == "-c" ]]; then
    CHECK_MODE=true
    echo "=== CHECK MODE - Verifying directories are in sync ==="
    echo "SMB:  $SMB_PATH"
    echo "NVMe: $NVME_PATH"
    echo ""

    # Count differences in each direction
    smb_to_nvme=$(rsync -a --itemize-changes --size-only -n "$SMB_PATH" "$NVME_PATH" | grep -c '^>' || true)
    nvme_to_smb=$(rsync -a --itemize-changes --size-only -n "$NVME_PATH" "$SMB_PATH" | grep -c '^>' || true)

    if [[ $smb_to_nvme -eq 0 && $nvme_to_smb -eq 0 ]]; then
        echo "✓ Directories are in sync!"
    else
        echo "✗ Directories are NOT in sync:"
        [[ $smb_to_nvme -gt 0 ]] && echo "  - $smb_to_nvme files newer/missing on NVMe"
        [[ $nvme_to_smb -gt 0 ]] && echo "  - $nvme_to_smb files newer/missing on SMB"
        echo ""
        echo "Run './sync_results.sh -n' to see details, or './sync_results.sh' to sync."
    fi
    exit 0
fi

echo "=== Syncing sampling_credit_results ==="
echo "SMB:  $SMB_PATH"
echo "NVMe: $NVME_PATH"
echo ""

# Sync SMB -> NVMe (copy new/updated files from SMB to NVMe)
echo ">>> Syncing SMB -> NVMe..."
rsync -rlD --itemize-changes --stats --size-only $DRY_RUN "$SMB_PATH" "$NVME_PATH" | grep -v '^\.'

echo ""

# Sync NVMe -> SMB (copy new/updated files from NVMe to SMB)
echo ">>> Syncing NVMe -> SMB..."
rsync -rlD --itemize-changes --stats --size-only $DRY_RUN "$NVME_PATH" "$SMB_PATH" | grep -v '^\.'

echo ""
echo "=== Sync complete ==="
