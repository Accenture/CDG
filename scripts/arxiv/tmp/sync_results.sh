#!/bin/bash
# Sync sampling_credit_results between SMB drive and NVMe on butters-compute
# Bidirectional sync - copies changed files in both directions
#
# Usage: ./sync_results.sh [--dry-run|-n]

set -e

SMB_PATH="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results/"
NVME_PATH="/eph/nvme0/sampling_credit_results/sampling_credit_results/"
DRY_RUN=""
if [[ "$1" == "--dry-run" || "$1" == "-n" ]]; then
    DRY_RUN="-n"
    echo "=== DRY RUN MODE - No files will be copied ==="
    echo ""
fi

echo "=== Syncing sampling_credit_results ==="
echo "SMB:  $SMB_PATH"
echo "NVMe: $NVME_PATH"
echo ""

# Sync SMB -> NVMe (copy new/updated files from SMB to NVMe)
echo ">>> Syncing SMB -> NVMe..."
rsync -a --itemize-changes --stats --size-only $DRY_RUN "$SMB_PATH" "$NVME_PATH"

echo ""

# Sync NVMe -> SMB (copy new/updated files from NVMe to SMB)
echo ">>> Syncing NVMe -> SMB..."
rsync -a --itemize-changes --stats --size-only $DRY_RUN "$NVME_PATH" "$SMB_PATH"

echo ""
echo "=== Sync complete ==="
