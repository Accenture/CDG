#!/bin/bash
# Sync sampling_credit_results between SMB drive and NVMe on butters-compute
# Bidirectional sync - copies changed files in both directions

set -e

SMB_PATH="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results/"
NVME_PATH="/eph/nvme0/sampling_credit_results/sampling_credit_results/"
VM_HOST="butters-compute"

echo "=== Syncing sampling_credit_results on $VM_HOST ==="
echo "SMB:  $SMB_PATH"
echo "NVMe: $NVME_PATH"
echo ""

# Sync SMB -> NVMe (copy new/updated files from SMB to NVMe)
echo ">>> Syncing SMB -> NVMe..."
ssh "$VM_HOST" "rsync -av --update \"$SMB_PATH\" \"$NVME_PATH\""

echo ""

# Sync NVMe -> SMB (copy new/updated files from NVMe to SMB)
echo ">>> Syncing NVMe -> SMB..."
ssh "$VM_HOST" "rsync -av --update \"$NVME_PATH\" \"$SMB_PATH\""

echo ""
echo "=== Sync complete ==="
