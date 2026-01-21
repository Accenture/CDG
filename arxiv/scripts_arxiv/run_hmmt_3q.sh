#!/bin/bash
# HMMT February 2025 Inference Script - Chunked Version (3 questions at a time)
# Run this in a screen session: screen -S inference && ./scripts/run_hmmt_3q.sh

set -e  # Exit on error

echo "========================================="
echo "HMMT Feb 2025 Inference (Chunked - 3Q)"
echo "========================================="
echo "Python: $(which python)"
echo "Output: /eph/nvme0/hmmt_feb_2025_run_512_results/"
echo "Log: /eph/nvme0/hmmt_feb_2025_run_512_results/hmmt2025run/inference.log"
echo "Chunk size: 3 questions per batch"
echo "========================================="
echo ""

# Run inference with chunking
python scripts/run_vllm_batch.py \
    --dataset /eph/nvme0/hmmt_feb_2025/hmmt_feb_2025.jsonl \
    --budget 512 \
    --rid hmmt2025run3qdeepseek8b \
    --max_tokens 64000 \
    --temperature 0.6 \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --tensor_parallel_size 8 \
    --chunk_size 3

echo ""
echo "========================================="
echo "✅ Inference completed successfully!"
echo "========================================="
