#!/bin/bash
# Run offline inference in parallel across multiple GPUs using data parallelism
# Usage: ./run_offline_parallel.sh [num_gpus] [budget_per_question]

NUM_GPUS=${1:-8}
BUDGET=${2:-256}
DATASET="aime_2025.jsonl"
OUTPUT_DIR="offline_results"
MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# Get total questions
NUM_QUESTIONS=$(wc -l < "$DATASET" | tr -d ' ')
echo "Total questions: $NUM_QUESTIONS"
echo "Using $NUM_GPUS GPUs with budget=$BUDGET per question"

mkdir -p "$OUTPUT_DIR"

# Distribute questions across GPUs
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate question range for this GPU
    questions_per_gpu=$(( (NUM_QUESTIONS + NUM_GPUS - 1) / NUM_GPUS ))
    start_qid=$((gpu_id * questions_per_gpu))
    end_qid=$((start_qid + questions_per_gpu - 1))

    if [ $end_qid -ge $NUM_QUESTIONS ]; then
        end_qid=$((NUM_QUESTIONS - 1))
    fi

    echo "GPU $gpu_id: questions $start_qid to $end_qid"

    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python -c "
import sys
sys.path.insert(0, 'deepconf')
from deepconf.examples.example_offline import main
import argparse

for qid in range($start_qid, $end_qid + 1):
    sys.argv = [
        'example_offline.py',
        '--qid', str(qid),
        '--rid', 'parallel_run',
        '--budget', '$BUDGET',
        '--dataset', '$DATASET',
        '--output_dir', '$OUTPUT_DIR',
        '--model', '$MODEL',
        '--tensor_parallel_size', '1',
        '--model_type', 'deepseek'
    ]
    try:
        main()
    except Exception as e:
        print(f'Error on qid {qid}: {e}')
" &> "$OUTPUT_DIR/gpu_${gpu_id}.log" &

    echo "Started GPU $gpu_id process (PID: $!)"
done

echo "All processes started. Monitor with: tail -f $OUTPUT_DIR/gpu_*.log"
echo "Wait for completion with: wait"
wait
echo "All done!"
