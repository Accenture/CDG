"""
Run offline inference for a single question - simpler alternative to shell script.
Can be used with GNU parallel or just a loop.

Usage:
    # Single question (auto-generates run ID: run001, run002, etc.)
    CUDA_VISIBLE_DEVICES=0 python run_offline_single.py --qid 0 --budget 256

    # Parallel with GNU parallel (8 GPUs, 30 questions)
    seq 0 29 | parallel -j8 'CUDA_VISIBLE_DEVICES=$(( {} % 8 )) python run_offline_single.py --qid {} --budget 256'

    # Specify custom run ID
    seq 0 29 | parallel -j8 'CUDA_VISIBLE_DEVICES=$(( {} % 8 )) python run_offline_single.py --qid {} --budget 256 --rid my_experiment'

Output directory structure:
    {output_dir}/
    └── {rid}/
        ├── qid0_{timestamp}.pkl
        ├── qid1_{timestamp}.pkl
        ├── qid2_{timestamp}.pkl
        └── ...

    Each .pkl file contains:
        - question: str
        - ground_truth: str
        - qid: int
        - run_id: str
        - all_traces: list of trace dicts
        - voting_results: dict
        - config: dict
        - ... (other DeepThinkOutput fields)
"""
import sys
import os

# Set HuggingFace cache to large disk
os.environ["HF_HOME"] = "/mnt/dev/model_ckpt/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/dev/model_ckpt/hf_cache"

# Add deepconf to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

import json
import pickle
import argparse
import re
from datetime import datetime
from vllm import SamplingParams
from deepconf import DeepThinkLLM


def get_next_run_id(output_dir: str) -> str:
    """Auto-generate next run ID (run001, run002, etc.)"""
    if not os.path.exists(output_dir):
        return "run001"

    existing = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    run_numbers = []
    for name in existing:
        match = re.match(r'^run(\d+)$', name)
        if match:
            run_numbers.append(int(match.group(1)))

    next_num = max(run_numbers, default=0) + 1
    return f"run{next_num:03d}"


def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        messages = [{"role": "user", "content": question}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(description='Run offline inference for a single question')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="aime_2025.jsonl")
    parser.add_argument('--qid', type=int, required=True)
    parser.add_argument('--rid', type=str, default=None, help="Run ID (auto-generates run001, run002, etc. if not specified)")
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--max_tokens', type=int, default=130000)
    parser.add_argument('--model_type', type=str, default="deepseek")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--output_dir', type=str, default="/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results")

    args = parser.parse_args()

    # Auto-generate run ID if not specified
    if args.rid is None:
        args.rid = get_next_run_id(args.output_dir)
        print(f"Auto-generated run ID: {args.rid}")

    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    if args.qid >= len(data) or args.qid < 0:
        raise ValueError(f"Question ID {args.qid} out of range (0-{len(data)-1})")

    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()

    print(f"Processing question {args.qid}: {question[:100]}...")

    # Initialize model
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True
    )

    prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

    sampling_params = SamplingParams(
        n=1,  # budget controls number of prompts in deepthink, not n
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Run inference
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=args.budget,
        window_size=args.window_size,
        sampling_params=sampling_params,
        compute_multiple_voting=True
    )

    # Save results
    run_dir = os.path.join(args.output_dir, args.rid)
    os.makedirs(run_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = result.to_dict()
    result_data.update({
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
    })

    result_filename = f"{run_dir}/qid{args.qid}_{timestamp}.pkl"

    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)

    print(f"Results saved to {result_filename}")


if __name__ == "__main__":
    main()
