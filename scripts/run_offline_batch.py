"""
Run offline inference for all questions using vLLM's native batching.
Much more efficient than spawning parallel processes.

Usage:
    # Run all questions with 256 samples each (auto-generates run ID)
    python scripts/run_offline_batch.py --budget 256

    # Specify GPU count and custom run ID
    python scripts/run_offline_batch.py --budget 256 --tensor_parallel_size 8 --rid my_experiment

    # Run subset of questions
    python scripts/run_offline_batch.py --budget 256 --qid_start 0 --qid_end 10

    # Save as human-readable JSON (in addition to pickle)
    python scripts/run_offline_batch.py --budget 256 --save_json

    # Save only JSON (no pickle files)
    python scripts/run_offline_batch.py --budget 256 --json_only

Output directory structure:
    {output_dir}/
    └── {rid}/
        ├── qid0_{timestamp}.pkl   (unless --json_only)
        ├── qid0_{timestamp}.json  (if --save_json or --json_only)
        └── ...
"""
import sys
import os

# Add scripts to path for config import
sys.path.insert(0, os.path.dirname(__file__))
from config import PathConfig

# Set HuggingFace cache to large disk (using centralized config)
PathConfig.setup_hf_env()

# Add deepconf to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

import json
import pickle
import argparse
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from vllm import SamplingParams
from deepconf import DeepThinkLLM
from deepconf.utils import process_batch_results_offline, compute_all_voting_results


def trace_to_json_serializable(trace: dict) -> dict:
    """Convert trace dict to JSON-serializable format."""
    result = {}
    for key, value in trace.items():
        if key == 'token_ids':
            result[key] = [int(t) for t in value] if value else []
        elif key == 'confs':
            result[key] = [round(float(c), 4) for c in value] if value else []
        elif key == 'text':
            if len(value) > 5000:
                result[key] = value[:2500] + "\n... [TRUNCATED] ...\n" + value[-2500:]
                result['text_truncated'] = True
                result['text_full_length'] = len(value)
            else:
                result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            result[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def save_result_as_json(result_data: dict, output_path: str) -> None:
    """Save result data as human-readable JSON file (summarized format)."""
    traces = result_data.get('all_traces', [])

    # Build trace summaries
    trace_summaries = []
    for i, trace in enumerate(traces):
        confs = trace.get('confs', [])
        summary = {
            'trace_index': i,
            'extracted_answer': trace.get('extracted_answer', None),
            'num_tokens': trace.get('num_tokens', 0),
            'stop_reason': trace.get('stop_reason', ''),
            'mean_confidence': round(float(np.mean(confs)), 4) if confs else None,
            'min_confidence': round(float(min(confs)), 4) if confs else None,
            'max_confidence': round(float(max(confs)), 4) if confs else None,
        }
        trace_summaries.append(summary)

    # Answer distribution
    answer_counts = defaultdict(int)
    for trace in traces:
        ans = trace.get('extracted_answer')
        if ans:
            answer_counts[str(ans)] += 1

    json_data = {
        'metadata': {
            'question': result_data.get('question', ''),
            'ground_truth': result_data.get('ground_truth', ''),
            'qid': result_data.get('qid', -1),
            'run_id': result_data.get('run_id', ''),
            'total_tokens': result_data.get('total_tokens', 0),
            'total_traces_count': result_data.get('total_traces_count', 0),
            'voted_answer': result_data.get('voted_answer', ''),
            'final_answer': result_data.get('final_answer', ''),
            'saved_at': datetime.now().isoformat(),
        },
        'config': result_data.get('config', {}),
        'voting_results': result_data.get('voting_results', {}),
        'trace_summaries': trace_summaries,
        'answer_distribution': dict(sorted(answer_counts.items(), key=lambda x: -x[1])[:20]),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


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
    parser = argparse.ArgumentParser(description='Run offline inference for all questions (batch mode)')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument('--dataset', type=str, default="aime_2025.jsonl")
    parser.add_argument('--qid_start', type=int, default=None, help="Start question ID (inclusive)")
    parser.add_argument('--qid_end', type=int, default=None, help="End question ID (exclusive)")
    parser.add_argument('--rid', type=str, default=None,
                        help="Run ID (auto-generates run001, run002, etc. if not specified)")
    parser.add_argument('--budget', type=int, default=256,
                        help="Number of samples per question")
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--max_tokens', type=int, default=130000)
    parser.add_argument('--model_type', type=str, default="deepseek")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_num_seqs', type=int, default=64,
                        help="Maximum number of concurrent sequences (limits memory usage)")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument('--output_dir', type=str,
                        default=PathConfig.OUTPUT_BASE)
    parser.add_argument('--save_json', action='store_true',
                        help="Also save results as human-readable JSON files")
    parser.add_argument('--json_only', action='store_true',
                        help="Save only JSON files (no pickle)")

    args = parser.parse_args()

    # Auto-generate run ID if not specified
    if args.rid is None:
        args.rid = get_next_run_id(args.output_dir)
    print(f"Run ID: {args.rid}")

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.rid)
    os.makedirs(run_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    # Determine question range
    qid_start = args.qid_start if args.qid_start is not None else 0
    qid_end = args.qid_end if args.qid_end is not None else len(data)
    qid_end = min(qid_end, len(data))

    questions_to_process = list(range(qid_start, qid_end))
    num_questions = len(questions_to_process)

    print(f"Processing questions {qid_start} to {qid_end-1} ({num_questions} questions)")
    print(f"Budget per question: {args.budget}")
    print(f"Total prompts: {num_questions * args.budget}")

    # Initialize model (single initialization for all questions)
    print("Initializing model...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Prepare all prompts
    print("Preparing prompts...")
    all_prompts = []
    prompt_to_qid = []  # Track which qid each prompt belongs to

    for qid in tqdm(questions_to_process, desc="Preparing prompts"):
        question = data[qid]['question']
        prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

        # Add `budget` copies of this prompt
        for _ in range(args.budget):
            all_prompts.append(prompt)
            prompt_to_qid.append(qid)

    print(f"Total prompts prepared: {len(all_prompts)}")

    # Create sampling parameters (one per prompt for different seeds)
    print("Creating sampling parameters...")
    import time as time_module

    sampling_params_list = []
    base_seed = time_module.time_ns()

    for i in range(len(all_prompts)):
        sp = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            logprobs=20,
            seed=base_seed + i
        )
        sampling_params_list.append(sp)

    # Run batch inference
    print(f"Running batch inference on {len(all_prompts)} prompts...")
    generation_start = time_module.time()

    vllm_outputs = deep_llm.llm.generate(all_prompts, sampling_params_list)

    generation_time = time_module.time() - generation_start
    print(f"Generation completed in {generation_time:.2f}s")
    print(f"Throughput: {len(all_prompts) / generation_time:.2f} prompts/s")

    # Group outputs by question ID
    print("Processing and saving results...")
    outputs_by_qid = {qid: [] for qid in questions_to_process}

    for i, output in enumerate(vllm_outputs):
        qid = prompt_to_qid[i]
        outputs_by_qid[qid].append(output)

    # Process and save results for each question
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for qid in tqdm(questions_to_process, desc="Saving results"):
        question_data = data[qid]
        question = question_data['question']
        ground_truth = str(question_data.get('answer', '')).strip()

        # Process results for this question
        qid_outputs = outputs_by_qid[qid]
        processed = process_batch_results_offline(qid_outputs, args.window_size)

        # Compute voting results
        voting_results = compute_all_voting_results(processed['traces'])

        # Build result data
        result_data = {
            'question': question,
            'ground_truth': ground_truth,
            'qid': qid,
            'run_id': args.rid,
            'all_traces': processed['traces'],
            'total_tokens': processed['total_tokens'],
            'total_traces_count': len(processed['traces']),
            'voting_results': voting_results,
            'config': {
                'model': args.model,
                'mode': 'offline_batch',
                'budget': args.budget,
                'window_size': args.window_size,
                'temperature': args.temperature,
                'top_p': args.top_p,
            }
        }

        # Get voted answer
        if 'majority' in voting_results and voting_results['majority']:
            result_data['voted_answer'] = voting_results['majority']['answer']
            result_data['final_answer'] = voting_results['majority']['answer']

        # Save pickle (unless json_only)
        if not args.json_only:
            result_filename = f"{run_dir}/qid{qid}_{timestamp}.pkl"
            with open(result_filename, 'wb') as f:
                pickle.dump(result_data, f)

        # Save JSON (if requested or json_only)
        if args.save_json or args.json_only:
            json_filename = f"{run_dir}/qid{qid}_{timestamp}.json"
            save_result_as_json(result_data, json_filename)

    save_format = "JSON only" if args.json_only else ("pickle + JSON" if args.save_json else "pickle")
    print(f"\nAll results saved to {run_dir}/ ({save_format})")
    print(f"Total generation time: {generation_time:.2f}s")
    print(f"Average time per question: {generation_time / num_questions:.2f}s")


if __name__ == "__main__":
    main()
