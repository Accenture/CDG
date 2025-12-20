"""
Run offline inference for questions in CHUNKS using vLLM's native batching.
Processes N questions at a time to reduce memory usage and save incremental results.

Usage:
    # Run with default chunk size (3 questions at a time)
    python scripts/run_offline_batch_3_q.py --budget 256

    # Custom chunk size (5 questions at a time)
    python scripts/run_offline_batch_3_q.py --budget 256 --chunk_size 5

    # Specify GPU count and custom run ID
    python scripts/run_offline_batch_3_q.py --budget 256 --tensor_parallel_size 8 --rid my_experiment

Output directory structure:
    {output_dir}/
    └── {rid}/
        ├── qid0_{timestamp}.pkl   (saved after chunk 1)
        ├── qid1_{timestamp}.pkl
        ├── qid2_{timestamp}.pkl
        ├── qid3_{timestamp}.pkl   (saved after chunk 2)
        └── ...
"""
import sys
import os

# Set HuggingFace cache to large disk
os.environ["HF_HOME"] = "/mnt/model_confs/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/model_confs/hf_cache"

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
import logging


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


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to both console and file"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'inference.log')

    # Create logger
    logger = logging.getLogger('hmmt_inference')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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
    parser.add_argument('--top_k', type=int, default=-1,
                        help="Top-k sampling parameter (-1 for disabled, per vLLM default)")
    parser.add_argument('--max_num_seqs', type=int, default=64,
                        help="Maximum number of concurrent sequences (limits memory usage)")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument('--output_dir', type=str,
                        default="/eph/nvme0/hmmt_feb_2025_run_512_results")
    parser.add_argument('--save_json', action='store_true',
                        help="Also save results as human-readable JSON files")
    parser.add_argument('--json_only', action='store_true',
                        help="Save only JSON files (no pickle)")
    parser.add_argument('--chunk_size', type=int, default=3,
                        help="Number of questions to process per chunk (reduces memory usage)")

    args = parser.parse_args()

    # Auto-generate run ID if not specified
    if args.rid is None:
        args.rid = get_next_run_id(args.output_dir)

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.rid)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(run_dir)

    logger.info("="*80)
    logger.info("HMMT FEBRUARY 2025 INFERENCE RUN")
    logger.info("="*80)
    logger.info(f"Run ID: {args.rid}")
    logger.info(f"Output directory: {run_dir}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    logger.info(f"Loaded {len(data)} questions")

    # Determine question range
    qid_start = args.qid_start if args.qid_start is not None else 0
    qid_end = args.qid_end if args.qid_end is not None else len(data)
    qid_end = min(qid_end, len(data))

    questions_to_process = list(range(qid_start, qid_end))
    num_questions = len(questions_to_process)

    # Split questions into chunks
    chunk_size = args.chunk_size
    num_chunks = (num_questions + chunk_size - 1) // chunk_size

    logger.info("="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Processing questions {qid_start} to {qid_end-1} ({num_questions} questions)")
    logger.info(f"Chunk size: {chunk_size} questions per batch")
    logger.info(f"Total chunks: {num_chunks}")
    logger.info(f"Budget per question: {args.budget}")
    logger.info(f"Prompts per chunk: {chunk_size * args.budget}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info("="*80)

    # Initialize model (single initialization for all chunks)
    logger.info("Initializing model...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    logger.info("Model initialized successfully")

    # Process questions in chunks
    import time as time_module
    import gc

    total_generation_time = 0
    questions_completed = 0

    # Process questions in chunks
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_questions)
        chunk_qids = questions_to_process[chunk_start:chunk_end]

        logger.info("\n" + "="*80)
        logger.info(f"CHUNK {chunk_idx + 1}/{num_chunks}")
        logger.info("="*80)
        logger.info(f"Processing questions {chunk_qids[0]} to {chunk_qids[-1]} ({len(chunk_qids)} questions)")
        logger.info(f"Prompts in this chunk: {len(chunk_qids) * args.budget}")

        # Prepare prompts for this chunk
        logger.info("Preparing prompts...")
        chunk_prompts = []
        prompt_to_qid = []

        for qid in chunk_qids:
            question = data[qid]['question']
            prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

            for _ in range(args.budget):
                chunk_prompts.append(prompt)
                prompt_to_qid.append(qid)

        logger.info(f"Total prompts in chunk: {len(chunk_prompts)}")

        # Create sampling parameters
        sampling_params_list = []
        base_seed = time_module.time_ns() + chunk_idx * 10000

        for i in range(len(chunk_prompts)):
            sp = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                logprobs=20,
                seed=base_seed + i
            )
            sampling_params_list.append(sp)

        # Run batch inference for this chunk
        logger.info("="*80)
        logger.info(f"BATCH INFERENCE - CHUNK {chunk_idx + 1}/{num_chunks}")
        logger.info("="*80)
        logger.info(f"Running batch inference on {len(chunk_prompts)} prompts...")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        generation_start = time_module.time()

        vllm_outputs = deep_llm.llm.generate(chunk_prompts, sampling_params_list)

        generation_time = time_module.time() - generation_start
        total_generation_time += generation_time

        logger.info(f"Generation completed in {generation_time:.2f}s")
        logger.info(f"Throughput: {len(chunk_prompts) / generation_time:.2f} prompts/s")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Group outputs by question ID
        logger.info("="*80)
        logger.info(f"PROCESSING AND SAVING RESULTS - CHUNK {chunk_idx + 1}/{num_chunks}")
        logger.info("="*80)
        outputs_by_qid = {qid: [] for qid in chunk_qids}

        for i, output in enumerate(vllm_outputs):
            qid = prompt_to_qid[i]
            outputs_by_qid[qid].append(output)

        # Process and save results for each question in this chunk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for qid in chunk_qids:
            questions_completed += 1

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question {questions_completed}/{num_questions} (qid={qid})")
            logger.info(f"{'='*80}")

            question_data = data[qid]
            question = question_data['question']
            ground_truth = str(question_data.get('answer', '')).strip()

            logger.info(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
            logger.info(f"Ground truth answer: {ground_truth}")

            # Process results for this question
            qid_outputs = outputs_by_qid[qid]
            logger.info(f"Processing {len(qid_outputs)} traces...")
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
                    'mode': 'offline_batch_chunked',
                    'budget': args.budget,
                    'chunk_size': chunk_size,
                    'window_size': args.window_size,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                }
            }

            # Get voted answer
            voted_answer = "N/A"
            if 'majority' in voting_results and voting_results['majority']:
                voted_answer = voting_results['majority']['answer']
                result_data['voted_answer'] = voted_answer
                result_data['final_answer'] = voted_answer

            logger.info(f"Voted answer: {voted_answer}")
            logger.info(f"Total tokens: {processed['total_tokens']:,}")
            logger.info(f"Valid traces: {len(processed['traces'])}")

            # Save pickle (unless json_only)
            if not args.json_only:
                result_filename = f"{run_dir}/qid{qid}_{timestamp}.pkl"
                with open(result_filename, 'wb') as f:
                    pickle.dump(result_data, f)
                logger.info(f"Saved pickle: qid{qid}_{timestamp}.pkl")

            # Save JSON (if requested or json_only)
            if args.save_json or args.json_only:
                json_filename = f"{run_dir}/qid{qid}_{timestamp}.json"
                save_result_as_json(result_data, json_filename)
                logger.info(f"Saved JSON: qid{qid}_{timestamp}.json")

            logger.info(f"✅ Question {questions_completed}/{num_questions} completed")

        # Clear memory after each chunk
        logger.info(f"\n🧹 Clearing memory after chunk {chunk_idx + 1}...")
        del vllm_outputs
        del chunk_prompts
        del sampling_params_list
        del outputs_by_qid
        gc.collect()
        logger.info(f"✅ Chunk {chunk_idx + 1}/{num_chunks} completed and memory cleared")

    # Final summary
    save_format = "JSON only" if args.json_only else ("pickle + JSON" if args.save_json else "pickle")
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"All results saved to {run_dir}/ ({save_format})")
    logger.info(f"Total chunks processed: {num_chunks}")
    logger.info(f"Total generation time: {total_generation_time:.2f}s")
    logger.info(f"Average time per question: {total_generation_time / num_questions:.2f}s")
    logger.info(f"Questions processed: {questions_completed}")
    logger.info("="*80)
    logger.info("✅ Successfully finishes all questions in hmmt_feb")
    logger.info("="*80)


if __name__ == "__main__":
    main()
