"""
Run offline inference for GPT-OSS models using Harmony format.
Processes N questions at a time to reduce memory usage and save incremental results.

Requirements:
    # Install special vLLM version for GPT-OSS
    uv pip install --pre vllm==0.10.1+gptoss \
        --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
        --index-strategy unsafe-best-match

    # Install harmony package
    uv pip install openai-harmony

Usage:
    python scripts/run_gpt_oss_batch.py \
        --model openai/gpt-oss-120b \
        --dataset /path/to/dataset.jsonl \
        --budget 512 \
        --rid gptoss120b_aime2025_512

Output directory structure:
    {output_dir}/
    └── {rid}/
        ├── qid0_{timestamp}.pkl   (saved after chunk 1)
        ├── qid1_{timestamp}.pkl
        └── ...
"""
import sys
import os

# Add scripts to path for config and inference_utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PathConfig

# Set HuggingFace cache to large disk (using centralized config)
PathConfig.setup_hf_env()

# Disable flashinfer sampler for GPT-OSS (per official docs)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Add deepconf to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepconf'))

import json
import pickle
import argparse
import re
import numpy as np
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from deepconf.utils import compute_all_voting_results

# Import shared utilities
from inference_utils import (
    get_next_run_id,
    get_completed_qids,
    setup_logging,
    save_result_as_json,
    filter_completed_questions,
)

# Harmony imports for GPT-OSS prompt formatting
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)

# Map string reasoning effort to enum (per gpt-oss/gpt_oss/chat.py)
REASONING_EFFORT_MAP = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def extract_answer_from_text(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    # Find the last \boxed{} occurrence
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def process_gpt_oss_outputs(vllm_outputs, encoding, window_size: int = 2048) -> dict:
    """Process GPT-OSS vLLM outputs and extract traces with confidence scores."""
    traces = []
    total_tokens = 0

    for output in vllm_outputs:
        for completion in output.outputs:
            # Get token IDs from completion
            token_ids = list(completion.token_ids)
            text = completion.text
            num_tokens = len(token_ids)
            total_tokens += num_tokens

            # Parse the Harmony-formatted output
            try:
                entries = encoding.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
                # Extract text content from entries
                full_text = ""
                for entry in entries:
                    if hasattr(entry, 'content'):
                        full_text += str(entry.content)
                    elif hasattr(entry, 'text'):
                        full_text += str(entry.text)
                if full_text:
                    text = full_text
            except Exception:
                # Fall back to raw text if parsing fails
                pass

            # Extract answer from boxed format
            extracted_answer = extract_answer_from_text(text)

            # Compute confidence scores from logprobs if available
            # Use same format as deepconf/utils.py:compute_confidence()
            # Returns -mean_logprob (negative log probability), where lower = more confident
            confs = []
            if completion.logprobs:
                for logprob_entry in completion.logprobs:
                    if logprob_entry:
                        # Take mean of all top-k logprobs (matching deepconf format)
                        mean_logprob = np.mean([lp.logprob for lp in logprob_entry.values()])
                        conf = round(-mean_logprob, 3)
                        confs.append(float(conf))

            # Compute min confidence over windows (lowest -logprob = most confident)
            min_conf = float('inf')
            if len(confs) >= window_size:
                for i in range(len(confs) - window_size + 1):
                    window_conf = np.mean(confs[i:i+window_size])
                    min_conf = min(min_conf, window_conf)
            elif confs:
                min_conf = np.mean(confs)

            trace = {
                'text': text,
                'token_ids': token_ids,
                'num_tokens': num_tokens,
                'confs': confs,
                'min_conf': min_conf,
                'extracted_answer': extracted_answer,
                'stop_reason': completion.finish_reason,
            }
            traces.append(trace)

    return {
        'traces': traces,
        'total_tokens': total_tokens,
    }


def prepare_prompt_token_ids(question: str, encoding, reasoning_effort: str = "high") -> list:
    """
    Prepare prompt token IDs for GPT-OSS using Harmony format.

    Per DeepConf paper:
    - Use provider's official system prompt
    - Append instruction: "Please reason step by step, and put your final answer within \\boxed{}."
    - Enable reasoning effort = high
    """
    # Build the instruction for developer message
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    # Create conversation with Harmony format
    # SystemContent.new() includes the default system prompt
    # Set reasoning effort on SystemContent (per gpt-oss/gpt_oss/chat.py)
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT_MAP[reasoning_effort])
    )

    # Developer content with instructions
    developer_content = DeveloperContent.new().with_instructions(instruction)

    # Build conversation
    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.DEVELOPER, developer_content),
        Message.from_role_and_content(Role.USER, question),
    ])

    # Render to token IDs
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    return prefill_ids


def main():
    parser = argparse.ArgumentParser(description='Run GPT-OSS offline inference with Harmony format')
    parser.add_argument('--model', type=str, default="openai/gpt-oss-120b")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help="Number of GPUs for tensor parallelism (1 for 120B on single H100)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--qid_start', type=int, default=None, help="Start question ID (inclusive)")
    parser.add_argument('--qid_end', type=int, default=None, help="End question ID (exclusive)")
    parser.add_argument('--rid', type=str, default=None,
                        help="Run ID (auto-generates run001, run002, etc. if not specified)")
    parser.add_argument('--budget', type=int, default=512,
                        help="Number of samples per question")
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--max_tokens', type=int, default=130000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--logprobs', type=int, default=20,
                        help="Number of top logprobs to return per token")
    parser.add_argument('--reasoning_effort', type=str, default="high",
                        help="Reasoning effort level (low, medium, high)")
    parser.add_argument('--max_num_seqs', type=int, default=64,
                        help="Maximum number of concurrent sequences")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument('--max_model_len', type=int, default=130000,
                        help="Maximum model context length")
    parser.add_argument('--enforce_eager', action='store_true',
                        help="Disable CUDA graphs to reduce memory usage during initialization")
    parser.add_argument('--output_dir', type=str, default=PathConfig.OUTPUT_BASE)
    parser.add_argument('--save_json', action='store_true',
                        help="Also save results as human-readable JSON files")
    parser.add_argument('--json_only', action='store_true',
                        help="Save only JSON files (no pickle)")
    parser.add_argument('--chunk_size', type=int, default=3,
                        help="Number of questions to process per chunk")
    parser.add_argument('--no_resume', action='store_true',
                        help="Disable auto-resume (reprocess all questions even if completed)")

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
    logger.info("GPT-OSS INFERENCE RUN (HARMONY FORMAT)")
    logger.info("="*80)
    logger.info(f"Run ID: {args.rid}")
    logger.info(f"Output directory: {run_dir}")

    # Load Harmony encoding for GPT-OSS
    logger.info("Loading Harmony encoding...")
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    logger.info(f"Loaded Harmony encoding with {len(stop_token_ids)} stop tokens")

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

    # Resume: skip already completed questions
    questions_to_process = filter_completed_questions(
        questions_to_process, run_dir, args.no_resume, logger
    )
    if not questions_to_process:
        logger.info("All questions already completed! Nothing to do.")
        return

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
    logger.info(f"Logprobs: {args.logprobs}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Reasoning effort: {args.reasoning_effort}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info("="*80)

    # Initialize vLLM model
    logger.info("Initializing vLLM model for GPT-OSS...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
    )
    logger.info("Model initialized successfully")

    # Process questions in chunks
    import time as time_module
    import gc

    total_generation_time = 0
    questions_completed = 0

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_questions)
        chunk_qids = questions_to_process[chunk_start:chunk_end]

        logger.info("\n" + "="*80)
        logger.info(f"CHUNK {chunk_idx + 1}/{num_chunks}")
        logger.info("="*80)
        logger.info(f"Processing questions {chunk_qids[0]} to {chunk_qids[-1]} ({len(chunk_qids)} questions)")
        logger.info(f"Prompts in this chunk: {len(chunk_qids) * args.budget}")

        # Prepare prompt token IDs for this chunk
        logger.info("Preparing prompts with Harmony format...")
        chunk_prompt_ids = []
        prompt_to_qid = []

        for qid in chunk_qids:
            question = data[qid]['question']
            prefill_ids = prepare_prompt_token_ids(question, encoding, args.reasoning_effort)

            for _ in range(args.budget):
                chunk_prompt_ids.append(prefill_ids)
                prompt_to_qid.append(qid)

        logger.info(f"Total prompts in chunk: {len(chunk_prompt_ids)}")

        # Create sampling parameters
        sampling_params_list = []
        base_seed = time_module.time_ns() + chunk_idx * 10000

        for i in range(len(chunk_prompt_ids)):
            sp = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                logprobs=args.logprobs,
                seed=base_seed + i,
                stop_token_ids=stop_token_ids,
            )
            sampling_params_list.append(sp)

        # Run batch inference for this chunk
        logger.info("="*80)
        logger.info(f"BATCH INFERENCE - CHUNK {chunk_idx + 1}/{num_chunks}")
        logger.info("="*80)
        logger.info(f"Running batch inference on {len(chunk_prompt_ids)} prompts...")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        generation_start = time_module.time()

        # GPT-OSS uses TokensPrompt for pre-tokenized inputs (vLLM 0.10.x API)
        token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in chunk_prompt_ids]
        vllm_outputs = llm.generate(
            prompts=token_prompts,
            sampling_params=sampling_params_list,
        )

        generation_time = time_module.time() - generation_start
        total_generation_time += generation_time

        logger.info(f"Generation completed in {generation_time:.2f}s")
        logger.info(f"Throughput: {len(chunk_prompt_ids) / generation_time:.2f} prompts/s")
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
            processed = process_gpt_oss_outputs(qid_outputs, encoding, args.window_size)

            # Compute truncation stats (traces that hit max_tokens limit)
            truncated_count = sum(1 for t in processed['traces'] if t.get('stop_reason') == 'length')
            truncation_rate = truncated_count / len(processed['traces']) if processed['traces'] else 0

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
                'truncated_count': truncated_count,
                'truncation_rate': truncation_rate,
                'voting_results': voting_results,
                'config': {
                    'model': args.model,
                    'mode': 'offline_batch_gpt_oss',
                    'budget': args.budget,
                    'chunk_size': chunk_size,
                    'window_size': args.window_size,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                    'logprobs': args.logprobs,
                    'max_tokens': args.max_tokens,
                    'reasoning_effort': args.reasoning_effort,
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
            if truncated_count > 0:
                logger.warning(f"⚠️ Truncated traces: {truncated_count}/{len(processed['traces'])} ({truncation_rate:.1%})")

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

            logger.info(f"Question {questions_completed}/{num_questions} completed")

        # Clear memory after each chunk
        logger.info(f"\nClearing memory after chunk {chunk_idx + 1}...")
        del vllm_outputs
        del chunk_prompt_ids
        del sampling_params_list
        del outputs_by_qid
        gc.collect()
        logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} completed and memory cleared")

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
    logger.info("GPT-OSS inference completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
