"""
Run inference for questions using Google Gemini API (google-genai SDK).
Supports both interactive and batch API modes.

Requirements:
    pip install google-genai

Usage:
    # Interactive mode (real-time, full price)
    python scripts/inference/run_gemini_batch.py \
        --model gemini-2.0-flash \
        --dataset /path/to/dataset.jsonl \
        --budget 64 \
        --rid gemini_aime2025

    # Batch mode (async, 50% cost savings)
    python scripts/inference/run_gemini_batch.py \
        --model gemini-2.0-flash \
        --dataset /path/to/dataset.jsonl \
        --budget 64 \
        --batch

Output directory structure:
    {output_dir}/
    └── {rid}/
        ├── qid0_{timestamp}.pkl
        ├── qid1_{timestamp}.pkl
        └── ...
"""
import sys
import os

# Add scripts to path for config and inference_utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PathConfig

import json
import pickle
import argparse
import re
import time as time_module
import gc
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import importlib.util

# New google-genai SDK
from google import genai
from google.genai import types

# Import utils directly to avoid vLLM dependency in deepconf/__init__.py
_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'deepconf', 'deepconf', 'utils.py')
_spec = importlib.util.spec_from_file_location("deepconf_utils", _utils_path)
_deepconf_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_deepconf_utils)
compute_all_voting_results = _deepconf_utils.compute_all_voting_results

# Import shared utilities
from inference_utils import (
    get_next_run_id,
    setup_logging,
    save_result_as_json,
    filter_completed_questions,
)


def extract_answer_from_text(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def prepare_prompt(question: str, model_name: str) -> str:
    """
    Prepare prompt for Gemini models.

    Per DeepConf paper:
    - Append instruction: "Please reason step by step, and put your final answer within \\boxed{}."
    """
    instruction = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return question + instruction


def create_gemini_client(api_key: str) -> genai.Client:
    """Create a Gemini client instance."""
    return genai.Client(api_key=api_key)


def create_generation_config(
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_tokens: int = 65536,
    logprobs: int = 20,
) -> types.GenerateContentConfig:
    """Create generation config with logprobs enabled."""
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_tokens,
        response_logprobs=True,
        logprobs=logprobs,
    )


def extract_logprobs_from_response(response) -> Tuple[List[float], int]:
    """
    Extract token-level confidence scores from Gemini response.

    Returns:
        confs: List of -logprob values (higher = less confident, matching deepconf format)
        num_tokens: Number of output tokens
    """
    confs = []
    num_tokens = 0

    try:
        if response.candidates:
            candidate = response.candidates[0]

            # Check for logprobs_result in the new SDK
            if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                logprobs_result = candidate.logprobs_result

                # Extract from chosen_candidates
                if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates:
                    for token_info in logprobs_result.chosen_candidates:
                        if hasattr(token_info, 'log_probability') and token_info.log_probability is not None:
                            conf = round(-token_info.log_probability, 3)
                            confs.append(float(conf))
                            num_tokens += 1

                # Alternative: top_candidates structure
                elif hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
                    for step in logprobs_result.top_candidates:
                        if hasattr(step, 'candidates') and step.candidates:
                            chosen = step.candidates[0]
                            if hasattr(chosen, 'log_probability') and chosen.log_probability is not None:
                                conf = round(-chosen.log_probability, 3)
                                confs.append(float(conf))
                                num_tokens += 1

    except Exception as e:
        pass

    return confs, num_tokens


def call_gemini_api(
    client: genai.Client,
    model_name: str,
    prompt: str,
    config: types.GenerateContentConfig,
    sample_idx: int,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Call Gemini API with retry logic.

    Returns a trace dict compatible with existing format.
    """
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            start_time = time_module.time()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            elapsed_time = time_module.time() - start_time

            # Extract text from response
            text = ""
            finish_reason = "unknown"

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)

            # Extract answer
            extracted_answer = extract_answer_from_text(text)

            # Extract logprobs and compute confidence scores
            confs, num_tokens_from_logprobs = extract_logprobs_from_response(response)

            # Use logprobs count if available, otherwise estimate
            if num_tokens_from_logprobs > 0:
                num_tokens = num_tokens_from_logprobs
            else:
                num_tokens = int(len(text.split()) * 1.3)

            # Compute min_conf over sliding windows (matching deepconf format)
            min_conf = float('inf')
            window_size = 2048
            if len(confs) >= window_size:
                for i in range(len(confs) - window_size + 1):
                    window_conf = sum(confs[i:i+window_size]) / window_size
                    min_conf = min(min_conf, window_conf)
            elif confs:
                min_conf = sum(confs) / len(confs)

            return {
                'text': text,
                'num_tokens': num_tokens,
                'confs': confs,
                'min_conf': min_conf,
                'extracted_answer': extracted_answer,
                'stop_reason': finish_reason,
                'sample_idx': sample_idx,
                'elapsed_time': elapsed_time,
                'success': True,
            }

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if 'quota' in error_str or 'rate' in error_str or '429' in error_str or 'resource_exhausted' in error_str:
                delay = min(delay * 2, 60)
                time_module.sleep(delay)
            elif 'invalid' in error_str or 'blocked' in error_str:
                break
            else:
                time_module.sleep(delay)

    return {
        'text': f"ERROR: {str(last_error)}",
        'num_tokens': 0,
        'confs': [],
        'min_conf': float('inf'),
        'extracted_answer': "",
        'stop_reason': 'error',
        'sample_idx': sample_idx,
        'elapsed_time': 0,
        'success': False,
        'error': str(last_error),
    }


def run_parallel_inference(
    client: genai.Client,
    model_name: str,
    prompts: List[str],
    config: types.GenerateContentConfig,
    max_workers: int = 8,
    delay_between_calls: float = 0.1,
) -> List[Dict[str, Any]]:
    """Run inference on multiple prompts in parallel using ThreadPoolExecutor."""
    results = [None] * len(prompts)

    def call_with_delay(idx: int, prompt: str) -> Dict[str, Any]:
        time_module.sleep(idx * delay_between_calls)
        return call_gemini_api(client, model_name, prompt, config, idx)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(call_with_delay, idx, prompt): idx
            for idx, prompt in enumerate(prompts)
        }

        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    'text': f"ERROR: {str(e)}",
                    'num_tokens': 0,
                    'confs': [],
                    'min_conf': float('inf'),
                    'extracted_answer': "",
                    'stop_reason': 'error',
                    'sample_idx': idx,
                    'elapsed_time': 0,
                    'success': False,
                    'error': str(e),
                }

    return results


def process_gemini_outputs(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process Gemini API outputs into standard format."""
    total_tokens = sum(t.get('num_tokens', 0) for t in traces)
    return {
        'traces': traces,
        'total_tokens': total_tokens,
    }


# ============================================================
# Batch API Functions
# ============================================================

def create_batch_request(
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    logprobs: int,
) -> Dict[str, Any]:
    """
    Create a single batch request in the format expected by Gemini Batch API.

    Per docs: config is at same level as contents, containing generation params.
    """
    return {
        'contents': [{
            'parts': [{'text': prompt}],
            'role': 'user'
        }],
        'config': {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_output_tokens': max_tokens,
            'response_logprobs': True,
            'logprobs': logprobs,
        }
    }


def extract_logprobs_from_batch_response(response) -> Tuple[List[float], int]:
    """Extract logprobs from a batch API response (same structure as interactive)."""
    return extract_logprobs_from_response(response)


def process_batch_response(response, sample_idx: int) -> Dict[str, Any]:
    """Process a single response from batch API results."""
    try:
        # Extract text
        text = ""
        finish_reason = "unknown"

        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)

        # Extract answer
        extracted_answer = extract_answer_from_text(text)

        # Extract logprobs
        confs, num_tokens_from_logprobs = extract_logprobs_from_batch_response(response)

        if num_tokens_from_logprobs > 0:
            num_tokens = num_tokens_from_logprobs
        else:
            num_tokens = int(len(text.split()) * 1.3)

        # Compute min_conf
        min_conf = float('inf')
        window_size = 2048
        if len(confs) >= window_size:
            for i in range(len(confs) - window_size + 1):
                window_conf = sum(confs[i:i+window_size]) / window_size
                min_conf = min(min_conf, window_conf)
        elif confs:
            min_conf = sum(confs) / len(confs)

        return {
            'text': text,
            'num_tokens': num_tokens,
            'confs': confs,
            'min_conf': min_conf,
            'extracted_answer': extracted_answer,
            'stop_reason': finish_reason,
            'sample_idx': sample_idx,
            'elapsed_time': 0,  # Not available for batch
            'success': True,
        }
    except Exception as e:
        return {
            'text': f"ERROR: {str(e)}",
            'num_tokens': 0,
            'confs': [],
            'min_conf': float('inf'),
            'extracted_answer': "",
            'stop_reason': 'error',
            'sample_idx': sample_idx,
            'elapsed_time': 0,
            'success': False,
            'error': str(e),
        }


def submit_single_batch(
    client: genai.Client,
    model_name: str,
    requests: List[Dict[str, Any]],
    batch_idx: int,
    total_batches: int,
    logger,
) -> str:
    """Submit a single batch job and return job name."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    display_name = f"gemini_inference_{timestamp}_batch{batch_idx+1}of{total_batches}"

    logger.info(f"Submitting batch {batch_idx+1}/{total_batches} with {len(requests)} requests...")

    batch_job = client.batches.create(
        model=f"models/{model_name}",
        src=requests,
        config={'display_name': display_name},
    )

    logger.info(f"  Batch {batch_idx+1} created: {batch_job.name}")
    return batch_job.name


def run_batch_inference(
    client: genai.Client,
    model_name: str,
    all_requests: List[Dict[str, Any]],
    request_metadata: List[Dict[str, Any]],
    logger,
    poll_interval: int = 30,
    max_batch_size: int = 5000,
    run_dir: str = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Submit batch job(s) and wait for completion.

    Automatically splits into multiple batch jobs if requests exceed max_batch_size
    to stay under the 20MB inline request limit.

    Args:
        client: Gemini client
        model_name: Model to use
        all_requests: List of batch request dicts
        request_metadata: List of {qid, sample_idx} for each request
        logger: Logger instance
        poll_interval: Seconds between status checks
        max_batch_size: Max requests per batch job (default 5000, ~15MB safe limit)

    Returns:
        Dict mapping qid -> list of trace results
    """
    total_requests = len(all_requests)
    num_batches = (total_requests + max_batch_size - 1) // max_batch_size

    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Max batch size: {max_batch_size}")
    logger.info(f"Number of batch jobs: {num_batches}")
    logger.info(f"Model: models/{model_name}")

    # Split into batches
    batches = []
    metadata_batches = []
    for i in range(num_batches):
        start_idx = i * max_batch_size
        end_idx = min(start_idx + max_batch_size, total_requests)
        batches.append(all_requests[start_idx:end_idx])
        metadata_batches.append(request_metadata[start_idx:end_idx])

    # Submit all batch jobs
    logger.info("\n" + "="*60)
    logger.info("SUBMITTING BATCH JOBS")
    logger.info("="*60)

    job_names = []
    for batch_idx, batch_requests in enumerate(batches):
        job_name = submit_single_batch(
            client, model_name, batch_requests, batch_idx, num_batches, logger
        )
        job_names.append(job_name)
        # Small delay between submissions to avoid rate limits
        if batch_idx < num_batches - 1:
            time_module.sleep(1)

    logger.info(f"All {num_batches} batch jobs submitted")

    # Save job names for recovery (in case script is killed)
    if run_dir:
        jobs_file = os.path.join(run_dir, "batch_jobs.txt")
        with open(jobs_file, 'w') as f:
            for jn in job_names:
                f.write(f"{jn}\n")
        logger.info(f"Job names saved to: {jobs_file}")
        logger.info("(Use recover_gemini_batch.py if script is interrupted)")

    # Wait for all batch jobs to complete (poll all in parallel)
    logger.info("\n" + "="*60)
    logger.info("WAITING FOR BATCH JOBS TO COMPLETE")
    logger.info("="*60)
    logger.info(f"Polling every {poll_interval}s...")

    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    # Track status of each batch
    batch_status = {job_name: None for job_name in job_names}
    batch_jobs = {job_name: None for job_name in job_names}
    start_time = time_module.time()
    poll_count = 0

    # Poll all jobs until all complete
    while True:
        poll_count += 1
        pending_jobs = [jn for jn, status in batch_status.items() if status is None]

        if not pending_jobs:
            break

        # Check status of all pending jobs
        for job_name in pending_jobs:
            batch_job = client.batches.get(name=job_name)
            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            if state_name in completed_states:
                batch_status[job_name] = state_name
                batch_jobs[job_name] = batch_job
                batch_idx = job_names.index(job_name)
                logger.info(f"  Batch {batch_idx+1}/{num_batches} finished: {state_name}")

        # Log progress
        completed_count = sum(1 for s in batch_status.values() if s is not None)
        elapsed_min = (time_module.time() - start_time) / 60

        if completed_count < num_batches:
            logger.info(f"[Poll #{poll_count}] Completed: {completed_count}/{num_batches} | Elapsed: {elapsed_min:.1f}min")
            time_module.sleep(poll_interval)

    total_wait_time = time_module.time() - start_time
    logger.info(f"All batch jobs completed in {total_wait_time:.0f}s ({total_wait_time/60:.1f}min)")

    # Process results from all batches
    results_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    total_success = 0
    total_errors = 0

    for batch_idx, (job_name, metadata_batch) in enumerate(zip(job_names, metadata_batches)):
        state_name = batch_status[job_name]
        batch_job = batch_jobs[job_name]

        if state_name == 'JOB_STATE_SUCCEEDED':
            if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'inlined_responses'):
                responses = batch_job.dest.inlined_responses
                logger.info(f"  Processing {len(responses)} responses from batch {batch_idx+1}...")

                for idx, response_wrapper in enumerate(responses):
                    if idx >= len(metadata_batch):
                        break

                    meta = metadata_batch[idx]
                    qid = meta['qid']
                    sample_idx = meta['sample_idx']

                    # Check for error in response wrapper
                    if hasattr(response_wrapper, 'error') and response_wrapper.error:
                        total_errors += 1
                        trace = {
                            'text': f"ERROR: {response_wrapper.error}",
                            'num_tokens': 0,
                            'confs': [],
                            'min_conf': float('inf'),
                            'extracted_answer': "",
                            'stop_reason': 'error',
                            'sample_idx': sample_idx,
                            'elapsed_time': 0,
                            'success': False,
                            'error': str(response_wrapper.error),
                        }
                    else:
                        response = response_wrapper.response if hasattr(response_wrapper, 'response') else response_wrapper
                        trace = process_batch_response(response, sample_idx)
                        if trace.get('success', True):
                            total_success += 1
                        else:
                            total_errors += 1

                    if qid not in results_by_qid:
                        results_by_qid[qid] = []
                    results_by_qid[qid].append(trace)
            else:
                logger.error(f"Batch {batch_idx+1} succeeded but no inlined responses found")
        else:
            logger.error(f"Batch {batch_idx+1} failed with state: {state_name}")
            if batch_job and hasattr(batch_job, 'error') and batch_job.error:
                logger.error(f"Error details: {batch_job.error}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total batches: {num_batches}")
    logger.info(f"Total responses: {total_success + total_errors}")
    logger.info(f"Successful: {total_success}, Errors: {total_errors}")
    logger.info(f"Questions with results: {len(results_by_qid)}")

    return results_by_qid


def main():
    parser = argparse.ArgumentParser(description='Run Gemini API inference')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash",
                        help="Gemini model name")
    parser.add_argument('--api_key', type=str, default=None,
                        help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--qid_start', type=int, default=None)
    parser.add_argument('--qid_end', type=int, default=None)
    parser.add_argument('--rid', type=str, default=None)
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--max_tokens', type=int, default=65536)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--logprobs', type=int, default=20)
    parser.add_argument('--delay_between_calls', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=PathConfig.OUTPUT_BASE)
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--json_only', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=1)
    parser.add_argument('--no_resume', action='store_true')
    # Batch API options
    parser.add_argument('--batch', action='store_true',
                        help="Use Batch API (50%% cost savings, async processing)")
    parser.add_argument('--poll_interval', type=int, default=30,
                        help="Seconds between batch job status checks")
    parser.add_argument('--max_batch_size', type=int, default=5000,
                        help="Max requests per batch job (default 5000, stay under 20MB limit)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: No API key provided. Use --api_key or set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    # Auto-generate run ID if not specified
    if args.rid is None:
        args.rid = get_next_run_id(args.output_dir)

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.rid)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(run_dir, logger_name='gemini_inference')

    logger.info("="*80)
    logger.info("GEMINI API INFERENCE RUN (google-genai SDK)")
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

    # Resume: skip already completed questions
    questions_to_process = filter_completed_questions(
        questions_to_process, run_dir, args.no_resume, logger
    )
    if not questions_to_process:
        logger.info("All questions already completed! Nothing to do.")
        return

    num_questions = len(questions_to_process)
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
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Logprobs: {args.logprobs}")
    logger.info(f"Max parallel workers: {args.max_workers}")
    logger.info(f"Mode: {'BATCH API (50% cost)' if args.batch else 'Interactive'}")
    logger.info("="*80)

    # Initialize Gemini client
    logger.info("Initializing Gemini client...")
    client = create_gemini_client(api_key)
    logger.info("Client initialized successfully")

    # ============================================================
    # BATCH MODE
    # ============================================================
    if args.batch:
        logger.info("\n" + "="*80)
        logger.info("BATCH MODE - Preparing all requests")
        logger.info("="*80)

        # Prepare all requests upfront
        all_requests = []
        request_metadata = []  # Track qid and sample_idx for each request

        for qid in questions_to_process:
            question_data = data[qid]
            question = question_data['question']
            prompt = prepare_prompt(question, args.model)

            for sample_idx in range(args.budget):
                req = create_batch_request(
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    logprobs=args.logprobs,
                )
                all_requests.append(req)
                request_metadata.append({'qid': qid, 'sample_idx': sample_idx})

        total_requests = len(all_requests)
        logger.info(f"Prepared {total_requests} requests ({num_questions} questions × {args.budget} samples)")

        # Submit and wait for batch job(s)
        results_by_qid = run_batch_inference(
            client=client,
            model_name=args.model,
            all_requests=all_requests,
            request_metadata=request_metadata,
            logger=logger,
            poll_interval=args.poll_interval,
            max_batch_size=args.max_batch_size,
            run_dir=run_dir,
        )

        # Process and save results for each question
        logger.info("\n" + "="*80)
        logger.info("Processing batch results")
        logger.info("="*80)

        for qid in questions_to_process:
            question_data = data[qid]
            question = question_data['question']
            ground_truth = str(question_data.get('answer', '')).strip()

            traces = results_by_qid.get(qid, [])

            if not traces:
                logger.warning(f"No results for qid={qid}")
                continue

            processed = process_gemini_outputs(traces)
            successful_traces = [t for t in traces if t.get('success', True)]
            failed_traces = [t for t in traces if not t.get('success', True)]
            traces_with_logprobs = [t for t in traces if t.get('confs')]
            truncated_count = sum(1 for t in traces if 'MAX_TOKENS' in str(t.get('stop_reason', '')))
            truncation_rate = truncated_count / len(traces) if traces else 0

            voting_results = compute_all_voting_results(processed['traces'])

            result_data = {
                'question': question,
                'ground_truth': ground_truth,
                'qid': qid,
                'run_id': args.rid,
                'all_traces': processed['traces'],
                'total_tokens': processed['total_tokens'],
                'total_traces_count': len(processed['traces']),
                'successful_traces_count': len(successful_traces),
                'failed_traces_count': len(failed_traces),
                'traces_with_logprobs': len(traces_with_logprobs),
                'truncated_count': truncated_count,
                'truncation_rate': truncation_rate,
                'voting_results': voting_results,
                'config': {
                    'model': args.model,
                    'mode': 'gemini_batch_api',
                    'budget': args.budget,
                    'window_size': args.window_size,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                    'logprobs': args.logprobs,
                    'max_tokens': args.max_tokens,
                }
            }

            voted_answer = "N/A"
            if 'majority' in voting_results and voting_results['majority']:
                voted_answer = voting_results['majority']['answer']
                result_data['voted_answer'] = voted_answer
                result_data['final_answer'] = voted_answer

            logger.info(f"qid={qid}: voted={voted_answer}, traces={len(traces)}, logprobs={len(traces_with_logprobs)}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if not args.json_only:
                result_filename = f"{run_dir}/qid{qid}_{timestamp}.pkl"
                with open(result_filename, 'wb') as f:
                    pickle.dump(result_data, f)

            if args.save_json or args.json_only:
                json_filename = f"{run_dir}/qid{qid}_{timestamp}.json"
                save_result_as_json(result_data, json_filename)

        logger.info("\n" + "="*80)
        logger.info("BATCH MODE COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to {run_dir}/")
        logger.info(f"Questions processed: {len(results_by_qid)}")
        return

    # ============================================================
    # INTERACTIVE MODE (existing logic)
    # ============================================================
    config = create_generation_config(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )

    # Process questions in chunks
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

        for qid in chunk_qids:
            questions_completed += 1
            question_data = data[qid]
            question = question_data['question']
            ground_truth = str(question_data.get('answer', '')).strip()

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question {questions_completed}/{num_questions} (qid={qid})")
            logger.info(f"{'='*80}")
            logger.info(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
            logger.info(f"Ground truth answer: {ground_truth}")

            prompt = prepare_prompt(question, args.model)
            prompts = [prompt] * args.budget

            logger.info(f"Running {args.budget} API calls with {args.max_workers} workers...")
            logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            generation_start = time_module.time()

            traces = run_parallel_inference(
                client=client,
                model_name=args.model,
                prompts=prompts,
                config=config,
                max_workers=args.max_workers,
                delay_between_calls=args.delay_between_calls,
            )

            generation_time = time_module.time() - generation_start
            total_generation_time += generation_time

            logger.info(f"Generation completed in {generation_time:.2f}s")
            logger.info(f"Throughput: {args.budget / generation_time:.2f} calls/s")

            processed = process_gemini_outputs(traces)

            successful_traces = [t for t in traces if t.get('success', True)]
            failed_traces = [t for t in traces if not t.get('success', True)]
            traces_with_logprobs = [t for t in traces if t.get('confs')]

            truncated_count = sum(1 for t in traces if 'MAX_TOKENS' in str(t.get('stop_reason', '')))
            truncation_rate = truncated_count / len(traces) if traces else 0

            voting_results = compute_all_voting_results(processed['traces'])

            result_data = {
                'question': question,
                'ground_truth': ground_truth,
                'qid': qid,
                'run_id': args.rid,
                'all_traces': processed['traces'],
                'total_tokens': processed['total_tokens'],
                'total_traces_count': len(processed['traces']),
                'successful_traces_count': len(successful_traces),
                'failed_traces_count': len(failed_traces),
                'traces_with_logprobs': len(traces_with_logprobs),
                'truncated_count': truncated_count,
                'truncation_rate': truncation_rate,
                'voting_results': voting_results,
                'config': {
                    'model': args.model,
                    'mode': 'gemini_api_batch',
                    'budget': args.budget,
                    'chunk_size': chunk_size,
                    'window_size': args.window_size,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                    'logprobs': args.logprobs,
                    'max_tokens': args.max_tokens,
                    'max_workers': args.max_workers,
                }
            }

            voted_answer = "N/A"
            if 'majority' in voting_results and voting_results['majority']:
                voted_answer = voting_results['majority']['answer']
                result_data['voted_answer'] = voted_answer
                result_data['final_answer'] = voted_answer

            logger.info(f"Voted answer: {voted_answer}")
            logger.info(f"Total tokens: {processed['total_tokens']:,}")
            logger.info(f"Successful traces: {len(successful_traces)}/{len(traces)}")
            logger.info(f"Traces with logprobs: {len(traces_with_logprobs)}/{len(traces)}")
            if failed_traces:
                logger.warning(f"Failed traces: {len(failed_traces)}")
            if truncated_count > 0:
                logger.warning(f"Truncated traces: {truncated_count}/{len(traces)} ({truncation_rate:.1%})")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if not args.json_only:
                result_filename = f"{run_dir}/qid{qid}_{timestamp}.pkl"
                with open(result_filename, 'wb') as f:
                    pickle.dump(result_data, f)
                logger.info(f"Saved pickle: qid{qid}_{timestamp}.pkl")

            if args.save_json or args.json_only:
                json_filename = f"{run_dir}/qid{qid}_{timestamp}.json"
                save_result_as_json(result_data, json_filename)
                logger.info(f"Saved JSON: qid{qid}_{timestamp}.json")

            logger.info(f"Question {questions_completed}/{num_questions} completed")

        gc.collect()
        logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} completed")

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
    logger.info("Gemini inference completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
