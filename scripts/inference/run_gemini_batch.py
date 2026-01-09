"""
Run inference for questions using Google Gemini API.
Processes N questions at a time to reduce memory usage and save incremental results.

Requirements:
    pip install google-generativeai

Usage:
    python scripts/inference/run_gemini_batch.py \
        --model gemini-2.0-flash-thinking-exp \
        --dataset /path/to/dataset.jsonl \
        --budget 64 \
        --rid gemini_aime2025

    # With custom API key (or set GOOGLE_API_KEY env var)
    python scripts/inference/run_gemini_batch.py \
        --api_key YOUR_API_KEY \
        --dataset /path/to/dataset.jsonl

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

import json
import pickle
import argparse
import re
import asyncio
import time as time_module
import gc
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import importlib.util

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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


def create_gemini_model(
    model_name: str,
    api_key: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_tokens: int = 65536,
    logprobs: int = 20,
) -> genai.GenerativeModel:
    """Create and configure a Gemini model instance."""
    genai.configure(api_key=api_key)

    # Base config
    config_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,
    }

    # Try to enable logprobs (not supported in older google.generativeai versions)
    try:
        test_config = genai.GenerationConfig(response_logprobs=True, logprobs=1)
        config_kwargs["response_logprobs"] = True
        config_kwargs["logprobs"] = logprobs
        print("Logprobs enabled")
    except TypeError:
        print("WARNING: Logprobs not supported in this API version. Confidence-based voting will not work.")

    generation_config = genai.GenerationConfig(**config_kwargs)

    # Disable all safety filters for math problems
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return model


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
        # Check if logprobs_result exists in the response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Try to get logprobs from the candidate
            if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                logprobs_result = candidate.logprobs_result

                # Extract chosen candidates' logprobs
                if hasattr(logprobs_result, 'chosen_candidates'):
                    for token_info in logprobs_result.chosen_candidates:
                        if hasattr(token_info, 'log_probability'):
                            # Convert to -logprob format (matching deepconf)
                            conf = round(-token_info.log_probability, 3)
                            confs.append(float(conf))
                            num_tokens += 1

                # Alternative: check top_candidates structure
                elif hasattr(logprobs_result, 'top_candidates'):
                    for step in logprobs_result.top_candidates:
                        if hasattr(step, 'candidates') and step.candidates:
                            # Take the first (chosen) candidate's logprob
                            chosen = step.candidates[0]
                            if hasattr(chosen, 'log_probability'):
                                conf = round(-chosen.log_probability, 3)
                                confs.append(float(conf))
                                num_tokens += 1

            # Fallback: try avg_logprobs if available
            if not confs and hasattr(candidate, 'avg_logprobs') and candidate.avg_logprobs is not None:
                # We don't have per-token, but we can note the average
                pass

    except Exception as e:
        # If logprobs extraction fails, return empty list
        pass

    return confs, num_tokens


def call_gemini_api(
    model: genai.GenerativeModel,
    prompt: str,
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
            response = model.generate_content(prompt)
            elapsed_time = time_module.time() - start_time

            # Extract text from response
            text = ""
            finish_reason = "unknown"

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                finish_reason = str(candidate.finish_reason.name) if candidate.finish_reason else "unknown"

            # Extract answer
            extracted_answer = extract_answer_from_text(text)

            # Extract logprobs and compute confidence scores
            confs, num_tokens_from_logprobs = extract_logprobs_from_response(response)

            # Use logprobs count if available, otherwise estimate
            if num_tokens_from_logprobs > 0:
                num_tokens = num_tokens_from_logprobs
            else:
                # Fallback: estimate from text
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
                'confs': confs,  # Token-level confidence scores
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

            # Check for rate limit errors
            if 'quota' in error_str or 'rate' in error_str or '429' in error_str:
                delay = min(delay * 2, 60)  # exponential backoff, max 60s
                time_module.sleep(delay)
            elif 'invalid' in error_str or 'blocked' in error_str:
                # Content blocked or invalid - don't retry
                break
            else:
                time_module.sleep(delay)

    # Return error trace
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
    model: genai.GenerativeModel,
    prompts: List[str],
    max_workers: int = 8,
    delay_between_calls: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Run inference on multiple prompts in parallel using ThreadPoolExecutor.
    """
    results = [None] * len(prompts)

    def call_with_delay(idx: int, prompt: str) -> Dict[str, Any]:
        # Add small delay to avoid rate limits
        time_module.sleep(idx * delay_between_calls)
        return call_gemini_api(model, prompt, idx)

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


def main():
    parser = argparse.ArgumentParser(description='Run Gemini API inference')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash-thinking-exp",
                        help="Gemini model name (e.g., gemini-2.0-flash-thinking-exp, gemini-1.5-pro)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--qid_start', type=int, default=None, help="Start question ID (inclusive)")
    parser.add_argument('--qid_end', type=int, default=None, help="End question ID (exclusive)")
    parser.add_argument('--rid', type=str, default=None,
                        help="Run ID (auto-generates run001, run002, etc. if not specified)")
    parser.add_argument('--budget', type=int, default=64,
                        help="Number of samples per question")
    parser.add_argument('--window_size', type=int, default=2048,
                        help="Window size for confidence computation (unused for Gemini)")
    parser.add_argument('--max_tokens', type=int, default=65536,
                        help="Maximum output tokens")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--max_workers', type=int, default=8,
                        help="Maximum parallel API calls")
    parser.add_argument('--logprobs', type=int, default=20,
                        help="Number of top logprobs to return per token (1-20)")
    parser.add_argument('--delay_between_calls', type=float, default=0.1,
                        help="Delay between parallel API calls (seconds)")
    parser.add_argument('--output_dir', type=str, default=PathConfig.OUTPUT_BASE)
    parser.add_argument('--save_json', action='store_true',
                        help="Also save results as human-readable JSON files")
    parser.add_argument('--json_only', action='store_true',
                        help="Save only JSON files (no pickle)")
    parser.add_argument('--chunk_size', type=int, default=1,
                        help="Number of questions to process per chunk")
    parser.add_argument('--no_resume', action='store_true',
                        help="Disable auto-resume (reprocess all questions even if completed)")

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
    logger.info("GEMINI API INFERENCE RUN")
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
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Logprobs: {args.logprobs}")
    logger.info(f"Max parallel workers: {args.max_workers}")
    logger.info("="*80)

    # Initialize Gemini model
    logger.info("Initializing Gemini model...")
    model = create_gemini_model(
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )
    logger.info("Model initialized successfully")

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

        # Process each question in the chunk
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

            # Prepare prompts
            prompt = prepare_prompt(question, args.model)
            prompts = [prompt] * args.budget

            # Run parallel inference
            logger.info(f"Running {args.budget} API calls with {args.max_workers} workers...")
            logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            generation_start = time_module.time()

            traces = run_parallel_inference(
                model=model,
                prompts=prompts,
                max_workers=args.max_workers,
                delay_between_calls=args.delay_between_calls,
            )

            generation_time = time_module.time() - generation_start
            total_generation_time += generation_time

            logger.info(f"Generation completed in {generation_time:.2f}s")
            logger.info(f"Throughput: {args.budget / generation_time:.2f} calls/s")

            # Process results
            processed = process_gemini_outputs(traces)

            # Count successful and failed traces
            successful_traces = [t for t in traces if t.get('success', True)]
            failed_traces = [t for t in traces if not t.get('success', True)]

            # Compute truncation stats
            truncated_count = sum(1 for t in traces if t.get('stop_reason') == 'MAX_TOKENS')
            truncation_rate = truncated_count / len(traces) if traces else 0

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
                'successful_traces_count': len(successful_traces),
                'failed_traces_count': len(failed_traces),
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

            # Get voted answer
            voted_answer = "N/A"
            if 'majority' in voting_results and voting_results['majority']:
                voted_answer = voting_results['majority']['answer']
                result_data['voted_answer'] = voted_answer
                result_data['final_answer'] = voted_answer

            logger.info(f"Voted answer: {voted_answer}")
            logger.info(f"Total tokens (estimated): {processed['total_tokens']:,}")
            logger.info(f"Successful traces: {len(successful_traces)}/{len(traces)}")
            if failed_traces:
                logger.warning(f"Failed traces: {len(failed_traces)}")
            if truncated_count > 0:
                logger.warning(f"Truncated traces: {truncated_count}/{len(traces)} ({truncation_rate:.1%})")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        gc.collect()
        logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} completed")

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
    logger.info("Gemini inference completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
