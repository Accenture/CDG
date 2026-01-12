"""
Recover results from Gemini Batch API jobs.

Use this script if the main inference script was killed while waiting for batch jobs.
Batch jobs continue running on Google's servers and results can be recovered.

Usage:
    # List recent batch jobs
    python scripts/inference/recover_gemini_batch.py --list

    # Recover specific job(s) by ID
    python scripts/inference/recover_gemini_batch.py --jobs batches/abc123 batches/def456

    # Recover all succeeded jobs from today
    python scripts/inference/recover_gemini_batch.py --recover-all

    # Recover with custom output directory
    python scripts/inference/recover_gemini_batch.py --jobs batches/abc123 --output_dir ./recovered
"""
import sys
import os
import json
import pickle
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PathConfig

from google import genai


def extract_answer_from_text(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def extract_logprobs_from_response(response) -> Tuple[List[float], int]:
    """Extract token-level confidence scores from Gemini response."""
    confs = []
    num_tokens = 0

    try:
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                logprobs_result = candidate.logprobs_result
                if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates:
                    for token_info in logprobs_result.chosen_candidates:
                        if hasattr(token_info, 'log_probability') and token_info.log_probability is not None:
                            conf = round(-token_info.log_probability, 3)
                            confs.append(float(conf))
                            num_tokens += 1
    except Exception:
        pass

    return confs, num_tokens


def process_response(response, sample_idx: int) -> Dict[str, Any]:
    """Process a single response into trace format."""
    try:
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

        extracted_answer = extract_answer_from_text(text)
        confs, num_tokens = extract_logprobs_from_response(response)

        if num_tokens == 0:
            num_tokens = int(len(text.split()) * 1.3)

        min_conf = float('inf')
        if confs:
            min_conf = sum(confs) / len(confs)

        return {
            'text': text,
            'num_tokens': num_tokens,
            'confs': confs,
            'min_conf': min_conf,
            'extracted_answer': extracted_answer,
            'stop_reason': finish_reason,
            'sample_idx': sample_idx,
            'elapsed_time': 0,
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


def list_batch_jobs(client: genai.Client, limit: int = 20):
    """List recent batch jobs."""
    print("\n" + "="*80)
    print("RECENT BATCH JOBS")
    print("="*80)
    print(f"{'Job Name':<50} {'State':<20} {'Display Name'}")
    print("-"*80)

    count = 0
    for job in client.batches.list():
        state = job.state.name if hasattr(job.state, 'name') else str(job.state)
        display_name = job.display_name if hasattr(job, 'display_name') else 'N/A'
        print(f"{job.name:<50} {state:<20} {display_name}")
        count += 1
        if count >= limit:
            print(f"... (showing first {limit} jobs)")
            break

    if count == 0:
        print("No batch jobs found.")

    print("="*80)


def recover_job(client: genai.Client, job_name: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Recover results from a single batch job.

    Returns:
        (state_name, list of trace dicts)
    """
    print(f"\nRecovering job: {job_name}")

    job = client.batches.get(name=job_name)
    state = job.state.name if hasattr(job.state, 'name') else str(job.state)

    print(f"  State: {state}")

    if state != 'JOB_STATE_SUCCEEDED':
        print(f"  Skipping - job not succeeded")
        return state, []

    traces = []

    if hasattr(job, 'dest') and hasattr(job.dest, 'inlined_responses'):
        responses = job.dest.inlined_responses
        print(f"  Found {len(responses)} responses")

        for idx, response_wrapper in enumerate(responses):
            if hasattr(response_wrapper, 'error') and response_wrapper.error:
                traces.append({
                    'text': f"ERROR: {response_wrapper.error}",
                    'num_tokens': 0,
                    'confs': [],
                    'min_conf': float('inf'),
                    'extracted_answer': "",
                    'stop_reason': 'error',
                    'sample_idx': idx,
                    'elapsed_time': 0,
                    'success': False,
                    'error': str(response_wrapper.error),
                })
            else:
                response = response_wrapper.response if hasattr(response_wrapper, 'response') else response_wrapper
                trace = process_response(response, idx)
                traces.append(trace)

        success_count = sum(1 for t in traces if t.get('success', True))
        print(f"  Processed: {success_count} successful, {len(traces) - success_count} errors")
    else:
        print(f"  No inlined responses found")

    return state, traces


def save_recovered_results(
    traces: List[Dict[str, Any]],
    output_dir: str,
    job_name: str,
):
    """Save recovered traces to pickle and JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a simple result structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = job_name.split('/')[-1][:8]  # Short job ID

    result_data = {
        'recovered_from': job_name,
        'all_traces': traces,
        'total_tokens': sum(t.get('num_tokens', 0) for t in traces),
        'total_traces_count': len(traces),
        'successful_traces_count': sum(1 for t in traces if t.get('success', True)),
        'traces_with_logprobs': sum(1 for t in traces if t.get('confs')),
        'recovered_at': datetime.now().isoformat(),
    }

    # Save pickle
    pkl_path = os.path.join(output_dir, f"recovered_{job_id}_{timestamp}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result_data, f)
    print(f"  Saved: {pkl_path}")

    # Save JSON summary
    json_path = os.path.join(output_dir, f"recovered_{job_id}_{timestamp}.json")
    json_data = {
        'recovered_from': job_name,
        'total_traces': len(traces),
        'successful': result_data['successful_traces_count'],
        'with_logprobs': result_data['traces_with_logprobs'],
        'answers': [t.get('extracted_answer', '') for t in traces[:20]],  # First 20
        'recovered_at': result_data['recovered_at'],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    return pkl_path


def main():
    parser = argparse.ArgumentParser(description='Recover Gemini Batch API results')
    parser.add_argument('--list', action='store_true',
                        help='List recent batch jobs')
    parser.add_argument('--jobs', nargs='+',
                        help='Job name(s) to recover (e.g., batches/abc123)')
    parser.add_argument('--recover-all', action='store_true',
                        help='Recover all succeeded jobs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for recovered results')
    parser.add_argument('--limit', type=int, default=20,
                        help='Max jobs to list/recover')

    args = parser.parse_args()

    # Initialize client
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Try to load from file
        key_file = os.path.join(os.path.dirname(__file__), 'gemini_api_key.txt')
        if os.path.exists(key_file):
            with open(key_file) as f:
                api_key = f.read().strip()
                os.environ['GOOGLE_API_KEY'] = api_key

    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or create gemini_api_key.txt")
        sys.exit(1)

    client = genai.Client()

    # List jobs
    if args.list:
        list_batch_jobs(client, args.limit)
        return

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(PathConfig.OUTPUT_BASE, f"recovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Recover specific jobs
    if args.jobs:
        print(f"\nRecovering {len(args.jobs)} job(s)...")
        print(f"Output directory: {output_dir}")

        all_traces = []
        for job_name in args.jobs:
            state, traces = recover_job(client, job_name)
            if traces:
                all_traces.extend(traces)
                save_recovered_results(traces, output_dir, job_name)

        print(f"\nTotal recovered: {len(all_traces)} traces")
        return

    # Recover all succeeded jobs
    if args.recover_all:
        print(f"\nRecovering all succeeded jobs...")
        print(f"Output directory: {output_dir}")

        count = 0
        total_traces = 0

        for job in client.batches.list():
            if count >= args.limit:
                break

            state = job.state.name if hasattr(job.state, 'name') else str(job.state)
            if state == 'JOB_STATE_SUCCEEDED':
                _, traces = recover_job(client, job.name)
                if traces:
                    save_recovered_results(traces, output_dir, job.name)
                    total_traces += len(traces)
                count += 1

        print(f"\nRecovered {count} jobs, {total_traces} total traces")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
