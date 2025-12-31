"""
Shared utility functions for inference scripts.

This module contains common functions used by:
- run_vllm_batch.py (DeepSeek/Qwen)
- run_gpt_oss_batch.py (GPT-OSS)
"""

import os
import re
import json
import logging
import sys
import numpy as np
from datetime import datetime
from collections import defaultdict


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


def get_completed_qids(run_dir: str) -> set:
    """Scan run directory for completed qid files and return set of completed qids."""
    completed = set()
    if not os.path.exists(run_dir):
        return completed

    for filename in os.listdir(run_dir):
        # Match both pickle and json files: qid{N}_{timestamp}.pkl or .json
        # Timestamp format is YYYYMMDD_HHMMSS (two underscore-separated parts)
        match = re.match(r'^qid(\d+)_\d+_\d+\.(pkl|json)$', filename)
        if match:
            completed.add(int(match.group(1)))

    return completed


def setup_logging(log_dir: str, logger_name: str = 'inference') -> logging.Logger:
    """Setup logging to both console and file"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'inference.log')

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler - append mode for resume support
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
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


def filter_completed_questions(questions_to_process: list, run_dir: str,
                                no_resume: bool, logger=None) -> list:
    """
    Filter out already completed questions for resume support.

    Args:
        questions_to_process: List of qids to process
        run_dir: Directory containing results
        no_resume: If True, skip filtering and process all
        logger: Optional logger for output (uses print if None)

    Returns:
        Filtered list of qids to process
    """
    log = logger.info if logger else print

    if no_resume:
        return questions_to_process

    completed_qids = get_completed_qids(run_dir)
    if completed_qids:
        original_count = len(questions_to_process)
        questions_to_process = [qid for qid in questions_to_process if qid not in completed_qids]
        skipped_count = original_count - len(questions_to_process)
        if skipped_count > 0:
            log(f"RESUME: Found {len(completed_qids)} completed questions in {run_dir}")
            log(f"RESUME: Skipping {skipped_count} already completed questions: {sorted(completed_qids)}")

    return questions_to_process
