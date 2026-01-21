"""
Forking Token Analysis for Sampling Credit

This script implements experiments to analyze forking tokens and their relationship
to model uncertainty/entropy.

Tasks:
1. Save reports as human-readable log files (JSON) instead of pickle
2. Define forking tokens based on the word cloud from the paper
3. Exp 1: Analyze if forking tokens have higher entropy in correct samples
4. Exp 2: Brute force search for optimal sampling threshold

Reference: https://arxiv.org/pdf/2506.01939

Usage:
    python analysis/forking_token_analysis.py --results_dir /path/to/results
    python analysis/forking_token_analysis.py --results_dir /path/to/results --exp1
    python analysis/forking_token_analysis.py --results_dir /path/to/results --exp2
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

from dynasor.core.evaluator import math_equal


# ============= FORKING TOKENS DEFINITION =============
# Based on the word cloud from https://arxiv.org/pdf/2506.01939
# Ordered by visual weight (larger = more important)

FORKING_TOKENS_TIER1 = [
    # Giants (largest font)
    "since", "thus", "suppose", "perhaps", "actually", "define", "which",
    "assume", "also", "given", "wait", "maybe", "let", "what", "however"
]

FORKING_TOKENS_TIER2 = [
    # Very large
    "using", "express", "specific", "denote", "unless", "consider", "earlier",
    "just", "the", "that", "we", "it"
]

FORKING_TOKENS_TIER3 = [
    # Medium
    "going", "solving", "recall", "calculate", "re", "note", "now", "how",
    "suppose", "according", "similar", "write", "initial", "but", "because", "yes"
]

FORKING_TOKENS_TIER4 = [
    # Small
    "complicated", "analysis", "make", "these", "so", "see", "this",
    "if", "when", "certain", "made", "considering", "therefore", "might",
    "something", "for", "calculated", "calculation", "my", "could", "found",
    "hence", "as", "adding", "leading", "think", "use", "doesn", "considered",
    "go", "try", "alternatively", "each", "or", "in"
]

# Combined list with all forking tokens
FORKING_TOKENS_ALL = (
    FORKING_TOKENS_TIER1 +
    FORKING_TOKENS_TIER2 +
    FORKING_TOKENS_TIER3 +
    FORKING_TOKENS_TIER4
)

# Create lowercase set for efficient lookup
FORKING_TOKENS_SET = set(t.lower() for t in FORKING_TOKENS_ALL)

# Tier weights for weighted analysis
FORKING_TOKEN_WEIGHTS = {}
for t in FORKING_TOKENS_TIER1:
    FORKING_TOKEN_WEIGHTS[t.lower()] = 4.0
for t in FORKING_TOKENS_TIER2:
    FORKING_TOKEN_WEIGHTS[t.lower()] = 3.0
for t in FORKING_TOKENS_TIER3:
    FORKING_TOKEN_WEIGHTS[t.lower()] = 2.0
for t in FORKING_TOKENS_TIER4:
    FORKING_TOKEN_WEIGHTS[t.lower()] = 1.0


def is_forking_token(token_str: str) -> bool:
    """Check if a token string is a forking token."""
    # Clean token: remove special chars, lowercase
    cleaned = token_str.strip().lower()
    # Remove common prefixes from tokenizers (e.g., 'Ġ' from GPT-2 style)
    cleaned = cleaned.lstrip('Ġ').lstrip('▁').lstrip('_')
    return cleaned in FORKING_TOKENS_SET


def get_forking_tier(token_str: str) -> Optional[int]:
    """Get tier (1-4) of forking token, or None if not a forking token."""
    cleaned = token_str.strip().lower().lstrip('Ġ').lstrip('▁').lstrip('_')
    if cleaned in FORKING_TOKEN_WEIGHTS:
        weight = FORKING_TOKEN_WEIGHTS[cleaned]
        if weight == 4.0:
            return 1
        elif weight == 3.0:
            return 2
        elif weight == 2.0:
            return 3
        else:
            return 4
    return None


# ============= HUMAN-READABLE REPORT SAVING =============

def trace_to_json_serializable(trace: Dict) -> Dict:
    """Convert trace dict to JSON-serializable format."""
    result = {}
    for key, value in trace.items():
        if key == 'token_ids':
            # Convert to list of ints
            result[key] = [int(t) for t in value] if value else []
        elif key == 'confs':
            # Round floats for readability
            result[key] = [round(float(c), 4) for c in value] if value else []
        elif key == 'text':
            # Truncate very long text for readability
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


def save_result_as_json(result_data: Dict, output_path: str, include_full_traces: bool = False) -> None:
    """
    Save result data as human-readable JSON file.

    Args:
        result_data: The result dictionary (same format as pickle)
        output_path: Path to save JSON file
        include_full_traces: If False, summarize traces instead of full content
    """
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
    }

    # Process traces
    traces = result_data.get('all_traces', [])
    if include_full_traces:
        json_data['all_traces'] = [trace_to_json_serializable(t) for t in traces]
    else:
        # Summarized traces (more human-readable)
        trace_summaries = []
        for i, trace in enumerate(traces):
            summary = {
                'trace_index': i,
                'extracted_answer': trace.get('extracted_answer', None),
                'num_tokens': trace.get('num_tokens', 0),
                'stop_reason': trace.get('stop_reason', ''),
                'mean_confidence': round(np.mean(trace.get('confs', [0])), 4) if trace.get('confs') else None,
                'min_confidence': round(min(trace.get('confs', [0])), 4) if trace.get('confs') else None,
                'max_confidence': round(max(trace.get('confs', [0])), 4) if trace.get('confs') else None,
            }
            trace_summaries.append(summary)
        json_data['trace_summaries'] = trace_summaries

        # Answer distribution
        answer_counts = defaultdict(int)
        for trace in traces:
            ans = trace.get('extracted_answer')
            if ans:
                answer_counts[str(ans)] += 1
        json_data['answer_distribution'] = dict(sorted(
            answer_counts.items(),
            key=lambda x: -x[1]
        )[:20])  # Top 20 answers

    # Write JSON with nice formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def convert_pickle_dir_to_json(pkl_dir: str, json_dir: str, include_full_traces: bool = False) -> None:
    """Convert all pickle files in a directory to JSON format."""
    os.makedirs(json_dir, exist_ok=True)

    pkl_files = list(Path(pkl_dir).glob("*.pkl"))
    print(f"Converting {len(pkl_files)} pickle files to JSON...")

    for pkl_file in pkl_files:
        if 'stats' in pkl_file.name or 'comparison' in pkl_file.name:
            continue

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            json_path = os.path.join(json_dir, pkl_file.stem + '.json')
            save_result_as_json(data, json_path, include_full_traces)

        except Exception as e:
            print(f"Warning: Failed to convert {pkl_file}: {e}")

    print(f"JSON files saved to {json_dir}")


# ============= DATA LOADING =============

def quick_parse(text: str) -> str:
    """Parse LaTeX text content."""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def load_results(results_dir: str, num_workers: int = None) -> Dict:
    """Load all result pickle files."""
    pkl_files = list(Path(results_dir).glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(pkl_files))

    results_by_qid = defaultdict(list)

    def load_single(filename):
        try:
            if 'stats' in filename.name or 'comparison' in filename.name:
                return None, None
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data.get('qid'), data
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            return None, None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(load_single, pkl_files)
        for qid, data in results:
            if qid is not None:
                results_by_qid[qid].append(data)

    return dict(results_by_qid)


def load_tokenizer(model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    """Load tokenizer for decoding tokens."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


# ============= EXPERIMENT 1: Forking Token Entropy Analysis =============

def analyze_forking_token_entropy(
    results_by_qid: Dict,
    tokenizer,
    only_correct: bool = True,
    window_size: int = 50  # Window around forking tokens for sample-level analysis
) -> Dict:
    """
    Exp 1: Analyze if forking tokens have higher entropy (lower confidence).

    Returns entropy distributions for:
    - Token-level: Forking tokens vs non-forking tokens
    - Sample-level: Average confidence of samples containing forking tokens
    - Per-tier breakdown

    Args:
        results_by_qid: Loaded results
        tokenizer: HuggingFace tokenizer for decoding
        only_correct: If True, only analyze traces with correct answers
        window_size: Window size around forking tokens for sample-level analysis
    """

    forking_confs = []       # Confidence values for forking tokens
    non_forking_confs = []   # Confidence values for non-forking tokens

    # Per-tier breakdown
    tier_confs = {1: [], 2: [], 3: [], 4: []}

    # Sample-level analysis: average confidence of windows around forking tokens
    sample_forking_window_confs = []      # Mean conf of windows containing forking tokens
    sample_non_forking_window_confs = []  # Mean conf of windows without forking tokens

    # Per-trace statistics
    trace_forking_ratio = []              # Ratio of forking tokens per trace
    trace_mean_forking_conf = []          # Mean confidence of forking tokens per trace
    trace_mean_non_forking_conf = []      # Mean confidence of non-forking tokens per trace

    total_traces_analyzed = 0
    total_tokens_analyzed = 0

    for qid, results_list in results_by_qid.items():
        for result in results_list:
            ground_truth = result.get('ground_truth', '')
            traces = result.get('all_traces', [])

            for trace in traces:
                answer = trace.get('extracted_answer')

                # Skip if filtering to correct only
                if only_correct:
                    if not answer or not ground_truth:
                        continue
                    try:
                        if not equal_func(answer, ground_truth):
                            continue
                    except:
                        if str(answer) != str(ground_truth):
                            continue

                token_ids = trace.get('token_ids', [])
                confs = trace.get('confs', [])

                if not token_ids or not confs or len(token_ids) != len(confs):
                    continue

                total_traces_analyzed += 1

                # Track forking token positions for this trace
                trace_forking_indices = []
                trace_forking_confs_local = []
                trace_non_forking_confs_local = []

                # Decode tokens and analyze
                for i, (tid, conf) in enumerate(zip(token_ids, confs)):
                    if tokenizer:
                        try:
                            token_str = tokenizer.decode([tid])
                        except:
                            token_str = ""
                    else:
                        # Fallback: can't decode without tokenizer
                        continue

                    total_tokens_analyzed += 1

                    tier = get_forking_tier(token_str)
                    if tier is not None:
                        forking_confs.append(conf)
                        tier_confs[tier].append(conf)
                        trace_forking_indices.append(i)
                        trace_forking_confs_local.append(conf)
                    else:
                        non_forking_confs.append(conf)
                        trace_non_forking_confs_local.append(conf)

                # === Sample-level analysis ===
                # Compute per-trace statistics
                if trace_forking_confs_local:
                    trace_mean_forking_conf.append(np.mean(trace_forking_confs_local))
                if trace_non_forking_confs_local:
                    trace_mean_non_forking_conf.append(np.mean(trace_non_forking_confs_local))

                total_tokens_in_trace = len(token_ids)
                if total_tokens_in_trace > 0:
                    trace_forking_ratio.append(len(trace_forking_indices) / total_tokens_in_trace)

                # Compute window-based confidence around forking tokens
                if trace_forking_indices and len(confs) > 0:
                    # Mark which positions are in a forking window
                    in_forking_window = set()
                    for idx in trace_forking_indices:
                        start = max(0, idx - window_size // 2)
                        end = min(len(confs), idx + window_size // 2 + 1)
                        for j in range(start, end):
                            in_forking_window.add(j)

                    # Compute mean confidence for forking windows vs non-forking regions
                    forking_window_confs = [confs[j] for j in in_forking_window]
                    non_forking_window_confs = [confs[j] for j in range(len(confs)) if j not in in_forking_window]

                    if forking_window_confs:
                        sample_forking_window_confs.append(np.mean(forking_window_confs))
                    if non_forking_window_confs:
                        sample_non_forking_window_confs.append(np.mean(non_forking_window_confs))

    # Compute statistics
    results = {
        'summary': {
            'total_traces_analyzed': total_traces_analyzed,
            'total_tokens_analyzed': total_tokens_analyzed,
            'num_forking_tokens': len(forking_confs),
            'num_non_forking_tokens': len(non_forking_confs),
            'only_correct_traces': only_correct,
        },
        'forking_tokens': {
            'mean_confidence': float(np.mean(forking_confs)) if forking_confs else None,
            'std_confidence': float(np.std(forking_confs)) if forking_confs else None,
            'median_confidence': float(np.median(forking_confs)) if forking_confs else None,
            'min_confidence': float(np.min(forking_confs)) if forking_confs else None,
            'max_confidence': float(np.max(forking_confs)) if forking_confs else None,
        },
        'non_forking_tokens': {
            'mean_confidence': float(np.mean(non_forking_confs)) if non_forking_confs else None,
            'std_confidence': float(np.std(non_forking_confs)) if non_forking_confs else None,
            'median_confidence': float(np.median(non_forking_confs)) if non_forking_confs else None,
            'min_confidence': float(np.min(non_forking_confs)) if non_forking_confs else None,
            'max_confidence': float(np.max(non_forking_confs)) if non_forking_confs else None,
        },
        'per_tier': {},
        # === SAMPLE-LEVEL ANALYSIS ===
        'sample_level': {
            # Per-trace mean confidence of forking vs non-forking tokens
            'per_trace_forking_mean': {
                'mean': float(np.mean(trace_mean_forking_conf)) if trace_mean_forking_conf else None,
                'std': float(np.std(trace_mean_forking_conf)) if trace_mean_forking_conf else None,
                'median': float(np.median(trace_mean_forking_conf)) if trace_mean_forking_conf else None,
                'count': len(trace_mean_forking_conf),
            },
            'per_trace_non_forking_mean': {
                'mean': float(np.mean(trace_mean_non_forking_conf)) if trace_mean_non_forking_conf else None,
                'std': float(np.std(trace_mean_non_forking_conf)) if trace_mean_non_forking_conf else None,
                'median': float(np.median(trace_mean_non_forking_conf)) if trace_mean_non_forking_conf else None,
                'count': len(trace_mean_non_forking_conf),
            },
            # Window-based analysis (windows around forking tokens)
            'forking_window_mean_conf': {
                'mean': float(np.mean(sample_forking_window_confs)) if sample_forking_window_confs else None,
                'std': float(np.std(sample_forking_window_confs)) if sample_forking_window_confs else None,
                'median': float(np.median(sample_forking_window_confs)) if sample_forking_window_confs else None,
                'count': len(sample_forking_window_confs),
            },
            'non_forking_window_mean_conf': {
                'mean': float(np.mean(sample_non_forking_window_confs)) if sample_non_forking_window_confs else None,
                'std': float(np.std(sample_non_forking_window_confs)) if sample_non_forking_window_confs else None,
                'median': float(np.median(sample_non_forking_window_confs)) if sample_non_forking_window_confs else None,
                'count': len(sample_non_forking_window_confs),
            },
            # Forking ratio per trace
            'forking_ratio_per_trace': {
                'mean': float(np.mean(trace_forking_ratio)) if trace_forking_ratio else None,
                'std': float(np.std(trace_forking_ratio)) if trace_forking_ratio else None,
                'min': float(np.min(trace_forking_ratio)) if trace_forking_ratio else None,
                'max': float(np.max(trace_forking_ratio)) if trace_forking_ratio else None,
            },
            'window_size': window_size,
        },
        'raw_data': {
            'forking_confs': forking_confs,
            'non_forking_confs': non_forking_confs,
            'tier_confs': tier_confs,
            'trace_mean_forking_conf': trace_mean_forking_conf,
            'trace_mean_non_forking_conf': trace_mean_non_forking_conf,
            'sample_forking_window_confs': sample_forking_window_confs,
            'sample_non_forking_window_confs': sample_non_forking_window_confs,
        }
    }

    # Per-tier statistics
    for tier in [1, 2, 3, 4]:
        confs = tier_confs[tier]
        results['per_tier'][f'tier_{tier}'] = {
            'count': len(confs),
            'mean_confidence': float(np.mean(confs)) if confs else None,
            'std_confidence': float(np.std(confs)) if confs else None,
        }

    # Hypothesis tests
    from scipy import stats

    # Token-level t-test
    if forking_confs and non_forking_confs:
        try:
            t_stat, p_value = stats.ttest_ind(forking_confs, non_forking_confs)
            results['hypothesis_test_token_level'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
            }
        except Exception as e:
            results['hypothesis_test_token_level'] = {'error': str(e)}

    # Sample-level t-test (per-trace mean confidence)
    if trace_mean_forking_conf and trace_mean_non_forking_conf:
        try:
            t_stat, p_value = stats.ttest_ind(trace_mean_forking_conf, trace_mean_non_forking_conf)
            results['hypothesis_test_sample_level'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
                'description': 'Comparing per-trace average confidence of forking vs non-forking tokens',
            }
        except Exception as e:
            results['hypothesis_test_sample_level'] = {'error': str(e)}

    # Window-level t-test
    if sample_forking_window_confs and sample_non_forking_window_confs:
        try:
            t_stat, p_value = stats.ttest_ind(sample_forking_window_confs, sample_non_forking_window_confs)
            results['hypothesis_test_window_level'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
                'description': f'Comparing mean confidence of {window_size}-token windows around forking tokens vs other regions',
            }
        except Exception as e:
            results['hypothesis_test_window_level'] = {'error': str(e)}

    return results


def plot_forking_histograms(analysis_results: Dict, output_path: str) -> None:
    """Plot histograms of forking vs non-forking token confidence (token-level and sample-level)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping histogram plot")
        return

    forking = analysis_results['raw_data']['forking_confs']
    non_forking = analysis_results['raw_data']['non_forking_confs']

    # Sample-level data
    trace_forking = analysis_results['raw_data'].get('trace_mean_forking_conf', [])
    trace_non_forking = analysis_results['raw_data'].get('trace_mean_non_forking_conf', [])

    if not forking or not non_forking:
        print("No data to plot")
        return

    # Sample if too many points for token-level
    max_points = 100000
    if len(forking) > max_points:
        forking = np.random.choice(forking, max_points, replace=False)
    if len(non_forking) > max_points:
        non_forking = np.random.choice(non_forking, max_points, replace=False)

    # Create 2x2 figure: top row = token-level, bottom row = sample-level
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Row 1: Token-Level Analysis ===
    # Histogram 1: Forking tokens (token-level)
    axes[0, 0].hist(forking, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(np.mean(forking), color='darkred', linestyle='--',
                       label=f'Mean: {np.mean(forking):.3f}')
    axes[0, 0].set_xlabel('Confidence (negative log prob)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Token-Level: Forking Tokens (n={len(forking):,})')
    axes[0, 0].legend()

    # Histogram 2: Non-forking tokens (token-level)
    axes[0, 1].hist(non_forking, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(np.mean(non_forking), color='darkblue', linestyle='--',
                       label=f'Mean: {np.mean(non_forking):.3f}')
    axes[0, 1].set_xlabel('Confidence (negative log prob)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Token-Level: Non-Forking Tokens (n={len(non_forking):,})')
    axes[0, 1].legend()

    # === Row 2: Sample-Level Analysis ===
    if trace_forking and trace_non_forking:
        # Histogram 3: Per-trace mean confidence of forking tokens
        axes[1, 0].hist(trace_forking, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].axvline(np.mean(trace_forking), color='darkred', linestyle='--',
                           label=f'Mean: {np.mean(trace_forking):.3f}')
        axes[1, 0].set_xlabel('Mean Confidence per Trace')
        axes[1, 0].set_ylabel('Count (Traces)')
        axes[1, 0].set_title(f'Sample-Level: Avg Conf of Forking Tokens (n={len(trace_forking):,} traces)')
        axes[1, 0].legend()

        # Histogram 4: Per-trace mean confidence of non-forking tokens
        axes[1, 1].hist(trace_non_forking, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(trace_non_forking), color='darkblue', linestyle='--',
                           label=f'Mean: {np.mean(trace_non_forking):.3f}')
        axes[1, 1].set_xlabel('Mean Confidence per Trace')
        axes[1, 1].set_ylabel('Count (Traces)')
        axes[1, 1].set_title(f'Sample-Level: Avg Conf of Non-Forking Tokens (n={len(trace_non_forking):,} traces)')
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No sample-level data', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'No sample-level data', ha='center', va='center')

    plt.suptitle('Exp 1: Forking vs Non-Forking Token Confidence Distribution\n(Top: Token-Level, Bottom: Sample-Level)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to {output_path}")


# ============= EXPERIMENT 2: Threshold Search =============

def threshold_weight_function(conf: float, threshold: float) -> float:
    """
    Apply different weighting based on threshold:
    - conf > threshold: weight = conf (y=x, reward confident tokens)
    - conf <= threshold: weight = 1/conf (y=1/x, penalize uncertain tokens)

    Note: conf is negative log prob, so LOWER = more confident
    """
    if conf <= threshold:
        # High confidence region: use y=x (or 1/conf since lower conf = more confident)
        return 1.0 / (conf + 0.01)
    else:
        # Low confidence region: use y=1/x (penalize)
        return 1.0 / (conf + 0.01)


def smooth_threshold_weight(conf: float, threshold: float, steepness: float = 2.0) -> float:
    """
    Smooth sigmoid blend between y=x and y=1/x behaviors.

    - conf < threshold: weight proportional to 1/conf (reward confident)
    - conf > threshold: weight proportional to 1/conf (but with penalty)

    Uses sigmoid to smooth the transition.
    """
    if conf <= 0:
        return 1.0

    # Sigmoid blend factor
    blend = 1.0 / (1.0 + np.exp(-steepness * (conf - threshold)))

    # y=x behavior (for low conf = high confidence)
    y_x = 1.0 / (conf + 0.01)

    # y=1/x behavior (for high conf = low confidence) - additional penalty
    y_inv = 1.0 / ((conf + 0.01) ** 2)

    return (1 - blend) * y_x + blend * y_inv


def evaluate_threshold(
    precomputed: Dict,
    threshold: float,
    weight_fn_type: str = 'smooth'
) -> Dict:
    """Evaluate a specific threshold value."""
    correct = 0
    total = 0

    for qid, data in precomputed.items():
        traces = data['traces']
        answers = data['answers']
        ground_truth = data['ground_truth']

        if not answers:
            continue

        total += 1

        # Compute weights using threshold
        weights = []
        for trace in traces:
            confs = trace.get('confs', [])
            if not confs:
                weights.append(1.0)
                continue

            if weight_fn_type == 'smooth':
                token_weights = [smooth_threshold_weight(c, threshold) for c in confs]
            else:
                token_weights = [threshold_weight_function(c, threshold) for c in confs]

            weights.append(np.mean(token_weights))

        # Weighted vote
        answer_weights = defaultdict(float)
        for answer, weight in zip(answers, weights):
            if answer is not None:
                answer_weights[str(answer)] += weight

        if not answer_weights:
            continue

        predicted = max(answer_weights.keys(), key=lambda x: answer_weights[x])

        try:
            is_correct = equal_func(predicted, ground_truth)
        except:
            is_correct = str(predicted) == str(ground_truth)

        if is_correct:
            correct += 1

    return {
        'threshold': threshold,
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
    }


def brute_force_threshold_search(
    results_by_qid: Dict,
    threshold_range: Tuple[float, float] = (0, 15),
    step: float = 0.1,
    weight_fn_type: str = 'smooth'
) -> Dict:
    """
    Exp 2: Brute force search for optimal threshold.

    Tests thresholds from 0 to 15 with 0.1 spacing.

    Returns:
        Dict with all results and best threshold
    """
    # Precompute trace data
    precomputed = {}
    for qid, results_list in results_by_qid.items():
        all_traces = []
        ground_truth = None

        for result in results_list:
            all_traces.extend(result.get('all_traces', []))
            if ground_truth is None:
                ground_truth = result.get('ground_truth', '')

        if not all_traces or not ground_truth:
            continue

        answers = []
        valid_traces = []
        for trace in all_traces:
            answer = trace.get('extracted_answer')
            if answer:
                answers.append(answer)
                valid_traces.append(trace)

        if answers:
            precomputed[qid] = {
                'traces': valid_traces,
                'ground_truth': ground_truth,
                'answers': answers,
            }

    print(f"Precomputed {len(precomputed)} questions for threshold search")

    # Generate thresholds
    start, end = threshold_range
    thresholds = np.arange(start, end + step, step)
    print(f"Testing {len(thresholds)} thresholds from {start} to {end}")

    # Evaluate each threshold
    results = []
    best_accuracy = 0
    best_threshold = None

    for threshold in thresholds:
        result = evaluate_threshold(precomputed, threshold, weight_fn_type)
        results.append(result)

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_threshold = threshold

        if int(threshold * 10) % 10 == 0:  # Print every 1.0
            print(f"  Threshold {threshold:.1f}: {100*result['accuracy']:.2f}%")

    return {
        'all_results': results,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'threshold_range': threshold_range,
        'step': step,
        'weight_fn_type': weight_fn_type,
    }


def plot_threshold_search(search_results: Dict, output_path: str) -> None:
    """Plot threshold search results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    results = search_results['all_results']
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, accuracies, 'b-', linewidth=1.5, alpha=0.8)

    # Mark best threshold
    best_t = search_results['best_threshold']
    best_acc = search_results['best_accuracy'] * 100
    plt.axvline(best_t, color='red', linestyle='--', alpha=0.7,
                label=f'Best: t={best_t:.2f} ({best_acc:.2f}%)')
    plt.scatter([best_t], [best_acc], color='red', s=100, zorder=5)

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Exp 2: Threshold Search for Sampling Weight Function')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Threshold search plot saved to {output_path}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser(description='Forking Token Analysis')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing result pickle files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results_dir/analysis)')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                        help='Model name for tokenizer')

    # Tasks
    parser.add_argument('--convert_to_json', action='store_true',
                        help='Task 1: Convert pickle files to JSON')
    parser.add_argument('--include_full_traces', action='store_true',
                        help='Include full trace data in JSON (larger files)')
    parser.add_argument('--exp1', action='store_true',
                        help='Run Exp 1: Forking token entropy analysis')
    parser.add_argument('--exp2', action='store_true',
                        help='Run Exp 2: Threshold brute force search')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')

    # Exp 1 options
    parser.add_argument('--only_correct', action='store_true', default=True,
                        help='Exp 1: Only analyze correct traces')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Exp 1: Window size around forking tokens for sample-level analysis')

    # Exp 2 options
    parser.add_argument('--threshold_start', type=float, default=0.0)
    parser.add_argument('--threshold_end', type=float, default=15.0)
    parser.add_argument('--threshold_step', type=float, default=0.1)

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all if specified
    if args.all:
        args.convert_to_json = True
        args.exp1 = True
        args.exp2 = True

    # Task 1: Convert to JSON
    if args.convert_to_json:
        print("\n" + "="*60)
        print("TASK 1: Converting pickle files to JSON")
        print("="*60)
        json_dir = os.path.join(args.output_dir, 'json_reports')
        convert_pickle_dir_to_json(args.results_dir, json_dir, args.include_full_traces)

    # Load data for experiments
    if args.exp1 or args.exp2:
        print("\nLoading results...")
        results_by_qid = load_results(args.results_dir)
        print(f"Loaded {len(results_by_qid)} questions")

    # Exp 1: Forking token entropy analysis
    if args.exp1:
        print("\n" + "="*60)
        print("EXP 1: Forking Token Entropy Analysis")
        print("="*60)

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(args.model)

        if tokenizer is None:
            print("ERROR: Could not load tokenizer. Exp 1 requires tokenizer to decode tokens.")
        else:
            analysis = analyze_forking_token_entropy(
                results_by_qid,
                tokenizer,
                only_correct=args.only_correct,
                window_size=args.window_size
            )

            # Print summary
            print("\n--- Summary ---")
            print(f"Total traces analyzed: {analysis['summary']['total_traces_analyzed']}")
            print(f"Total tokens analyzed: {analysis['summary']['total_tokens_analyzed']}")
            print(f"Forking tokens: {analysis['summary']['num_forking_tokens']}")
            print(f"Non-forking tokens: {analysis['summary']['num_non_forking_tokens']}")

            print("\n--- Forking Tokens ---")
            ft = analysis['forking_tokens']
            print(f"  Mean confidence: {ft['mean_confidence']:.4f}" if ft['mean_confidence'] else "  No data")
            print(f"  Std: {ft['std_confidence']:.4f}" if ft['std_confidence'] else "")

            print("\n--- Non-Forking Tokens (Token-Level) ---")
            nft = analysis['non_forking_tokens']
            print(f"  Mean confidence: {nft['mean_confidence']:.4f}" if nft['mean_confidence'] else "  No data")
            print(f"  Std: {nft['std_confidence']:.4f}" if nft['std_confidence'] else "")

            # Sample-level results
            print("\n" + "="*60)
            print("SAMPLE-LEVEL ANALYSIS")
            print("="*60)

            sl = analysis.get('sample_level', {})
            if sl:
                ptf = sl.get('per_trace_forking_mean', {})
                ptnf = sl.get('per_trace_non_forking_mean', {})

                print("\n--- Per-Trace Mean Confidence of Forking Tokens ---")
                if ptf.get('mean') is not None:
                    print(f"  Mean across traces: {ptf['mean']:.4f}")
                    print(f"  Std: {ptf['std']:.4f}")
                    print(f"  Median: {ptf['median']:.4f}")
                    print(f"  Count: {ptf['count']} traces")
                else:
                    print("  No data")

                print("\n--- Per-Trace Mean Confidence of Non-Forking Tokens ---")
                if ptnf.get('mean') is not None:
                    print(f"  Mean across traces: {ptnf['mean']:.4f}")
                    print(f"  Std: {ptnf['std']:.4f}")
                    print(f"  Median: {ptnf['median']:.4f}")
                    print(f"  Count: {ptnf['count']} traces")
                else:
                    print("  No data")

                # Window-based analysis
                fwm = sl.get('forking_window_mean_conf', {})
                nfwm = sl.get('non_forking_window_mean_conf', {})
                window_size = sl.get('window_size', 50)

                print(f"\n--- Window-Based Analysis (window={window_size} tokens) ---")
                if fwm.get('mean') is not None:
                    print(f"  Windows around forking tokens: mean={fwm['mean']:.4f}, std={fwm['std']:.4f}")
                if nfwm.get('mean') is not None:
                    print(f"  Windows without forking tokens: mean={nfwm['mean']:.4f}, std={nfwm['std']:.4f}")

                # Forking ratio
                fr = sl.get('forking_ratio_per_trace', {})
                if fr.get('mean') is not None:
                    print(f"\n--- Forking Ratio per Trace ---")
                    print(f"  Mean: {fr['mean']*100:.2f}% of tokens are forking tokens")
                    print(f"  Range: {fr['min']*100:.2f}% - {fr['max']*100:.2f}%")

            # Hypothesis tests
            print("\n" + "="*60)
            print("HYPOTHESIS TESTS")
            print("="*60)

            if 'hypothesis_test_token_level' in analysis and 'p_value' in analysis['hypothesis_test_token_level']:
                ht = analysis['hypothesis_test_token_level']
                print(f"\n--- Token-Level T-Test ---")
                print(f"  t-statistic: {ht['t_statistic']:.4f}")
                print(f"  p-value: {ht['p_value']:.6f}")
                print(f"  Significant at 0.05: {ht['significant_at_0.05']}")

            if 'hypothesis_test_sample_level' in analysis and 'p_value' in analysis['hypothesis_test_sample_level']:
                ht = analysis['hypothesis_test_sample_level']
                print(f"\n--- Sample-Level T-Test (per-trace mean conf) ---")
                print(f"  t-statistic: {ht['t_statistic']:.4f}")
                print(f"  p-value: {ht['p_value']:.6f}")
                print(f"  Significant at 0.05: {ht['significant_at_0.05']}")

            if 'hypothesis_test_window_level' in analysis and 'p_value' in analysis['hypothesis_test_window_level']:
                ht = analysis['hypothesis_test_window_level']
                print(f"\n--- Window-Level T-Test ---")
                print(f"  t-statistic: {ht['t_statistic']:.4f}")
                print(f"  p-value: {ht['p_value']:.6f}")
                print(f"  Significant at 0.05: {ht['significant_at_0.05']}")

            # Save results
            exp1_path = os.path.join(args.output_dir, 'exp1_forking_analysis.json')
            # Remove raw data for JSON (too large)
            analysis_save = {k: v for k, v in analysis.items() if k != 'raw_data'}
            with open(exp1_path, 'w') as f:
                json.dump(analysis_save, f, indent=2)
            print(f"\nExp 1 results saved to {exp1_path}")

            # Plot histograms
            hist_path = os.path.join(args.output_dir, 'exp1_histograms.png')
            plot_forking_histograms(analysis, hist_path)

    # Exp 2: Threshold search
    if args.exp2:
        print("\n" + "="*60)
        print("EXP 2: Threshold Brute Force Search")
        print("="*60)

        search_results = brute_force_threshold_search(
            results_by_qid,
            threshold_range=(args.threshold_start, args.threshold_end),
            step=args.threshold_step,
        )

        print(f"\n--- Best Result ---")
        print(f"Best threshold: {search_results['best_threshold']:.2f}")
        print(f"Best accuracy: {100*search_results['best_accuracy']:.2f}%")

        # Save results
        exp2_path = os.path.join(args.output_dir, 'exp2_threshold_search.json')
        with open(exp2_path, 'w') as f:
            json.dump(search_results, f, indent=2)
        print(f"\nExp 2 results saved to {exp2_path}")

        # Plot
        plot_path = os.path.join(args.output_dir, 'exp2_threshold_plot.png')
        plot_threshold_search(search_results, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
