#!/usr/bin/env python3
"""
Exp 3: CDG Hyperparameter Sweep

Standalone script that reads directly from Yu's pickle directories.
Sweeps alpha, beta, and position_pct to find optimal CDG parameters.

Beta values: 1, 2, 3, 4, 5, 10, 15, 20, 25, 30
Alpha values: 0.5, 1.0
Position pct: 10, 20 (prioritize alpha=0.5, position=10 first)

Formula: score = count^alpha * mean(mean_conf + beta * gradient)
Where gradient = mean(last P%) - mean(first P%)

Usage:
    python exp_cdg_sweep.py              # Run full sweep
    python exp_cdg_sweep.py --no-compute # Load cache, print tables only
    python exp_cdg_sweep.py --resume     # Resume from partial cache if interrupted

Cache files:
    results/cache/cdg_sweep/sweep_cache.json    # Final cache for --no-compute
    results/cache/cdg_sweep/partial_cache.json  # Partial cache for --resume
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import json
from datetime import datetime

# Import centralized config
from config import (
    DATASETS, RESULT_BASE_PATH, BETA_VALUES, ALPHA_VALUES, POSITION_PCT_VALUES
)

# dynasor for math evaluation: pip install git+https://github.com/hao-ai-lab/Dynasor.git
from dynasor.core.evaluator import math_equal

# Output directory for results (local - small files only)
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'cache' / 'cdg_sweep'

# Stable cache file (no timestamp - for quick figure regeneration)
CACHE_FILE = OUTPUT_DIR / 'sweep_cache.json'

# Partial cache file (for resume support)
PARTIAL_CACHE_FILE = OUTPUT_DIR / 'partial_cache.json'

# Config file path (compatible with HyperparamConfig for run_all_experiments.sh)
# This goes to the results directory so other scripts can read it
CONFIG_OUTPUT_DIR = Path(RESULT_BASE_PATH)
CONFIG_FILENAME = 'cdg_hyperparams.json'


# ============================================================================
# HELPER FUNCTIONS (matching Yu's implementation exactly)
# ============================================================================

def quick_parse(text: str) -> str:
    """Parse LaTeX text command."""
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
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        try:
            return math_equal(answer, ground_truth)
        except:
            return str(answer).strip() == str(ground_truth).strip()


def compute_gradient(confs, percentile=10):
    """
    Compute gradient: mean(last percentile%) - mean(first percentile%)
    Exactly matching Yu's implementation.
    """
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


# ============================================================================
# DATA LOADING (deterministic - matching Yu's implementation)
# ============================================================================

def load_dataset(results_dir: str, position_pct: int = 10) -> dict:
    """
    Load dataset and precompute metrics for all traces.
    Exactly matching Yu's load_dataset function.
    """
    results_dir = Path(results_dir)

    # Find pickle files - try direct glob first, then subdirectories
    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    if not pkl_files:
        return {}

    question_data = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        qid = data['qid']
        gt = data['ground_truth']

        traces_list = []
        for trace in data['all_traces']:
            answer = trace.get('extracted_answer')
            confs = trace.get('confs', [])

            if answer is not None and len(confs) >= 10:
                mean_conf = np.mean(confs)
                gradient = compute_gradient(confs, percentile=position_pct)
                traces_list.append({
                    'answer': answer,
                    'mean_conf': mean_conf,
                    'gradient': gradient,
                })

        question_data[qid] = {'gt': gt, 'traces': traces_list}

    return question_data


# ============================================================================
# CDG VOTING (exactly matching Yu's implementation)
# ============================================================================

def cdg_voting(question_data: dict, alpha: float = 0.5, beta: float = 10) -> tuple:
    """
    CDG (Count-Dampened Gradient) Voting.

    Formula: score = count^alpha * mean(mean_conf + beta * gradient)

    Returns:
        (correct_count, total_count, per_question_results)
    """
    correct = 0
    total = 0
    results = {}

    for qid, qdata in question_data.items():
        gt = qdata['gt']
        traces = qdata['traces']

        if not traces:
            continue

        total += 1

        answer_scores = defaultdict(list)
        for t in traces:
            score = t['mean_conf'] + beta * t['gradient']
            answer_scores[t['answer']].append(score)

        if answer_scores:
            final_scores = {
                ans: (len(scores) ** alpha) * np.mean(scores)
                for ans, scores in answer_scores.items()
            }
            winner = max(final_scores.items(), key=lambda x: x[1])[0]
        else:
            winner = None

        is_correct = equal_func(str(winner), str(gt)) if winner else False
        if is_correct:
            correct += 1

        results[qid] = {
            'correct': is_correct,
            'winner': winner,
            'gt': gt,
        }

    return correct, total, results


# ============================================================================
# PARTIAL CACHE FUNCTIONS
# ============================================================================

def save_partial_cache(all_results: list, completed_configs: list, timestamp: str):
    """Save partial results for resume support."""
    with open(PARTIAL_CACHE_FILE, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'beta_values': BETA_VALUES,
            'alpha_values': ALPHA_VALUES,
            'position_pct_values': POSITION_PCT_VALUES,
            'results': all_results,
            'completed_configs': completed_configs,  # list of (position_pct, alpha) tuples
            'is_partial': True,
        }, f, indent=2)
    print(f'  [Partial cache saved: {len(completed_configs)} configs done]')


def load_partial_cache():
    """Load partial cache if available."""
    if not PARTIAL_CACHE_FILE.exists():
        return [], []

    with open(PARTIAL_CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

    if not cache_data.get('is_partial', False):
        return [], []

    completed = cache_data.get('completed_configs', [])
    # Convert lists back to tuples for comparison
    completed = [tuple(c) for c in completed]
    print(f'Found partial cache: {len(completed)} configs completed')
    return cache_data.get('results', []), completed


# ============================================================================
# MAIN SWEEP
# ============================================================================

def run_sweep(resume=False):
    """Run full hyperparameter sweep."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 100)
    print('EXP 3: CDG HYPERPARAMETER SWEEP')
    print('=' * 100)
    print()
    print(f'Beta values: {BETA_VALUES}')
    print(f'Alpha values: {ALPHA_VALUES}')
    print(f'Position pct values: {POSITION_PCT_VALUES}')
    print(f'Models: {list(DATASETS.keys())}')
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    # Try to resume from partial cache
    all_results = []
    completed_configs = []
    if resume:
        all_results, completed_configs = load_partial_cache()
        if completed_configs:
            print(f'Resuming from partial cache. Completed: {completed_configs}')
        else:
            print('No partial cache found, starting fresh.')

    # Prioritize alpha=0.5, position=10 first
    position_order = [10, 20]
    alpha_order = [0.5, 1.0]

    for position_pct in position_order:
        for alpha in alpha_order:
            config_key = (position_pct, alpha)

            # Skip already completed configs
            if config_key in completed_configs:
                print()
                print('#' * 100)
                print(f'# POSITION PCT = {position_pct}, ALPHA = {alpha} [SKIPPED - in cache]')
                print('#' * 100)
                continue

            print()
            print('#' * 100)
            print(f'# POSITION PCT = {position_pct}')
            print('#' * 100)

            print()
            print(f'--- Alpha = {alpha}, Position = {position_pct} ---')
            print()

            for model_name, datasets in DATASETS.items():
                print(f'\n{model_name.upper()}')
                print('-' * 80)

                # Header
                header = f"{'Dataset':<15}"
                for beta in BETA_VALUES:
                    header += f"β={beta:<6}"
                print(header)

                model_data = {}

                for dataset_name, dataset_path in datasets.items():
                    # Load data once per dataset/position_pct combination
                    question_data = load_dataset(dataset_path, position_pct=position_pct)

                    if not question_data:
                        print(f"{dataset_name:<15} [NO DATA]")
                        continue

                    row = f"{dataset_name:<15}"

                    for beta in BETA_VALUES:
                        correct, total, _ = cdg_voting(question_data, alpha=alpha, beta=beta)
                        acc = correct / total if total > 0 else 0
                        row += f"{100*acc:.1f}%   "

                        # Store result
                        result = {
                            'model': model_name,
                            'dataset': dataset_name,
                            'alpha': alpha,
                            'beta': beta,
                            'position_pct': position_pct,
                            'correct': correct,
                            'total': total,
                            'accuracy': acc,
                        }
                        all_results.append(result)

                    print(row)

            # Save partial cache after each (position_pct, alpha) config completes
            completed_configs.append(config_key)
            save_partial_cache(all_results, completed_configs, timestamp)

    # Save all results to JSON
    output_file = OUTPUT_DIR / f'cdg_sweep_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'beta_values': BETA_VALUES,
            'alpha_values': ALPHA_VALUES,
            'position_pct_values': POSITION_PCT_VALUES,
            'results': all_results,
        }, f, indent=2)
    print(f'\nResults saved to: {output_file}')

    # Find and print best parameters per model
    print()
    print('=' * 100)
    print('BEST PARAMETERS PER MODEL')
    print('=' * 100)

    best_params = {}
    for model_name in DATASETS.keys():
        model_results = [r for r in all_results if r['model'] == model_name]

        # Aggregate across datasets for each param combination
        param_scores = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in model_results:
            key = (r['alpha'], r['beta'], r['position_pct'])
            param_scores[key]['correct'] += r['correct']
            param_scores[key]['total'] += r['total']

        # Find best
        best_key = None
        best_acc = -1
        for key, scores in param_scores.items():
            acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_key = key

        if best_key:
            alpha, beta, position_pct = best_key
            scores = param_scores[best_key]
            best_params[model_name] = {
                'alpha': alpha,
                'beta': beta,
                'position_pct': position_pct,
                'correct': scores['correct'],
                'total': scores['total'],
                'accuracy': best_acc,
            }
            print(f"{model_name:<15} alpha={alpha}, beta={beta}, position_pct={position_pct} "
                  f"=> {scores['correct']}/{scores['total']} ({100*best_acc:.1f}%)")

    # Save best params (simple format)
    best_params_file = OUTPUT_DIR / f'best_params_{timestamp}.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'\nBest params saved to: {best_params_file}')

    # Also save in HyperparamConfig format (compatible with run_all_experiments.sh)
    hyperparam_config = {
        'version': '1.0',
        'updated_at': datetime.now().isoformat(),
        'defaults': {
            'alpha': 0.5,
            'beta': 10,
            'position_pct': 20,
        },
        'models': {}
    }

    for model_name, params in best_params.items():
        hyperparam_config['models'][model_name] = {
            'alpha': params['alpha'],
            'beta': params['beta'],
            'position_pct': params['position_pct'],
            'accuracy': params['accuracy'],
            'tuned_at': datetime.now().isoformat(),
            'tuning_details': {
                'correct': params['correct'],
                'total': params['total'],
                'sweep_type': 'beta',
                'beta_values': BETA_VALUES,
                'alpha_values': ALPHA_VALUES,
                'position_pct_values': POSITION_PCT_VALUES,
            }
        }

    # Save to CONFIG_OUTPUT_DIR (Yu's directory)
    CONFIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_file = CONFIG_OUTPUT_DIR / CONFIG_FILENAME
    with open(config_file, 'w') as f:
        json.dump(hyperparam_config, f, indent=2)
    print(f'HyperparamConfig saved to: {config_file}')

    # Also save locally
    local_config_file = OUTPUT_DIR / CONFIG_FILENAME
    with open(local_config_file, 'w') as f:
        json.dump(hyperparam_config, f, indent=2)
    print(f'Local copy saved to: {local_config_file}')

    # Save to stable cache file (for --figures-only mode)
    cache_data = {
        'timestamp': timestamp,
        'beta_values': BETA_VALUES,
        'alpha_values': ALPHA_VALUES,
        'position_pct_values': POSITION_PCT_VALUES,
        'results': all_results,
        'best_params': best_params,
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f'\n*** CACHE SAVED: {CACHE_FILE} ***')
    print('Use --figures-only to regenerate figures without re-running sweep')

    # Clean up partial cache
    if PARTIAL_CACHE_FILE.exists():
        PARTIAL_CACHE_FILE.unlink()
        print('Partial cache cleaned up.')

    return all_results, best_params


def load_cache():
    """Load results from cache file."""
    if not CACHE_FILE.exists():
        print(f'ERROR: Cache file not found: {CACHE_FILE}')
        print('Run without --figures-only first to generate cache.')
        return None, None

    with open(CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

    print(f'Loaded cache from: {CACHE_FILE}')
    print(f'  Timestamp: {cache_data.get("timestamp", "unknown")}')
    print(f'  Results: {len(cache_data["results"])} entries')

    return cache_data['results'], cache_data.get('best_params', {})


def print_tables_from_cache(all_results: list):
    """Print sweep tables from cached results (same format as during computation)."""
    print()
    print('=' * 100)
    print('SWEEP RESULTS (from cache)')
    print('=' * 100)

    for position_pct in POSITION_PCT_VALUES:
        print()
        print('#' * 100)
        print(f'# POSITION PCT = {position_pct}')
        print('#' * 100)

        for alpha in ALPHA_VALUES:
            print()
            print(f'--- Alpha = {alpha}, Position = {position_pct} ---')
            print()

            for model_name in DATASETS.keys():
                print(f'\n{model_name.upper()}')
                print('-' * 80)

                # Header
                header = f"{'Dataset':<15}"
                for beta in BETA_VALUES:
                    header += f"β={beta:<6}"
                print(header)

                for dataset_name in DATASETS[model_name].keys():
                    row = f"{dataset_name:<15}"

                    for beta in BETA_VALUES:
                        # Find result in cache
                        r = next((x for x in all_results
                                  if x['model'] == model_name and x['dataset'] == dataset_name
                                  and x['alpha'] == alpha and x['beta'] == beta
                                  and x['position_pct'] == position_pct), None)
                        if r:
                            row += f"{100*r['accuracy']:.1f}%   "
                        else:
                            row += f"{'N/A':<7}"
                    print(row)

    # Print best params
    print()
    print('=' * 100)
    print('BEST PARAMETERS PER MODEL')
    print('=' * 100)

    for model_name in DATASETS.keys():
        model_results = [r for r in all_results if r['model'] == model_name]
        if not model_results:
            continue

        # Find best
        best = max(model_results, key=lambda x: x['accuracy'])
        total_correct = sum(r['correct'] for r in model_results
                           if r['alpha'] == best['alpha'] and r['beta'] == best['beta']
                           and r['position_pct'] == best['position_pct'])
        total = sum(r['total'] for r in model_results
                   if r['alpha'] == best['alpha'] and r['beta'] == best['beta']
                   and r['position_pct'] == best['position_pct'])

        print(f"{model_name:<15} alpha={best['alpha']}, beta={best['beta']}, position_pct={best['position_pct']} "
              f"=> {total_correct}/{total} ({100*best['accuracy']:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDG Hyperparameter Sweep')
    parser.add_argument('--no-compute', action='store_true',
                        help='Load from cache, print tables & regenerate figures (no computation)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from partial cache (if interrupted)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_compute:
        print('=' * 100)
        print('NO-COMPUTE MODE: Loading from cache')
        print('=' * 100)
        all_results, best_params = load_cache()
        if all_results is None:
            sys.exit(1)
        print_tables_from_cache(all_results)
    else:
        all_results, best_params = run_sweep(resume=args.resume)
