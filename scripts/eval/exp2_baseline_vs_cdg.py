#!/usr/bin/env python3
"""
Exp 2: Baseline Methods vs CDG Comparison

Compares four methods across different trace budgets (8 -> 512):
1. Majority Vote - simple count-based voting
2. Mean Weighted - weighted vote using mean confidence
3. Top10 Tail Filtered - DeepConf's best method (filter top 10% by tail conf)
4. CDG - Count-Dampened Gradient (our method)

Outputs:
1. Scaling trend figure: accuracy vs trace count for each method
2. Scaling trend table: detailed numbers for all trace counts
3. Full budget (512) comparison table: all methods × all datasets

Usage:
    python exp2_baseline_vs_cdg.py              # Run full analysis
    python exp2_baseline_vs_cdg.py --no-compute # Load cache, print tables & regenerate figures
    python exp2_baseline_vs_cdg.py --resume     # Resume from partial cache if interrupted

Cache files:
    results/exp2_baseline_vs_cdg/cache.json         # Final cache for --no-compute
    results/exp2_baseline_vs_cdg/partial_cache.json # Partial cache for --resume
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys
import json
import random
from datetime import datetime

# Use dynasor from conda
sys.path.insert(0, '/mnt/yuenvs/deepconf/lib/python3.10/site-packages')
from dynasor.core.evaluator import math_equal


# ============================================================================
# HARDCODED DATA PATHS (Yu's directories)
# ============================================================================

DATASETS = {
    'deepseek8b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2024_run_512_results_deepseek8b',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/aime_2025_run_512_results_newpara',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/bruno_2025_run_512_results_deepseek8b',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/deepseek/hmmt_feb_2025_run_512_results',
    },
    'gptoss20b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason',
    },
    'gemma3_27b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2024_512',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_aime2025_512',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_bruno2025_512',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforGEMMA/gemma3_27b_hmmt2025_512',
    },
    'qwq32b': {
        'aime2024': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2024_512',
        'aime2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_aime2025_512',
        'bruno2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_bruno2025_512',
        'hmmt2025': '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/result_ready_paper/resultsforqwq/qwq32b_hmmt2025_512',
    },
}

# CDG optimal parameters per model
CDG_PARAMS = {
    'deepseek8b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gptoss20b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gemma3_27b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
    'qwq32b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
}

TRACE_COUNTS = [8, 16, 32, 64, 128, 256, 512]
NUM_SUBSAMPLES = 5  # Number of random subsamples for variance estimation
RANDOM_SEED = 42

METHODS = ['majority', 'mean_weighted', 'top10_tail', 'cdg']
METHOD_LABELS = {
    'majority': 'Majority',
    'mean_weighted': 'Mean Weighted',
    'top10_tail': 'Top10 Tail',
    'cdg': 'CDG',
}

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'exp2_baseline_vs_cdg'

# Stable cache file (no timestamp - for quick figure regeneration)
CACHE_FILE = OUTPUT_DIR / 'cache.json'

# Partial cache file (for resume support)
PARTIAL_CACHE_FILE = OUTPUT_DIR / 'partial_cache.json'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_parse(text: str) -> str:
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    answer = quick_parse(str(answer))
    ground_truth = quick_parse(str(ground_truth))
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    try:
        return math_equal(answer, ground_truth)
    except:
        return str(answer).strip() == str(ground_truth).strip()


def calculate_mean_confidence(trace):
    try:
        if 'confs' in trace and trace['confs']:
            return np.mean(trace['confs'])
        return 0.0
    except:
        return 0.0


def calculate_tail_confidence(trace, tail_tokens=2048):
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except:
        return 0.0


def compute_gradient(confs, percentile=10):
    if not confs or len(confs) < 10:
        return 0.0
    n = len(confs)
    cutoff = max(1, int(n * percentile / 100))
    return np.mean(confs[-cutoff:]) - np.mean(confs[:cutoff])


def simple_majority_vote(answers):
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def weighted_majority_vote(answers, weights):
    if not answers:
        return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights:
        return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def filter_top_confidence(traces, top_percent=0.1):
    if not traces:
        return []
    confidences = [calculate_tail_confidence(t) for t in traces]
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    return [t for t, c in zip(traces, confidences) if c >= threshold]


def cdg_vote(traces, alpha, beta, position_pct):
    if not traces:
        return None
    answer_scores = defaultdict(list)
    for trace in traces:
        answer = trace.get('extracted_answer')
        confs = trace.get('confs', [])
        if answer is not None and len(confs) >= 10:
            mean_conf = np.mean(confs)
            gradient = compute_gradient(confs, percentile=position_pct)
            score = mean_conf + beta * gradient
            answer_scores[answer].append(score)
    if not answer_scores:
        return None
    final_scores = {
        ans: (len(scores) ** alpha) * np.mean(scores)
        for ans, scores in answer_scores.items()
    }
    return max(final_scores.items(), key=lambda x: x[1])[0]


# ============================================================================
# DATA LOADING AND EVALUATION
# ============================================================================

def load_all_questions(dataset_path: str) -> list:
    """Load all questions from a dataset."""
    results_dir = Path(dataset_path)
    pkl_files = sorted(results_dir.glob('qid*.pkl'))
    if not pkl_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                pkl_files = sorted(subdir.glob('qid*.pkl'))
                if pkl_files:
                    break

    questions = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        questions.append(data)

    return questions


def evaluate_methods(traces: list, gt: str, cdg_params: dict) -> dict:
    """Evaluate all methods on a set of traces."""
    valid_traces = [t for t in traces if t.get('extracted_answer')]
    if not valid_traces:
        return None

    answers = [t['extracted_answer'] for t in valid_traces]
    results = {}

    # 1. Majority Vote
    winner = simple_majority_vote(answers)
    results['majority'] = equal_func(str(winner), str(gt)) if winner else False

    # 2. Mean Weighted
    mean_confs = [calculate_mean_confidence(t) for t in valid_traces]
    if any(c > 0 for c in mean_confs):
        winner = weighted_majority_vote(answers, mean_confs)
        results['mean_weighted'] = equal_func(str(winner), str(gt)) if winner else False
    else:
        results['mean_weighted'] = False

    # 3. Top 10% Tail
    top_traces = filter_top_confidence(valid_traces, 0.1)
    if top_traces:
        top_answers = [t['extracted_answer'] for t in top_traces]
        top_confs = [calculate_tail_confidence(t) for t in top_traces]
        if any(c > 0 for c in top_confs):
            winner = weighted_majority_vote(top_answers, top_confs)
            results['top10_tail'] = equal_func(str(winner), str(gt)) if winner else False
        else:
            results['top10_tail'] = False
    else:
        results['top10_tail'] = False

    # 4. CDG
    winner = cdg_vote(valid_traces, cdg_params['alpha'], cdg_params['beta'], cdg_params['position_pct'])
    results['cdg'] = equal_func(str(winner), str(gt)) if winner else False

    return results


def save_partial_cache(all_results: list, completed_models: list, timestamp: str):
    """Save partial results for resume support."""
    with open(PARTIAL_CACHE_FILE, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'trace_counts': TRACE_COUNTS,
            'num_subsamples': NUM_SUBSAMPLES,
            'cdg_params': CDG_PARAMS,
            'results': all_results,
            'completed_models': completed_models,
            'is_partial': True,
        }, f, indent=2)
    print(f'  [Partial cache saved: {len(completed_models)} models done]')


def load_partial_cache():
    """Load partial cache if available."""
    if not PARTIAL_CACHE_FILE.exists():
        return None, []

    with open(PARTIAL_CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

    if not cache_data.get('is_partial', False):
        return None, []

    print(f'Found partial cache: {cache_data.get("completed_models", [])} completed')
    return cache_data.get('results', []), cache_data.get('completed_models', [])


def run_analysis(resume=False):
    """Run full scaling analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print('=' * 100)
    print('EXP 2: BASELINE VS CDG - SCALING ANALYSIS')
    print('=' * 100)
    print()
    print(f'Trace counts: {TRACE_COUNTS}')
    print(f'Subsamples per count: {NUM_SUBSAMPLES}')
    print(f'Methods: {", ".join(METHOD_LABELS.values())}')
    print()

    # Try to resume from partial cache
    all_results = []
    completed_models = []
    if resume:
        all_results, completed_models = load_partial_cache()
        if completed_models:
            print(f'Resuming from partial cache. Skipping: {completed_models}')
        else:
            print('No partial cache found, starting fresh.')

    for model_name, datasets in DATASETS.items():
        # Skip already completed models
        if model_name in completed_models:
            print(f'\n{"="*80}')
            print(f'{model_name.upper()} [SKIPPED - already in cache]')
            print(f'{"="*80}')
            continue

        print(f'\n{"="*80}')
        print(f'{model_name.upper()}')
        print(f'{"="*80}')

        cdg_params = CDG_PARAMS[model_name]
        print(f'CDG params: alpha={cdg_params["alpha"]}, beta={cdg_params["beta"]}, position_pct={cdg_params["position_pct"]}')

        for dataset_name, dataset_path in datasets.items():
            print(f'\n--- {dataset_name} ---')

            questions = load_all_questions(dataset_path)
            if not questions:
                print(f'  [NO DATA]')
                continue

            print(f'  Loaded {len(questions)} questions')

            for trace_count in TRACE_COUNTS:
                method_correct = defaultdict(list)

                for subsample_idx in range(NUM_SUBSAMPLES):
                    # Evaluate each question
                    for q in questions:
                        gt = q['ground_truth']
                        all_traces = q['all_traces']

                        # Subsample traces
                        if trace_count >= len(all_traces):
                            sampled_traces = all_traces
                        else:
                            sampled_traces = random.sample(all_traces, trace_count)

                        results = evaluate_methods(sampled_traces, gt, cdg_params)
                        if results:
                            for method, correct in results.items():
                                method_correct[method].append(1 if correct else 0)

                # Compute accuracy for each method
                row = f'  n={trace_count:<4}'
                for method in METHODS:
                    if method_correct[method]:
                        acc = np.mean(method_correct[method])
                        std = np.std(method_correct[method]) / np.sqrt(len(method_correct[method]))
                        row += f'  {METHOD_LABELS[method]}: {100*acc:.1f}%'

                        all_results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'trace_count': trace_count,
                            'method': method,
                            'accuracy': acc,
                            'std': std,
                            'n_samples': len(method_correct[method]),
                        })
                print(row)

        # Save partial cache after each model completes
        completed_models.append(model_name)
        save_partial_cache(all_results, completed_models, timestamp)

    # Save final results
    output_file = OUTPUT_DIR / f'results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'trace_counts': TRACE_COUNTS,
            'num_subsamples': NUM_SUBSAMPLES,
            'cdg_params': CDG_PARAMS,
            'results': all_results,
        }, f, indent=2)
    print(f'\nResults saved to: {output_file}')

    # Save to stable cache file (for --figures-only mode)
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'trace_counts': TRACE_COUNTS,
            'num_subsamples': NUM_SUBSAMPLES,
            'cdg_params': CDG_PARAMS,
            'results': all_results,
        }, f, indent=2)
    print(f'\n*** CACHE SAVED: {CACHE_FILE} ***')
    print('Use --figures-only to regenerate outputs without re-running analysis')

    # Clean up partial cache
    if PARTIAL_CACHE_FILE.exists():
        PARTIAL_CACHE_FILE.unlink()
        print('Partial cache cleaned up.')

    return all_results


def load_cache():
    """Load results from cache file."""
    if not CACHE_FILE.exists():
        print(f'ERROR: Cache file not found: {CACHE_FILE}')
        print('Run without --figures-only first to generate cache.')
        return None

    with open(CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

    print(f'Loaded cache from: {CACHE_FILE}')
    print(f'  Timestamp: {cache_data.get("timestamp", "unknown")}')
    print(f'  Results: {len(cache_data["results"])} entries')

    return cache_data['results']


# ============================================================================
# OUTPUT: SCALING TREND FIGURE
# ============================================================================

def generate_scaling_figure(all_results: list, output_dir: Path):
    """Generate scaling trend figure: accuracy vs trace count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    models = list(DATASETS.keys())
    datasets = list(DATASETS[models[0]].keys())

    fig, axes = plt.subplots(len(models), len(datasets),
                              figsize=(4 * len(datasets), 3 * len(models)),
                              squeeze=False)

    method_styles = {
        'majority': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'mean_weighted': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
        'top10_tail': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
        'cdg': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
    }

    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            for method in METHODS:
                style = method_styles[method]
                # Filter results for this model/dataset/method
                method_results = [r for r in all_results
                                  if r['model'] == model and r['dataset'] == dataset and r['method'] == method]

                if method_results:
                    method_results.sort(key=lambda x: x['trace_count'])
                    x = [r['trace_count'] for r in method_results]
                    y = [100 * r['accuracy'] for r in method_results]

                    ax.plot(x, y,
                           color=style['color'],
                           marker=style['marker'],
                           linestyle=style['linestyle'],
                           label=METHOD_LABELS[method],
                           linewidth=2, markersize=5)

            ax.set_xscale('log', base=2)
            ax.set_xlabel('Number of Traces', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(f'{model} - {dataset}', fontsize=11)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(TRACE_COUNTS)
            ax.set_xticklabels([str(t) for t in TRACE_COUNTS], fontsize=8)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('Accuracy vs Number of Traces', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'scaling_trend.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f'Scaling figure saved to: {fig_path}')
    plt.close(fig)


# ============================================================================
# OUTPUT: SCALING TREND TABLE
# ============================================================================

def generate_scaling_table(all_results: list, output_dir: Path):
    """Generate scaling trend table as text and CSV."""
    models = list(DATASETS.keys())
    datasets = list(DATASETS[models[0]].keys())

    lines = []
    lines.append('=' * 120)
    lines.append('SCALING TREND TABLE: Accuracy (%) at Different Trace Counts')
    lines.append('=' * 120)

    for model in models:
        lines.append(f'\n{model.upper()}')
        lines.append('-' * 100)

        for dataset in datasets:
            lines.append(f'\n  {dataset}:')

            # Header
            header = '    Traces  '
            for method in METHODS:
                header += f'{METHOD_LABELS[method]:<15}'
            lines.append(header)

            for trace_count in TRACE_COUNTS:
                row = f'    {trace_count:<8}'
                for method in METHODS:
                    r = next((x for x in all_results
                              if x['model'] == model and x['dataset'] == dataset
                              and x['trace_count'] == trace_count and x['method'] == method), None)
                    if r:
                        row += f'{100*r["accuracy"]:.1f}%          '
                    else:
                        row += f'{"N/A":<15}'
                lines.append(row)

    # Print to console
    table_text = '\n'.join(lines)
    print(table_text)

    # Save as text file
    txt_path = output_dir / 'scaling_table.txt'
    with open(txt_path, 'w') as f:
        f.write(table_text)
    print(f'\nScaling table saved to: {txt_path}')

    # Save as CSV
    csv_lines = ['model,dataset,trace_count,method,accuracy,std']
    for r in all_results:
        csv_lines.append(f'{r["model"]},{r["dataset"]},{r["trace_count"]},{r["method"]},{r["accuracy"]:.4f},{r.get("std", 0):.4f}')

    csv_path = output_dir / 'scaling_table.csv'
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f'CSV saved to: {csv_path}')


# ============================================================================
# OUTPUT: 512-ONLY COMPARISON TABLE
# ============================================================================

def generate_full_budget_table(all_results: list, output_dir: Path):
    """Generate 512-only comparison table: all methods × all datasets."""
    models = list(DATASETS.keys())
    datasets = list(DATASETS[models[0]].keys())

    # Filter to 512 traces only
    results_512 = [r for r in all_results if r['trace_count'] == 512]

    lines = []
    lines.append('')
    lines.append('=' * 140)
    lines.append('FULL BUDGET (512 TRACES) COMPARISON: All Methods × All Datasets')
    lines.append('=' * 140)

    # Per-model tables
    for model in models:
        lines.append(f'\n{model.upper()}')
        lines.append('-' * 120)

        # Header
        header = f'{"Dataset":<15}'
        for method in METHODS:
            header += f'{METHOD_LABELS[method]:<18}'
        header += f'{"CDG vs Maj":<12}{"CDG vs Top10":<12}'
        lines.append(header)

        model_totals = {m: {'correct': 0, 'total': 0} for m in METHODS}

        for dataset in datasets:
            row = f'{dataset:<15}'

            method_accs = {}
            for method in METHODS:
                r = next((x for x in results_512
                          if x['model'] == model and x['dataset'] == dataset and x['method'] == method), None)
                if r:
                    acc = r['accuracy']
                    method_accs[method] = acc
                    row += f'{100*acc:.1f}%             '
                    # For totals (approximate)
                    model_totals[method]['correct'] += acc * r.get('n_samples', 1)
                    model_totals[method]['total'] += r.get('n_samples', 1)
                else:
                    method_accs[method] = 0
                    row += f'{"N/A":<18}'

            # Deltas
            cdg_vs_maj = method_accs.get('cdg', 0) - method_accs.get('majority', 0)
            cdg_vs_top10 = method_accs.get('cdg', 0) - method_accs.get('top10_tail', 0)
            row += f'{100*cdg_vs_maj:+.1f}%       {100*cdg_vs_top10:+.1f}%'
            lines.append(row)

        # Model totals
        lines.append('-' * 120)
        row = f'{"TOTAL":<15}'
        total_accs = {}
        for method in METHODS:
            if model_totals[method]['total'] > 0:
                acc = model_totals[method]['correct'] / model_totals[method]['total']
                total_accs[method] = acc
                row += f'{100*acc:.1f}%             '
            else:
                total_accs[method] = 0
                row += f'{"N/A":<18}'

        cdg_vs_maj = total_accs.get('cdg', 0) - total_accs.get('majority', 0)
        cdg_vs_top10 = total_accs.get('cdg', 0) - total_accs.get('top10_tail', 0)
        row += f'{100*cdg_vs_maj:+.1f}%       {100*cdg_vs_top10:+.1f}%'
        lines.append(row)

    # Grand summary across all models
    lines.append('')
    lines.append('=' * 140)
    lines.append('GRAND SUMMARY (512 traces)')
    lines.append('=' * 140)

    header = f'{"Model":<15}'
    for method in METHODS:
        header += f'{METHOD_LABELS[method]:<15}'
    header += f'{"CDG vs Maj":<12}{"CDG vs Top10":<12}'
    lines.append(header)
    lines.append('-' * 120)

    for model in models:
        row = f'{model:<15}'
        model_accs = {}

        for method in METHODS:
            model_results = [r for r in results_512 if r['model'] == model and r['method'] == method]
            if model_results:
                avg_acc = np.mean([r['accuracy'] for r in model_results])
                model_accs[method] = avg_acc
                row += f'{100*avg_acc:.1f}%          '
            else:
                model_accs[method] = 0
                row += f'{"N/A":<15}'

        cdg_vs_maj = model_accs.get('cdg', 0) - model_accs.get('majority', 0)
        cdg_vs_top10 = model_accs.get('cdg', 0) - model_accs.get('top10_tail', 0)
        row += f'{100*cdg_vs_maj:+.1f}%       {100*cdg_vs_top10:+.1f}%'
        lines.append(row)

    # Print to console
    table_text = '\n'.join(lines)
    print(table_text)

    # Save as text file
    txt_path = output_dir / 'full_budget_512_table.txt'
    with open(txt_path, 'w') as f:
        f.write(table_text)
    print(f'\n512-only table saved to: {txt_path}')


# ============================================================================
# OUTPUT: FULL BUDGET BAR CHART
# ============================================================================

def generate_full_budget_figure(all_results: list, output_dir: Path):
    """Generate bar chart comparing methods at 512 traces."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    # Filter to 512 traces only
    results_512 = [r for r in all_results if r['trace_count'] == 512]

    models = list(DATASETS.keys())
    x = np.arange(len(models))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(METHODS):
        accs = []
        for model in models:
            model_results = [r for r in results_512 if r['model'] == model and r['method'] == method]
            if model_results:
                avg_acc = np.mean([r['accuracy'] for r in model_results])
                accs.append(100 * avg_acc)
            else:
                accs.append(0)
        ax.bar(x + i * width, accs, width, label=METHOD_LABELS[method], color=colors[i])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Method Comparison at Full Budget (512 Traces)', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    fig_path = output_dir / 'full_budget_512_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f'512-only figure saved to: {fig_path}')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 2: Baseline vs CDG Comparison')
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
        all_results = load_cache()
        if all_results is None:
            sys.exit(1)
    else:
        all_results = run_analysis(resume=args.resume)

    # Generate all outputs
    print('\n' + '=' * 100)
    print('GENERATING OUTPUTS')
    print('=' * 100)

    generate_scaling_figure(all_results, OUTPUT_DIR)
    generate_scaling_table(all_results, OUTPUT_DIR)
    generate_full_budget_table(all_results, OUTPUT_DIR)
    generate_full_budget_figure(all_results, OUTPUT_DIR)

    print('\n' + '=' * 100)
    print('ALL OUTPUTS GENERATED')
    print('=' * 100)
    print(f'Output directory: {OUTPUT_DIR}')
    print('Files:')
    print('  - scaling_trend.png/pdf     (accuracy vs trace count figure)')
    print('  - scaling_table.txt/csv     (detailed numbers for all trace counts)')
    print('  - full_budget_512_table.txt (512-only comparison)')
    print('  - full_budget_512_comparison.png/pdf (512-only bar chart)')
