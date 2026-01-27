"""
Evaluation Config - Centralized paths and parameters for eval scripts.

This config defines:
1. RESULT_BASE_PATH - Base directory for all inference result pickle files
2. DATASETS - Model/dataset path mappings (relative to RESULT_BASE_PATH)
3. CDG_PARAMS - Optimal CDG hyperparameters per model
4. Shared constants (trace counts, methods, etc.)

Usage:
    from config import DATASETS, CDG_PARAMS, RESULT_BASE_PATH
"""

from pathlib import Path

# ============================================================================
# BASE PATH FOR INFERENCE RESULTS
# ============================================================================

# Base directory containing all inference result pickle files.
# Change this to your local path where the results are stored.
RESULT_BASE_PATH = "/path/to/result"

# ============================================================================
# DATASET PATHS (relative to RESULT_BASE_PATH)
# ============================================================================
# Each model has results for 4 benchmarks: aime2024, aime2025, brumo2025, hmmt2025
# Directory structure: RESULT_BASE_PATH / <model_subdir> / <dataset_subdir>

DATASET_PATHS = {
    'deepseek8b': {
        # DeepSeek R1 8B results
        'aime2024': 'deepseek/aime_2024_run_512_results_deepseek8b',
        'aime2025': 'deepseek/aime_2025_run_512_results_newpara',
        'brumo2025': 'deepseek/bruno_2025_run_512_results_deepseek8b',
        'hmmt2025': 'deepseek/hmmt_feb_2025_run_512_results',
    },
    'gptoss20b': {
        # GPT-OSS 20B results (no high reasoning, topk=40)
        'aime2024': 'gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_aime_2024_gptoss_topk_40_no_highreason',
        'aime2025': 'gptoss_NO_high_reason_topk_40/aime_2025_run_512_results_gptoss20b_topk_40_no_highreason',
        'brumo2025': 'gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_bruno_2025_gptoss_topk_40_no_highreason',
        'hmmt2025': 'gptoss_NO_high_reason_topk_40/results_gpt_oss_topk_40_no_highreason_paper/result_hmmt_2025_gptoss_topk_40_no_highreason',
    },
    'gemma3_27b': {
        # Gemma 3 27B results
        'aime2024': 'resultsforGEMMA/gemma3_27b_aime2024_512',
        'aime2025': 'resultsforGEMMA/gemma3_27b_aime2025_512',
        'brumo2025': 'resultsforGEMMA/gemma3_27b_bruno2025_512',
        'hmmt2025': 'resultsforGEMMA/gemma3_27b_hmmt2025_512',
    },
    'qwq32b': {
        # QwQ 32B results
        'aime2024': 'resultsforqwq/qwq32b_aime2024_512',
        'aime2025': 'resultsforqwq/qwq32b_aime2025_512',
        'brumo2025': 'resultsforqwq/qwq32b_bruno2025_512',
        'hmmt2025': 'resultsforqwq/qwq32b_hmmt2025_512',
    },
}


def get_datasets_dict() -> dict:
    """
    Build DATASETS dict with full absolute paths.
    Returns dict of model -> dataset -> absolute_path.
    """
    datasets = {}
    for model, dataset_map in DATASET_PATHS.items():
        datasets[model] = {}
        for dataset, rel_path in dataset_map.items():
            datasets[model][dataset] = str(Path(RESULT_BASE_PATH) / rel_path)
    return datasets


# Full paths dict (for backward compatibility with existing scripts)
DATASETS = get_datasets_dict()


# ============================================================================
# CDG HYPERPARAMETERS (optimal values per model)
# ============================================================================
# alpha: count dampening exponent (count^alpha)
# beta: gradient weight (mean_conf + beta * gradient)
# position_pct: percentile for gradient computation (first/last P%)

CDG_PARAMS = {
    'deepseek8b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gptoss20b': {'alpha': 0.5, 'beta': 10, 'position_pct': 10},
    'gemma3_27b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
    'qwq32b': {'alpha': 0.5, 'beta': 3, 'position_pct': 10},
}


# ============================================================================
# EXPERIMENT CONSTANTS
# ============================================================================

# Trace counts for scaling experiments
TRACE_COUNTS = [8, 16, 32, 64, 128, 256, 512]

# Number of random subsamples for variance estimation
NUM_SUBSAMPLES = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Voting methods
METHODS = ['majority', 'mean_weighted', 'top10_tail', 'cdg']

METHOD_LABELS = {
    'majority': 'Majority',
    'mean_weighted': 'Mean Weighted',
    'top10_tail': 'Top10 Tail',
    'cdg': 'CDG',
}

# Display names for models
MODEL_NAMES = {
    'deepseek8b': 'DeepSeek R1 8B',
    'gptoss20b': 'gpt-oss 20B',
    'gemma3_27b': 'Gemma 3 27B',
    'qwq32b': 'QwQ 32B',
}

# Display names for datasets
DATASET_NAMES = {
    'aime2024': 'AIME 2024',
    'aime2025': 'AIME 2025',
    'brumo2025': 'BRUMO 2025',
    'hmmt2025': 'HMMT 2025',
}


# ============================================================================
# CDG HYPERPARAMETER SWEEP VALUES
# ============================================================================

BETA_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
ALPHA_VALUES = [0.5, 1.0]
POSITION_PCT_VALUES = [10, 20]


# ============================================================================
# OUTPUT DIRECTORIES (relative to repo root)
# ============================================================================

# Repo root directory (sampling_credit/)
REPO_ROOT = Path(__file__).parent.parent.parent

# Results directory for cached eval outputs
RESULTS_DIR = REPO_ROOT / 'results'

# Cache directories for each experiment
CACHE_DIR = RESULTS_DIR / 'cache'

OUTPUT_DIRS = {
    'exp_voting_methods': CACHE_DIR / 'voting_methods',
    'exp_cdg_sweep': CACHE_DIR / 'cdg_sweep',
    'util_histogram_cache': CACHE_DIR / 'histogram',
}

# Figures output directory
FIGURES_DIR = RESULTS_DIR / 'figures'

# Appendix figures output directory
APPENDIX_DIR = FIGURES_DIR / 'appendix'


# ============================================================================
# DYNASOR PACKAGE (for math_equal function)
# ============================================================================
# Install via: pip install git+https://github.com/hao-ai-lab/Dynasor.git
# Usage: from dynasor.core.evaluator import math_equal
