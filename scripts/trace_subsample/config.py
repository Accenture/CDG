"""
Configuration for trace subsampling experiments.
"""

# Subset sizes to generate
SUBSET_SIZES = [8, 16, 32, 64, 128, 256]

# Number of versions per subset size
NUM_VERSIONS = 5

# Base random seeds for reproducibility (one per version)
# Seeds chosen to be well-spaced prime numbers
VERSION_SEEDS = {
    1: 42,
    2: 137,
    3: 293,
    4: 461,
    5: 631,
}

# Source experiments to process (512 trace runs)
SOURCE_EXPERIMENTS = [
    # Qwen 32B
    "qwen32b_aime2024_512",
    "qwen32b_aime2025_512",
    "qwen32b_bruno2025_512",
    "qwen32b_hmmt2025_512",
    # DeepSeek 8B
    "deepseek8b_aime2024_512",
    "deepseek8b_aime2025_512",
    "deepseek8b_bruno2025_512",
    "deepseek8b_hmmt2025_512",
    # Gemma 3 27B
    "gemma3_27b_aime2024_512",
    "gemma3_27b_aime2025_512",
    "gemma3_27b_bruno2025_512",
    "gemma3_27b_hmmt2025_512",
    # GPTOSS 20B (with high reason) - partial
    "gptoss20b_aime2025_512",
    "gptoss20b_bruno2025_512",
    # GPTOSS 20B (no high reason)
    "gptoss20b_nohighreason_aime2024_512",
    "gptoss20b_nohighreason_aime2025_512",
    "gptoss20b_nohighreason_bruno2025_512",
    "gptoss20b_nohighreason_hmmt2025_512",
]

# Paths (to be set when running on VM)
# These are relative to the results base directory
RESULTS_BASE = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results"
OUTPUT_SUBDIR = "subset_trace"
