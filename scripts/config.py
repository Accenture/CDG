"""
Centralized configuration for all paths and settings.

This config file consolidates all hardcoded paths used across the codebase.
Modify this file to change paths when running on different machines.

Usage:
    from config import PathConfig, ModelConfig, DatasetConfig

    # Access paths
    hf_cache = PathConfig.HF_CACHE
    output_dir = PathConfig.OUTPUT_DIR

    # Access dataset paths
    aime2025_path = DatasetConfig.get_dataset_path("aime_2025")
"""
import os
from pathlib import Path


class PathConfig:
    """
    Central configuration for all file system paths.

    Values are read from environment variables (set by config.sh).
    Defaults are provided as fallback.

    To change paths, modify config.sh (the single source of truth).
    """

    # ============================================================
    # BASE DIRECTORIES (read from env vars set by config.sh)
    # ============================================================

    # Base directory for datasets (JSONL files) - use fast NVMe
    DATASETS_BASE = os.environ.get("DATASETS_BASE", "/eph/nvme0/datasets")

    # Base directory for inference results output
    OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/mnt/batch/tasks/shared/LS_root/mounts/clusters/butters-compute/code/Users/minghao.a.liu/sampling_credit_results")

    # ============================================================
    # HUGGINGFACE CACHE PATHS
    # ============================================================

    # HuggingFace home directory (for model downloads) - use NVMe (27TB free, faster)
    HF_CACHE = os.environ.get("HF_HOME", "/eph/nvme0/hf_cache")

    # Transformers cache (usually same as HF_CACHE)
    TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", HF_CACHE)

    # ============================================================
    # DATASET DIRECTORIES
    # ============================================================

    # Individual dataset directories
    AIME_2025_DIR = os.path.join(DATASETS_BASE, "aime_2025")
    AIME_2024_DIR = os.path.join(DATASETS_BASE, "aime_2024")
    HMMT_2025_DIR = os.path.join(DATASETS_BASE, "hmmt_feb_2025")
    BRUNO_2025_DIR = os.path.join(DATASETS_BASE, "bruno_2025")

    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================

    @classmethod
    def setup_hf_env(cls):
        """Set HuggingFace environment variables. Call this at the start of scripts."""
        os.environ["HF_HOME"] = cls.HF_CACHE
        os.environ["TRANSFORMERS_CACHE"] = cls.TRANSFORMERS_CACHE

    @classmethod
    def ensure_dirs_exist(cls):
        """Create all necessary directories if they don't exist."""
        dirs = [
            cls.DATASETS_BASE,
            cls.HF_CACHE,
            cls.OUTPUT_BASE,
            cls.AIME_2025_DIR,
            cls.AIME_2024_DIR,
            cls.HMMT_2025_DIR,
            cls.BRUNO_2025_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


class DatasetConfig:
    """
    Configuration for dataset paths and metadata.
    """

    # Dataset definitions: name -> (directory, filename, huggingface_id)
    DATASETS = {
        "aime_2025": {
            "dir": PathConfig.AIME_2025_DIR,
            "filename": "aime_2025.jsonl",
            "hf_id": "MathArena/aime_2025",
            "split": "train",
        },
        "aime_2024": {
            "dir": PathConfig.AIME_2024_DIR,
            "filename": "aime_2024.jsonl",
            "hf_id": ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
            "split": "train",
        },
        "hmmt_2025": {
            "dir": PathConfig.HMMT_2025_DIR,
            "filename": "hmmt_feb_2025.jsonl",
            "hf_id": "MathArena/hmmt_feb_2025",
            "split": "train",
        },
        "bruno_2025": {
            "dir": PathConfig.BRUNO_2025_DIR,
            "filename": "bruno_2025.jsonl",
            "hf_id": "MathArena/brumo_2025",
            "split": "train",
        },
    }

    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get full path to dataset JSONL file."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")

        ds = cls.DATASETS[dataset_name]
        return os.path.join(ds["dir"], ds["filename"])

    @classmethod
    def get_dataset_dir(cls, dataset_name: str) -> str:
        """Get directory for dataset."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASETS[dataset_name]["dir"]

    @classmethod
    def list_datasets(cls) -> list:
        """List all available dataset names."""
        return list(cls.DATASETS.keys())


class ModelConfig:
    """
    Configuration for model settings per DeepConf paper Table 11.
    """

    # Model configurations: model_name -> settings
    MODELS = {
        "deepseek-8b": {
            "hf_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "model_type": "deepseek",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": -1,  # disabled
            "max_tokens": 64000,
        },
        "qwen3-8b": {
            "hf_id": "Qwen/Qwen3-8B",
            "model_type": "qwen",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32000,
        },
        "qwen3-32b": {
            "hf_id": "Qwen/Qwen3-32B",
            "model_type": "qwen",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32000,
        },
        "gpt-oss-20b": {
            "hf_id": "openai/gpt-oss-20b",
            "model_type": "gpt-oss",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "max_tokens": 130000,
            "reasoning_effort": "high",
        },
        "gpt-oss-120b": {
            "hf_id": "openai/gpt-oss-120b",
            "model_type": "gpt-oss",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "max_tokens": 130000,
            "reasoning_effort": "high",
        },
        "qwq-32b": {
            "hf_id": "Qwen/QwQ-32B",
            "model_type": "qwq",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
        },
        "qwen25-32b": {
            "hf_id": "Qwen/Qwen2.5-32B",
            "model_type": "qwen",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
        "deepseek-r1-llama-70b": {
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "model_type": "deepseek",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": -1,
            "max_tokens": 32768,
        },
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get configuration for a specific model."""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name].copy()

    @classmethod
    def list_models(cls) -> list:
        """List all available model names."""
        return list(cls.MODELS.keys())


# ============================================================
# SHELL SCRIPT HELPER - Export paths as environment variables
# ============================================================

def print_shell_exports():
    """Print shell export commands for use in bash scripts."""
    print("# Add these to your shell script or source this output")
    print(f'export DATASETS_BASE="{PathConfig.DATASETS_BASE}"')
    print(f'export OUTPUT_BASE="{PathConfig.OUTPUT_BASE}"')
    print(f'export HF_CACHE="{PathConfig.HF_CACHE}"')
    print()
    print("# Dataset paths")
    for name in DatasetConfig.list_datasets():
        path = DatasetConfig.get_dataset_path(name)
        var_name = f"DATASET_{name.upper().replace('-', '_')}"
        print(f'export {var_name}="{path}"')


if __name__ == "__main__":
    # When run directly, print shell exports
    print_shell_exports()
