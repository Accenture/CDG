"""
Data Preparation Config - Paths for preparing benchmark datasets.

Defines source (HuggingFace) and target (local JSONL) paths for datasets.

Usage:
    from config import DatasetConfig
    path = DatasetConfig.get_dataset_path("aime_2025")
"""
import os

# ============================================================================
# BASE DIRECTORY
# ============================================================================

# Base directory for prepared datasets (JSONL files).
DATASETS_BASE = os.environ.get("DATASETS_BASE", "/path/to/datasets")


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

class DatasetConfig:
    """
    Configuration for dataset paths and HuggingFace sources.
    """

    # Dataset definitions: name -> (directory, filename, huggingface_id)
    DATASETS = {
        "aime_2025": {
            "dir": os.path.join(DATASETS_BASE, "aime_2025"),
            "filename": "aime_2025.jsonl",
            "hf_id": "MathArena/aime_2025",
            "split": "train",
        },
        "aime_2024": {
            "dir": os.path.join(DATASETS_BASE, "aime_2024"),
            "filename": "aime_2024.jsonl",
            "hf_id": ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
            "split": "train",
        },
        "hmmt_2025": {
            "dir": os.path.join(DATASETS_BASE, "hmmt_feb_2025"),
            "filename": "hmmt_feb_2025.jsonl",
            "hf_id": "MathArena/hmmt_feb_2025",
            "split": "train",
        },
        "brumo_2025": {
            "dir": os.path.join(DATASETS_BASE, "brumo_2025"),
            "filename": "brumo_2025.jsonl",
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
