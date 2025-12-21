"""
Prepare AIME 2025 dataset in JSONL format for deepconf experiments.
"""
import json
import os
import sys

# Add scripts to path for config import (parent folder contains config.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DatasetConfig

from datasets import load_dataset


def main():
    # Get dataset config
    ds_config = DatasetConfig.DATASETS["aime_2025"]

    # Load dataset
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset(ds_config["hf_id"], split=ds_config["split"])

    # Create output directory
    output_dir = ds_config["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Convert to JSONL
    output_file = os.path.join(output_dir, ds_config["filename"])
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            entry = {
                "question": example["problem"],
                "answer": str(example["answer"])
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Converted {len(dataset)} examples to {output_file}")

    # Print first example for verification
    print("\nFirst example:")
    print(f"Question: {dataset[0]['problem'][:200]}...")
    print(f"Answer: {dataset[0]['answer']}")


if __name__ == "__main__":
    main()
