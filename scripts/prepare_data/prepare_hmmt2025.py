"""
Prepare HMMT February 2025 dataset in JSONL format for deepconf experiments.

HMMT (Harvard-MIT Mathematics Tournament) February 2025 contains 30 competition problems
covering Algebra, Combinatorics, Geometry, and Number Theory.
"""
import json
import os
import sys
from pathlib import Path

# Add scripts to path for config import (parent folder contains config.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DatasetConfig

from datasets import load_dataset


def main():
    # Get dataset config
    ds_config = DatasetConfig.DATASETS["hmmt_2025"]

    # Load dataset
    print("Loading HMMT February 2025 dataset...")
    dataset = load_dataset(ds_config["hf_id"], split=ds_config["split"])

    print(f"Loaded {len(dataset)} problems")

    # Create output directory
    output_dir = Path(ds_config["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSONL
    output_file = output_dir / ds_config["filename"]
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            entry = {
                "question": example["problem"],
                "answer": str(example["answer"])
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(dataset)} examples to {output_file}")

    # Print dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS:")
    print("="*80)

    # Count problem types
    problem_types_count = {}
    for example in dataset:
        for ptype in example["problem_type"]:
            problem_types_count[ptype] = problem_types_count.get(ptype, 0) + 1

    print(f"\nTotal problems: {len(dataset)}")
    print(f"\nProblem type distribution:")
    for ptype, count in sorted(problem_types_count.items()):
        print(f"  {ptype}: {count}")

    # Print first example for verification
    print("\n" + "="*80)
    print("FIRST EXAMPLE:")
    print("="*80)
    print(f"\nProblem Index: {dataset[0]['problem_idx']}")
    print(f"Problem Type: {', '.join(dataset[0]['problem_type'])}")
    print(f"\nQuestion:\n{dataset[0]['problem'][:300]}...")
    print(f"\nAnswer: {dataset[0]['answer']}")

    # Print second example
    print("\n" + "="*80)
    print("SECOND EXAMPLE:")
    print("="*80)
    print(f"\nProblem Index: {dataset[1]['problem_idx']}")
    print(f"Problem Type: {', '.join(dataset[1]['problem_type'])}")
    print(f"\nQuestion:\n{dataset[1]['problem'][:300]}...")
    print(f"\nAnswer: {dataset[1]['answer']}")

    print("\n" + "="*80)
    print(f"✅ Dataset prepared successfully!")
    print(f"📁 Output file: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
