"""
Prepare AIME 2024 dataset in JSONL format for deepconf experiments.
"""
import json
import os
import sys

from config import DatasetConfig

from datasets import load_dataset, concatenate_datasets


def main():
    # Get dataset config
    ds_config = DatasetConfig.DATASETS["aime_2024"]

    # Load both AIME 2024 datasets from MathArena (hf_id is a list for this dataset)
    hf_ids = ds_config["hf_id"]
    print("Loading AIME 2024 I dataset...")
    dataset_i = load_dataset(hf_ids[0], split=ds_config["split"])
    print("Loading AIME 2024 II dataset...")
    dataset_ii = load_dataset(hf_ids[1], split=ds_config["split"])

    print(f"Loaded {len(dataset_i)} questions from AIME I")
    print(f"Loaded {len(dataset_ii)} questions from AIME II")

    # Combine both datasets
    dataset = concatenate_datasets([dataset_i, dataset_ii])
    print(f"Total questions: {len(dataset)}")

    # Convert to JSONL
    output_dir = ds_config["dir"]
    os.makedirs(output_dir, exist_ok=True)
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
