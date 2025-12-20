"""
Prepare AIME 2024 dataset in JSONL format for deepconf experiments.
"""
import json
import os
from datasets import load_dataset, concatenate_datasets

def main():
    # Load both AIME 2024 datasets from MathArena
    print("Loading AIME 2024 I dataset...")
    dataset_i = load_dataset("MathArena/aime_2024_I", split="train")
    print("Loading AIME 2024 II dataset...")
    dataset_ii = load_dataset("MathArena/aime_2024_II", split="train")

    print(f"Loaded {len(dataset_i)} questions from AIME I")
    print(f"Loaded {len(dataset_ii)} questions from AIME II")

    # Combine both datasets
    dataset = concatenate_datasets([dataset_i, dataset_ii])
    print(f"Total questions: {len(dataset)}")

    # Convert to JSONL
    output_dir = "/mnt/aime_2024"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "aime_2024.jsonl")

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
