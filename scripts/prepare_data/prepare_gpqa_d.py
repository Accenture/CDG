"""
Prepare GPQA Diamond dataset in JSONL format for deepconf experiments.
"""
import json
import os
from datasets import load_dataset

def main():
    # Load dataset
    print("Loading GPQA Diamond dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    # Convert to JSONL
    output_dir = "/mnt/gpqa_d"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gpqa_diamond.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            entry = {
                "question": example["Question"],
                "answer": str(example["Correct Answer"])
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Converted {len(dataset)} examples to {output_file}")

    # Print first example for verification
    print("\nFirst example:")
    print(f"Question: {dataset[0]['Question'][:200]}...")
    print(f"Answer: {dataset[0]['Correct Answer']}")


if __name__ == "__main__":
    main()
