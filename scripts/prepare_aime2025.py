"""
Prepare AIME 2025 dataset in JSONL format for deepconf experiments.
"""
import json
from datasets import load_dataset

def main():
    # Load dataset
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset("MathArena/aime_2025", split="train")

    # Convert to JSONL
    output_file = "aime_2025.jsonl"
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
