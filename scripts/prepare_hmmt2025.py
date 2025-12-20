"""
Prepare HMMT February 2025 dataset in JSONL format for deepconf experiments.

HMMT (Harvard-MIT Mathematics Tournament) February 2025 contains 30 competition problems
covering Algebra, Combinatorics, Geometry, and Number Theory.
"""
import json
from datasets import load_dataset
from pathlib import Path


def main():
    # Load dataset
    print("Loading HMMT February 2025 dataset...")
    dataset = load_dataset("MathArena/hmmt_feb_2025", split="train")

    print(f"Loaded {len(dataset)} problems")

    # Create output directory
    output_dir = Path("/eph/nvme0/hmmt_feb_2025")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSONL
    output_file = output_dir / "hmmt_feb_2025.jsonl"
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
