#!/usr/bin/env python3
"""
Subsample traces from 512-trace experiments to create smaller trace sets.

Creates subsets of 8/16/32/64/128/256 traces with 5 versions each.
Processes file-by-file to minimize memory usage.

Usage:
    python scripts/trace_subsample/subsample.py

    # Process specific experiment
    python scripts/trace_subsample/subsample.py --experiment qwen32b_aime2025_512

    # Custom paths
    python scripts/trace_subsample/subsample.py --results_base /path/to/results
"""
import os
import sys
import pickle
import random
import argparse
import gc
from glob import glob
from datetime import datetime

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'deepconf', 'deepconf'))

# Import directly from utils to avoid vllm dependency
from utils import compute_all_voting_results
from config import SUBSET_SIZES, NUM_VERSIONS, VERSION_SEEDS, SOURCE_EXPERIMENTS, RESULTS_BASE, OUTPUT_SUBDIR


def get_seed_for_file(version_seed: int, experiment: str, qid: int) -> int:
    """Generate deterministic seed for a specific file and version."""
    # Combine version seed with hash of experiment name and qid
    exp_hash = hash(experiment) & 0xFFFFFFFF  # Keep positive
    return version_seed + exp_hash + qid * 1000


def subsample_traces(traces: list, n: int, seed: int) -> list:
    """Randomly select n traces from the full trace list."""
    random.seed(seed)
    if n >= len(traces):
        return traces.copy()
    return random.sample(traces, n)


def process_single_file(
    pkl_path: str,
    experiment: str,
    output_base: str,
    subset_sizes: list,
    num_versions: int,
    version_seeds: dict,
    dry_run: bool = False
) -> dict:
    """
    Process a single pickle file and create all subsampled versions.

    Returns dict with stats about processing.
    """
    stats = {'files_created': 0, 'errors': []}

    # Load the pickle file
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        stats['errors'].append(f"Failed to load {pkl_path}: {e}")
        return stats

    # Extract info
    all_traces = data.get('all_traces', [])
    qid = data.get('qid', -1)
    original_count = len(all_traces)

    if original_count == 0:
        stats['errors'].append(f"No traces in {pkl_path}")
        return stats

    # Get original filename timestamp part
    basename = os.path.basename(pkl_path)
    # Format: qid{N}_{timestamp}.pkl
    timestamp = basename.split('_', 1)[1].replace('.pkl', '')

    # Process each subset size and version
    for subset_size in subset_sizes:
        if subset_size > original_count:
            continue

        for version in range(1, num_versions + 1):
            # Create output directory
            output_dir_name = f"{experiment}_subset{subset_size}v{version}"
            output_dir = os.path.join(output_base, output_dir_name)

            if not dry_run:
                os.makedirs(output_dir, exist_ok=True)

            # Generate seed for this specific file + version
            seed = get_seed_for_file(version_seeds[version], experiment, qid)

            # Subsample traces
            subset_traces = subsample_traces(all_traces, subset_size, seed)

            # Recompute voting results
            voting_results = compute_all_voting_results(subset_traces)

            # Compute total tokens for subset
            total_tokens = sum(t.get('num_tokens', 0) for t in subset_traces)

            # Build new result data
            subset_data = {
                'question': data.get('question', ''),
                'ground_truth': data.get('ground_truth', ''),
                'qid': qid,
                'run_id': output_dir_name,
                'all_traces': subset_traces,
                'total_tokens': total_tokens,
                'total_traces_count': len(subset_traces),
                'voting_results': voting_results,
                'config': data.get('config', {}).copy(),
                # Subsample metadata
                'subsample_info': {
                    'source_experiment': experiment,
                    'source_file': basename,
                    'source_trace_count': original_count,
                    'subset_size': subset_size,
                    'version': version,
                    'seed': seed,
                    'created_at': datetime.now().isoformat(),
                }
            }

            # Update config to reflect subset
            subset_data['config']['budget'] = subset_size
            subset_data['config']['subsampled_from'] = original_count

            # Get voted answer
            if 'majority' in voting_results and voting_results['majority']:
                subset_data['voted_answer'] = voting_results['majority']['answer']
                subset_data['final_answer'] = voting_results['majority']['answer']

            # Save
            output_path = os.path.join(output_dir, f"qid{qid}_{timestamp}.pkl")

            if not dry_run:
                with open(output_path, 'wb') as f:
                    pickle.dump(subset_data, f)

            stats['files_created'] += 1

    # Clear memory
    del data
    del all_traces
    gc.collect()

    return stats


def process_experiment(
    experiment: str,
    results_base: str,
    output_base: str,
    subset_sizes: list,
    num_versions: int,
    version_seeds: dict,
    dry_run: bool = False
) -> dict:
    """Process all pickle files for an experiment."""
    exp_dir = os.path.join(results_base, experiment)

    if not os.path.exists(exp_dir):
        return {'error': f"Experiment directory not found: {exp_dir}"}

    # Find all pickle files
    pkl_files = sorted(glob(os.path.join(exp_dir, "qid*.pkl")))

    if not pkl_files:
        return {'error': f"No pickle files found in {exp_dir}"}

    print(f"\n{'='*60}")
    print(f"Processing: {experiment}")
    print(f"{'='*60}")
    print(f"Found {len(pkl_files)} pickle files")
    print(f"Subset sizes: {subset_sizes}")
    print(f"Versions per size: {num_versions}")
    print(f"Total output files per question: {len(subset_sizes) * num_versions}")
    print(f"{'='*60}")

    total_stats = {'files_created': 0, 'errors': [], 'files_processed': 0}

    for i, pkl_path in enumerate(pkl_files, 1):
        basename = os.path.basename(pkl_path)
        print(f"\n[{i}/{len(pkl_files)}] Processing {basename}...")

        stats = process_single_file(
            pkl_path=pkl_path,
            experiment=experiment,
            output_base=output_base,
            subset_sizes=subset_sizes,
            num_versions=num_versions,
            version_seeds=version_seeds,
            dry_run=dry_run
        )

        total_stats['files_created'] += stats['files_created']
        total_stats['errors'].extend(stats['errors'])
        total_stats['files_processed'] += 1

        print(f"  Created {stats['files_created']} subset files")
        if stats['errors']:
            for err in stats['errors']:
                print(f"  ERROR: {err}")

        # Clear memory after each file
        gc.collect()

    return total_stats


def main():
    parser = argparse.ArgumentParser(description='Subsample traces from 512-trace experiments')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Specific experiment to process (default: all from config)')
    parser.add_argument('--results_base', type=str, default=RESULTS_BASE,
                        help='Base directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without creating files')
    parser.add_argument('--subset_sizes', type=str, default=None,
                        help='Comma-separated subset sizes (default: 8,16,32,64,128,256)')

    args = parser.parse_args()

    # Parse subset sizes if provided
    subset_sizes = SUBSET_SIZES
    if args.subset_sizes:
        subset_sizes = [int(x.strip()) for x in args.subset_sizes.split(',')]

    # Determine output base
    output_base = os.path.join(args.results_base, OUTPUT_SUBDIR)

    # Determine experiments to process
    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = SOURCE_EXPERIMENTS

    print("="*60)
    print("TRACE SUBSAMPLING")
    print("="*60)
    print(f"Results base: {args.results_base}")
    print(f"Output base: {output_base}")
    print(f"Experiments: {experiments}")
    print(f"Subset sizes: {subset_sizes}")
    print(f"Versions: {NUM_VERSIONS}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)

    if args.dry_run:
        print("\n*** DRY RUN - No files will be created ***\n")

    # Create output directory
    if not args.dry_run:
        os.makedirs(output_base, exist_ok=True)

    # Process each experiment
    grand_total = {'files_created': 0, 'errors': [], 'experiments_processed': 0}

    for experiment in experiments:
        stats = process_experiment(
            experiment=experiment,
            results_base=args.results_base,
            output_base=output_base,
            subset_sizes=subset_sizes,
            num_versions=NUM_VERSIONS,
            version_seeds=VERSION_SEEDS,
            dry_run=args.dry_run
        )

        if 'error' in stats:
            print(f"\nSKIPPED {experiment}: {stats['error']}")
            grand_total['errors'].append(stats['error'])
        else:
            grand_total['files_created'] += stats['files_created']
            grand_total['errors'].extend(stats['errors'])
            grand_total['experiments_processed'] += 1

            print(f"\n{experiment} completed:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Subset files created: {stats['files_created']}")
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Experiments processed: {grand_total['experiments_processed']}")
    print(f"Total subset files created: {grand_total['files_created']}")
    print(f"Total errors: {len(grand_total['errors'])}")
    if grand_total['errors']:
        print("\nErrors:")
        for err in grand_total['errors'][:10]:
            print(f"  - {err}")
        if len(grand_total['errors']) > 10:
            print(f"  ... and {len(grand_total['errors']) - 10} more")
    print("="*60)


if __name__ == "__main__":
    main()
