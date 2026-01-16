#!/usr/bin/env python3
"""
Lazy Run All Figures - Single command to generate everything.

Usage:
    python scripts/eval/run_all_figures.py          # Generate all figures
    python scripts/eval/run_all_figures.py --lazy   # Quick mode, key figures only
    python scripts/eval/run_all_figures.py --list   # List what would be generated
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent))

# Configuration
DEFAULT_MODEL = 'qwen32b'
DEFAULT_DATASET = 'aime2025'


def run_command(cmd: list, description: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {description}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False


def generate_all_figures(model: str, dataset: str, cache_only: bool = False,
                         dry_run: bool = False, no_show: bool = True):
    """Generate all figures."""

    common_args = ['--no-show'] if no_show else []
    if cache_only:
        common_args.append('--cache-only')

    tasks = []

    # ==========================================================================
    # 1. Hyperparam Figures
    # ==========================================================================
    hp_output = str(RESULTS_BASE / "fig_hyperparam")

    # Main paper: one model × 4 datasets
    tasks.append({
        'description': f'Hyperparam: Main paper ({model} × 4 datasets)',
        'cmd': [
            sys.executable, str(SCRIPT_DIR / "fig_hyperparam.py"),
            '--output-dir', hp_output,
            '--main-paper', '--model', model,
        ] + common_args,
    })

    # Main paper: one dataset × 4 models
    tasks.append({
        'description': f'Hyperparam: Main paper ({dataset} × 4 models)',
        'cmd': [
            sys.executable, str(SCRIPT_DIR / "fig_hyperparam.py"),
            '--output-dir', hp_output,
            '--main-paper', '--dataset', dataset,
        ] + common_args,
    })

    if not cache_only:
        # Full grid + individual figures
        tasks.append({
            'description': 'Hyperparam: Full appendix grid + individuals',
            'cmd': [
                sys.executable, str(SCRIPT_DIR / "fig_hyperparam.py"),
                '--output-dir', hp_output,
            ] + common_args,
        })

    # ==========================================================================
    # 2. Histogram Figures
    # ==========================================================================
    hist_output = str(RESULTS_BASE / "fig_histogram")

    # Main model/dataset
    tasks.append({
        'description': f'Histogram: {model} on {dataset}',
        'cmd': [
            sys.executable, str(SCRIPT_DIR / "fig_histogram.py"),
            '--output-dir', hist_output,
            '--model', model,
            '--dataset', dataset,
        ] + common_args,
    })

    if not cache_only:
        # All combinations
        tasks.append({
            'description': 'Histogram: All model-dataset combinations',
            'cmd': [
                sys.executable, str(SCRIPT_DIR / "fig_histogram.py"),
                '--output-dir', hist_output,
                '--all',
            ] + common_args,
        })

    # ==========================================================================
    # 3. Subsample Figures
    # ==========================================================================
    sub_output = str(RESULTS_BASE / "fig_subsample")

    # Main paper (single row)
    tasks.append({
        'description': f'Subsample: Main paper ({model})',
        'cmd': [
            sys.executable, str(SCRIPT_DIR / "fig_subsample.py"),
            '--output-dir', sub_output,
            '--main-paper', '--model', model,
            '--key-methods-only',
        ] + common_args,
    })

    # Improvement plot
    tasks.append({
        'description': f'Subsample: Improvement plot ({model})',
        'cmd': [
            sys.executable, str(SCRIPT_DIR / "fig_subsample.py"),
            '--output-dir', sub_output,
            '--improvement', '--model', model,
        ] + common_args,
    })

    if not cache_only:
        # Full grid
        tasks.append({
            'description': 'Subsample: Full grid (appendix)',
            'cmd': [
                sys.executable, str(SCRIPT_DIR / "fig_subsample.py"),
                '--output-dir', sub_output,
                '--grid',
            ] + common_args,
        })

        # Method comparison
        tasks.append({
            'description': f'Subsample: Method comparison ({dataset})',
            'cmd': [
                sys.executable, str(SCRIPT_DIR / "fig_subsample.py"),
                '--output-dir', sub_output,
                '--method-comparison', '--dataset', dataset,
            ] + common_args,
        })

        # Compact heatmap
        tasks.append({
            'description': 'Subsample: Compact heatmap',
            'cmd': [
                sys.executable, str(SCRIPT_DIR / "fig_subsample.py"),
                '--output-dir', sub_output,
                '--compact',
            ] + common_args,
        })

    # ==========================================================================
    # Execute
    # ==========================================================================

    print("=" * 70)
    print("FIGURE GENERATION")
    print("=" * 70)
    print(f"Mode: {'CACHE-ONLY (skip eval on miss)' if cache_only else 'FULL (evaluate if needed)'}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Tasks: {len(tasks)}")
    print("=" * 70)

    # Create output directories
    for d in ['fig_hyperparam', 'fig_histogram', 'fig_subsample']:
        (RESULTS_BASE / d).mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    success_count = 0

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['description']}")

        if run_command(task['cmd'], task['description'], dry_run):
            success_count += 1
            print("  ✓ Done")
        else:
            print("  ✗ Failed")

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {success_count}/{len(tasks)} tasks")
    print(f"Time: {elapsed.total_seconds():.1f}s")
    print(f"\nOutput directories:")
    print(f"  {RESULTS_BASE / 'fig_hyperparam'}")
    print(f"  {RESULTS_BASE / 'fig_histogram'}")
    print(f"  {RESULTS_BASE / 'fig_subsample'}")
    print("=" * 70)

    return success_count == len(tasks)


def list_outputs():
    """List what would be generated."""
    print("=" * 70)
    print("EXPECTED OUTPUTS")
    print("=" * 70)

    outputs = {
        'fig_hyperparam': [
            'main_model_{model}.pdf           - Main paper: one model × 4 datasets',
            'main_dataset_{dataset}.pdf       - Main paper: one dataset × 4 models',
            'appendix_full_grid.pdf           - Full 4×4 grid for appendix',
            'individual/{model}_{dataset}.pdf - Individual plots',
            'hyperparam_data.json             - Numerical data',
        ],
        'fig_histogram': [
            'histogram_{model}_{dataset}.pdf  - Combined 6-panel figure',
            'individual/{metric}_*.pdf        - Individual metric plots',
            'histogram_data_*.json            - Numerical data + stats',
        ],
        'fig_subsample': [
            'design2_main_model_{model}.pdf   - Main paper scaling plot',
            'design4_improvement_{model}.pdf  - Improvement over baseline',
            'design1_grid_appendix.pdf        - Full grid (appendix)',
            'design3_method_comparison_*.pdf  - Method comparison view',
            'design5_compact_heatmap.pdf      - Summary heatmap',
            'subsample_data.json              - Numerical data',
        ],
    }

    for category, files in outputs.items():
        print(f"\n{RESULTS_BASE / category}/")
        for f in files:
            print(f"  {f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate all paper figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_figures.py              # Full generation
  python run_all_figures.py --cache-only # Fast: skip eval on cache miss
  python run_all_figures.py --list       # Show expected outputs
  python run_all_figures.py --dry-run    # Show commands without running
        """
    )

    parser.add_argument('--cache-only', '-c', action='store_true',
                        help='Only use cached results, skip evaluation on cache miss (fast mode)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Model for main figures (default: {DEFAULT_MODEL})')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help=f'Dataset for main figures (default: {DEFAULT_DATASET})')
    parser.add_argument('--list', action='store_true',
                        help='List expected outputs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--show', action='store_true',
                        help='Display figures (default: no display)')

    args = parser.parse_args()

    if args.list:
        list_outputs()
        return

    success = generate_all_figures(
        model=args.model,
        dataset=args.dataset,
        cache_only=args.cache_only,
        dry_run=args.dry_run,
        no_show=not args.show,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
