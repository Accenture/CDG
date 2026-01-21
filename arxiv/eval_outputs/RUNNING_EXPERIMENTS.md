# Running Experiments Log
**Started:** 2026-01-15
**Updated:** 2026-01-15 2PM

## Data Fix (1/15 2PM)

Fixed 3 experiments with corrected data:
- `deepseek8b_aime2025_512` - replaced
- `gptoss20b_aime2025_512` - replaced
- `gptoss20b_hmmt2025_512` - replaced

```bash
./scripts/fix_results.sh --execute  # Done
./scripts/trace_subsample/run_subsample.sh  # Regenerates subsamples
```

## Verified Optimal Parameters (1/15 2PM)

| Model | Alpha | Beta | Pos% |
|-------|-------|------|------|
| DeepSeek | 0.5 | 10 | 10 |
| Gptoss20b | 0.5 | 10 | 10 |
| Gemma3 | 0.5 | 3 | 10 |
| QwQ | 0.5 | 3 | 10 |

## Screen Sessions

| Screen Name | Experiment | Command | Status |
|-------------|------------|---------|--------|
| `eval` (88293) | Exp 3 - Hyperparam tuning | `bash scripts/eval/run_all_experiments.sh --exp 3` | Running |
| `eval2` (90385) | Exp 2 - Subsample scaling | `bash scripts/eval/run_all_experiments.sh --exp 2` | Running |
| `eval_sp` (91468) | Special 1-2, 1-3-1, 1-3-2 | `bash scripts/eval/run_special_experiments.sh` | Running |
| `eval_sp2` (93977) | Special 1-4 (beta=0 ablation) | `bash scripts/eval/run_special_experiments.sh --exp 1-4` | Running |

## New Special Experiments (1/15 2PM)

| Experiment | Description | Command |
|------------|-------------|---------|
| **1-2-2** | Main eval with verified optimal params | `./run_special_experiments.sh --exp 1-2-2` |
| **2-2** | Subsample scaling with verified params | `./run_special_experiments.sh --exp 2-2` |

## Check Progress

```bash
# Attach to screen
screen -r eval      # Exp 3
screen -r eval2     # Exp 2
screen -r eval_sp   # Special 1-2, 1-3-1, 1-3-2
screen -r eval_sp2  # Special 1-4

# Detach from screen: Ctrl+A, then D
```

## Output Files

### Exp 2 (Subsample Scaling)
- `results/eval_outputs/exp2_subsample_*.txt`
- `results/eval_outputs/subsample_scaling_*.png`

### Exp 3 (Hyperparam Tuning)
- `results/eval_outputs/exp3_hyperparam_*.txt`
- `results/eval_outputs/hyperparam_sweep_pos10_*.json`
- `results/eval_outputs/hyperparam_sweep_pos10_*.png`
- `results/eval_outputs/hyperparam_sweep_pos20_*.json`
- `results/eval_outputs/hyperparam_sweep_pos20_*.png`

### Special Experiments
- `results/eval_outputs/special/exp1-2_model_specific_*.txt`
- `results/eval_outputs/special/exp1-2-2_verified_optimal_*.txt`
- `results/eval_outputs/special/exp1-3-1_general_pos20_*.txt`
- `results/eval_outputs/special/exp1-3-2_general_pos10_*.txt`
- `results/eval_outputs/special/exp1-4_beta0_ablation_*.txt`
- `results/eval_outputs/special/exp2-2_subsample_verified_*.txt`

## Cache Location
- `{OUTPUT_BASE}/.eval_cache/results/` - Cached evaluation results
- `{OUTPUT_BASE}/.eval_cache/features/` - Cached extracted features

## Quick Check Commands

```bash
# List output files
ls -la results/eval_outputs/
ls -la results/eval_outputs/special/

# Tail latest output
tail -f results/eval_outputs/exp3_hyperparam_*.txt
tail -f results/eval_outputs/special/exp1-4_*.txt

# Check if processes are running
ps aux | grep python
```

## Dependency Graph

```
512-trace pkl files
        │
        ├──────────────────┬────────────────────┐
        │                  │                    │
        ▼                  ▼                    ▼
   Subtrace            Exp 3              Special 1-2-2
   Creation          (HP sweep)          (hardcoded)
        │                  │                    │
        │                  ▼                    │
        │         cdg_hyperparams.json         │
        │                  │                    │
        ▼                  ▼                    ▼
  Special 2-2         Exp 1 & 2              DONE
  (hardcoded)     (use tuned params)
        │                  │
        ▼                  ▼
      DONE               DONE
```

## Notes
- Exp 1 (main evaluation) depends on Exp 3 results (tuned hyperparams)
- Special experiments 1-2-2 and 2-2 use hardcoded verified params (independent of Exp 3)
- Exp 2 and Exp 3 are independent, can run in parallel
- Special 2-2 needs subtrace creation to complete first
- All special experiments (1-2, 1-2-2, 1-3-1, 1-3-2, 1-4) are independent
