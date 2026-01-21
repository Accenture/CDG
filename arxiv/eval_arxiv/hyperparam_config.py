"""
Hyperparameter configuration manager for CDG method.

Stores and retrieves optimal hyperparameters (alpha, beta, position_pct) per model.
Used to persist results from Exp 3 (hyperparameter tuning) for use in Exp 1/2.

Usage:
    from hyperparam_config import HyperparamConfig

    # Save optimal params for a model (from Exp 3)
    config = HyperparamConfig(results_dir)
    config.set_model_params('qwen32b', alpha=0.5, beta=20, position_pct=20, accuracy=0.85)
    config.save()

    # Load params for a model (in Exp 1/2)
    config = HyperparamConfig(results_dir)
    params = config.get_model_params('qwen32b')
    # Returns: {'alpha': 0.5, 'beta': 20, 'position_pct': 20}

    # Get params with fallback to defaults
    params = config.get_model_params('unknown_model', use_defaults=True)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List


# Default hyperparameters (used when no tuning data available)
DEFAULT_HYPERPARAMS = {
    'alpha': 0.5,
    'beta': 10,
    'position_pct': 20,
}

CONFIG_FILENAME = "cdg_hyperparams.json"


class HyperparamConfig:
    """
    Manages per-model hyperparameter configurations.

    Config file structure:
    {
        "version": "1.0",
        "updated_at": "2024-01-01T12:00:00",
        "defaults": {"alpha": 0.5, "beta": 10, "position_pct": 20},
        "models": {
            "qwen32b": {
                "alpha": 0.5,
                "beta": 20,
                "position_pct": 20,
                "accuracy": 0.85,
                "tuned_at": "2024-01-01T12:00:00",
                "tuning_details": {...}
            },
            ...
        }
    }
    """

    def __init__(self, results_dir: str):
        """
        Initialize config manager.

        Args:
            results_dir: Base results directory where config file is stored
        """
        self.results_dir = Path(results_dir)
        self.config_path = self.results_dir / CONFIG_FILENAME
        self.config = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing config or create new one."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")

        # Create new config
        return {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'defaults': DEFAULT_HYPERPARAMS.copy(),
            'models': {},
        }

    def save(self):
        """Save config to disk."""
        self.config['updated_at'] = datetime.now().isoformat()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Saved hyperparameter config to: {self.config_path}")

    def set_model_params(self, model: str, alpha: float, beta: float,
                         position_pct: int, accuracy: float = None,
                         tuning_details: dict = None):
        """
        Set optimal hyperparameters for a model.

        Args:
            model: Model name (e.g., 'qwen32b', 'deepseek8b')
            alpha: Optimal alpha value
            beta: Optimal beta value
            position_pct: Optimal position percentile
            accuracy: Accuracy achieved with these params (optional)
            tuning_details: Additional tuning info (optional)
        """
        self.config['models'][model] = {
            'alpha': alpha,
            'beta': beta,
            'position_pct': position_pct,
            'accuracy': accuracy,
            'tuned_at': datetime.now().isoformat(),
        }

        if tuning_details:
            self.config['models'][model]['tuning_details'] = tuning_details

    def get_model_params(self, model: str, use_defaults: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get hyperparameters for a model.

        Args:
            model: Model name
            use_defaults: If True, return defaults when model not found

        Returns:
            Dict with 'alpha', 'beta', 'position_pct' keys, or None if not found
        """
        if model in self.config['models']:
            params = self.config['models'][model]
            return {
                'alpha': params['alpha'],
                'beta': params['beta'],
                'position_pct': params['position_pct'],
            }

        if use_defaults:
            return self.config.get('defaults', DEFAULT_HYPERPARAMS).copy()

        return None

    def get_all_models(self) -> List[str]:
        """Get list of all models with tuned parameters."""
        return list(self.config['models'].keys())

    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get full info for a model including accuracy and tuning details."""
        return self.config['models'].get(model)

    def set_defaults(self, alpha: float, beta: float, position_pct: int):
        """Set default hyperparameters (used when model-specific not available)."""
        self.config['defaults'] = {
            'alpha': alpha,
            'beta': beta,
            'position_pct': position_pct,
        }

    def print_summary(self):
        """Print summary of all configured hyperparameters."""
        print("\n" + "=" * 70)
        print("CDG HYPERPARAMETER CONFIGURATION")
        print("=" * 70)
        print(f"Config file: {self.config_path}")
        print(f"Last updated: {self.config.get('updated_at', 'N/A')}")

        defaults = self.config.get('defaults', DEFAULT_HYPERPARAMS)
        print(f"\nDefaults: alpha={defaults['alpha']}, beta={defaults['beta']}, "
              f"position_pct={defaults['position_pct']}")

        models = self.config.get('models', {})
        if models:
            print(f"\nPer-model configurations ({len(models)} models):")
            print(f"{'Model':<20} {'Alpha':<8} {'Beta':<8} {'Pos%':<8} {'Accuracy':<10}")
            print("-" * 60)
            for model, params in sorted(models.items()):
                acc = params.get('accuracy')
                acc_str = f"{100*acc:.1f}%" if acc else "N/A"
                print(f"{model:<20} {params['alpha']:<8} {params['beta']:<8} "
                      f"{params['position_pct']:<8} {acc_str:<10}")
        else:
            print("\nNo per-model configurations yet. Run Exp 3 to tune hyperparameters.")

        print("=" * 70)


def find_best_params_from_sweep(results: list, model: str) -> dict:
    """
    Find the best hyperparameters from a sweep for a specific model.

    Args:
        results: List of result dicts from sweep_beta or similar
        model: Model name to filter by

    Returns:
        Dict with best alpha, beta, position_pct, and accuracy
    """
    # Filter results for this model
    model_results = [r for r in results if r.get('model') == model]

    if not model_results:
        return None

    # Find best by accuracy
    best = max(model_results, key=lambda r: r.get('accuracy', 0))

    return {
        'alpha': best.get('alpha', DEFAULT_HYPERPARAMS['alpha']),
        'beta': best.get('beta', DEFAULT_HYPERPARAMS['beta']),
        'position_pct': best.get('position_pct', DEFAULT_HYPERPARAMS['position_pct']),
        'accuracy': best.get('accuracy', 0),
        'correct': best.get('correct', 0),
        'total': best.get('total', 0),
    }


def find_best_params_across_datasets(results: list, model: str) -> dict:
    """
    Find hyperparameters that work best across all datasets for a model.

    Aggregates accuracy across datasets and finds params that maximize
    total correct answers.

    Args:
        results: List of result dicts from sweep
        model: Model name

    Returns:
        Dict with best params and aggregated accuracy
    """
    # Filter for this model
    model_results = [r for r in results if r.get('model') == model]

    if not model_results:
        return None

    # Group by (alpha, beta, position_pct)
    from collections import defaultdict
    param_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'datasets': []})

    for r in model_results:
        key = (r.get('alpha'), r.get('beta'), r.get('position_pct'))
        param_stats[key]['correct'] += r.get('correct', 0)
        param_stats[key]['total'] += r.get('total', 0)
        param_stats[key]['datasets'].append(r.get('dataset'))

    # Find best params (maximize total correct)
    best_key = max(param_stats.keys(),
                   key=lambda k: param_stats[k]['correct'] / param_stats[k]['total']
                   if param_stats[k]['total'] > 0 else 0)

    best_stats = param_stats[best_key]

    return {
        'alpha': best_key[0],
        'beta': best_key[1],
        'position_pct': best_key[2],
        'accuracy': best_stats['correct'] / best_stats['total'] if best_stats['total'] > 0 else 0,
        'correct': best_stats['correct'],
        'total': best_stats['total'],
        'datasets': list(set(best_stats['datasets'])),
    }
