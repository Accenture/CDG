"""
Caching and resume support for evaluation scripts.

Provides disk-based caching for:
- Extracted features per run (expensive to recompute)
- Evaluation results per (run, method, params) combination

Cache is stored in a dedicated directory alongside results.

Usage:
    from eval_cache import EvalCache

    cache = EvalCache(results_dir)

    # Check if we have cached features
    features = cache.load_features(run_id)
    if features is None:
        features = extract_features(...)
        cache.save_features(run_id, features)

    # Check if we have cached results
    result = cache.load_result(run_id, method, params)
    if result is None:
        result = evaluate(...)
        cache.save_result(run_id, method, params, result)
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class EvalCache:
    """
    Disk-based cache for evaluation results and intermediate features.

    Cache structure:
        {results_dir}/.eval_cache/
            features/
                {run_id}.pkl          # Extracted features per run
            results/
                {run_id}_{method}_{params_hash}.json  # Evaluation results
            meta.json                 # Cache metadata
    """

    CACHE_DIR = ".eval_cache"
    FEATURES_DIR = "features"
    RESULTS_DIR = "results"
    META_FILE = "meta.json"
    VERSION = "1.0"

    def __init__(self, results_dir: str, enabled: bool = True):
        """
        Initialize cache.

        Args:
            results_dir: Base results directory
            enabled: If False, all cache operations are no-ops
        """
        self.results_dir = Path(results_dir)
        self.enabled = enabled
        self.cache_dir = self.results_dir / self.CACHE_DIR
        self.features_dir = self.cache_dir / self.FEATURES_DIR
        self.results_cache_dir = self.cache_dir / self.RESULTS_DIR

        if enabled:
            self._ensure_dirs()

    def _ensure_dirs(self):
        """Create cache directories if they don't exist."""
        self.cache_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        self.results_cache_dir.mkdir(exist_ok=True)

    def _params_hash(self, params: dict) -> str:
        """Create a short hash of parameters for cache key."""
        # Sort keys for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def _get_run_mtime(self, run_id: str) -> Optional[float]:
        """Get modification time of newest pkl file in run directory."""
        # Check both main dir and subset_trace subdir
        run_path = self.results_dir / run_id
        if not run_path.exists():
            run_path = self.results_dir / "subset_trace" / run_id
        if not run_path.exists():
            return None

        pkl_files = list(run_path.glob("*.pkl"))
        if not pkl_files:
            return None

        return max(f.stat().st_mtime for f in pkl_files)

    # ========================================
    # Features Cache (extracted trace features)
    # ========================================

    def _features_path(self, run_id: str) -> Path:
        return self.features_dir / f"{run_id}.pkl"

    def load_features(self, run_id: str) -> Optional[Dict]:
        """
        Load cached features for a run.

        Returns None if:
        - Cache disabled
        - Cache file doesn't exist
        - Cache is stale (run files modified after cache)
        """
        if not self.enabled:
            return None

        cache_path = self._features_path(run_id)
        if not cache_path.exists():
            return None

        # Check staleness
        run_mtime = self._get_run_mtime(run_id)
        if run_mtime is None:
            return None

        cache_mtime = cache_path.stat().st_mtime
        if run_mtime > cache_mtime:
            # Run was modified after cache was created
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            # Validate structure
            if isinstance(data, dict) and 'features' in data:
                return data['features']
        except Exception:
            pass

        return None

    def save_features(self, run_id: str, features: Dict):
        """Save extracted features to cache."""
        if not self.enabled:
            return

        cache_path = self._features_path(run_id)
        data = {
            'features': features,
            'run_id': run_id,
            'cached_at': datetime.now().isoformat(),
            'version': self.VERSION,
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            # Cache write failures are non-fatal
            pass

    # ========================================
    # Results Cache (evaluation results)
    # ========================================

    def _result_path(self, run_id: str, method: str, params: dict) -> Path:
        params_hash = self._params_hash(params)
        return self.results_cache_dir / f"{run_id}_{method}_{params_hash}.json"

    def load_result(self, run_id: str, method: str, params: dict) -> Optional[Dict]:
        """
        Load cached evaluation result.

        Returns None if cache miss or stale.
        """
        if not self.enabled:
            return None

        cache_path = self._result_path(run_id, method, params)
        if not cache_path.exists():
            return None

        # Check staleness
        run_mtime = self._get_run_mtime(run_id)
        if run_mtime is None:
            return None

        cache_mtime = cache_path.stat().st_mtime
        if run_mtime > cache_mtime:
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'result' in data:
                return data['result']
        except Exception:
            pass

        return None

    def save_result(self, run_id: str, method: str, params: dict, result: Dict):
        """Save evaluation result to cache."""
        if not self.enabled:
            return

        cache_path = self._result_path(run_id, method, params)
        data = {
            'result': result,
            'run_id': run_id,
            'method': method,
            'params': params,
            'cached_at': datetime.now().isoformat(),
            'version': self.VERSION,
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ========================================
    # Batch Operations
    # ========================================

    def get_cached_results(self, run_ids: List[str], methods: List[str],
                           params: dict) -> Dict[str, Dict[str, Dict]]:
        """
        Load all cached results for given runs and methods.

        Returns:
            {run_id: {method: result_dict, ...}, ...}
        """
        cached = {}
        for run_id in run_ids:
            cached[run_id] = {}
            for method in methods:
                result = self.load_result(run_id, method, params)
                if result is not None:
                    cached[run_id][method] = result
        return cached

    def get_missing_tasks(self, run_ids: List[str], methods: List[str],
                          params: dict) -> List[tuple]:
        """
        Get list of (run_id, method) pairs that are not cached.

        Returns:
            List of (run_id, method) tuples that need computation
        """
        missing = []
        for run_id in run_ids:
            for method in methods:
                if self.load_result(run_id, method, params) is None:
                    missing.append((run_id, method))
        return missing

    # ========================================
    # Cache Management
    # ========================================

    def clear(self):
        """Clear all cache files."""
        if not self.enabled:
            return

        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self._ensure_dirs()

    def clear_results(self):
        """Clear only results cache (keep features)."""
        if not self.enabled:
            return

        import shutil
        if self.results_cache_dir.exists():
            shutil.rmtree(self.results_cache_dir)
        self.results_cache_dir.mkdir(exist_ok=True)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.cache_dir.exists():
            return {'enabled': False}

        features_files = list(self.features_dir.glob("*.pkl"))
        results_files = list(self.results_cache_dir.glob("*.json"))

        features_size = sum(f.stat().st_size for f in features_files)
        results_size = sum(f.stat().st_size for f in results_files)

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'features_count': len(features_files),
            'features_size_mb': round(features_size / 1024 / 1024, 2),
            'results_count': len(results_files),
            'results_size_mb': round(results_size / 1024 / 1024, 2),
        }
