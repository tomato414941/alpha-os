"""Automatic evaluation universe selection via correlation clustering.

Selects a diverse subset of assets for cross-asset alpha evaluation.
Criteria: low pairwise correlation, long data history, diverse volatility.

The eval universe is computed once and cached to disk so that all consumers
(generator, admission, manual tests) use the same fixed set of assets.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

_EVAL_UNIVERSE_PATH = DATA_DIR / "eval_universe.json"

# Module-level cache — loaded once per process
EVAL_UNIVERSE: list[str] = []


def load_cached_eval_universe() -> list[str]:
    """Load eval universe from disk cache. Returns empty list if not cached."""
    global EVAL_UNIVERSE
    if EVAL_UNIVERSE:
        return EVAL_UNIVERSE
    if not _EVAL_UNIVERSE_PATH.exists():
        return []
    try:
        raw = json.loads(_EVAL_UNIVERSE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list) and all(isinstance(s, str) for s in raw):
            EVAL_UNIVERSE = raw
            return raw
    except Exception:
        pass
    return []


def save_eval_universe(selected: list[str]) -> None:
    """Save eval universe to disk cache."""
    global EVAL_UNIVERSE
    EVAL_UNIVERSE = selected
    _EVAL_UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EVAL_UNIVERSE_PATH.write_text(json.dumps(selected), encoding="utf-8")
    logger.info("Eval universe cached: %d assets -> %s", len(selected), _EVAL_UNIVERSE_PATH)


def select_eval_universe(
    data: dict[str, np.ndarray],
    candidates: list[str],
    *,
    n_clusters: int = 20,
    min_finite_days: int = 500,
    lookback: int = 504,
) -> list[str]:
    """Select diverse evaluation universe via hierarchical clustering.

    Parameters
    ----------
    data : {signal_name: values_array} full data dict.
    candidates : list of signal names to consider.
    n_clusters : target number of clusters (= universe size).
    min_finite_days : minimum finite data points required.
    lookback : days of return history for correlation computation.

    Returns
    -------
    List of selected signal names, one per cluster.
    """
    # Filter to assets with enough data
    valid = []
    returns = {}
    for sig in candidates:
        arr = data.get(sig)
        if arr is None:
            continue
        finite_idx = np.where(np.isfinite(arr))[0]
        if len(finite_idx) < min_finite_days:
            continue
        # Use only the valid (finite) portion of prices
        first_valid = int(finite_idx[0])
        valid_prices = arr[first_valid:]
        rets = np.diff(valid_prices) / valid_prices[:-1]
        rets = np.where(np.isfinite(rets), rets, 0.0)
        returns[sig] = rets
        valid.append(sig)

    if len(valid) <= n_clusters:
        logger.info("Eval universe: %d valid assets (<= %d clusters), using all", len(valid), n_clusters)
        return valid

    # Build return matrix (right-aligned, padded with 0)
    n = len(valid)
    R = np.zeros((n, lookback))
    for i, name in enumerate(valid):
        r = returns[name]
        k = min(len(r), lookback)
        R[i, lookback - k:] = r[-k:]

    # Correlation matrix → distance
    corr = np.corrcoef(R)
    corr = np.nan_to_num(corr)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, n_clusters, criterion="maxclust")
    except ImportError:
        logger.warning("scipy not available, falling back to random selection")
        rng = np.random.default_rng(42)
        return list(rng.choice(valid, size=min(n_clusters, len(valid)), replace=False))

    # Pick best representative from each cluster:
    # prefer longest data history, then highest volatility (more signal)
    clusters: dict[int, tuple[str, int, float]] = {}
    for i, name in enumerate(valid):
        label = int(labels[i])
        r = returns[name]
        n_days = len(r)
        vol = float(np.std(r[-252:]) * np.sqrt(252)) if len(r) >= 252 else 0.0
        score = n_days + vol * 1000  # prefer long history + high vol
        if label not in clusters or score > clusters[label][2]:
            clusters[label] = (name, n_days, score)

    selected = [clusters[label][0] for label in sorted(clusters)]

    # Log summary
    sel_idx = [valid.index(s) for s in selected]
    sel_corr = corr[np.ix_(sel_idx, sel_idx)]
    upper = sel_corr[np.triu_indices(len(selected), k=1)]
    mean_corr = float(np.mean(np.abs(upper))) if len(upper) > 0 else 0.0

    logger.info(
        "Eval universe: %d candidates -> %d clusters -> %d selected (mean |corr|=%.3f)",
        len(valid), n_clusters, len(selected), mean_corr,
    )

    return selected
