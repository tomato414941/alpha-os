"""Behavior descriptor computation for MAP-Elites archive.

3 axes: feature_bucket, holding_half_life, complexity.
"""
from __future__ import annotations

import numpy as np

from ..dsl.expr import Expr
from ..dsl.features import collect_feature_names
from ..dsl.generator import _collect_nodes


N_FEAT_BUCKETS = 100


def compute_behavior(
    signal: np.ndarray,
    expr: Expr,
    feature_subset: frozenset[str] | None = None,
) -> np.ndarray:
    """Compute 3D behavior descriptor for an alpha signal.

    Axes:
      0: feature_bucket — hash of features actually used by the expression
      1: holding_half_life — signal autocorrelation half-life (days)
      2: complexity — expression tree node count
    """
    return np.array([
        float(_feature_bucket(expr)),
        _holding_half_life(signal),
        float(len(_collect_nodes(expr))),
    ])


def _feature_bucket(expr: Expr) -> int:
    """Hash the features actually referenced by the expression."""
    names = collect_feature_names(expr)
    if not names:
        return 0
    return hash(frozenset(names)) % N_FEAT_BUCKETS


def _holding_half_life(signal: np.ndarray) -> float:
    s = signal[~np.isnan(signal)]
    if len(s) < 10:
        return 0.0
    if s.std() == 0:
        return 0.0
    ac1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    if np.isnan(ac1) or ac1 <= 0 or ac1 >= 1.0:
        return 0.0
    return float(-np.log(2) / np.log(ac1))
