"""Centralized alpha signal evaluation and normalization.

Eliminates duplicated parse → evaluate → sanitize → normalize logic
that was previously copy-pasted across pipeline, paper, forward, and simulator.
"""
from __future__ import annotations

import logging

import numpy as np

from ..dsl import parse
from ..dsl.expr import Expr

logger = logging.getLogger(__name__)

# Sentinel fitness for failed evaluations (used by GP and backtest ranking).
FAILED_FITNESS = -999.0


class EvaluationError(Exception):
    """Raised when an alpha expression cannot be evaluated."""


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize a raw signal to [-1, 1] via std-scaling.

    If std == 0, falls back to sign(signal) clipped to [-1, 1].
    """
    std = signal.std()
    if std > 0:
        return np.clip(signal / std, -1, 1)
    return np.clip(np.sign(signal), -1, 1)


def evaluate_expression(
    expr: Expr,
    data: dict[str, np.ndarray],
    n_days: int,
) -> np.ndarray:
    """Evaluate a DSL expression and sanitize the output.

    Returns a 1-D float array of length n_days.
    Raises EvaluationError on any failure (missing feature, length mismatch, etc.).
    """
    try:
        sig = expr.evaluate(data)
        sig = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0)
        if sig.ndim == 0:
            sig = np.full(n_days, float(sig))
        if len(sig) != n_days:
            raise EvaluationError(
                f"Signal length {len(sig)} != expected {n_days}"
            )
        return sig
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(str(exc)) from exc


def evaluate_alpha(
    expression: str,
    data: dict[str, np.ndarray],
    n_days: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Parse, evaluate, and optionally normalize an alpha expression.

    Convenience wrapper combining parse + evaluate + normalize.
    """
    expr = parse(expression)
    sig = evaluate_expression(expr, data, n_days)
    if normalize:
        sig = normalize_signal(sig)
    return sig
