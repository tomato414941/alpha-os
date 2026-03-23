from __future__ import annotations

import numpy as np

from . import parse
from .expr import Expr

FAILED_FITNESS = -999.0


class EvaluationError(Exception):
    pass


def sanitize_signal(signal: np.ndarray | float) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(signal, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std > 0:
        return np.clip(signal / std, -1, 1)
    return np.clip(np.sign(signal), -1, 1)


def evaluate_expression(
    expr: Expr,
    data: dict[str, np.ndarray],
    n_days: int,
) -> np.ndarray:
    try:
        sig = sanitize_signal(expr.evaluate(data))
        if sig.ndim == 0:
            sig = np.full(n_days, float(sig))
        if len(sig) != n_days:
            raise EvaluationError(f"Signal length {len(sig)} != expected {n_days}")
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
    expr = parse(expression)
    sig = evaluate_expression(expr, data, n_days)
    if normalize:
        sig = normalize_signal(sig)
    return sig
