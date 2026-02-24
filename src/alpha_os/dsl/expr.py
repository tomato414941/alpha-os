from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tokens import UNARY_OPS, BINARY_OPS, ROLLING_OPS, PAIR_ROLLING_OPS


class Expr:
    """Base class for expression nodes."""

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Feature(Expr):
    """Leaf node: a signal name like 'nvda', 'vix_close'."""

    name: str

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if self.name not in data:
            raise KeyError(f"Feature '{self.name}' not found in data")
        return np.asarray(data[self.name], dtype=np.float64)

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant(Expr):
    """Leaf node: a scalar value."""

    value: float

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        # Return a scalar; NumPy broadcasting handles the rest
        return np.float64(self.value)

    def __repr__(self) -> str:
        # Emit integer form when possible: 0.5 -> "0.5", 2.0 -> "2.0"
        return str(self.value)


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation: (neg x), (abs x), etc."""

    op: str
    child: Expr

    def __post_init__(self) -> None:
        if self.op not in UNARY_OPS:
            raise ValueError(f"Unknown unary op: {self.op}")

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        x = self.child.evaluate(data)
        return _eval_unary(self.op, x)

    def __repr__(self) -> str:
        return f"({self.op} {self.child!r})"


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary operation: (add x y), (sub x y), etc."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in BINARY_OPS:
            raise ValueError(f"Unknown binary op: {self.op}")

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        lhs = self.left.evaluate(data)
        rhs = self.right.evaluate(data)
        return _eval_binary(self.op, lhs, rhs)

    def __repr__(self) -> str:
        return f"({self.op} {self.left!r} {self.right!r})"


@dataclass(frozen=True)
class RollingOp(Expr):
    """Rolling window operation: (mean_20 x), (roc_10 x), etc."""

    op: str
    window: int
    child: Expr

    def __post_init__(self) -> None:
        if self.op not in ROLLING_OPS:
            raise ValueError(f"Unknown rolling op: {self.op}")

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        x = self.child.evaluate(data)
        return _eval_rolling(self.op, self.window, x)

    def __repr__(self) -> str:
        return f"({self.op}_{self.window} {self.child!r})"


@dataclass(frozen=True)
class PairRollingOp(Expr):
    """Pair rolling operation: (corr_60 x y)."""

    op: str
    window: int
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in PAIR_ROLLING_OPS:
            raise ValueError(f"Unknown pair rolling op: {self.op}")

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        lhs = self.left.evaluate(data)
        rhs = self.right.evaluate(data)
        return _eval_pair_rolling(self.op, self.window, lhs, rhs)

    def __repr__(self) -> str:
        return f"({self.op}_{self.window} {self.left!r} {self.right!r})"


# ---------------------------------------------------------------------------
# Evaluation helpers (pure NumPy / pandas)
# ---------------------------------------------------------------------------

def _eval_unary(op: str, x: np.ndarray) -> np.ndarray:
    if op == "neg":
        return -x
    if op == "abs":
        return np.abs(x)
    if op == "sign":
        return np.sign(x)
    if op == "log":
        # Safe signed log: sign(x) * log1p(|x|)
        return np.sign(x) * np.log1p(np.abs(x))
    if op == "zscore":
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / (s + 1e-10)
    raise ValueError(f"Unknown unary op: {op}")


def _eval_binary(op: str, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "div":
        # Safe division: avoid zero
        return lhs / (rhs + 1e-10 * np.sign(rhs + 1e-10))
    if op == "max":
        return np.maximum(lhs, rhs)
    if op == "min":
        return np.minimum(lhs, rhs)
    raise ValueError(f"Unknown binary op: {op}")


def _eval_rolling(op: str, window: int, x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)

    if op == "mean":
        return s.rolling(window, min_periods=window).mean().to_numpy()
    if op == "std":
        return s.rolling(window, min_periods=window).std(ddof=1).to_numpy()
    if op == "ts_max":
        return s.rolling(window, min_periods=window).max().to_numpy()
    if op == "ts_min":
        return s.rolling(window, min_periods=window).min().to_numpy()
    if op == "delta":
        result = np.empty_like(x, dtype=np.float64)
        result[:window] = np.nan
        result[window:] = x[window:] - x[:-window]
        return result
    if op == "roc":
        past = np.empty_like(x, dtype=np.float64)
        past[:window] = np.nan
        past[window:] = x[:-window]
        return (x - past) / (np.abs(past) + 1e-10)
    if op == "rank":
        return (
            s.rolling(window, min_periods=window)
            .apply(_percentile_rank, raw=True)
            .to_numpy()
        )
    if op == "ema":
        return s.ewm(span=window, adjust=False).mean().to_numpy()
    raise ValueError(f"Unknown rolling op: {op}")


def _percentile_rank(arr: np.ndarray) -> float:
    last = arr[-1]
    return np.sum(arr <= last) / len(arr)


def _eval_pair_rolling(
    op: str, window: int, lhs: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    sl = pd.Series(lhs)
    sr = pd.Series(rhs)

    if op == "corr":
        return sl.rolling(window, min_periods=window).corr(sr).to_numpy()
    if op == "cov":
        return sl.rolling(window, min_periods=window).cov(sr).to_numpy()
    raise ValueError(f"Unknown pair rolling op: {op}")
