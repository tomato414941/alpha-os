"""Classical indicator producer — computes well-known signals and writes to prediction store.

This is an independent producer. It doesn't import the pipeline,
the registry, or the DSL. It reads raw data and writes predictions.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np

from ..config import Config, DATA_DIR
from ..data.signal_client import build_signal_client_from_config
from ..data.store import DataStore
from ..data.universe import init_universe
from ..data.eval_universe import load_cached_eval_universe
from .store import Prediction, PredictionStore, SignalMeta

logger = logging.getLogger(__name__)


# ── Classical signal functions ──
# Each takes a data dict and returns a scalar (today's signal value).
# Normalization is the pipeline's responsibility, not the producer's.

def _zscore(arr: np.ndarray, window: int = 60) -> float:
    """Z-score of the last value over a trailing window."""
    if len(arr) < window:
        return 0.0
    recent = arr[-window:]
    m, s = float(np.nanmean(recent)), float(np.nanstd(recent))
    if s < 1e-12:
        return 0.0
    return float((arr[-1] - m) / s)


def _rank(arr: np.ndarray, window: int = 20) -> float:
    """Percentile rank of the last value over a trailing window."""
    if len(arr) < window:
        return 0.5
    recent = arr[-window:]
    val = arr[-1]
    return float(np.mean(recent <= val))


def _roc(arr: np.ndarray, period: int = 20) -> float:
    """Rate of change over period days."""
    if len(arr) < period + 1:
        return 0.0
    prev = arr[-period - 1]
    if abs(prev) < 1e-12:
        return 0.0
    return float((arr[-1] - prev) / abs(prev))


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int = 60) -> float:
    """Trailing correlation between two series."""
    n = min(len(a), len(b), window)
    if n < 20:
        return 0.0
    x, y = a[-n:], b[-n:]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _rolling_std(arr: np.ndarray, window: int = 20) -> float:
    """Trailing standard deviation."""
    if len(arr) < window:
        return 0.0
    return float(np.nanstd(arr[-window:]))


# ── Signal definitions ──

CLASSICAL_SIGNALS = {
    "classical_mean_rev_rsi": {
        "compute": lambda data: -_rank(_roc_array(data.get("sp500"), 5), 20),
        "horizon": 1,
        "description": "Short-term mean reversion (RSI-like)",
    },
    "classical_mean_rev_zscore": {
        "compute": lambda data: -_zscore(data.get("sp500"), 60),
        "horizon": 20,
        "description": "Long-term mean reversion",
    },
    "classical_carry_short": {
        "compute": lambda data: -_last(data.get("tsy_yield_2y")),
        "horizon": 20,
        "description": "Low short rates → risk-on",
    },
    "classical_dollar_weak": {
        "compute": lambda data: -_roc(data.get("dxy"), 20),
        "horizon": 20,
        "description": "Dollar weakness → assets rise",
    },
    "classical_fear_greed_smooth": {
        "compute": lambda data: _rolling_mean_rank(data.get("fear_greed"), rank_w=20, mean_w=60),
        "horizon": 20,
        "description": "Smoothed Fear & Greed rank",
    },
    "classical_low_vol": {
        "compute": lambda data: -_rolling_std(data.get("sp500"), 20),
        "horizon": 20,
        "description": "Low volatility → long",
    },
    "classical_gold_dxy_corr": {
        "compute": lambda data: _rolling_corr(data.get("gold"), data.get("dxy"), 60),
        "horizon": 20,
        "description": "Gold-dollar correlation",
    },
}


def _roc_array(arr: np.ndarray | None, period: int = 5) -> np.ndarray:
    """Rate of change as array (for rank computation)."""
    if arr is None or len(arr) < period + 1:
        return np.array([0.0])
    result = np.zeros(len(arr))
    result[period:] = (arr[period:] - arr[:-period]) / (np.abs(arr[:-period]) + 1e-12)
    return result


def _last(arr: np.ndarray | None) -> float:
    """Last finite value."""
    if arr is None or len(arr) == 0:
        return 0.0
    val = float(arr[-1])
    return val if np.isfinite(val) else 0.0


def _rolling_mean_rank(arr: np.ndarray | None, rank_w: int = 20, mean_w: int = 60) -> float:
    """Mean of rolling rank — smoothed rank indicator."""
    if arr is None or len(arr) < rank_w + mean_w:
        return 0.5
    ranks = np.array([_rank(arr[:i + 1], rank_w) for i in range(len(arr))])
    return float(np.nanmean(ranks[-mean_w:]))


def _compute_signal(signal_def: dict, data: dict[str, np.ndarray]) -> float | None:
    """Safely compute a classical signal value."""
    try:
        value = signal_def["compute"](data)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        if isinstance(value, np.floating) and np.isfinite(value):
            return float(value)
    except Exception:
        pass
    return None


def produce_classical_predictions(
    config: Config,
    *,
    today: str | None = None,
    assets: list[str] | None = None,
) -> int:
    """Compute all classical indicators and write to prediction store.

    Returns the number of predictions written.
    """
    t0 = time.perf_counter()
    today = today or date.today().isoformat()

    # Load data
    client = build_signal_client_from_config(config.api)
    init_universe(client)

    eval_assets = assets or load_cached_eval_universe()
    if not eval_assets:
        logger.warning("No eval universe cached")
        return 0

    # Load features needed by classical signals
    needed_features = {"sp500", "tsy_yield_2y", "dxy", "fear_greed", "gold", "vix_close"}
    all_needed = sorted(set(eval_assets) | needed_features)

    db_path = DATA_DIR / "alpha_cache.db"
    store = DataStore(db_path, client)
    matrix = store.get_matrix(all_needed)
    store.close()

    # Don't fillna on price columns
    eval_set = set(eval_assets)
    for col in matrix.columns:
        if col not in eval_set:
            matrix[col] = matrix[col].fillna(0)
    data = {col: matrix[col].values for col in matrix.columns}

    # Compute and write predictions
    pred_store = PredictionStore()
    predictions: list[Prediction] = []

    for signal_id, signal_def in CLASSICAL_SIGNALS.items():
        value = _compute_signal(signal_def, data)
        if value is None:
            continue

        pred_store.register_signal(SignalMeta(
            signal_id=signal_id,
            source="classical",
            definition=signal_def["description"],
            horizon=signal_def["horizon"],
        ))

        for asset in eval_assets:
            predictions.append(Prediction(
                signal_id=signal_id,
                date=today,
                asset=asset,
                value=value,
                horizon=signal_def["horizon"],
            ))

    n_written = pred_store.write(predictions) if predictions else 0
    pred_store.close()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Classical producer: %d signals -> %d predictions, %.1fs",
        len(CLASSICAL_SIGNALS), n_written, elapsed,
    )
    return n_written
