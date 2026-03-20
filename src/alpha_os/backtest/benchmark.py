"""Market benchmark construction for residual return evaluation."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def equal_weight_benchmark(
    price_arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute equal-weight benchmark returns from multiple price series.

    Each asset contributes 1/N weight. NaN prices are excluded from the
    average for that day (asset may have shorter history).

    Parameters
    ----------
    price_arrays : {asset_name: price_array} for each asset in the universe.

    Returns
    -------
    benchmark_returns : (T-1,) daily returns of the equal-weight portfolio.
    """
    if not price_arrays:
        return np.array([])

    all_returns: list[np.ndarray] = []
    max_len = 0
    for name, prices in price_arrays.items():
        if len(prices) < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(prices) / prices[:-1]
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        all_returns.append(rets)
        max_len = max(max_len, len(rets))

    if not all_returns:
        return np.array([])

    # Align to same length (right-align, pad left with nan)
    aligned = np.full((len(all_returns), max_len), np.nan)
    for i, rets in enumerate(all_returns):
        aligned[i, max_len - len(rets):] = rets

    # Equal-weight average, ignoring nan
    benchmark = np.nanmean(aligned, axis=0)
    benchmark = np.nan_to_num(benchmark, nan=0.0)
    return benchmark


def build_benchmark_returns(
    data: dict[str, np.ndarray],
    asset_signals: list[str],
) -> np.ndarray:
    """Build benchmark returns from a data dict and list of price signal names.

    Parameters
    ----------
    data : {signal_name: values_array} as loaded by DataStore.
    asset_signals : list of price signal names to include in benchmark,
        e.g. ["btc_ohlcv", "eth_btc", "sol_usdt", "spy", "qqq"].

    Returns
    -------
    benchmark_returns : (T-1,) daily returns.
    """
    price_arrays: dict[str, np.ndarray] = {}
    for sig in asset_signals:
        arr = data.get(sig)
        if arr is not None and len(arr) >= 2:
            # Use only finite values
            clean = np.where(np.isfinite(arr), arr, np.nan)
            price_arrays[sig] = clean
        else:
            logger.debug("Benchmark: skipping %s (missing or too short)", sig)

    return equal_weight_benchmark(price_arrays)
