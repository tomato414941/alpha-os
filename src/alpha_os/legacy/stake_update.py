"""Pure helpers for legacy stake updates."""
from __future__ import annotations

import sqlite3

import numpy as np


STAKE_LOOKBACK_DAYS = 60
MIN_STAKE_OBSERVATIONS = 10


def compute_daily_marginal_contributions(
    observation_db_path: str,
    hypothesis_ids: list[str],
    stakes: dict[str, float],
    date: str,
) -> dict[str, float]:
    """Compute leave-one-out marginal contribution for one observation date."""
    conn = sqlite3.connect(observation_db_path)
    rows = conn.execute(
        "SELECT hypothesis_id, signal_value, daily_return "
        "FROM hypothesis_observations WHERE date = ?",
        (date,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    signals: dict[str, float] = {}
    for hypothesis_id, sig_val, daily_ret in rows:
        if (
            hypothesis_id in hypothesis_ids
            and hypothesis_id in stakes
            and stakes[hypothesis_id] > 0
            and np.isfinite(sig_val)
        ):
            signals[hypothesis_id] = sig_val

    if len(signals) < 2:
        return {}

    price_return = None
    for _hypothesis_id, sig_val, daily_ret in rows:
        if abs(sig_val) > 1e-8 and np.isfinite(daily_ret):
            price_return = daily_ret / sig_val
            break
    if price_return is None:
        return {}

    ids = list(signals.keys())
    stake_arr = np.array([stakes.get(hypothesis_id, 0.0) for hypothesis_id in ids])
    sig_arr = np.array([signals[hypothesis_id] for hypothesis_id in ids])
    total_stake = stake_arr.sum()
    if total_stake <= 0:
        return {}

    portfolio = float(np.dot(stake_arr, sig_arr) / total_stake)
    full_pnl = portfolio * price_return

    marginals: dict[str, float] = {}
    for i, hypothesis_id in enumerate(ids):
        remaining_stake = total_stake - stake_arr[i]
        if remaining_stake <= 0:
            marginals[hypothesis_id] = full_pnl
            continue
        portfolio_without = float(
            (np.dot(stake_arr, sig_arr) - stake_arr[i] * sig_arr[i]) / remaining_stake
        )
        pnl_without = portfolio_without * price_return
        marginals[hypothesis_id] = full_pnl - pnl_without

    return marginals


def compute_rolling_marginal_stake(
    marginal_history: list[float],
    *,
    lookback: int = STAKE_LOOKBACK_DAYS,
    prior_stake: float = 0.0,
) -> float:
    """Compute stake from the rolling mean of marginal contributions."""
    if len(marginal_history) < MIN_STAKE_OBSERVATIONS:
        return prior_stake
    recent = marginal_history[-lookback:]
    mean_marginal = float(np.mean(recent))
    return max(mean_marginal, 0.0)
