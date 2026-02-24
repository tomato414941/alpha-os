"""Probability of Backtest Overfitting (PBO) — Bailey et al. (2015).

Uses combinatorial purged cross-validation (CPCV) to estimate the
probability that a strategy selected in-sample underperforms out-of-sample.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..backtest.engine import BacktestEngine
from ..backtest import metrics


@dataclass
class PBOResult:
    pbo: float
    n_combinations: int
    logit_distribution: list[float]
    oos_sharpe_distribution: list[float]


def _split_into_blocks(n: int, n_blocks: int) -> list[tuple[int, int]]:
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n
        blocks.append((start, end))
    return blocks


def probability_of_backtest_overfitting(
    signals: np.ndarray,
    prices: np.ndarray,
    engine: BacktestEngine,
    n_blocks: int = 10,
    max_combinations: int = 100,
) -> PBOResult:
    """Estimate PBO via CPCV.

    Parameters
    ----------
    signals : (n_strategies, n_days) signal matrix
    prices : (n_days,) price array
    engine : BacktestEngine instance
    n_blocks : number of time blocks for CPCV
    max_combinations : cap on number of IS/OOS splits to evaluate

    Returns
    -------
    PBOResult with PBO estimate and supporting distributions.
    """
    n_strategies, n_days = signals.shape
    if n_strategies < 2 or n_days < n_blocks * 10:
        return PBOResult(1.0, 0, [], [])

    blocks = _split_into_blocks(n_days, n_blocks)
    half = n_blocks // 2

    # Generate IS/OOS splits: choose half blocks for IS, rest for OOS
    all_combos = list(itertools.combinations(range(n_blocks), half))
    rng = np.random.default_rng(42)
    if len(all_combos) > max_combinations:
        indices = rng.choice(len(all_combos), max_combinations, replace=False)
        combos = [all_combos[i] for i in indices]
    else:
        combos = all_combos

    logits: list[float] = []
    oos_sharpes: list[float] = []

    for is_blocks in combos:
        oos_blocks = [i for i in range(n_blocks) if i not in is_blocks]

        # Build IS and OOS index arrays
        is_idx = np.concatenate([np.arange(blocks[b][0], blocks[b][1]) for b in is_blocks])
        oos_idx = np.concatenate([np.arange(blocks[b][0], blocks[b][1]) for b in oos_blocks])

        # Evaluate all strategies IS
        is_sharpes = np.zeros(n_strategies)
        for s in range(n_strategies):
            is_sig = signals[s, is_idx]
            is_prices = prices[is_idx]
            if len(is_prices) < 10:
                continue
            result = engine.run(is_sig, is_prices)
            is_sharpes[s] = result.sharpe

        # Best strategy IS
        best_is = int(np.argmax(is_sharpes))

        # Evaluate best strategy OOS
        oos_sig = signals[best_is, oos_idx]
        oos_prices = prices[oos_idx]
        if len(oos_prices) < 10:
            continue
        oos_result = engine.run(oos_sig, oos_prices)
        oos_sharpe = oos_result.sharpe
        oos_sharpes.append(oos_sharpe)

        # Rank of IS-best in OOS
        oos_all_sharpes = np.zeros(n_strategies)
        for s in range(n_strategies):
            sig_s = signals[s, oos_idx]
            res_s = engine.run(sig_s, oos_prices)
            oos_all_sharpes[s] = res_s.sharpe

        rank_oos = np.sum(oos_all_sharpes <= oos_sharpe) / n_strategies
        # Logit: log(rank / (1 - rank)), clamped
        rank_clamped = np.clip(rank_oos, 0.01, 0.99)
        logit = np.log(rank_clamped / (1 - rank_clamped))
        logits.append(float(logit))

    if not logits:
        return PBOResult(1.0, 0, [], [])

    # PBO = fraction of logits <= 0 (IS-best underperforms median OOS)
    pbo = float(np.mean(np.array(logits) <= 0))

    return PBOResult(
        pbo=pbo,
        n_combinations=len(combos),
        logit_distribution=logits,
        oos_sharpe_distribution=oos_sharpes,
    )
