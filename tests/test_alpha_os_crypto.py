from __future__ import annotations

import sys
from pathlib import Path


def test_crypto_runs_on_checked_in_data():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto.backtest import run_backtest
    from strategies.crypto.contribution import run_symbol_contribution
    from strategies.crypto.data import load_daily_market_bars
    from strategies.crypto.robustness import run_robustness_check
    from strategies.crypto.momentum import (
        SevenDayMomentumStrategy,
        SevenDayMomentumWithThirtyDayTrendStrategy,
        SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy,
    )

    market_bars = load_daily_market_bars()
    result = run_backtest(SevenDayMomentumStrategy(), market_bars)
    candidate = run_backtest(
        SevenDayMomentumWithThirtyDayTrendStrategy(),
        market_bars,
        lookback_days=30,
    )
    skfolio_candidate = run_backtest(
        SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy(),
        market_bars,
        lookback_days=30,
    )

    assert len(result.steps) == 723
    assert result.summary.total_return > 0.0
    assert result.summary.max_drawdown < 0.0
    assert len(candidate.steps) == 700
    assert candidate.summary.max_drawdown < 0.0
    assert len(skfolio_candidate.steps) == 700
    assert skfolio_candidate.summary.max_drawdown < 0.0

    robustness_rows = run_robustness_check(market_bars=market_bars)
    assert robustness_rows

    contribution_rows = run_symbol_contribution(market_bars=market_bars)
    assert {row.symbol for row in contribution_rows} == {"BTCUSDT", "ETHUSDT"}
