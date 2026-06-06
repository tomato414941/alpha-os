from __future__ import annotations

import sys
from pathlib import Path


def test_crypto_momentum_baseline_runs_on_checked_in_data():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto_momentum_baseline.backtest import run_backtest
    from strategies.crypto_momentum_baseline.data import load_daily_closes
    from strategies.crypto_momentum_baseline.strategy import SevenDayMomentumStrategy

    result = run_backtest(SevenDayMomentumStrategy(), load_daily_closes())

    assert len(result.steps) == 723
    assert result.summary.total_return > 0.0
    assert result.summary.max_drawdown < 0.0
