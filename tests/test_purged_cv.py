from __future__ import annotations

import numpy as np

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.validation.purged_cv import purged_walk_forward


def test_purged_walk_forward_includes_distributional_metrics():
    rng = np.random.default_rng(42)
    n = 400
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    signal = rng.normal(0.0, 1.0, n)

    engine = BacktestEngine(CostModel(commission_pct=0.0, slippage_pct=0.0))
    result = purged_walk_forward(
        signal,
        prices,
        engine,
        n_folds=5,
        embargo=5,
        min_train=120,
    )

    assert result.n_folds > 0
    assert result.oos_cvar_95 <= 0.0
    assert 0.0 <= result.oos_tail_hit_rate <= 1.0
