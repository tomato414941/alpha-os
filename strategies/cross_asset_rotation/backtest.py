from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alpha_os.trading_strategy import TradingStrategy

from strategies.cross_asset_rotation.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR
from strategies.cross_asset_rotation.strategy import (
    CrossAssetRiskOnOffStrategy,
    CrossAssetTopMomentumStrategy,
    CrossAssetVolAdjustedMomentumStrategy,
)
from strategies.daily_close.backtest import (
    DailyCloseDecisionInput,
    TargetWeights,
    run_backtest,
)
from strategies.daily_close.data import load_daily_market_bars


@dataclass(frozen=True)
class StrategyVariant:
    factory: Callable[[], TradingStrategy[DailyCloseDecisionInput, TargetWeights]]
    lookback_days: int


VARIANTS = {
    "top_momentum_126_252": StrategyVariant(
        factory=lambda: CrossAssetTopMomentumStrategy(
            momentum_lookback_days=126,
            trend_lookback_days=252,
        ),
        lookback_days=252,
    ),
    "vol_adjusted_momentum_126_252": StrategyVariant(
        factory=lambda: CrossAssetVolAdjustedMomentumStrategy(
            momentum_lookback_days=126,
            trend_lookback_days=252,
            volatility_lookback_days=63,
        ),
        lookback_days=252,
    ),
    "risk_on_off_126": StrategyVariant(
        factory=lambda: CrossAssetRiskOnOffStrategy(lookback_days=126),
        lookback_days=126,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    args = parser.parse_args()

    market_bars = load_daily_market_bars(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    for name, variant in VARIANTS.items():
        result = run_backtest(
            variant.factory(),
            market_bars,
            lookback_days=variant.lookback_days,
        )
        summary = result.summary
        print(f"variant={name}")
        print(f"steps={len(result.steps)}")
        print(f"total_return={summary.total_return:.6f}")
        print(f"annualized_return={summary.annualized_return:.6f}")
        print(f"annualized_volatility={summary.annualized_volatility:.6f}")
        print(f"sharpe={summary.sharpe:.6f}")
        print(f"max_drawdown={summary.max_drawdown:.6f}")
        print(f"mean_daily_turnover={summary.mean_daily_turnover:.6f}")
        print("")


if __name__ == "__main__":
    main()
