from __future__ import annotations

import argparse
from pathlib import Path

from strategies.cash_rotation.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR
from strategies.cash_rotation.strategy import RiskOnOffRotationStrategy
from strategies.daily_close.backtest import run_backtest
from strategies.daily_close.data import load_daily_market_bars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--lookback-days", type=int, default=126)
    args = parser.parse_args()

    market_bars = load_daily_market_bars(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    result = run_backtest(
        RiskOnOffRotationStrategy(lookback_days=args.lookback_days),
        market_bars,
        lookback_days=args.lookback_days,
    )
    summary = result.summary
    print(f"steps={len(result.steps)}")
    print(f"total_return={summary.total_return:.6f}")
    print(f"annualized_return={summary.annualized_return:.6f}")
    print(f"annualized_volatility={summary.annualized_volatility:.6f}")
    print(f"sharpe={summary.sharpe:.6f}")
    print(f"max_drawdown={summary.max_drawdown:.6f}")
    print(f"mean_daily_turnover={summary.mean_daily_turnover:.6f}")


if __name__ == "__main__":
    main()
