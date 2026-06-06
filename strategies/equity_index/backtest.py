from __future__ import annotations

import argparse
from pathlib import Path

from strategies.daily_close.backtest import run_backtest
from strategies.daily_close.data import load_daily_market_bars
from strategies.equity_index.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR
from strategies.equity_index.strategy import PositiveTrendTopMomentumStrategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--momentum-lookback-days", type=int, default=63)
    parser.add_argument("--trend-lookback-days", type=int, default=126)
    args = parser.parse_args()

    market_bars = load_daily_market_bars(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    lookback_days = max(args.momentum_lookback_days, args.trend_lookback_days)
    result = run_backtest(
        PositiveTrendTopMomentumStrategy(
            momentum_lookback_days=args.momentum_lookback_days,
            trend_lookback_days=args.trend_lookback_days,
        ),
        market_bars,
        lookback_days=lookback_days,
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
