from __future__ import annotations

import argparse
from pathlib import Path

from strategies.crypto.data import EXPANDED_SYMBOLS, LOCAL_DATASET_DIR
from strategies.crypto_pair_spread.strategy import PairSpread, ZScorePairSpreadStrategy
from strategies.daily_close.backtest import run_backtest
from strategies.daily_close.data import load_daily_market_bars


DEFAULT_PAIRS = (
    PairSpread("BTCUSDT", "ETHUSDT"),
    PairSpread("SOLUSDT", "ETHUSDT"),
    PairSpread("DOGEUSDT", "BTCUSDT"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(EXPANDED_SYMBOLS))
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--entry-zscore", type=float, default=1.5)
    args = parser.parse_args()

    market_bars = load_daily_market_bars(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    result = run_backtest(
        ZScorePairSpreadStrategy(
            pairs=DEFAULT_PAIRS,
            lookback_days=args.lookback_days,
            entry_zscore=args.entry_zscore,
        ),
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
