from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.allocation import EqualWeightAllocator
from strategies.crypto.backtest import run_backtest
from strategies.crypto.data import (
    DATASET_DIR,
    DEFAULT_SYMBOLS,
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto.metrics import BacktestSummary
from strategies.crypto.momentum import TrendFilteredMomentumStrategy


@dataclass(frozen=True)
class RobustnessRow:
    sample: str
    momentum_lookback_days: int
    trend_lookback_days: int
    transaction_cost_rate: float
    steps: int
    summary: BacktestSummary


def run_robustness_check(
    *,
    dataset_dir: Path = DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    market_bars: tuple[DailyMarketBar, ...] | None = None,
) -> tuple[RobustnessRow, ...]:
    bars = market_bars or load_daily_market_bars(
        dataset_dir=dataset_dir,
        symbols=symbols,
    )
    rows: list[RobustnessRow] = []
    for sample_name, sample_bars in _samples(bars).items():
        for momentum_lookback_days, trend_lookback_days in (
            (3, 20),
            (7, 20),
            (7, 30),
            (14, 30),
            (14, 60),
        ):
            for transaction_cost_rate in (0.0, 0.001, 0.002):
                strategy = TrendFilteredMomentumStrategy(
                    momentum_lookback_days=momentum_lookback_days,
                    trend_lookback_days=trend_lookback_days,
                    allocator=EqualWeightAllocator(),
                )
                result = run_backtest(
                    strategy,
                    sample_bars,
                    lookback_days=trend_lookback_days,
                    transaction_cost_rate=transaction_cost_rate,
                )
                rows.append(
                    RobustnessRow(
                        sample=sample_name,
                        momentum_lookback_days=momentum_lookback_days,
                        trend_lookback_days=trend_lookback_days,
                        transaction_cost_rate=transaction_cost_rate,
                        steps=len(result.steps),
                        summary=result.summary,
                    )
                )
    return tuple(rows)


def _samples(
    market_bars: tuple[DailyMarketBar, ...],
) -> dict[str, tuple[DailyMarketBar, ...]]:
    return {
        "all": market_bars,
        "2024": tuple(
            bar for bar in market_bars if bar.timestamp.startswith("2024-")
        ),
        "2025": tuple(
            bar for bar in market_bars if bar.timestamp.startswith("2025-")
        ),
        "2026": tuple(
            bar for bar in market_bars if bar.timestamp.startswith("2026-")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    args = parser.parse_args()

    print(
        "sample,momentum_lookback_days,trend_lookback_days,"
        "transaction_cost_rate,steps,total_return,sharpe,max_drawdown,"
        "mean_daily_turnover"
    )
    for row in run_robustness_check(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    ):
        print(
            f"{row.sample},"
            f"{row.momentum_lookback_days},"
            f"{row.trend_lookback_days},"
            f"{row.transaction_cost_rate:.4f},"
            f"{row.steps},"
            f"{row.summary.total_return:.6f},"
            f"{row.summary.sharpe:.6f},"
            f"{row.summary.max_drawdown:.6f},"
            f"{row.summary.mean_daily_turnover:.6f}"
        )


if __name__ == "__main__":
    main()
