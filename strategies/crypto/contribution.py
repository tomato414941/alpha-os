from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.backtest import BacktestStep, run_backtest
from strategies.crypto.data import (
    DATASET_DIR,
    DEFAULT_SYMBOLS,
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto.variants import CURRENT_VARIANT, VARIANTS


@dataclass(frozen=True)
class SymbolContribution:
    symbol: str
    total_gross_contribution: float
    mean_weight: float
    active_days: int
    max_weight: float


def summarize_symbol_contributions(
    steps: tuple[BacktestStep, ...],
) -> tuple[SymbolContribution, ...]:
    symbols = sorted(
        {
            symbol
            for step in steps
            for symbol in (
                set(step.target.target_weights)
                | set(step.gross_contribution_by_symbol)
            )
        }
    )
    rows = []
    for symbol in symbols:
        weights = tuple(
            step.target.target_weights.get(symbol, 0.0)
            for step in steps
        )
        rows.append(
            SymbolContribution(
                symbol=symbol,
                total_gross_contribution=sum(
                    step.gross_contribution_by_symbol.get(symbol, 0.0)
                    for step in steps
                ),
                mean_weight=sum(weights) / len(weights) if weights else 0.0,
                active_days=sum(1 for weight in weights if weight > 0.0),
                max_weight=max(weights) if weights else 0.0,
            )
        )
    return tuple(
        sorted(rows, key=lambda row: row.total_gross_contribution)
    )


def run_symbol_contribution(
    *,
    dataset_dir: Path = DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    variant: str = CURRENT_VARIANT,
    market_bars: tuple[DailyMarketBar, ...] | None = None,
) -> tuple[SymbolContribution, ...]:
    bars = (
        market_bars
        if market_bars is not None
        else load_daily_market_bars(dataset_dir=dataset_dir, symbols=symbols)
    )
    strategy_variant = VARIANTS[variant]
    result = run_backtest(
        strategy_variant.factory(),
        bars,
        lookback_days=strategy_variant.lookback_days,
    )
    return summarize_symbol_contributions(result.steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--variant", default=CURRENT_VARIANT)
    args = parser.parse_args()

    print("symbol,total_gross_contribution,mean_weight,active_days,max_weight")
    for row in run_symbol_contribution(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
        variant=args.variant,
    ):
        print(
            f"{row.symbol},"
            f"{row.total_gross_contribution:.6f},"
            f"{row.mean_weight:.6f},"
            f"{row.active_days},"
            f"{row.max_weight:.6f}"
        )


if __name__ == "__main__":
    main()
