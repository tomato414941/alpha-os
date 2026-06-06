from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.contribution import run_symbol_contribution
from strategies.crypto.data import (
    DATASET_DIR,
    DEFAULT_SYMBOLS,
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto.variants import CURRENT_VARIANT


@dataclass(frozen=True)
class SymbolEligibilityProfile:
    symbol: str
    first_timestamp: str
    last_timestamp: str
    history_days: int
    total_return: float
    realized_volatility: float
    max_drawdown: float
    total_gross_contribution: float | None = None


def summarize_symbol_eligibility(
    market_bars: tuple[DailyMarketBar, ...],
    *,
    variant: str | None = None,
) -> tuple[SymbolEligibilityProfile, ...]:
    contribution_by_symbol = (
        {
            row.symbol: row.total_gross_contribution
            for row in run_symbol_contribution(
                variant=variant,
                market_bars=market_bars,
            )
        }
        if variant is not None
        else {}
    )
    symbols = sorted({symbol for bar in market_bars for symbol in bar.closes})
    return tuple(
        sorted(
            (
                _symbol_profile(
                    market_bars,
                    symbol=symbol,
                    total_gross_contribution=contribution_by_symbol.get(symbol),
                )
                for symbol in symbols
            ),
            key=lambda profile: (
                profile.total_gross_contribution
                if profile.total_gross_contribution is not None
                else profile.total_return
            ),
        )
    )


def _symbol_profile(
    market_bars: tuple[DailyMarketBar, ...],
    *,
    symbol: str,
    total_gross_contribution: float | None,
) -> SymbolEligibilityProfile:
    rows = tuple(
        (bar.timestamp, bar.closes[symbol])
        for bar in market_bars
        if symbol in bar.closes
    )
    returns = tuple(
        (current_close / previous_close) - 1.0
        for (_, previous_close), (_, current_close) in zip(rows[:-1], rows[1:])
        if previous_close > 0.0
    )
    mean_return = sum(returns) / len(returns) if returns else 0.0
    realized_volatility = (
        math.sqrt(
            sum((daily_return - mean_return) ** 2 for daily_return in returns)
            / len(returns)
        )
        if returns
        else 0.0
    )
    peak = 0.0
    max_drawdown = 0.0
    for _, close in rows:
        peak = max(peak, close)
        if peak > 0.0:
            max_drawdown = min(max_drawdown, (close / peak) - 1.0)
    first_close = rows[0][1] if rows else 0.0
    last_close = rows[-1][1] if rows else 0.0
    total_return = (last_close / first_close) - 1.0 if first_close > 0.0 else 0.0
    return SymbolEligibilityProfile(
        symbol=symbol,
        first_timestamp=rows[0][0] if rows else "",
        last_timestamp=rows[-1][0] if rows else "",
        history_days=len(rows),
        total_return=total_return,
        realized_volatility=realized_volatility,
        max_drawdown=max_drawdown,
        total_gross_contribution=total_gross_contribution,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--variant", default=CURRENT_VARIANT)
    parser.add_argument(
        "--no-contribution",
        action="store_true",
        help="Show only close-derived diagnostics.",
    )
    args = parser.parse_args()

    market_bars = load_daily_market_bars(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    print(
        "symbol,first_timestamp,last_timestamp,history_days,total_return,"
        "realized_volatility,max_drawdown,total_gross_contribution"
    )
    for row in summarize_symbol_eligibility(
        market_bars,
        variant=None if args.no_contribution else args.variant,
    ):
        contribution = (
            ""
            if row.total_gross_contribution is None
            else f"{row.total_gross_contribution:.6f}"
        )
        print(
            f"{row.symbol},"
            f"{row.first_timestamp},"
            f"{row.last_timestamp},"
            f"{row.history_days},"
            f"{row.total_return:.6f},"
            f"{row.realized_volatility:.6f},"
            f"{row.max_drawdown:.6f},"
            f"{contribution}"
        )


if __name__ == "__main__":
    main()
