from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from strategies.cash_rotation.data import DEFAULT_SYMBOLS as CASH_SYMBOLS
from strategies.cash_rotation.data import LOCAL_DATASET_DIR as CASH_DATASET_DIR
from strategies.cash_rotation.strategy import RiskOnOffRotationStrategy
from strategies.cross_asset_rotation.backtest import VARIANTS as CROSS_ASSET_VARIANTS
from strategies.cross_asset_rotation.data import DEFAULT_SYMBOLS as CROSS_ASSET_SYMBOLS
from strategies.cross_asset_rotation.data import LOCAL_DATASET_DIR as CROSS_ASSET_DATASET_DIR
from strategies.crypto.backtest import run_backtest as run_crypto_backtest
from strategies.crypto.data import EXPANDED_SYMBOLS as CRYPTO_SYMBOLS
from strategies.crypto.data import LOCAL_DATASET_DIR as CRYPTO_DATASET_DIR
from strategies.crypto.data import load_daily_market_bars as load_crypto_bars
from strategies.crypto.variants import VARIANTS as CRYPTO_VARIANTS
from strategies.crypto_pair_spread.backtest import DEFAULT_PAIRS
from strategies.crypto_pair_spread.strategy import ZScorePairSpreadStrategy
from strategies.daily_close.backtest import run_backtest as run_daily_close_backtest
from strategies.daily_close.data import DailyMarketBar, load_daily_market_bars
from strategies.daily_close.metrics import BacktestSummary, summarize_backtest
from strategies.equity_index.data import DEFAULT_SYMBOLS as EQUITY_SYMBOLS
from strategies.equity_index.data import LOCAL_DATASET_DIR as EQUITY_DATASET_DIR
from strategies.equity_index.strategy import PositiveTrendTopMomentumStrategy


SMALLER_CRYPTO_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
)


@dataclass(frozen=True)
class LeaderboardRow:
    group: str
    candidate: str
    selection_basis: str
    symbols: str
    steps: int
    total_return: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float
    best_benchmark_total_symbol: str
    best_benchmark_total_return: float
    excess_total_return: float
    best_benchmark_sharpe_symbol: str
    best_benchmark_sharpe: float
    excess_sharpe: float


@dataclass(frozen=True)
class Benchmark:
    symbol: str
    summary: BacktestSummary


def leaderboard_rows() -> tuple[LeaderboardRow, ...]:
    rows: list[LeaderboardRow] = []
    rows.extend(_crypto_rows())
    rows.extend(_crypto_pair_spread_rows())
    rows.extend(_equity_index_rows())
    rows.extend(_cash_rotation_rows())
    rows.extend(_cross_asset_rows())
    return tuple(sorted(rows, key=lambda row: row.sharpe, reverse=True))


def _crypto_rows() -> tuple[LeaderboardRow, ...]:
    rows: list[LeaderboardRow] = []
    expanded_bars = load_crypto_bars(
        dataset_dir=CRYPTO_DATASET_DIR,
        symbols=CRYPTO_SYMBOLS,
    )
    smaller_bars = load_crypto_bars(
        dataset_dir=CRYPTO_DATASET_DIR,
        symbols=SMALLER_CRYPTO_SYMBOLS,
    )
    for variant_name, selection_basis, symbols, bars in (
        (
            "7d_momentum_30d_trend",
            "fixed_expanded_universe",
            CRYPTO_SYMBOLS,
            expanded_bars,
        ),
        (
            "7d_momentum_30d_trend_skfolio_max_ratio",
            "fixed_expanded_universe",
            CRYPTO_SYMBOLS,
            expanded_bars,
        ),
        (
            "7d_momentum_30d_trend_skfolio_max_ratio_eligible",
            "rolling_asset_quality",
            CRYPTO_SYMBOLS,
            expanded_bars,
        ),
        (
            "7d_momentum_30d_trend_skfolio_min_variance",
            "fixed_expanded_universe",
            CRYPTO_SYMBOLS,
            expanded_bars,
        ),
        (
            "7d_momentum_30d_trend_skfolio_max_ratio",
            "manual_same_period_exclusion",
            SMALLER_CRYPTO_SYMBOLS,
            smaller_bars,
        ),
    ):
        variant = CRYPTO_VARIANTS[variant_name]
        result = run_crypto_backtest(
            variant.factory(),
            bars,
            lookback_days=variant.lookback_days,
        )
        rows.append(
            _row(
                group="crypto",
                candidate=variant_name,
                selection_basis=selection_basis,
                symbols=symbols,
                steps=len(result.steps),
                summary=result.summary,
                benchmarks=_benchmarks(
                    bars,
                    symbols=symbols,
                    lookback_days=variant.lookback_days,
                ),
            )
        )
    return tuple(rows)


def _crypto_pair_spread_rows() -> tuple[LeaderboardRow, ...]:
    bars = load_daily_market_bars(
        dataset_dir=CRYPTO_DATASET_DIR,
        symbols=CRYPTO_SYMBOLS,
    )
    rows: list[LeaderboardRow] = []
    for entry_zscore in (1.0, 1.5, 2.0):
        lookback_days = 60
        result = run_daily_close_backtest(
            ZScorePairSpreadStrategy(
                pairs=DEFAULT_PAIRS,
                lookback_days=lookback_days,
                entry_zscore=entry_zscore,
            ),
            bars,
            lookback_days=lookback_days,
        )
        rows.append(
            _row(
                group="crypto_pair_spread",
                candidate=f"zscore_pair_spread_{entry_zscore:g}",
                selection_basis="fixed_pairs",
                symbols=CRYPTO_SYMBOLS,
                steps=len(result.steps),
                summary=result.summary,
                benchmarks=_benchmarks(
                    bars,
                    symbols=CRYPTO_SYMBOLS,
                    lookback_days=lookback_days,
                ),
            )
        )
    return tuple(rows)


def _equity_index_rows() -> tuple[LeaderboardRow, ...]:
    bars = load_daily_market_bars(
        dataset_dir=EQUITY_DATASET_DIR,
        symbols=EQUITY_SYMBOLS,
    )
    rows: list[LeaderboardRow] = []
    for momentum_lookback_days, trend_lookback_days in (
        (63, 126),
        (126, 252),
        (21, 63),
    ):
        lookback_days = max(momentum_lookback_days, trend_lookback_days)
        result = run_daily_close_backtest(
            PositiveTrendTopMomentumStrategy(
                momentum_lookback_days=momentum_lookback_days,
                trend_lookback_days=trend_lookback_days,
            ),
            bars,
            lookback_days=lookback_days,
        )
        rows.append(
            _row(
                group="equity_index",
                candidate=f"top_momentum_{momentum_lookback_days}_{trend_lookback_days}",
                selection_basis="fixed_etf_universe",
                symbols=EQUITY_SYMBOLS,
                steps=len(result.steps),
                summary=result.summary,
                benchmarks=_benchmarks(
                    bars,
                    symbols=EQUITY_SYMBOLS,
                    lookback_days=lookback_days,
                ),
            )
        )
    return tuple(rows)


def _cash_rotation_rows() -> tuple[LeaderboardRow, ...]:
    bars = load_daily_market_bars(
        dataset_dir=CASH_DATASET_DIR,
        symbols=CASH_SYMBOLS,
    )
    rows: list[LeaderboardRow] = []
    for lookback_days in (63, 126, 252):
        result = run_daily_close_backtest(
            RiskOnOffRotationStrategy(lookback_days=lookback_days),
            bars,
            lookback_days=lookback_days,
        )
        rows.append(
            _row(
                group="cash_rotation",
                candidate=f"risk_on_off_{lookback_days}",
                selection_basis="fixed_etf_universe",
                symbols=CASH_SYMBOLS,
                steps=len(result.steps),
                summary=result.summary,
                benchmarks=_benchmarks(
                    bars,
                    symbols=CASH_SYMBOLS,
                    lookback_days=lookback_days,
                ),
            )
        )
    return tuple(rows)


def _cross_asset_rows() -> tuple[LeaderboardRow, ...]:
    bars = load_daily_market_bars(
        dataset_dir=CROSS_ASSET_DATASET_DIR,
        symbols=CROSS_ASSET_SYMBOLS,
    )
    rows: list[LeaderboardRow] = []
    for name, variant in CROSS_ASSET_VARIANTS.items():
        result = run_daily_close_backtest(
            variant.factory(),
            bars,
            lookback_days=variant.lookback_days,
        )
        rows.append(
            _row(
                group="cross_asset_rotation",
                candidate=name,
                selection_basis="fixed_mixed_universe",
                symbols=CROSS_ASSET_SYMBOLS,
                steps=len(result.steps),
                summary=result.summary,
                benchmarks=_benchmarks(
                    bars,
                    symbols=CROSS_ASSET_SYMBOLS,
                    lookback_days=variant.lookback_days,
                ),
            )
        )
    return tuple(rows)


def _row(
    *,
    group: str,
    candidate: str,
    selection_basis: str,
    symbols: tuple[str, ...],
    steps: int,
    summary: BacktestSummary,
    benchmarks: tuple[Benchmark, ...],
) -> LeaderboardRow:
    best_total = max(benchmarks, key=lambda benchmark: benchmark.summary.total_return)
    best_sharpe = max(benchmarks, key=lambda benchmark: benchmark.summary.sharpe)
    return LeaderboardRow(
        group=group,
        candidate=candidate,
        selection_basis=selection_basis,
        symbols=" ".join(symbols),
        steps=steps,
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        mean_daily_turnover=summary.mean_daily_turnover,
        best_benchmark_total_symbol=best_total.symbol,
        best_benchmark_total_return=best_total.summary.total_return,
        excess_total_return=summary.total_return - best_total.summary.total_return,
        best_benchmark_sharpe_symbol=best_sharpe.symbol,
        best_benchmark_sharpe=best_sharpe.summary.sharpe,
        excess_sharpe=summary.sharpe - best_sharpe.summary.sharpe,
    )


def _benchmarks(
    market_bars: tuple[DailyMarketBar, ...],
    *,
    symbols: tuple[str, ...],
    lookback_days: int,
) -> tuple[Benchmark, ...]:
    return tuple(
        Benchmark(
            symbol=symbol,
            summary=_buy_and_hold_summary(
                market_bars,
                symbol=symbol,
                lookback_days=lookback_days,
            ),
        )
        for symbol in symbols
    )


def _buy_and_hold_summary(
    market_bars: tuple[DailyMarketBar, ...],
    *,
    symbol: str,
    lookback_days: int,
) -> BacktestSummary:
    rewards = []
    equities = []
    equity = 1.0
    for index in range(lookback_days, len(market_bars) - 1):
        current_bar = market_bars[index]
        next_bar = market_bars[index + 1]
        if symbol not in current_bar.closes or symbol not in next_bar.closes:
            continue
        current_close = current_bar.closes[symbol]
        if current_close <= 0.0:
            continue
        reward = (next_bar.closes[symbol] / current_close) - 1.0
        rewards.append(reward)
        equity *= 1.0 + reward
        equities.append(equity)
    return summarize_backtest(
        rewards=tuple(rewards),
        equities=tuple(equities),
        transaction_costs=tuple(0.0 for _ in rewards),
        transaction_cost_rate=0.001,
    )


def write_csv(rows: Iterable[LeaderboardRow]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=tuple(LeaderboardRow.__dataclass_fields__),
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row.__dict__)


def write_markdown(rows: Iterable[LeaderboardRow]) -> None:
    headers = (
        "group",
        "candidate",
        "selection",
        "total",
        "sharpe",
        "drawdown",
        "turnover",
        "best total bh",
        "excess total",
        "best sharpe bh",
        "excess sharpe",
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                (
                    row.group,
                    row.candidate,
                    row.selection_basis,
                    f"{row.total_return:.6f}",
                    f"{row.sharpe:.6f}",
                    f"{row.max_drawdown:.6f}",
                    f"{row.mean_daily_turnover:.6f}",
                    (
                        f"{row.best_benchmark_total_symbol} "
                        f"{row.best_benchmark_total_return:.6f}"
                    ),
                    f"{row.excess_total_return:.6f}",
                    (
                        f"{row.best_benchmark_sharpe_symbol} "
                        f"{row.best_benchmark_sharpe:.6f}"
                    ),
                    f"{row.excess_sharpe:.6f}",
                )
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args()

    rows = leaderboard_rows()
    writers: dict[str, Callable[[Iterable[LeaderboardRow]], None]] = {
        "markdown": write_markdown,
        "csv": write_csv,
    }
    writers[args.format](rows)


if __name__ == "__main__":
    main()
