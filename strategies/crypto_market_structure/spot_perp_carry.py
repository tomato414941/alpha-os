from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.data import LOCAL_DATASET_DIR as SPOT_DATASET_DIR
from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR as PERP_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.crypto_market_structure.funding_carry import (
    FundingCarryDecisionInput,
    FundingCarryResult,
    PositiveFundingCarryStrategy,
    FundingCarryTargetWeights,
)
from strategies.daily_close.metrics import summarize_backtest


@dataclass(frozen=True)
class SpotDay:
    timestamp: str
    close: float


def load_spot_days(
    *,
    dataset_dir: Path = SPOT_DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> dict[str, tuple[SpotDay, ...]]:
    rows_by_symbol: dict[str, tuple[SpotDay, ...]] = {}
    for symbol in symbols:
        path = dataset_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows_by_symbol[symbol] = tuple(
                SpotDay(timestamp=str(row["timestamp"]), close=float(row["close"]))
                for row in reader
            )
    return rows_by_symbol


def run_spot_perp_carry_screen(
    *,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float = 0.0002,
    top_n_values: tuple[int, ...] = (1, 2, 3),
    rebalance_days_values: tuple[int, ...] = (1, 3, 7, 14),
    paired_leg_cost_rate: float = 0.0004,
    capital_per_notional: float = 2.0,
) -> tuple[FundingCarryResult, ...]:
    spot_by_symbol_and_timestamp = {
        symbol: {row.timestamp: row for row in rows}
        for symbol, rows in spot_rows_by_symbol.items()
    }
    perp_by_symbol_and_timestamp = {
        symbol: {row.timestamp: row for row in rows}
        for symbol, rows in perp_rows_by_symbol.items()
    }
    symbols = tuple(sorted(set(spot_by_symbol_and_timestamp) & set(perp_by_symbol_and_timestamp)))
    timestamps = sorted(
        set.intersection(
            *(
                set(spot_by_symbol_and_timestamp[symbol])
                & set(perp_by_symbol_and_timestamp[symbol])
                for symbol in symbols
            )
        )
    )
    results = []
    for top_n in top_n_values:
        for rebalance_days in rebalance_days_values:
            results.append(
                _run_candidate(
                    spot_by_symbol_and_timestamp,
                    perp_by_symbol_and_timestamp,
                    symbols=symbols,
                    timestamps=timestamps,
                    candidate=f"spot_perp_positive_funding_top_{top_n}_{rebalance_days}d",
                    min_funding_rate_sum=min_funding_rate_sum,
                    top_n=top_n,
                    rebalance_days=rebalance_days,
                    paired_leg_cost_rate=paired_leg_cost_rate,
                    capital_per_notional=capital_per_notional,
                )
            )
    return tuple(sorted(results, key=lambda result: result.sharpe, reverse=True))


def write_spot_perp_carry_results(
    results: tuple[FundingCarryResult, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "steps",
                "total_return",
                "sharpe",
                "max_drawdown",
                "mean_daily_turnover",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.candidate,
                    result.steps,
                    f"{result.total_return:.10f}",
                    f"{result.sharpe:.10f}",
                    f"{result.max_drawdown:.10f}",
                    f"{result.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def _run_candidate(
    spot_by_symbol_and_timestamp: dict[str, dict[str, SpotDay]],
    perp_by_symbol_and_timestamp: dict[str, dict[str, MarketStructureDay]],
    *,
    symbols: tuple[str, ...],
    timestamps: list[str],
    candidate: str,
    min_funding_rate_sum: float,
    top_n: int,
    rebalance_days: int,
    paired_leg_cost_rate: float,
    capital_per_notional: float,
) -> FundingCarryResult:
    rewards: list[float] = []
    equities: list[float] = []
    transaction_costs: list[float] = []
    current_weights: dict[str, float] = {}
    target = FundingCarryTargetWeights(target_weights={})
    equity = 1.0
    strategy = PositiveFundingCarryStrategy(
        min_funding_rate_sum=min_funding_rate_sum,
        top_n=top_n,
    )
    for index, timestamp in enumerate(timestamps[:-1]):
        if index % rebalance_days == 0:
            target = strategy.decide(
                FundingCarryDecisionInput(
                    rows_by_symbol={
                        symbol: perp_by_symbol_and_timestamp[symbol][timestamp]
                        for symbol in symbols
                    },
                    current_weights=dict(current_weights),
                )
            )
        next_timestamp = timestamps[index + 1]
        gross_reward = sum(
            weight
            * _spot_perp_pair_return(
                current_spot=spot_by_symbol_and_timestamp[symbol][timestamp],
                next_spot=spot_by_symbol_and_timestamp[symbol][next_timestamp],
                current_perp=perp_by_symbol_and_timestamp[symbol][timestamp],
                next_perp=perp_by_symbol_and_timestamp[symbol][next_timestamp],
                capital_per_notional=capital_per_notional,
            )
            for symbol, weight in target.target_weights.items()
        )
        turnover = _turnover(current_weights, target.target_weights)
        transaction_cost = turnover * paired_leg_cost_rate * 2.0 / capital_per_notional
        reward = gross_reward - transaction_cost
        equity *= 1.0 + reward
        rewards.append(reward)
        equities.append(equity)
        transaction_costs.append(transaction_cost)
        current_weights = dict(target.target_weights)
    summary = summarize_backtest(
        rewards=tuple(rewards),
        equities=tuple(equities),
        transaction_costs=tuple(transaction_costs),
        transaction_cost_rate=paired_leg_cost_rate,
    )
    return FundingCarryResult(
        candidate=candidate,
        steps=len(rewards),
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        mean_daily_turnover=summary.mean_daily_turnover,
    )


def _spot_perp_pair_return(
    *,
    current_spot: SpotDay,
    next_spot: SpotDay,
    current_perp: MarketStructureDay,
    next_perp: MarketStructureDay,
    capital_per_notional: float,
) -> float:
    spot_return = (next_spot.close / current_spot.close) - 1.0
    perp_return = (next_perp.close / current_perp.close) - 1.0
    pair_notional_return = spot_return - perp_return + next_perp.funding_rate_sum
    return pair_notional_return / capital_per_notional


def _turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in current_weights.keys() | target_weights.keys()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dataset-dir", type=Path, default=SPOT_DATASET_DIR)
    parser.add_argument("--perp-dataset-dir", type=Path, default=PERP_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--paired-leg-cost-rate", type=float, default=0.0004)
    parser.add_argument("--capital-per-notional", type=float, default=2.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "spot_perp_carry.csv",
    )
    args = parser.parse_args()

    spot_rows_by_symbol = load_spot_days(
        dataset_dir=args.spot_dataset_dir,
        symbols=tuple(args.symbols),
    )
    perp_rows_by_symbol = load_market_structure_days(
        dataset_dir=args.perp_dataset_dir,
        symbols=tuple(args.symbols),
    )
    results = run_spot_perp_carry_screen(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=args.min_funding_rate_sum,
        paired_leg_cost_rate=args.paired_leg_cost_rate,
        capital_per_notional=args.capital_per_notional,
    )
    write_spot_perp_carry_results(results, output_path=args.output_path)
    for result in results:
        print(
            result.candidate,
            result.steps,
            f"{result.total_return:.6f}",
            f"{result.sharpe:.6f}",
            f"{result.max_drawdown:.6f}",
            f"{result.mean_daily_turnover:.6f}",
        )


if __name__ == "__main__":
    main()
