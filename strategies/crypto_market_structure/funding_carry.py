from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.daily_close.metrics import summarize_backtest


@dataclass(frozen=True)
class FundingCarryDecisionInput:
    rows_by_symbol: dict[str, MarketStructureDay]
    current_weights: dict[str, float]


@dataclass(frozen=True)
class FundingCarryTargetWeights:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class PositiveFundingCarryStrategy(
    TradingStrategy[FundingCarryDecisionInput, FundingCarryTargetWeights]
):
    min_funding_rate_sum: float
    top_n: int

    def decide(self, strategy_input: FundingCarryDecisionInput) -> FundingCarryTargetWeights:
        candidates = [
            (symbol, row.funding_rate_sum + max(row.premium_close, 0.0))
            for symbol, row in strategy_input.rows_by_symbol.items()
            if row.funding_rate_sum >= self.min_funding_rate_sum
        ]
        selected = tuple(
            symbol
            for symbol, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[
                : self.top_n
            ]
        )
        if not selected:
            return FundingCarryTargetWeights(target_weights={})
        weight = 1.0 / len(selected)
        return FundingCarryTargetWeights(
            target_weights={symbol: weight for symbol in selected}
        )


@dataclass(frozen=True)
class FundingCarryResult:
    candidate: str
    steps: int
    total_return: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


def run_funding_carry_screen(
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    *,
    min_funding_rate_sum: float = 0.0002,
    top_n_values: tuple[int, ...] = (1, 2, 3),
    rebalance_days_values: tuple[int, ...] = (1, 3, 7, 14),
    transaction_cost_rate: float = 0.0004,
) -> tuple[FundingCarryResult, ...]:
    timestamps = sorted(
        set.intersection(
            *(
                {row.timestamp for row in rows}
                for rows in rows_by_symbol.values()
                if rows
            )
        )
    )
    rows_by_symbol_and_timestamp = {
        symbol: {row.timestamp: row for row in rows}
        for symbol, rows in rows_by_symbol.items()
    }
    results = []
    for top_n in top_n_values:
        for rebalance_days in rebalance_days_values:
            results.append(
                _run_candidate(
                    rows_by_symbol_and_timestamp,
                    timestamps=timestamps,
                    candidate=f"positive_funding_carry_top_{top_n}_{rebalance_days}d",
                    min_funding_rate_sum=min_funding_rate_sum,
                    top_n=top_n,
                    rebalance_days=rebalance_days,
                    transaction_cost_rate=transaction_cost_rate,
                )
            )
    return tuple(sorted(results, key=lambda result: result.sharpe, reverse=True))


def write_funding_carry_results(
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
    rows_by_symbol_and_timestamp: dict[str, dict[str, MarketStructureDay]],
    *,
    timestamps: list[str],
    candidate: str,
    min_funding_rate_sum: float,
    top_n: int,
    rebalance_days: int,
    transaction_cost_rate: float,
) -> FundingCarryResult:
    rewards: list[float] = []
    equities: list[float] = []
    transaction_costs: list[float] = []
    current_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    equity = 1.0
    for index, timestamp in enumerate(timestamps[:-1]):
        if index % rebalance_days == 0:
            target_weights = PositiveFundingCarryStrategy(
                min_funding_rate_sum=min_funding_rate_sum,
                top_n=top_n,
            ).decide(
                FundingCarryDecisionInput(
                    rows_by_symbol={
                        symbol: rows_by_timestamp[timestamp]
                        for symbol, rows_by_timestamp in rows_by_symbol_and_timestamp.items()
                    },
                    current_weights=dict(current_weights),
                )
            ).target_weights
        next_timestamp = timestamps[index + 1]
        gross_reward = sum(
            weight
            * _spot_long_perp_short_return(
                rows_by_symbol_and_timestamp[symbol][timestamp],
                rows_by_symbol_and_timestamp[symbol][next_timestamp],
            )
            for symbol, weight in target_weights.items()
        )
        transaction_cost = (
            _turnover(current_weights, target_weights) * transaction_cost_rate
        )
        reward = gross_reward - transaction_cost
        equity *= 1.0 + reward
        rewards.append(reward)
        equities.append(equity)
        transaction_costs.append(transaction_cost)
        current_weights = dict(target_weights)
    summary = summarize_backtest(
        rewards=tuple(rewards),
        equities=tuple(equities),
        transaction_costs=tuple(transaction_costs),
        transaction_cost_rate=transaction_cost_rate,
    )
    return FundingCarryResult(
        candidate=candidate,
        steps=len(rewards),
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        mean_daily_turnover=summary.mean_daily_turnover,
    )


def _spot_long_perp_short_return(
    current: MarketStructureDay,
    next_row: MarketStructureDay,
) -> float:
    funding_received = next_row.funding_rate_sum
    premium_change = (
        (next_row.premium_close - current.premium_close)
        / (1.0 + max(current.premium_close, -0.99))
    )
    return funding_received - premium_change


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
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--transaction-cost-rate", type=float, default=0.0004)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "funding_carry.csv",
    )
    args = parser.parse_args()

    rows_by_symbol = load_market_structure_days(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    results = run_funding_carry_screen(
        rows_by_symbol,
        min_funding_rate_sum=args.min_funding_rate_sum,
        transaction_cost_rate=args.transaction_cost_rate,
    )
    write_funding_carry_results(results, output_path=args.output_path)
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
