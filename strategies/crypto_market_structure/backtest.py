from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.crypto_market_structure.strategy import (
    MarketStructureDecisionInput,
    MarketStructureRankStrategy,
    MarketStructureTargetWeights,
)
from strategies.daily_close.metrics import BacktestSummary, summarize_backtest


@dataclass(frozen=True)
class BacktestStep:
    timestamp: str
    target: MarketStructureTargetWeights
    reward: float
    gross_reward: float
    transaction_cost: float
    returns_by_symbol: dict[str, float]
    equity: float


@dataclass(frozen=True)
class BacktestResult:
    steps: tuple[BacktestStep, ...]
    summary: BacktestSummary


@dataclass(frozen=True)
class Variant:
    factory: Callable[[], MarketStructureRankStrategy]
    lookback_days: int
    rebalance_days: int = 1


VARIANTS = {
    "funding_premium_flow_top_2": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "funding_rate_sum": 1.0,
                "premium_close": 1.0,
                "taker_buy_imbalance": 1.0,
                "volume_ratio_20d": 0.5,
            },
            top_n=2,
        ),
        lookback_days=20,
    ),
    "funding_premium_flow_top_2_weekly": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "funding_rate_sum": 1.0,
                "premium_close": 1.0,
                "taker_buy_imbalance": 1.0,
                "volume_ratio_20d": 0.5,
            },
            top_n=2,
        ),
        lookback_days=20,
        rebalance_days=7,
    ),
    "flow_top_3": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "taker_buy_imbalance": 1.0,
                "volume_ratio_20d": 0.5,
            },
            top_n=3,
        ),
        lookback_days=20,
    ),
    "flow_top_3_weekly": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "taker_buy_imbalance": 1.0,
                "volume_ratio_20d": 0.5,
            },
            top_n=3,
        ),
        lookback_days=20,
        rebalance_days=7,
    ),
    "premium_funding_top_2": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "funding_rate_sum": 1.0,
                "premium_close": 1.0,
            },
            top_n=2,
        ),
        lookback_days=20,
    ),
    "premium_funding_top_2_weekly": Variant(
        factory=lambda: MarketStructureRankStrategy(
            feature_weights={
                "funding_rate_sum": 1.0,
                "premium_close": 1.0,
            },
            top_n=2,
        ),
        lookback_days=20,
        rebalance_days=7,
    ),
}


class MarketStructureBacktest:
    def __init__(
        self,
        rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
        *,
        lookback_days: int,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._rows_by_symbol = rows_by_symbol
        self._rows_by_symbol_and_timestamp = {
            symbol: {row.timestamp: row for row in rows}
            for symbol, rows in rows_by_symbol.items()
        }
        self._timestamps = sorted(
            set.intersection(
                *(
                    set(rows_by_timestamp)
                    for rows_by_timestamp in self._rows_by_symbol_and_timestamp.values()
                )
            )
        )
        self._lookback_days = lookback_days
        self._transaction_cost_rate = transaction_cost_rate
        self._index = lookback_days
        self._equity = 1.0
        self._current_weights: dict[str, float] = {}

    def reset(self) -> MarketStructureDecisionInput | None:
        self._index = self._lookback_days
        self._equity = 1.0
        self._current_weights = {}
        if not self.can_step():
            return None
        return self._decision_input()

    def can_step(self) -> bool:
        return self._index < len(self._timestamps) - 1

    def step(self, target: MarketStructureTargetWeights) -> BacktestStep:
        returns_by_symbol = self._returns_by_symbol()
        gross_reward = sum(
            target.target_weights.get(symbol, 0.0) * symbol_return
            for symbol, symbol_return in returns_by_symbol.items()
        )
        transaction_cost = (
            _turnover(self._current_weights, target.target_weights)
            * self._transaction_cost_rate
        )
        reward = gross_reward - transaction_cost
        self._equity *= 1.0 + reward
        self._current_weights = dict(target.target_weights)
        step = BacktestStep(
            timestamp=self._timestamps[self._index],
            target=target,
            reward=reward,
            gross_reward=gross_reward,
            transaction_cost=transaction_cost,
            returns_by_symbol=returns_by_symbol,
            equity=self._equity,
        )
        self._index += 1
        return step

    def _decision_input(self) -> MarketStructureDecisionInput:
        current_timestamp = self._timestamps[self._index]
        history_by_symbol: dict[str, tuple[MarketStructureDay, ...]] = {}
        for symbol, rows_by_timestamp in self._rows_by_symbol_and_timestamp.items():
            rows = [
                rows_by_timestamp[timestamp]
                for timestamp in self._timestamps[: self._index + 1]
                if timestamp in rows_by_timestamp and timestamp <= current_timestamp
            ]
            if len(rows) >= self._lookback_days:
                history_by_symbol[symbol] = tuple(rows[-self._lookback_days :])
        return MarketStructureDecisionInput(
            history_by_symbol=history_by_symbol,
            current_weights=dict(self._current_weights),
            equity=self._equity,
        )

    def _returns_by_symbol(self) -> dict[str, float]:
        current_timestamp = self._timestamps[self._index]
        next_timestamp = self._timestamps[self._index + 1]
        returns_by_symbol = {}
        for symbol, rows_by_timestamp in self._rows_by_symbol_and_timestamp.items():
            current = rows_by_timestamp[current_timestamp]
            next_row = rows_by_timestamp[next_timestamp]
            returns_by_symbol[symbol] = (
                (next_row.close / current.close) - 1.0
                if current.close > 0.0
                else 0.0
            )
        return returns_by_symbol


def run_backtest(
    strategy: TradingStrategy[MarketStructureDecisionInput, MarketStructureTargetWeights],
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    *,
    lookback_days: int,
    rebalance_days: int = 1,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    backtest = MarketStructureBacktest(
        rows_by_symbol,
        lookback_days=lookback_days,
        transaction_cost_rate=transaction_cost_rate,
    )
    decision_input = backtest.reset()
    if decision_input is None:
        return _summarize((), transaction_cost_rate=transaction_cost_rate)
    steps: list[BacktestStep] = []
    while backtest.can_step():
        target = (
            strategy.decide(decision_input)
            if len(steps) % rebalance_days == 0
            else MarketStructureTargetWeights(dict(backtest._current_weights))
        )
        steps.append(backtest.step(target))
        decision_input = backtest._decision_input()
    return _summarize(tuple(steps), transaction_cost_rate=transaction_cost_rate)


def _summarize(
    steps: tuple[BacktestStep, ...],
    *,
    transaction_cost_rate: float,
) -> BacktestResult:
    return BacktestResult(
        steps=steps,
        summary=summarize_backtest(
            rewards=tuple(step.reward for step in steps),
            equities=tuple(step.equity for step in steps),
            transaction_costs=tuple(step.transaction_cost for step in steps),
            transaction_cost_rate=transaction_cost_rate,
        ),
    )


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
    args = parser.parse_args()

    rows_by_symbol = load_market_structure_days(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    for name, variant in VARIANTS.items():
        result = run_backtest(
            variant.factory(),
            rows_by_symbol,
            lookback_days=variant.lookback_days,
            rebalance_days=variant.rebalance_days,
        )
        print(
            name,
            len(result.steps),
            f"{result.summary.total_return:.6f}",
            f"{result.summary.sharpe:.6f}",
            f"{result.summary.max_drawdown:.6f}",
            f"{result.summary.mean_daily_turnover:.6f}",
        )


if __name__ == "__main__":
    main()
