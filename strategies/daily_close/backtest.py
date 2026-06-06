from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

from strategies.daily_close.data import DailyMarketBar
from strategies.daily_close.metrics import BacktestSummary, summarize_backtest


@dataclass(frozen=True)
class DailyCloseDecisionInput:
    closes_by_symbol: dict[str, tuple[float, ...]]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class TargetWeights:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class BacktestStep:
    timestamp: str
    target: TargetWeights
    reward: float
    gross_reward: float
    transaction_cost: float
    returns_by_symbol: dict[str, float]
    gross_contribution_by_symbol: dict[str, float]
    equity: float


@dataclass(frozen=True)
class BacktestResult:
    steps: tuple[BacktestStep, ...]
    summary: BacktestSummary


class DailyCloseBacktest:
    def __init__(
        self,
        market_bars: tuple[DailyMarketBar, ...],
        *,
        lookback_days: int,
        initial_equity: float = 1.0,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._market_bars = market_bars
        self._lookback_days = lookback_days
        self._transaction_cost_rate = transaction_cost_rate
        self._index = lookback_days
        self._equity = initial_equity
        self._current_weights: dict[str, float] = {}

    def reset(self) -> DailyCloseDecisionInput | None:
        self._index = self._lookback_days
        self._equity = 1.0
        self._current_weights = {}
        if not self.can_step():
            return None
        return self._decision_input()

    def can_step(self) -> bool:
        return self._index < len(self._market_bars) - 1

    def _decision_input(self) -> DailyCloseDecisionInput:
        lookback_window = self._market_bars[
            self._index - self._lookback_days : self._index + 1
        ]
        symbols = sorted(set.intersection(*(set(bar.closes) for bar in lookback_window)))
        return DailyCloseDecisionInput(
            closes_by_symbol={
                symbol: tuple(bar.closes[symbol] for bar in lookback_window)
                for symbol in symbols
            },
            current_weights=dict(self._current_weights),
            equity=self._equity,
        )

    def step(self, target: TargetWeights) -> BacktestStep:
        returns_by_symbol = self._returns_by_symbol()
        gross_reward = sum(
            target.target_weights.get(symbol, 0.0) * symbol_return
            for symbol, symbol_return in returns_by_symbol.items()
        )
        gross_contribution_by_symbol = {
            symbol: target.target_weights.get(symbol, 0.0) * symbol_return
            for symbol, symbol_return in returns_by_symbol.items()
        }
        transaction_cost = (
            _turnover(self._current_weights, target.target_weights)
            * self._transaction_cost_rate
        )
        reward = gross_reward - transaction_cost
        self._equity *= 1.0 + reward
        self._current_weights = dict(target.target_weights)
        step = BacktestStep(
            timestamp=self._market_bars[self._index].timestamp,
            target=target,
            reward=reward,
            gross_reward=gross_reward,
            transaction_cost=transaction_cost,
            returns_by_symbol=returns_by_symbol,
            gross_contribution_by_symbol=gross_contribution_by_symbol,
            equity=self._equity,
        )
        self._index += 1
        return step

    def _returns_by_symbol(self) -> dict[str, float]:
        current_bar = self._market_bars[self._index]
        next_bar = self._market_bars[self._index + 1]
        returns_by_symbol = {}
        for symbol in sorted(set(current_bar.closes) & set(next_bar.closes)):
            current_close = current_bar.closes[symbol]
            next_close = next_bar.closes[symbol]
            returns_by_symbol[symbol] = (
                (next_close / current_close) - 1.0
                if current_close > 0.0
                else 0.0
            )
        return returns_by_symbol


def run_backtest(
    strategy: TradingStrategy[DailyCloseDecisionInput, TargetWeights],
    market_bars: tuple[DailyMarketBar, ...],
    *,
    lookback_days: int,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    backtest = DailyCloseBacktest(
        market_bars,
        lookback_days=lookback_days,
        transaction_cost_rate=transaction_cost_rate,
    )
    decision_input = backtest.reset()
    if decision_input is None:
        return _summarize((), transaction_cost_rate=transaction_cost_rate)
    steps: list[BacktestStep] = []
    while backtest.can_step():
        target = strategy.decide(decision_input)
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
