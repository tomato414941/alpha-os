from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_momentum_baseline.data import DailyClose, load_daily_closes
from strategies.crypto_momentum_baseline.strategy import (
    MomentumDecisionInput,
    SevenDayMomentumStrategy,
    TargetWeights,
)


@dataclass(frozen=True)
class BacktestStep:
    timestamp: str
    target: TargetWeights
    reward: float
    transaction_cost: float
    equity: float


@dataclass(frozen=True)
class BacktestResult:
    steps: tuple[BacktestStep, ...]
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


class DailyCloseBacktest:
    def __init__(
        self,
        closes_by_symbol: dict[str, list[DailyClose]],
        *,
        lookback_days: int = 7,
        initial_equity: float = 1.0,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._closes_by_symbol = closes_by_symbol
        self._symbols = tuple(closes_by_symbol)
        self._lookback_days = lookback_days
        self._initial_equity = initial_equity
        self._transaction_cost_rate = transaction_cost_rate
        self._index = lookback_days
        self._equity = initial_equity
        self._current_weights: dict[str, float] = {}

    def reset(self) -> MomentumDecisionInput | None:
        self._index = self._lookback_days
        self._equity = self._initial_equity
        self._current_weights = {}
        if not self.can_step():
            return None
        return self._decision_input()

    def can_step(self) -> bool:
        min_length = min(len(closes) for closes in self._closes_by_symbol.values())
        return self._index < min_length - 1

    def timestamp(self) -> str:
        first_symbol = self._symbols[0]
        return self._closes_by_symbol[first_symbol][self._index].timestamp

    def _decision_input(self) -> MomentumDecisionInput:
        return MomentumDecisionInput(
            closes_by_symbol={
                symbol: tuple(
                    row.close
                    for row in closes[self._index - self._lookback_days : self._index + 1]
                )
                for symbol, closes in self._closes_by_symbol.items()
            },
            current_weights=dict(self._current_weights),
            equity=self._equity,
        )

    def step(self, target: TargetWeights) -> BacktestStep:
        gross_reward = 0.0
        for symbol, weight in target.target_weights.items():
            closes = self._closes_by_symbol[symbol]
            current_close = closes[self._index].close
            next_close = closes[self._index + 1].close
            if current_close <= 0.0:
                continue
            gross_reward += weight * ((next_close / current_close) - 1.0)

        turnover = _turnover(self._current_weights, target.target_weights)
        transaction_cost = turnover * self._transaction_cost_rate
        reward = gross_reward - transaction_cost
        self._equity *= 1.0 + reward
        self._current_weights = dict(target.target_weights)

        step = BacktestStep(
            timestamp=self.timestamp(),
            target=target,
            reward=reward,
            transaction_cost=transaction_cost,
            equity=self._equity,
        )
        self._index += 1
        return step


def run_backtest(
    strategy: TradingStrategy[MomentumDecisionInput, TargetWeights],
    closes_by_symbol: dict[str, list[DailyClose]],
    *,
    lookback_days: int = 7,
    initial_equity: float = 1.0,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    backtest = DailyCloseBacktest(
        closes_by_symbol,
        lookback_days=lookback_days,
        initial_equity=initial_equity,
        transaction_cost_rate=transaction_cost_rate,
    )
    decision_input = backtest.reset()
    if decision_input is None:
        return _summarize(())

    steps: list[BacktestStep] = []
    while backtest.can_step():
        target = strategy.decide(decision_input)
        step = backtest.step(target)
        steps.append(step)
        decision_input = backtest._decision_input()

    return _summarize(tuple(steps))


def _turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    symbols = current_weights.keys() | target_weights.keys()
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in symbols
    )


def _summarize(steps: tuple[BacktestStep, ...]) -> BacktestResult:
    if not steps:
        return BacktestResult(
            steps=steps,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            mean_daily_turnover=0.0,
        )

    returns = [step.reward for step in steps]
    total_return = steps[-1].equity - 1.0
    annualized_return = (steps[-1].equity ** (365.0 / len(steps))) - 1.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    annualized_volatility = sqrt(variance) * sqrt(365.0)
    sharpe = (
        mean_return / sqrt(variance) * sqrt(365.0)
        if variance > 0.0
        else 0.0
    )
    max_drawdown = _max_drawdown([step.equity for step in steps])
    mean_daily_turnover = (
        sum(step.transaction_cost / 0.001 for step in steps) / len(steps)
    )

    return BacktestResult(
        steps=steps,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        mean_daily_turnover=mean_daily_turnover,
    )


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
    return max_drawdown


def main() -> None:
    result = run_backtest(SevenDayMomentumStrategy(), load_daily_closes())
    print(f"steps={len(result.steps)}")
    print(f"total_return={result.total_return:.6f}")
    print(f"annualized_return={result.annualized_return:.6f}")
    print(f"annualized_volatility={result.annualized_volatility:.6f}")
    print(f"sharpe={result.sharpe:.6f}")
    print(f"max_drawdown={result.max_drawdown:.6f}")
    print(f"mean_daily_turnover={result.mean_daily_turnover:.6f}")


if __name__ == "__main__":
    main()
