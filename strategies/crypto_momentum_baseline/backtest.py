from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_momentum_baseline.accounting import PortfolioAccounting
from strategies.crypto_momentum_baseline.data import (
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto_momentum_baseline.metrics import (
    BacktestSummary,
    summarize_backtest,
)
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
    summary: BacktestSummary


class DailyMarketBacktest:
    def __init__(
        self,
        market_bars: tuple[DailyMarketBar, ...],
        *,
        lookback_days: int = 7,
        initial_equity: float = 1.0,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._market_bars = market_bars
        self._lookback_days = lookback_days
        self._transaction_cost_rate = transaction_cost_rate
        self._index = lookback_days
        self._accounting = PortfolioAccounting(
            initial_equity=initial_equity,
            transaction_cost_rate=transaction_cost_rate,
        )

    def reset(self) -> MomentumDecisionInput | None:
        self._index = self._lookback_days
        if not self.can_step():
            return None
        return self._decision_input()

    def can_step(self) -> bool:
        return self._index < len(self._market_bars) - 1

    def timestamp(self) -> str:
        return self._market_bars[self._index].timestamp

    def _decision_input(self) -> MomentumDecisionInput:
        return MomentumDecisionInput(
            closes_by_symbol={
                symbol: tuple(
                    bar.closes[symbol]
                    for bar in self._market_bars[
                        self._index - self._lookback_days : self._index + 1
                    ]
                )
                for symbol in self._market_bars[self._index].closes
            },
            current_weights=self._accounting.current_weights,
            equity=self._accounting.equity,
        )

    def step(self, target: TargetWeights) -> BacktestStep:
        accounting_result = self._accounting.apply(
            target,
            returns_by_symbol=self._returns_by_symbol(),
        )

        step = BacktestStep(
            timestamp=self.timestamp(),
            target=target,
            reward=accounting_result.reward,
            transaction_cost=accounting_result.transaction_cost,
            equity=accounting_result.equity,
        )
        self._index += 1
        return step

    def _returns_by_symbol(self) -> dict[str, float]:
        returns_by_symbol: dict[str, float] = {}
        current_bar = self._market_bars[self._index]
        next_bar = self._market_bars[self._index + 1]
        for symbol, current_close in current_bar.closes.items():
            next_close = next_bar.closes[symbol]
            returns_by_symbol[symbol] = (
                (next_close / current_close) - 1.0
                if current_close > 0.0
                else 0.0
            )
        return returns_by_symbol


def run_backtest(
    strategy: TradingStrategy[MomentumDecisionInput, TargetWeights],
    market_bars: tuple[DailyMarketBar, ...],
    *,
    lookback_days: int = 7,
    initial_equity: float = 1.0,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    backtest = DailyMarketBacktest(
        market_bars,
        lookback_days=lookback_days,
        initial_equity=initial_equity,
        transaction_cost_rate=transaction_cost_rate,
    )
    decision_input = backtest.reset()
    if decision_input is None:
        return _summarize((), transaction_cost_rate=transaction_cost_rate)

    steps: list[BacktestStep] = []
    while backtest.can_step():
        target = strategy.decide(decision_input)
        step = backtest.step(target)
        steps.append(step)
        decision_input = backtest._decision_input()

    return _summarize(
        tuple(steps),
        transaction_cost_rate=transaction_cost_rate,
    )


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


def main() -> None:
    result = run_backtest(SevenDayMomentumStrategy(), load_daily_market_bars())
    print(f"steps={len(result.steps)}")
    print(f"total_return={result.summary.total_return:.6f}")
    print(f"annualized_return={result.summary.annualized_return:.6f}")
    print(f"annualized_volatility={result.summary.annualized_volatility:.6f}")
    print(f"sharpe={result.summary.sharpe:.6f}")
    print(f"max_drawdown={result.summary.max_drawdown:.6f}")
    print(f"mean_daily_turnover={result.summary.mean_daily_turnover:.6f}")


if __name__ == "__main__":
    main()
