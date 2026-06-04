from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class MarketObservation:
    prices: dict[str, float]


@dataclass(frozen=True)
class PortfolioAction:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class BacktestStep:
    equity: float
    action: PortfolioAction
    reward: float
    observation: MarketObservation


@dataclass(frozen=True)
class WorldStep:
    observation: MarketObservation
    reward: float
    done: bool
    equity: float


class EqualWeightLongOnlyStrategy:
    def decide(self, strategy_input: MarketObservation) -> PortfolioAction:
        tradable_symbols = tuple(
            symbol for symbol, price in strategy_input.prices.items() if price > 0.0
        )
        if not tradable_symbols:
            return PortfolioAction(target_weights={})

        weight = 1.0 / len(tradable_symbols)
        return PortfolioAction(
            target_weights={symbol: weight for symbol in tradable_symbols}
        )


def _portfolio_return(
    target_weights: dict[str, float],
    current_prices: dict[str, float],
    next_prices: dict[str, float],
) -> float:
    portfolio_return = 0.0
    for symbol, weight in target_weights.items():
        current_price = current_prices.get(symbol)
        next_price = next_prices.get(symbol)
        if current_price is None or next_price is None or current_price <= 0.0:
            continue
        portfolio_return += weight * ((next_price / current_price) - 1.0)
    return portfolio_return


class MarketBacktestWorld:
    def __init__(
        self,
        observations: Iterable[MarketObservation],
        *,
        initial_equity: float = 1.0,
    ) -> None:
        self._observations = list(observations)
        self._initial_equity = initial_equity
        self._index = 0
        self._equity = initial_equity

    def reset(self) -> MarketObservation | None:
        self._index = 0
        self._equity = self._initial_equity
        if not self._observations:
            return None
        return self._observations[0]

    def can_step(self) -> bool:
        return self._index < len(self._observations) - 1

    def step(self, action: PortfolioAction) -> WorldStep:
        if self._index >= len(self._observations) - 1:
            return WorldStep(
                observation=self._observations[-1],
                reward=0.0,
                done=True,
                equity=self._equity,
            )

        current_observation = self._observations[self._index]
        next_observation = self._observations[self._index + 1]
        reward = _portfolio_return(
            action.target_weights,
            current_observation.prices,
            next_observation.prices,
        )
        self._equity *= 1.0 + reward
        self._index += 1

        return WorldStep(
            observation=next_observation,
            reward=reward,
            done=self._index >= len(self._observations) - 1,
            equity=self._equity,
        )


def backtest_strategy(
    strategy: TradingStrategy[MarketObservation, PortfolioAction],
    observations: Iterable[MarketObservation],
    *,
    initial_equity: float = 1.0,
) -> list[BacktestStep]:
    world = MarketBacktestWorld(observations, initial_equity=initial_equity)
    observation = world.reset()
    if observation is None or not world.can_step():
        return []

    steps: list[BacktestStep] = []

    while world.can_step():
        action = strategy.decide(observation)
        world_step = world.step(action)
        steps.append(
            BacktestStep(
                equity=world_step.equity,
                action=action,
                reward=world_step.reward,
                observation=observation,
            )
        )
        observation = world_step.observation

    return steps


def main() -> None:
    observations = [
        MarketObservation(prices={"BTC": 100.0, "ETH": 50.0}),
        MarketObservation(prices={"BTC": 110.0, "ETH": 55.0}),
        MarketObservation(prices={"BTC": 99.0, "ETH": 60.5}),
    ]

    steps = backtest_strategy(EqualWeightLongOnlyStrategy(), observations)
    for step in steps:
        print(step)


if __name__ == "__main__":
    main()
