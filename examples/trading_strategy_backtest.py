from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class MarketPriceFrame:
    prices: dict[str, float]


@dataclass(frozen=True)
class MarketObservation:
    prices: dict[str, float]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class PortfolioAction:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class BacktestStep:
    equity: float
    action: PortfolioAction
    reward: float
    transaction_cost: float
    observation: MarketObservation


@dataclass(frozen=True)
class WorldStep:
    observation: MarketObservation
    reward: float
    transaction_cost: float
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


def _turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    symbols = current_weights.keys() | target_weights.keys()
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in symbols
    )


class MarketBacktestWorld:
    def __init__(
        self,
        price_frames: Iterable[MarketPriceFrame],
        *,
        initial_equity: float = 1.0,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._price_frames = list(price_frames)
        self._initial_equity = initial_equity
        self._transaction_cost_rate = transaction_cost_rate
        self._index = 0
        self._equity = initial_equity
        self._current_weights: dict[str, float] = {}

    def _observation(self) -> MarketObservation:
        return MarketObservation(
            prices=self._price_frames[self._index].prices,
            current_weights=dict(self._current_weights),
            equity=self._equity,
        )

    def reset(self) -> MarketObservation | None:
        self._index = 0
        self._equity = self._initial_equity
        self._current_weights = {}
        if not self._price_frames:
            return None
        return self._observation()

    def can_step(self) -> bool:
        return self._index < len(self._price_frames) - 1

    def step(self, action: PortfolioAction) -> WorldStep:
        if self._index >= len(self._price_frames) - 1:
            return WorldStep(
                observation=self._observation(),
                reward=0.0,
                transaction_cost=0.0,
                done=True,
                equity=self._equity,
            )

        current_prices = self._price_frames[self._index].prices
        next_prices = self._price_frames[self._index + 1].prices
        gross_reward = _portfolio_return(
            action.target_weights,
            current_prices,
            next_prices,
        )
        transaction_cost = (
            _turnover(self._current_weights, action.target_weights)
            * self._transaction_cost_rate
        )
        reward = gross_reward - transaction_cost
        self._equity *= 1.0 + reward
        self._current_weights = dict(action.target_weights)
        self._index += 1

        return WorldStep(
            observation=self._observation(),
            reward=reward,
            transaction_cost=transaction_cost,
            done=self._index >= len(self._price_frames) - 1,
            equity=self._equity,
        )


def backtest_strategy(
    strategy: TradingStrategy[MarketObservation, PortfolioAction],
    price_frames: Iterable[MarketPriceFrame],
    *,
    initial_equity: float = 1.0,
    transaction_cost_rate: float = 0.001,
) -> list[BacktestStep]:
    world = MarketBacktestWorld(
        price_frames,
        initial_equity=initial_equity,
        transaction_cost_rate=transaction_cost_rate,
    )
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
                transaction_cost=world_step.transaction_cost,
                observation=observation,
            )
        )
        observation = world_step.observation

    return steps


def main() -> None:
    price_frames = [
        MarketPriceFrame(prices={"BTC": 100.0, "ETH": 50.0}),
        MarketPriceFrame(prices={"BTC": 110.0, "ETH": 55.0}),
        MarketPriceFrame(prices={"BTC": 99.0, "ETH": 60.5}),
    ]

    steps = backtest_strategy(EqualWeightLongOnlyStrategy(), price_frames)
    for step in steps:
        print(step)


if __name__ == "__main__":
    main()
