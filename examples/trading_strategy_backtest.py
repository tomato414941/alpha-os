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


def backtest_strategy(
    strategy: TradingStrategy[MarketObservation, PortfolioAction],
    observations: Iterable[MarketObservation],
    *,
    initial_equity: float = 1.0,
) -> list[BacktestStep]:
    observation_list = list(observations)
    if len(observation_list) < 2:
        return []

    equity = initial_equity
    steps: list[BacktestStep] = []

    for current_observation, next_observation in zip(
        observation_list,
        observation_list[1:],
    ):
        action = strategy.decide(current_observation)
        realized_return = _portfolio_return(
            action.target_weights,
            current_observation.prices,
            next_observation.prices,
        )
        equity *= 1.0 + realized_return
        steps.append(BacktestStep(equity=equity, action=action))

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
