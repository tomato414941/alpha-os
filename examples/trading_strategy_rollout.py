from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class MarketObservation:
    prices: dict[str, float]
    current_weights: dict[str, float]


@dataclass(frozen=True)
class PortfolioAction:
    target_weights: dict[str, float]


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


def rollout_strategy(
    strategy: TradingStrategy[MarketObservation, PortfolioAction],
    observations: Iterable[MarketObservation],
) -> list[PortfolioAction]:
    return [strategy.decide(observation) for observation in observations]


def main() -> None:
    observations = [
        MarketObservation(
            prices={"BTC": 100.0, "ETH": 50.0},
            current_weights={"BTC": 0.0, "ETH": 0.0},
        ),
        MarketObservation(
            prices={"BTC": 110.0, "ETH": 55.0},
            current_weights={"BTC": 0.5, "ETH": 0.5},
        ),
    ]

    actions = rollout_strategy(EqualWeightLongOnlyStrategy(), observations)
    for action in actions:
        print(action)


if __name__ == "__main__":
    main()
