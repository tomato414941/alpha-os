from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_os.trading_strategy import TradingStrategy


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class MarketObservation:
    prices: dict[str, float]
    cash: float


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float


class BuyDipOrderStrategy:
    def __init__(self, *, symbol: str, reference_price: float, quantity: float) -> None:
        self._symbol = symbol
        self._reference_price = reference_price
        self._quantity = quantity

    def decide(self, strategy_input: MarketObservation) -> list[Order]:
        price = strategy_input.prices.get(self._symbol)
        if price is None or price >= self._reference_price:
            return []
        return [Order(symbol=self._symbol, side="buy", quantity=self._quantity)]


def decide_orders(
    strategy: TradingStrategy[MarketObservation, list[Order]],
    observation: MarketObservation,
) -> list[Order]:
    return strategy.decide(observation)


def main() -> None:
    observation = MarketObservation(prices={"BTC": 95.0}, cash=1_000.0)
    orders = decide_orders(
        BuyDipOrderStrategy(symbol="BTC", reference_price=100.0, quantity=0.1),
        observation,
    )
    print(orders)


if __name__ == "__main__":
    main()
