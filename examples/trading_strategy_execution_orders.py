from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_os.trading_strategy import TradingStrategy


OrderStyle = Literal["market", "limit"]
Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BrokerObservation:
    symbol: str
    target_quantity: float
    filled_quantity: float
    last_price: float


@dataclass(frozen=True)
class ExecutionOrder:
    symbol: str
    side: Side
    quantity: float
    order_style: OrderStyle


class FillRemainingExecutionStrategy:
    def __init__(self, *, order_style: OrderStyle) -> None:
        self._order_style = order_style

    def decide(self, strategy_input: BrokerObservation) -> list[ExecutionOrder]:
        remaining = strategy_input.target_quantity - strategy_input.filled_quantity
        if remaining == 0.0:
            return []

        side: Side = "buy" if remaining > 0.0 else "sell"
        return [
            ExecutionOrder(
                symbol=strategy_input.symbol,
                side=side,
                quantity=abs(remaining),
                order_style=self._order_style,
            )
        ]


def decide_execution_orders(
    strategy: TradingStrategy[BrokerObservation, list[ExecutionOrder]],
    observation: BrokerObservation,
) -> list[ExecutionOrder]:
    return strategy.decide(observation)


def main() -> None:
    observation = BrokerObservation(
        symbol="BTC",
        target_quantity=1.0,
        filled_quantity=0.25,
        last_price=100.0,
    )
    orders = decide_execution_orders(
        FillRemainingExecutionStrategy(order_style="limit"),
        observation,
    )
    print(orders)


if __name__ == "__main__":
    main()
