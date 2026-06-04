from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_os.trading_strategy import TradingStrategy


OrderStyle = Literal["market", "limit"]
Urgency = Literal["low", "normal", "high"]


@dataclass(frozen=True)
class RiskObservation:
    prices: dict[str, float]
    current_weights: dict[str, float]
    risk_score: float


@dataclass(frozen=True)
class ExecutionPreference:
    urgency: Urgency
    order_style: OrderStyle


@dataclass(frozen=True)
class TradingIntent:
    target_weights: dict[str, float]
    execution: ExecutionPreference


class RiskOffStrategy:
    def decide(self, strategy_input: RiskObservation) -> TradingIntent:
        if strategy_input.risk_score >= 0.8:
            return TradingIntent(
                target_weights={
                    symbol: 0.0 for symbol in strategy_input.current_weights
                },
                execution=ExecutionPreference(urgency="high", order_style="market"),
            )

        tradable_symbols = tuple(
            symbol for symbol, price in strategy_input.prices.items() if price > 0.0
        )
        if not tradable_symbols:
            return TradingIntent(
                target_weights={},
                execution=ExecutionPreference(urgency="normal", order_style="limit"),
            )

        weight = 1.0 / len(tradable_symbols)
        return TradingIntent(
            target_weights={symbol: weight for symbol in tradable_symbols},
            execution=ExecutionPreference(urgency="normal", order_style="limit"),
        )


def decide_trading_intent(
    strategy: TradingStrategy[RiskObservation, TradingIntent],
    observation: RiskObservation,
) -> TradingIntent:
    return strategy.decide(observation)


def main() -> None:
    observation = RiskObservation(
        prices={"BTC": 100.0, "ETH": 50.0},
        current_weights={"BTC": 0.5, "ETH": 0.5},
        risk_score=0.9,
    )

    intent = decide_trading_intent(RiskOffStrategy(), observation)
    print(intent)


if __name__ == "__main__":
    main()
