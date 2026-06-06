from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_os.trading_strategy import TradingStrategy


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class SpreadObservation:
    long_symbol: str
    short_symbol: str
    spread_zscore: float


@dataclass(frozen=True)
class SpreadLeg:
    symbol: str
    side: Side
    quantity: float


@dataclass(frozen=True)
class SpreadTradeIntent:
    legs: tuple[SpreadLeg, ...]
    reason: str


class MeanReversionSpreadStrategy:
    def __init__(self, *, entry_zscore: float, quantity: float) -> None:
        self._entry_zscore = entry_zscore
        self._quantity = quantity

    def decide(self, strategy_input: SpreadObservation) -> SpreadTradeIntent | None:
        if strategy_input.spread_zscore < self._entry_zscore:
            return None

        return SpreadTradeIntent(
            legs=(
                SpreadLeg(
                    symbol=strategy_input.long_symbol,
                    side="buy",
                    quantity=self._quantity,
                ),
                SpreadLeg(
                    symbol=strategy_input.short_symbol,
                    side="sell",
                    quantity=self._quantity,
                ),
            ),
            reason="mean_reversion_entry",
        )


def decide_spread_trade(
    strategy: TradingStrategy[SpreadObservation, SpreadTradeIntent | None],
    observation: SpreadObservation,
) -> SpreadTradeIntent | None:
    return strategy.decide(observation)


def main() -> None:
    observation = SpreadObservation(
        long_symbol="BTC",
        short_symbol="ETH",
        spread_zscore=2.2,
    )
    intent = decide_spread_trade(
        MeanReversionSpreadStrategy(entry_zscore=2.0, quantity=1.0),
        observation,
    )
    print(intent)


if __name__ == "__main__":
    main()
