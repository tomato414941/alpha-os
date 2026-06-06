from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class PriceObservation:
    price: float


@dataclass(frozen=True)
class PositionAction:
    target_position: float


class BreakoutStatefulStrategy:
    def __init__(self, *, breakout_fraction: float) -> None:
        self._breakout_fraction = breakout_fraction
        self._high_watermark: float | None = None

    def decide(self, strategy_input: PriceObservation) -> PositionAction:
        previous_high = self._high_watermark
        self._high_watermark = max(
            strategy_input.price,
            self._high_watermark if self._high_watermark is not None else strategy_input.price,
        )
        if previous_high is None:
            return PositionAction(target_position=0.0)
        if strategy_input.price > previous_high * (1.0 + self._breakout_fraction):
            return PositionAction(target_position=1.0)
        return PositionAction(target_position=0.0)


def decide_positions(
    strategy: TradingStrategy[PriceObservation, PositionAction],
    observations: list[PriceObservation],
) -> list[PositionAction]:
    return [strategy.decide(observation) for observation in observations]


def main() -> None:
    actions = decide_positions(
        BreakoutStatefulStrategy(breakout_fraction=0.01),
        [PriceObservation(price=100.0), PriceObservation(price=102.0)],
    )
    print(actions)


if __name__ == "__main__":
    main()
