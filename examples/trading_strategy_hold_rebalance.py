from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class PortfolioObservation:
    target_weights: dict[str, float]
    current_weights: dict[str, float]


@dataclass(frozen=True)
class Hold:
    reason: str


@dataclass(frozen=True)
class Rebalance:
    target_weights: dict[str, float]


Decision = Hold | Rebalance


class DriftAwareRebalanceStrategy:
    def __init__(self, *, rebalance_threshold: float) -> None:
        self._rebalance_threshold = rebalance_threshold

    def decide(self, strategy_input: PortfolioObservation) -> Decision:
        symbols = (
            strategy_input.target_weights.keys()
            | strategy_input.current_weights.keys()
        )
        max_drift = max(
            (
                abs(
                    strategy_input.target_weights.get(symbol, 0.0)
                    - strategy_input.current_weights.get(symbol, 0.0)
                )
                for symbol in symbols
            ),
            default=0.0,
        )
        if max_drift < self._rebalance_threshold:
            return Hold(reason="within_threshold")
        return Rebalance(target_weights=strategy_input.target_weights)


def decide_rebalance(
    strategy: TradingStrategy[PortfolioObservation, Decision],
    observation: PortfolioObservation,
) -> Decision:
    return strategy.decide(observation)


def main() -> None:
    observation = PortfolioObservation(
        target_weights={"BTC": 0.5, "ETH": 0.5},
        current_weights={"BTC": 0.7, "ETH": 0.3},
    )
    decision = decide_rebalance(
        DriftAwareRebalanceStrategy(rebalance_threshold=0.1),
        observation,
    )
    print(decision)


if __name__ == "__main__":
    main()
