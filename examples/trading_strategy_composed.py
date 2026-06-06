from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class MarketObservation:
    features_by_symbol: dict[str, dict[str, float]]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class AlphaScore:
    scores: dict[str, float]


@dataclass(frozen=True)
class PortfolioTarget:
    target_weights: dict[str, float]


class MomentumAlphaModel:
    def score(self, observation: MarketObservation) -> AlphaScore:
        return AlphaScore(
            scores={
                symbol: features.get("return_7d", 0.0)
                for symbol, features in observation.features_by_symbol.items()
            }
        )


class LongOnlyScoreAllocator:
    def allocate(self, scores: AlphaScore) -> PortfolioTarget:
        positive_scores = {
            symbol: score for symbol, score in scores.scores.items() if score > 0.0
        }
        score_sum = sum(positive_scores.values())
        if score_sum <= 0.0:
            return PortfolioTarget(target_weights={})

        return PortfolioTarget(
            target_weights={
                symbol: score / score_sum
                for symbol, score in positive_scores.items()
            }
        )


class MomentumAllocatedStrategy:
    def __init__(
        self,
        *,
        alpha_model: MomentumAlphaModel,
        allocator: LongOnlyScoreAllocator,
    ) -> None:
        self._alpha_model = alpha_model
        self._allocator = allocator

    def decide(self, strategy_input: MarketObservation) -> PortfolioTarget:
        scores = self._alpha_model.score(strategy_input)
        return self._allocator.allocate(scores)


def decide_portfolio_target(
    strategy: TradingStrategy[MarketObservation, PortfolioTarget],
    observation: MarketObservation,
) -> PortfolioTarget:
    return strategy.decide(observation)


def main() -> None:
    observation = MarketObservation(
        features_by_symbol={
            "BTC": {"return_7d": 0.04},
            "ETH": {"return_7d": 0.02},
            "SOL": {"return_7d": -0.01},
        },
        current_weights={},
        equity=1.0,
    )
    target = decide_portfolio_target(
        MomentumAllocatedStrategy(
            alpha_model=MomentumAlphaModel(),
            allocator=LongOnlyScoreAllocator(),
        ),
        observation,
    )
    print(target)


if __name__ == "__main__":
    main()
