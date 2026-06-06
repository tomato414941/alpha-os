from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class FeatureBatch:
    features_by_symbol: dict[str, dict[str, float]]


@dataclass(frozen=True)
class AlphaScore:
    scores: dict[str, float]


class MomentumAlphaModel:
    def decide(self, strategy_input: FeatureBatch) -> AlphaScore:
        return AlphaScore(
            scores={
                symbol: features.get("return_7d", 0.0)
                for symbol, features in strategy_input.features_by_symbol.items()
            }
        )


def score_alpha(
    model: TradingStrategy[FeatureBatch, AlphaScore],
    features: FeatureBatch,
) -> AlphaScore:
    return model.decide(features)


def main() -> None:
    features = FeatureBatch(
        features_by_symbol={
            "BTC": {"return_7d": 0.04},
            "ETH": {"return_7d": -0.02},
        }
    )
    scores = score_alpha(MomentumAlphaModel(), features)
    print(scores)


if __name__ == "__main__":
    main()
