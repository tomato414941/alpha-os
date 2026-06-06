from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaScore:
    scores: dict[str, float]


@dataclass(frozen=True)
class PortfolioTarget:
    target_weights: dict[str, float]


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


def allocate_portfolio(
    allocator: LongOnlyScoreAllocator,
    scores: AlphaScore,
) -> PortfolioTarget:
    return allocator.allocate(scores)


def main() -> None:
    target = allocate_portfolio(
        LongOnlyScoreAllocator(),
        AlphaScore(scores={"BTC": 0.04, "ETH": 0.02, "SOL": -0.01}),
    )
    print(target)


if __name__ == "__main__":
    main()
