from __future__ import annotations

from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy


@dataclass(frozen=True)
class PortfolioRiskObservation:
    net_exposure: float
    hedge_symbol: str


@dataclass(frozen=True)
class HedgeAction:
    symbol: str
    target_notional: float


class ExposureHedgeStrategy:
    def __init__(self, *, max_exposure: float) -> None:
        self._max_exposure = max_exposure

    def decide(self, strategy_input: PortfolioRiskObservation) -> HedgeAction | None:
        excess_exposure = strategy_input.net_exposure - self._max_exposure
        if excess_exposure <= 0.0:
            return None
        return HedgeAction(
            symbol=strategy_input.hedge_symbol,
            target_notional=-excess_exposure,
        )


def decide_hedge(
    strategy: TradingStrategy[PortfolioRiskObservation, HedgeAction | None],
    observation: PortfolioRiskObservation,
) -> HedgeAction | None:
    return strategy.decide(observation)


def main() -> None:
    observation = PortfolioRiskObservation(net_exposure=1.4, hedge_symbol="BTC-PERP")
    hedge = decide_hedge(ExposureHedgeStrategy(max_exposure=1.0), observation)
    print(hedge)


if __name__ == "__main__":
    main()
