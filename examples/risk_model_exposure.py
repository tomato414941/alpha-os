from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioSnapshot:
    notionals: dict[str, float]
    equity: float


@dataclass(frozen=True)
class RiskEstimate:
    gross_exposure: float
    net_exposure: float


class ExposureRiskModel:
    def estimate(self, portfolio: PortfolioSnapshot) -> RiskEstimate:
        if portfolio.equity <= 0.0:
            return RiskEstimate(gross_exposure=0.0, net_exposure=0.0)

        gross_notional = sum(abs(notional) for notional in portfolio.notionals.values())
        net_notional = sum(portfolio.notionals.values())
        return RiskEstimate(
            gross_exposure=gross_notional / portfolio.equity,
            net_exposure=net_notional / portfolio.equity,
        )


def estimate_risk(
    model: ExposureRiskModel,
    portfolio: PortfolioSnapshot,
) -> RiskEstimate:
    return model.estimate(portfolio)


def main() -> None:
    portfolio = PortfolioSnapshot(notionals={"BTC": 600.0, "ETH": -200.0}, equity=1_000.0)
    risk = estimate_risk(ExposureRiskModel(), portfolio)
    print(risk)


if __name__ == "__main__":
    main()
