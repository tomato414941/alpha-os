from dataclasses import dataclass


@dataclass
class CostModel:
    commission_pct: float = 0.10
    slippage_pct: float = 0.05

    @property
    def one_way_cost(self) -> float:
        return (self.commission_pct + self.slippage_pct) / 100.0

    def round_trip_cost(self, turnover: float) -> float:
        return turnover * self.one_way_cost
