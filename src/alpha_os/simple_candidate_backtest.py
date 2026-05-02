from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio_allocation import EqualWeightLongOnlyAllocator, PositionCandidate


@dataclass(frozen=True)
class SimpleCandidateBacktestResult:
    daily_returns: pd.DataFrame

    @property
    def total_net_return(self) -> float:
        return float((1.0 + self.daily_returns["net_return"]).prod() - 1.0)

    @property
    def max_drawdown(self) -> float:
        equity = (1.0 + self.daily_returns["net_return"]).cumprod()
        return float((equity / equity.cummax() - 1.0).min())

    @property
    def mean_daily_turnover(self) -> float:
        return float(self.daily_returns["turnover"].mean())


def run_equal_weight_long_only_backtest(
    *,
    returns_by_subject: dict[str, pd.Series],
    candidates_by_date: dict[str, tuple[PositionCandidate, ...]],
    cost_bps_per_unit_turnover: float = 0.0,
    gross_exposure_cap: float = 1.0,
) -> SimpleCandidateBacktestResult:
    subject_ids = tuple(sorted(returns_by_subject))
    if not subject_ids:
        return SimpleCandidateBacktestResult(daily_returns=_empty_daily_returns())

    return_frame = pd.DataFrame(
        {
            subject_id: returns_by_subject[subject_id].astype(float)
            for subject_id in subject_ids
        }
    ).dropna(how="all")
    if return_frame.empty:
        return SimpleCandidateBacktestResult(daily_returns=_empty_daily_returns())

    allocator = EqualWeightLongOnlyAllocator(gross_exposure_cap=gross_exposure_cap)
    previous_weights = {subject_id: 0.0 for subject_id in subject_ids}
    rows: list[dict[str, float | str]] = []
    for date, row in return_frame.iterrows():
        date_key = str(date.date()) if hasattr(date, "date") else str(date)
        candidates = candidates_by_date.get(
            date_key,
            tuple(
                PositionCandidate(subject_id=subject_id, direction="flat")
                for subject_id in subject_ids
            ),
        )
        allocation = allocator.allocate(candidates)
        weights = {
            subject_id: float(allocation.target_weights.get(subject_id, 0.0))
            for subject_id in subject_ids
        }
        gross_return = float(
            sum(
                weights[subject_id] * float(row.get(subject_id, 0.0))
                for subject_id in subject_ids
            )
        )
        turnover = float(
            sum(
                abs(weights[subject_id] - previous_weights[subject_id])
                for subject_id in subject_ids
            )
        )
        cost = turnover * float(cost_bps_per_unit_turnover) / 10_000.0
        rows.append(
            {
                "date": date_key,
                "gross_return": gross_return,
                "turnover": turnover,
                "net_return": gross_return - cost,
                "active_assets": float(
                    sum(1 for value in weights.values() if value > 0.0)
                ),
            }
        )
        previous_weights = weights

    daily_returns = pd.DataFrame(rows).set_index("date")
    return SimpleCandidateBacktestResult(daily_returns=daily_returns)


def _empty_daily_returns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=("gross_return", "turnover", "net_return", "active_assets"),
        dtype=float,
    )
