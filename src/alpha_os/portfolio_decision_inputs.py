from __future__ import annotations

from datetime import datetime
from statistics import pstdev
import numpy as np

from .portfolio_decision import (
    DependenceInput,
    HistoricalReturnInput,
    ObservedPortfolioInputs,
    PortfolioPositionState,
    PortfolioState,
)


def merge_observed_inputs(
    *inputs: ObservedPortfolioInputs,
    dependence_inputs: tuple[DependenceInput, ...] = (),
    historical_return_inputs: tuple[HistoricalReturnInput, ...] = (),
) -> ObservedPortfolioInputs:
    merged_history_by_subject: dict[str, HistoricalReturnInput] = {
        item.subject_id: item
        for item in historical_return_inputs
    }
    for item in inputs:
        for history in item.historical_return_inputs:
            merged_history_by_subject[history.subject_id] = history
    return ObservedPortfolioInputs(
        predictive_signals=tuple(
            signal
            for item in inputs
            for signal in item.predictive_signals
        ),
        risk_inputs=tuple(
            risk_input
            for item in inputs
            for risk_input in item.risk_inputs
        ),
        cost_inputs=tuple(
            cost_input
            for item in inputs
            for cost_input in item.cost_inputs
        ),
        uncertainty_inputs=tuple(
            uncertainty_input
            for item in inputs
            for uncertainty_input in item.uncertainty_inputs
        ),
        model_uncertainty_inputs=tuple(
            model_uncertainty_input
            for item in inputs
            for model_uncertainty_input in item.model_uncertainty_inputs
        ),
        structural_uncertainty_inputs=tuple(
            structural_uncertainty_input
            for item in inputs
            for structural_uncertainty_input in item.structural_uncertainty_inputs
        ),
        dependence_inputs=dependence_inputs,
        historical_return_inputs=tuple(
            merged_history_by_subject[subject_id]
            for subject_id in sorted(merged_history_by_subject)
        ),
    )


def build_runtime_observed_dependence_inputs(
    *,
    subject_ids: tuple[str, ...],
    observation_series_by_subject: dict[str, dict[str, float]] | None = None,
) -> tuple[DependenceInput, ...]:
    if len(subject_ids) < 2:
        return ()
    if observation_series_by_subject is None:
        return ()
    items: list[DependenceInput] = []
    for left_index, left_subject_id in enumerate(subject_ids):
        left_series = observation_series_by_subject.get(left_subject_id)
        if not left_series:
            continue
        for right_subject_id in subject_ids[left_index + 1 :]:
            right_series = observation_series_by_subject.get(right_subject_id)
            if not right_series:
                continue
            correlation = aligned_series_correlation(left_series, right_series)
            if correlation is None:
                continue
            items.append(
                DependenceInput(
                    name="rolling_return_correlation",
                    left_subject_id=left_subject_id,
                    right_subject_id=right_subject_id,
                    value=max(correlation, 0.0),
                    basis="correlation",
                )
            )
    return tuple(items)


def volatility_scaled_market_impact_bps(realized_volatility: float) -> float:
    level = max(realized_volatility, 0.0) * 100.0
    return float(min(max(level, 1.0), 100.0))


def realized_observation_volatility(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def portfolio_state_from_decision_details(details: object) -> PortfolioState | None:
    if not isinstance(details, dict):
        return None
    payload = details.get("portfolio_state")
    if not isinstance(payload, dict):
        return None
    positions_payload = payload.get("positions")
    positions: list[PortfolioPositionState] = []
    if isinstance(positions_payload, list):
        for item in positions_payload:
            if not isinstance(item, dict):
                continue
            subject_id = item.get("subject_id")
            weight = item.get("weight")
            if not isinstance(subject_id, str) or not isinstance(weight, (int, float)):
                continue
            positions.append(
                PortfolioPositionState(
                    subject_id=subject_id,
                    weight=float(weight),
                    notional=(
                        None
                        if not isinstance(item.get("notional"), (int, float))
                        else float(item["notional"])
                    ),
                    quantity=(
                        None
                        if not isinstance(item.get("quantity"), (int, float))
                        else float(item["quantity"])
                    ),
                )
            )
    return PortfolioState(
        portfolio_id=(
            payload["portfolio_id"]
            if isinstance(payload.get("portfolio_id"), str)
            else None
        ),
        as_of=payload["as_of"] if isinstance(payload.get("as_of"), str) else None,
        positions=tuple(positions),
        capital_base=(
            float(payload["capital_base"])
            if isinstance(payload.get("capital_base"), (int, float))
            else 1.0
        ),
        gross_limit=(
            float(payload["gross_limit"])
            if isinstance(payload.get("gross_limit"), (int, float))
            else None
        ),
        net_limit=(
            float(payload["net_limit"])
            if isinstance(payload.get("net_limit"), (int, float))
            else None
        ),
        rebalance_step=(
            int(payload["rebalance_step"])
            if isinstance(payload.get("rebalance_step"), int)
            else 0
        ),
        holding_period_days=(
            int(payload["holding_period_days"])
            if isinstance(payload.get("holding_period_days"), int)
            else 0
        ),
        recent_turnover=(
            float(payload["recent_turnover"])
            if isinstance(payload.get("recent_turnover"), (int, float))
            else 0.0
        ),
        current_drawdown=(
            float(payload["current_drawdown"])
            if isinstance(payload.get("current_drawdown"), (int, float))
            else 0.0
        ),
    )


def holding_period_days(*, previous_as_of: str | None, next_as_of: str | None) -> int:
    if previous_as_of is None or next_as_of is None:
        return 0
    try:
        previous_dt = datetime.fromisoformat(previous_as_of)
        next_dt = datetime.fromisoformat(next_as_of)
    except ValueError:
        return 0
    delta = next_dt - previous_dt
    return max(delta.days, 0)


def aligned_series_correlation(
    left_series: dict[str, float],
    right_series: dict[str, float],
) -> float | None:
    common_keys = sorted(set(left_series) & set(right_series))
    if len(common_keys) < 2:
        return None
    left = np.asarray([left_series[key] for key in common_keys], dtype=float)
    right = np.asarray([right_series[key] for key in common_keys], dtype=float)
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
