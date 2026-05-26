from __future__ import annotations

from datetime import datetime
from statistics import pstdev
import numpy as np

from .portfolio_decision import (
    CostInput,
    DependenceInput,
    HistoricalReturnInput,
    ModelUncertaintyInput,
    ObservedPortfolioInputs,
    PortfolioPositionState,
    PortfolioState,
    PredictiveSignalInput,
    RiskInput,
    UncertaintyInput,
)
from .store import MetaPredictionMetricState, MetaPredictionState


def build_runtime_observed_inputs(
    *,
    meta_prediction: MetaPredictionState,
    metric: MetaPredictionMetricState | None,
    subject_id: str,
    target_id: str,
    aggregation_kind: str,
    risk_window: int,
    realized_volatility: float,
) -> ObservedPortfolioInputs:
    confidence = signal_confidence(metric)
    uncertainty_components = uncertainty_proxy_components(
        metric=metric,
        meta_prediction=meta_prediction,
    )
    estimate_std = expected_return_estimate_std(
        signal_value=meta_prediction.value,
        components=uncertainty_components,
    )
    model_uncertainty_components = model_uncertainty_proxy_components(meta_prediction)
    model_error = model_error_std(
        signal_value=meta_prediction.value,
        components=model_uncertainty_components,
    )
    market_impact_bps = volatility_scaled_market_impact_bps(realized_volatility)
    return ObservedPortfolioInputs(
        predictive_signals=(
            PredictiveSignalInput(
                source_id=aggregation_kind,
                source_kind="meta_prediction",
                subject_id=subject_id,
                target_id=target_id,
                value=meta_prediction.value,
                confidence=confidence,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name=f"realized_vol_{risk_window}",
                subject_id=subject_id,
                value=realized_volatility,
                horizon_days=risk_window,
                unit="vol",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="market_impact",
                subject_id=subject_id,
                value=market_impact_bps,
                basis="per_notional",
                unit="bps",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                subject_id=subject_id,
                source_id=aggregation_kind,
                target_id=target_id,
                estimate_std=estimate_std,
                basis="per_signal",
                proxy_components=uncertainty_components,
            ),
        ),
        model_uncertainty_inputs=(
            ModelUncertaintyInput(
                subject_id=subject_id,
                source_id=aggregation_kind,
                target_id=target_id,
                model_error=model_error,
                basis="per_model",
                proxy_components=model_uncertainty_components,
            ),
        ),
        structural_uncertainty_inputs=(),
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


def signal_confidence(metric: MetaPredictionMetricState | None) -> float | None:
    if metric is None:
        return None
    return float(min(max(metric.corr, 0.0), 1.0))


def sample_coverage_uncertainty(metric: MetaPredictionMetricState | None) -> float:
    if metric is None:
        return 1.0
    if metric.window_size <= 0:
        return 1.0
    coverage = min(float(metric.sample_count) / float(metric.window_size), 1.0)
    return float(max(0.0, 1.0 - coverage))


def ensemble_disagreement(meta_prediction: MetaPredictionState) -> float:
    details = meta_prediction.details
    if not isinstance(details, dict):
        return 1.0
    contributors = details.get("contributors")
    if not isinstance(contributors, list) or not contributors:
        return 1.0
    mean_sign = 0
    if meta_prediction.value > 0.0:
        mean_sign = 1
    elif meta_prediction.value < 0.0:
        mean_sign = -1
    if mean_sign == 0:
        return 1.0
    disagreements = 0
    total = 0
    for item in contributors:
        if not isinstance(item, dict):
            continue
        prediction = item.get("prediction")
        if not isinstance(prediction, (int, float)):
            continue
        total += 1
        contributor_sign = 0
        if prediction > 0.0:
            contributor_sign = 1
        elif prediction < 0.0:
            contributor_sign = -1
        if contributor_sign != mean_sign:
            disagreements += 1
    if total == 0:
        return 1.0
    return float(disagreements) / float(total)


def contributor_dispersion(meta_prediction: MetaPredictionState) -> float:
    contributors = _contributor_items(meta_prediction)
    predictions = [
        float(item["prediction"])
        for item in contributors
        if isinstance(item.get("prediction"), (int, float))
    ]
    if not predictions:
        return 1.0
    if len(predictions) == 1:
        return 0.0
    mean_abs_prediction = float(np.mean(np.abs(predictions)))
    dispersion = float(pstdev(predictions))
    scale = mean_abs_prediction + dispersion
    if scale <= 0.0:
        return 1.0
    return float(min(max(dispersion / scale, 0.0), 1.0))


def contributor_concentration(meta_prediction: MetaPredictionState) -> float:
    contributors = _contributor_items(meta_prediction)
    if not contributors:
        return 1.0
    normalized_weights = _normalized_contributor_weights(contributors)
    contributor_count = len(normalized_weights)
    if contributor_count <= 1:
        return 1.0
    effective_count = 1.0 / sum(weight * weight for weight in normalized_weights)
    normalized_breadth = (effective_count - 1.0) / float(contributor_count - 1)
    return float(1.0 - min(max(normalized_breadth, 0.0), 1.0))


def specification_concentration(meta_prediction: MetaPredictionState) -> float:
    contributors = _contributor_items(meta_prediction)
    if not contributors:
        return 1.0
    specification_weights: dict[str, float] = {}
    normalized_weights = _normalized_contributor_weights(contributors)
    for item, weight in zip(contributors, normalized_weights, strict=False):
        specification_id = _specification_id(item)
        specification_weights[specification_id] = (
            specification_weights.get(specification_id, 0.0) + weight
        )
    if len(specification_weights) <= 1:
        return 1.0
    grouped_weights = list(specification_weights.values())
    effective_count = 1.0 / sum(weight * weight for weight in grouped_weights)
    normalized_breadth = (effective_count - 1.0) / float(len(grouped_weights) - 1)
    return float(1.0 - min(max(normalized_breadth, 0.0), 1.0))


def top_model_share(meta_prediction: MetaPredictionState) -> float:
    contributors = _contributor_items(meta_prediction)
    if not contributors:
        return 1.0
    return float(max(_normalized_contributor_weights(contributors), default=1.0))


def uncertainty_proxy_components(
    *,
    metric: MetaPredictionMetricState | None,
    meta_prediction: MetaPredictionState,
) -> dict[str, float]:
    return {
        "sample_coverage": sample_coverage_uncertainty(metric),
        "ensemble_disagreement": ensemble_disagreement(meta_prediction),
        "contributor_dispersion": contributor_dispersion(meta_prediction),
        "contributor_concentration": contributor_concentration(meta_prediction),
    }


def expected_return_estimate_std(
    *,
    signal_value: float,
    components: dict[str, float],
) -> float:
    level = max(abs(signal_value), 1e-6)
    proxy_level = mean_uncertainty_proxy_level(components)
    return float(level * proxy_level)


def mean_uncertainty_proxy_level(components: dict[str, float]) -> float:
    values = [
        max(float(value), 0.0)
        for value in components.values()
        if isinstance(value, (int, float))
    ]
    if not values:
        return 1.0
    return float(sum(values) / len(values))


def model_uncertainty_proxy_components(
    meta_prediction: MetaPredictionState,
) -> dict[str, float]:
    return {
        "model_prediction_dispersion": contributor_dispersion(meta_prediction),
        "model_weight_concentration": contributor_concentration(meta_prediction),
        "specification_weight_concentration": specification_concentration(meta_prediction),
        "top_model_share": top_model_share(meta_prediction),
    }


def model_error_std(
    *,
    signal_value: float,
    components: dict[str, float],
) -> float:
    level = max(abs(signal_value), 1e-6)
    proxy_level = mean_uncertainty_proxy_level(components)
    return float(level * proxy_level)


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


def _contributor_items(meta_prediction: MetaPredictionState) -> list[dict[str, object]]:
    details = meta_prediction.details
    if not isinstance(details, dict):
        return []
    contributors = details.get("contributors")
    if not isinstance(contributors, list):
        return []
    return [item for item in contributors if isinstance(item, dict)]


def _normalized_contributor_weights(
    contributors: list[dict[str, object]],
) -> list[float]:
    weights = [
        float(item["weight"])
        for item in contributors
        if isinstance(item.get("weight"), (int, float))
    ]
    if len(weights) != len(contributors):
        weights = [1.0 for _ in contributors]
    positive_weights = [max(weight, 0.0) for weight in weights]
    total_weight = float(sum(positive_weights))
    if total_weight <= 0.0:
        return [1.0 / float(len(contributors)) for _ in contributors]
    return [weight / total_weight for weight in positive_weights]


def _specification_id(item: dict[str, object]) -> str:
    signal_id = item.get("signal_id")
    if not isinstance(signal_id, str) or not signal_id:
        return "unknown"
    return signal_id.split("@", 1)[0]
