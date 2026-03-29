from __future__ import annotations

from dataclasses import dataclass

from .portfolio_decision import (
    CostInput,
    DependenceInput,
    PortfolioDecisionInput,
    PortfolioDecisionOutput,
    PortfolioTarget,
    PredictiveSignalInput,
    RiskInput,
    UncertaintyInput,
)


@dataclass(frozen=True)
class RuleBasedPortfolioPolicy:
    signal_scale: float = 1.0
    max_abs_weight: float = 1.0
    risk_aversion: float = 1.0
    drawdown_aversion: float = 1.0
    uncertainty_aversion: float = 1.0
    dependence_aversion: float = 1.0
    slippage_aversion: float = 1.0
    turnover_aversion: float = 1.0
    state_turnover_aversion: float = 1.0


def apply_rule_based_policy(
    decision_input: PortfolioDecisionInput,
    policy: RuleBasedPortfolioPolicy | None = None,
) -> PortfolioDecisionOutput:
    policy = policy or RuleBasedPortfolioPolicy()
    subject_signals = _aggregate_subject_signals(decision_input.predictive_signals)
    current_weights = decision_input.portfolio_state.weights_by_subject
    subject_ids = tuple(
        sorted(set(current_weights) | set(subject_signals))
    )
    gross_cap = _global_risk_limit(decision_input.risk_inputs, "gross_exposure_cap")
    turnover_penalty = _global_cost_value(
        decision_input.cost_inputs,
        "turnover_penalty",
    )
    state_turnover = max(decision_input.portfolio_state.recent_turnover, 0.0)
    current_drawdown = max(decision_input.portfolio_state.current_drawdown, 0.0)

    provisional_targets: list[PortfolioTarget] = []
    for subject_id in subject_ids:
        signal_value = subject_signals.get(subject_id, 0.0)
        current_weight = current_weights.get(subject_id, 0.0)
        raw_weight = _clip(
            policy.signal_scale * signal_value,
            -policy.max_abs_weight,
            policy.max_abs_weight,
        )

        risk_shrink = _shrink_from_level(
            _mean_risk_value(decision_input.risk_inputs, subject_id),
            policy.risk_aversion,
        )
        uncertainty_shrink = _shrink_from_level(
            _mean_uncertainty_value(decision_input.uncertainty_inputs, subject_id),
            policy.uncertainty_aversion,
        )
        dependence_shrink = _shrink_from_level(
            _mean_dependence_value(decision_input.dependence_inputs, subject_id),
            policy.dependence_aversion,
        )
        drawdown_shrink = _shrink_from_level(
            current_drawdown,
            policy.drawdown_aversion,
        )
        risk_scale = (
            risk_shrink
            * drawdown_shrink
            * uncertainty_shrink
            * dependence_shrink
        )
        target_weight = raw_weight * risk_scale

        no_trade_band = _subject_cost_value(
            decision_input.cost_inputs,
            "no_trade_band",
            subject_id,
        )
        slippage_penalty = _subject_cost_value(
            decision_input.cost_inputs,
            "expected_slippage",
            subject_id,
        )
        delta = target_weight - current_weight
        if abs(delta) <= no_trade_band:
            target_weight = current_weight
            delta = 0.0
        else:
            delta *= _shrink_from_level(
                state_turnover,
                policy.state_turnover_aversion,
            )
            delta *= _shrink_from_level(
                _cost_level(slippage_penalty) + max(turnover_penalty, 0.0),
                policy.slippage_aversion + policy.turnover_aversion,
            )
            target_weight = current_weight + delta

        entry_allowed = not (
            abs(current_weight) == 0.0 and abs(target_weight) == 0.0
        )
        provisional_targets.append(
            PortfolioTarget(
                subject_id=subject_id,
                target_weight=float(target_weight),
                position_delta=float(target_weight - current_weight),
                entry_allowed=entry_allowed,
                risk_scale=float(risk_scale),
            )
        )

    if gross_cap is not None and gross_cap > 0.0:
        provisional_targets = _apply_gross_cap(
            provisional_targets,
            gross_cap,
            current_weights,
        )

    return PortfolioDecisionOutput(
        portfolio_id=decision_input.portfolio_id,
        as_of=decision_input.as_of,
        targets=tuple(provisional_targets),
    )


def _aggregate_subject_signals(
    predictive_signals: tuple[PredictiveSignalInput, ...],
) -> dict[str, float]:
    observed_subject_ids: set[str] = set()
    weighted_values: dict[str, float] = {}
    weights: dict[str, float] = {}
    for signal in predictive_signals:
        observed_subject_ids.add(signal.subject_id)
        confidence = signal.confidence if signal.confidence is not None else 1.0
        confidence = max(confidence, 0.0)
        weighted_values[signal.subject_id] = (
            weighted_values.get(signal.subject_id, 0.0)
            + signal.value * confidence
        )
        weights[signal.subject_id] = weights.get(signal.subject_id, 0.0) + confidence
    return {
        subject_id: (
            float(weighted_values[subject_id] / weights[subject_id])
            if weights.get(subject_id, 0.0) > 0.0
            else 0.0
        )
        for subject_id in observed_subject_ids
    }


def _mean_risk_value(
    risk_inputs: tuple[RiskInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(risk_input.value, 0.0)
        for risk_input in risk_inputs
        if risk_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_uncertainty_value(
    uncertainty_inputs: tuple[UncertaintyInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(uncertainty_input.value, 0.0)
        for uncertainty_input in uncertainty_inputs
        if uncertainty_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_dependence_value(
    dependence_inputs: tuple[DependenceInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(dependence_input.value, 0.0)
        for dependence_input in dependence_inputs
        if dependence_input.left_subject_id == subject_id
        or dependence_input.right_subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _subject_cost_value(
    cost_inputs: tuple[CostInput, ...],
    name: str,
    subject_id: str,
) -> float:
    subject_specific = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id == subject_id
    ]
    if subject_specific:
        return float(sum(subject_specific) / len(subject_specific))
    global_values = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id is None
    ]
    return float(sum(global_values) / len(global_values)) if global_values else 0.0


def _global_cost_value(
    cost_inputs: tuple[CostInput, ...],
    name: str,
) -> float:
    values = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id is None
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _global_risk_limit(
    risk_inputs: tuple[RiskInput, ...],
    name: str,
) -> float | None:
    values = [
        risk_input.value
        for risk_input in risk_inputs
        if risk_input.name == name and risk_input.subject_id is None
    ]
    if not values:
        return None
    return float(min(values))


def _cost_level(value: float) -> float:
    return max(value / 10000.0, 0.0)


def _shrink_from_level(level: float, aversion: float) -> float:
    if level <= 0.0 or aversion <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + aversion * level))


def _apply_gross_cap(
    targets: list[PortfolioTarget],
    gross_cap: float,
    current_weights: dict[str, float],
) -> list[PortfolioTarget]:
    gross = float(sum(abs(target.target_weight) for target in targets))
    if gross <= gross_cap or gross == 0.0:
        return targets
    scale = gross_cap / gross
    capped_targets: list[PortfolioTarget] = []
    for target in targets:
        capped_targets.append(
            PortfolioTarget(
                subject_id=target.subject_id,
                target_weight=float(target.target_weight * scale),
                position_delta=float(
                    target.target_weight * scale
                    - current_weights.get(target.subject_id, 0.0)
                ),
                target_notional=target.target_notional,
                target_quantity=target.target_quantity,
                entry_allowed=target.entry_allowed,
                risk_scale=target.risk_scale,
            )
        )
    return capped_targets


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))
