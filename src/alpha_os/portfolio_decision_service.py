from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET, default_runtime_asset
from .decision_backtest import constrained_targets_by_subject, hold_position_target
from .meta_aggregation_service import DEFAULT_PRIMARY_AGGREGATION_KIND
from .portfolio_construction_config import PortfolioConstructionSpec
from .portfolio_decision_inputs import (
    build_runtime_observed_inputs,
    build_runtime_observed_dependence_inputs,
    holding_period_days,
    merge_observed_inputs,
    no_trade_band_from_market_impact_bps,
    portfolio_state_from_decision_details,
    realized_observation_volatility,
    volatility_scaled_market_impact_bps,
)
from .portfolio_decision import (
    CostInput,
    DependenceInput,
    HistoricalReturnInput,
    ModelUncertaintyInput,
    ObservationSpec,
    ObservedPortfolioInputs,
    PortfolioDecisionInput,
    PortfolioDecisionAssumptions,
    PortfolioDecisionOutput,
    PortfolioPositionState,
    PortfolioState,
    PortfolioTarget,
    RiskInput,
    SubjectObservationBinding,
    SubjectSet,
    PredictiveSignalInput,
    UncertaintyInput,
)
from .trading_strategy import TradingStrategySpec
from .portfolio_sizing_policy import (
    PortfolioSizingPolicy,
    apply_portfolio_sizing_policy,
)
from .store import EvaluationStore, MetaPredictionMetricState, MetaPredictionState
from .universe_contract import validate_subject_set_universe_contract


def _trading_strategy_trace_document(
    trading_strategy: TradingStrategySpec,
) -> dict[str, object]:
    portfolio = trading_strategy.portfolio
    policy = portfolio.to_portfolio_policy()
    risk = policy.risk_policy
    friction = portfolio.rebalance_friction_policy
    execution = portfolio.execution_policy
    holding = portfolio.holding_cost_policy
    return {
        "strategy_id": trading_strategy.strategy_id,
        "label": trading_strategy.label,
        "subject_set_id": trading_strategy.subject_set_id,
        "target_id": trading_strategy.target_id,
        "selection_kind": policy.selection_policy.selection_kind,
        "sizing_method": policy.sizing_policy.sizing_method,
        "rebalance": policy.rebalance_policy.rebalance,
        "top_k": policy.selection_policy.top_k,
        "long_only": risk.long_only,
        "direction_mode": risk.direction_mode,
        "gross_exposure_cap": risk.gross_exposure_cap,
        "target_vol": risk.target_vol,
        "gross_leverage_cap": risk.gross_leverage_cap,
        "net_exposure_target": risk.net_exposure_target,
        "turnover_friction": friction.turnover_friction,
        "no_trade_band": friction.no_trade_band,
        "market_impact_bps": execution.market_impact_bps,
        "fee_bps": execution.fee_bps,
        "bid_ask_spread_bps": execution.bid_ask_spread_bps,
        "funding_bps_per_step": holding.funding_bps_per_step,
        "borrow_fee_bps_per_step": holding.borrow_fee_bps_per_step,
    }


@dataclass(frozen=True)
class RuntimeDecisionBuildConfig:
    aggregation_kind: str = DEFAULT_PRIMARY_AGGREGATION_KIND
    risk_window: int = 20
    dependence_window: int = 20
    subject_set: SubjectSet | None = None


def build_portfolio_decision_input(
    store: EvaluationStore,
    *,
    runtime_asset: str | None = None,
    target_id: str = DEFAULT_TARGET,
    portfolio_id: str | None = None,
    subject_id: str | None = None,
    portfolio_state: PortfolioState,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
) -> PortfolioDecisionInput | None:
    config = config or RuntimeDecisionBuildConfig()
    assumptions = assumptions or PortfolioDecisionAssumptions()
    resolved_subject_id = DEFAULT_SUBJECT_ID if subject_id is None else subject_id
    resolved_runtime_asset = (
        default_runtime_asset(resolved_subject_id)
        if runtime_asset is None
        else runtime_asset
    )
    subject_set = _resolved_subject_set(
        runtime_asset=resolved_runtime_asset,
        subject_id=resolved_subject_id,
        configured_subject_set=config.subject_set,
    )
    observed_inputs_by_subject = []
    observed_as_of_values: list[str] = []
    observation_series_by_subject: dict[str, dict[str, float]] = {}
    for item in subject_set.bindings:
        meta_prediction = _latest_meta_prediction(
            store,
            subject_id=item.subject_id,
            asset=item.asset,
            target_id=target_id,
            aggregation_kind=config.aggregation_kind,
        )
        if meta_prediction is None:
            continue
        metric = _meta_metric(
            store,
            subject_id=item.subject_id,
            asset=item.asset,
            target_id=target_id,
            aggregation_kind=config.aggregation_kind,
        )
        realized_vol = _realized_observation_volatility(
            store,
            subject_id=item.subject_id,
            asset=item.asset,
            target_id=target_id,
            window_size=config.risk_window,
        )
        observed_inputs_by_subject.append(
            build_runtime_observed_inputs(
                meta_prediction=meta_prediction,
                metric=metric,
                subject_id=item.subject_id,
                target_id=target_id,
                aggregation_kind=config.aggregation_kind,
                risk_window=config.risk_window,
                realized_volatility=realized_vol,
            )
        )
        observed_as_of_values.append(meta_prediction.updated_at)
        observation_series = _observation_series_by_date(
            store,
            subject_id=item.subject_id,
            asset=item.asset,
            target_id=target_id,
            window_size=max(config.dependence_window, config.risk_window, 60),
        )
        if observation_series:
            observation_series_by_subject[item.subject_id] = observation_series
    if not observed_inputs_by_subject:
        return None

    observed_dependence_inputs = _observed_dependence_inputs(
        store,
        subject_set=subject_set,
        target_id=target_id,
        portfolio_state=portfolio_state,
        config=config,
    )

    observed_inputs = merge_observed_inputs(
        *observed_inputs_by_subject,
        dependence_inputs=observed_dependence_inputs,
        historical_return_inputs=_historical_return_inputs(
            observation_series_by_subject=observation_series_by_subject,
            subject_ids=subject_set.subject_ids,
        ),
    )

    return PortfolioDecisionInput(
        portfolio_id=portfolio_id,
        as_of=max(observed_as_of_values),
        portfolio_state=portfolio_state,
        observed_inputs=observed_inputs,
        assumptions=assumptions,
    )


def build_portfolio_decision_output(
    store: EvaluationStore,
    *,
    runtime_asset: str | None = None,
    target_id: str = DEFAULT_TARGET,
    portfolio_id: str | None = None,
    subject_id: str | None = None,
    portfolio_state: PortfolioState,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
    sizing_policy: PortfolioSizingPolicy | None = None,
) -> PortfolioDecisionOutput | None:
    decision_input = build_portfolio_decision_input(
        store,
        runtime_asset=runtime_asset,
        target_id=target_id,
        portfolio_id=portfolio_id,
        subject_id=subject_id,
        portfolio_state=portfolio_state,
        config=config,
        assumptions=assumptions,
    )
    if decision_input is None:
        return None
    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=sizing_policy,
    )
    return PortfolioDecisionOutput(
        portfolio_id=portfolio_id,
        as_of=decision_output.as_of,
        targets=decision_output.targets,
        sizing_diagnostics=decision_output.sizing_diagnostics,
    )


def build_portfolio_decision_input_from_compressed_belief(
    store: EvaluationStore,
    *,
    compressed_belief_id: str,
    portfolio_id: str | None = None,
    portfolio_state: PortfolioState,
    assumptions: PortfolioDecisionAssumptions | None = None,
) -> PortfolioDecisionInput:
    assumptions = assumptions or PortfolioDecisionAssumptions()
    belief_state = store.get_compressed_belief(compressed_belief_id)
    if belief_state is None:
        raise ValueError(f"unknown compressed belief: {compressed_belief_id}")
    belief = belief_state.belief
    predictive_signals = []
    risk_inputs = []
    cost_inputs = []
    uncertainty_inputs = []
    model_uncertainty_inputs = []
    observation_series_by_subject: dict[str, dict[str, float]] = {}
    subject_set = _subject_set_for_compressed_belief(store, signal_discovery_id=belief.signal_discovery_id)
    target_id_by_subject = {
        component.subject_id: component.target_id
        for component in belief.components
    }
    asset_by_subject = (
        {} if subject_set is None else subject_set.asset_by_subject
    )
    for component in belief.components:
        confidence = min(max(component.confidence, 0.0), 1.0)
        predictive_signals.append(
            PredictiveSignalInput(
                source_id=belief.compressed_belief_id,
                source_kind="compressed_belief",
                subject_id=component.subject_id,
                target_id=component.target_id,
                value=component.belief_value,
                confidence=confidence,
            )
        )
        uncertainty_inputs.append(
            UncertaintyInput(
                subject_id=component.subject_id,
                source_id=belief.compressed_belief_id,
                target_id=component.target_id,
                estimate_std=abs(component.belief_value) * (1.0 - confidence),
                basis="compressed_belief",
                proxy_components={
                    "compressed_confidence_gap": 1.0 - confidence,
                },
            )
        )
        model_uncertainty_inputs.append(
            ModelUncertaintyInput(
                subject_id=component.subject_id,
                source_id=belief.compressed_belief_id,
                target_id=component.target_id,
                model_error=(
                    0.35 / float(max(component.signal_contribution_count, 1))
                    + 0.15 / float(max(component.family_count, 1))
                    + 0.25 / float(max(component.cluster_count, 1))
                    + 0.15 / float(max(component.effective_belief_count, 1.0))
                    + 0.10 * max(0.0, 1.0 - component.diversity_score)
                ),
                basis="compressed_belief",
                proxy_components={
                    "signal_contribution_count_inverse": 1.0
                    / float(max(component.signal_contribution_count, 1)),
                    "family_count_inverse": 1.0 / float(max(component.family_count, 1)),
                    "cluster_count_inverse": 1.0 / float(max(component.cluster_count, 1)),
                    "effective_belief_count_inverse": 1.0
                    / float(max(component.effective_belief_count, 1.0)),
                    "diversity_gap": max(0.0, 1.0 - component.diversity_score),
                },
            )
        )
        asset = asset_by_subject.get(component.subject_id)
        if asset is not None:
            realized_volatility = _realized_observation_volatility(
                store,
                subject_id=component.subject_id,
                asset=asset,
                target_id=component.target_id,
                window_size=20,
            )
            risk_inputs.append(
                RiskInput(
                    name="realized_vol_20",
                    subject_id=component.subject_id,
                    value=realized_volatility,
                    horizon_days=20,
                    unit="vol",
                )
            )
            market_impact_bps = volatility_scaled_market_impact_bps(realized_volatility)
            cost_inputs.extend(
                (
                    CostInput(
                        name="market_impact",
                        subject_id=component.subject_id,
                        value=market_impact_bps,
                        basis="per_notional",
                        unit="bps",
                    ),
                    CostInput(
                        name="no_trade_band",
                        subject_id=component.subject_id,
                        value=no_trade_band_from_market_impact_bps(market_impact_bps),
                        basis="per_delta_weight",
                        unit="weight",
                    ),
                )
            )
            observation_series = _observation_series_by_date(
                store,
                subject_id=component.subject_id,
                asset=asset,
                target_id=component.target_id,
                window_size=20,
            )
            if observation_series:
                observation_series_by_subject[component.subject_id] = observation_series
    if not predictive_signals:
        raise ValueError(
            f"compressed belief does not contain usable components: {compressed_belief_id}"
        )
    dependence_inputs = build_runtime_observed_dependence_inputs(
        subject_ids=tuple(
            subject_id
            for subject_id in sorted(
                set(target_id_by_subject)
                | {position.subject_id for position in portfolio_state.positions}
            )
            if subject_id in observation_series_by_subject
        ),
        observation_series_by_subject=observation_series_by_subject,
    )
    observed_inputs = ObservedPortfolioInputs(
        predictive_signals=tuple(predictive_signals),
        risk_inputs=tuple(risk_inputs),
        cost_inputs=tuple(cost_inputs),
        uncertainty_inputs=tuple(uncertainty_inputs),
        model_uncertainty_inputs=tuple(model_uncertainty_inputs),
        dependence_inputs=dependence_inputs,
        historical_return_inputs=_historical_return_inputs(
            observation_series_by_subject=observation_series_by_subject,
            subject_ids=tuple(sorted(observation_series_by_subject)),
        ),
    )
    return PortfolioDecisionInput(
        portfolio_id=portfolio_id,
        as_of=belief.created_at,
        portfolio_state=portfolio_state,
        observed_inputs=observed_inputs,
        assumptions=assumptions,
    )


def apply_decision_output_constraints(
    decision_output: PortfolioDecisionOutput,
    *,
    portfolio_state: PortfolioState,
    subject_set: SubjectSet | None = None,
    portfolio_construction: PortfolioConstructionSpec | None = None,
    risk_by_subject: dict[str, float] | None = None,
) -> PortfolioDecisionOutput:
    if portfolio_construction is None:
        return decision_output
    rebalance_interval = max(int(portfolio_construction.rebalance_interval_steps), 1)
    current_weights = portfolio_state.weights_by_subject
    capital_base = max(float(portfolio_state.capital_base), 0.0)
    subject_ids = tuple(
        dict.fromkeys(
            tuple(current_weights)
            + tuple(item.subject_id for item in decision_output.targets)
        )
    )
    if ((max(int(portfolio_state.rebalance_step), 1) - 1) % rebalance_interval) != 0:
        targets = tuple(
            hold_position_target(
                subject_id=subject_id,
                current_weight=current_weights.get(subject_id, 0.0),
                capital_base=capital_base,
            )
            for subject_id in subject_ids
        )
        return PortfolioDecisionOutput(
            portfolio_id=decision_output.portfolio_id,
            as_of=decision_output.as_of,
            targets=targets,
            sizing_diagnostics=decision_output.sizing_diagnostics,
        )
    target_by_subject = {
        item.subject_id: item
        for item in decision_output.targets
    }
    normalized_targets = tuple(
        target_by_subject.get(
            subject_id,
            PortfolioTarget(
                subject_id=subject_id,
                target_weight=0.0,
                position_delta=0.0,
                target_notional=0.0,
                entry_allowed=False,
                risk_scale=1.0,
            ),
        )
        for subject_id in subject_ids
    )
    constrained = constrained_targets_by_subject(
        normalized_targets,
        current_weights=current_weights,
        capital_base=capital_base,
        gross_exposure_cap=portfolio_construction.gross_exposure_cap,
        gross_leverage_cap=portfolio_construction.gross_leverage_cap,
        net_exposure_target=portfolio_construction.net_exposure_target,
        target_vol=portfolio_construction.target_vol,
        risk_by_subject=risk_by_subject,
        risk_budget=portfolio_construction.risk_budget,
        constraint_boundary=portfolio_construction.constraint_boundary,
        long_only=portfolio_construction.long_only,
        direction_mode=portfolio_construction.direction_mode,
        top_k=portfolio_construction.top_k,
        active_overlay=portfolio_construction.active_overlay,
        asset_class_by_subject=(
            {} if subject_set is None else subject_set.asset_class_by_subject
        ),
        cluster_by_subject=(
            {} if subject_set is None else subject_set.cluster_by_subject
        ),
        asset_class_weight_caps=dict(portfolio_construction.asset_class_weight_caps),
        cluster_weight_caps=dict(portfolio_construction.cluster_weight_caps),
    )
    return PortfolioDecisionOutput(
        portfolio_id=decision_output.portfolio_id,
        as_of=decision_output.as_of,
        targets=tuple(
            constrained.get(
                subject_id,
                hold_position_target(
                    subject_id=subject_id,
                    current_weight=current_weights.get(subject_id, 0.0),
                    capital_base=capital_base,
                ),
            )
            for subject_id in subject_ids
        ),
        sizing_diagnostics=decision_output.sizing_diagnostics,
    )


def _subject_set_for_compressed_belief(
    store: EvaluationStore,
    *,
    signal_discovery_id: str,
) -> SubjectSet | None:
    discovery_state = store.get_signal_discovery_spec(signal_discovery_id)
    if discovery_state is None:
        return None
    subject_set_state = store.get_subject_set(discovery_state.definition.subject_set_id)
    if subject_set_state is None:
        return None
    validate_subject_set_universe_contract(subject_set_state.definition)
    return subject_set_state.definition


def build_runtime_portfolio_state(
    store: EvaluationStore,
    *,
    portfolio_id: str,
    aggregation_kind: str,
) -> PortfolioState:
    decisions = store.get_latest_portfolio_decisions(
        portfolio_id=portfolio_id,
        aggregation_kind=aggregation_kind,
    )
    if not decisions:
        return PortfolioState(portfolio_id=portfolio_id, positions=())
    state_snapshot = portfolio_state_from_decision_details(decisions[0].details)
    next_as_of = decisions[0].as_of
    return PortfolioState(
        portfolio_id=portfolio_id,
        as_of=next_as_of,
        positions=tuple(
            PortfolioPositionState(
                subject_id=item.subject_id,
                weight=item.target_weight,
                notional=item.target_notional,
                quantity=item.target_quantity,
            )
            for item in decisions
        ),
        capital_base=(
            1.0 if state_snapshot is None else state_snapshot.capital_base
        ),
        gross_limit=(
            None if state_snapshot is None else state_snapshot.gross_limit
        ),
        net_limit=(
            None if state_snapshot is None else state_snapshot.net_limit
        ),
        rebalance_step=(
            1 if state_snapshot is None else state_snapshot.rebalance_step + 1
        ),
        holding_period_days=holding_period_days(
            previous_as_of=None if state_snapshot is None else state_snapshot.as_of,
            next_as_of=next_as_of,
        ),
        recent_turnover=(
            float(sum(abs(item.position_delta) for item in decisions))
            if state_snapshot is None
            else state_snapshot.recent_turnover
        ),
        current_drawdown=(
            0.0 if state_snapshot is None else state_snapshot.current_drawdown
        ),
    )


def persist_portfolio_decision_output(
    store: EvaluationStore,
    *,
    decision_output: PortfolioDecisionOutput,
    target_id: str,
    aggregation_kind: str,
    portfolio_state: PortfolioState | None = None,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
    decision_input: PortfolioDecisionInput | None = None,
    sizing_method: str | None = None,
    sizing_engine: str | None = None,
    trading_strategy: TradingStrategySpec | None = None,
    recorded_at: str | None = None,
) -> None:
    config = config or RuntimeDecisionBuildConfig()
    assumptions = assumptions or PortfolioDecisionAssumptions()
    portfolio_id = decision_output.portfolio_id or "default"
    as_of = decision_output.as_of or ""
    details_json = json.dumps(
        {
            "config": asdict(config),
            "portfolio_state": None
            if portfolio_state is None
            else asdict(portfolio_state),
            "sizing_method": sizing_method,
            "sizing_engine": sizing_engine,
            "strategy": None
            if trading_strategy is None
            else _trading_strategy_trace_document(trading_strategy),
            "observed_inputs": None
            if decision_input is None
            else {
                "predictive_signals": [
                    asdict(item)
                    for item in decision_input.observed_inputs.predictive_signals
                ],
                "risk_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.risk_inputs
                ],
                "cost_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.cost_inputs
                ],
                "uncertainty_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.uncertainty_inputs
                ],
                "model_uncertainty_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.model_uncertainty_inputs
                ],
                "structural_uncertainty_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.structural_uncertainty_inputs
                ],
                "dependence_inputs": [
                    asdict(item)
                    for item in decision_input.observed_inputs.dependence_inputs
                ],
            },
            "input_summary": None
            if decision_input is None
            else _decision_input_summary(decision_input),
            "assumptions": {
                "risk_inputs": [asdict(item) for item in assumptions.risk_inputs],
                "cost_inputs": [asdict(item) for item in assumptions.cost_inputs],
                "uncertainty_inputs": [
                    asdict(item) for item in assumptions.uncertainty_inputs
                ],
                "model_uncertainty_inputs": [
                    asdict(item) for item in assumptions.model_uncertainty_inputs
                ],
                "structural_uncertainty_inputs": [
                    asdict(item) for item in assumptions.structural_uncertainty_inputs
                ],
                "dependence_inputs": [
                    asdict(item) for item in assumptions.dependence_inputs
                ],
            },
        },
        sort_keys=True,
    )
    for target in decision_output.targets:
        store.upsert_portfolio_decision(
            portfolio_id=portfolio_id,
            subject_id=target.subject_id,
            target_id=target_id,
            aggregation_kind=aggregation_kind,
            as_of=as_of,
            target_weight=target.target_weight,
            position_delta=target.position_delta,
            target_notional=target.target_notional,
            target_quantity=target.target_quantity,
            entry_allowed=target.entry_allowed,
            risk_scale=target.risk_scale,
            details_json=details_json,
            recorded_at=recorded_at,
        )


def _decision_input_summary(
    decision_input: PortfolioDecisionInput,
) -> dict[str, object]:
    subject_ids = sorted(
        {
            *(signal.subject_id for signal in decision_input.predictive_signals),
            *(
                risk_input.subject_id
                for risk_input in decision_input.risk_inputs
                if risk_input.subject_id is not None
            ),
            *(
                cost_input.subject_id
                for cost_input in decision_input.cost_inputs
                if cost_input.subject_id is not None
            ),
            *(
                uncertainty_input.subject_id
                for uncertainty_input in decision_input.uncertainty_inputs
                if uncertainty_input.subject_id is not None
            ),
            *(
                model_uncertainty_input.subject_id
                for model_uncertainty_input in decision_input.model_uncertainty_inputs
                if model_uncertainty_input.subject_id is not None
            ),
            *(
                structural_uncertainty_input.subject_id
                for structural_uncertainty_input in decision_input.structural_uncertainty_inputs
                if structural_uncertainty_input.subject_id is not None
            ),
        }
    )
    return {
        "subjects": {
            subject_id: {
                "predictive_signal": _subject_signal_summary(
                    decision_input,
                    subject_id,
                ),
                "risk_inputs": _named_subject_values(
                    decision_input.risk_inputs,
                    subject_id,
                ),
                "cost_inputs": _named_subject_values(
                    decision_input.cost_inputs,
                    subject_id,
                ),
                "uncertainty_inputs": _named_subject_values(
                    decision_input.uncertainty_inputs,
                    subject_id,
                    value_attr="estimate_std",
                ),
                "uncertainty_proxies": _uncertainty_proxy_summary(
                    decision_input.uncertainty_inputs,
                    subject_id,
                ),
                "model_uncertainty_inputs": _named_subject_values(
                    decision_input.model_uncertainty_inputs,
                    subject_id,
                    value_attr="model_error",
                ),
                "model_uncertainty_proxies": _proxy_component_summary(
                    decision_input.model_uncertainty_inputs,
                    subject_id,
                ),
                "structural_uncertainty_inputs": _named_subject_values(
                    decision_input.structural_uncertainty_inputs,
                    subject_id,
                    value_attr="structural_error",
                ),
                "structural_uncertainty_proxies": _proxy_component_summary(
                    decision_input.structural_uncertainty_inputs,
                    subject_id,
                ),
            }
            for subject_id in subject_ids
        },
        "dependence_inputs": [
            {
                "name": item.name,
                "left_subject_id": item.left_subject_id,
                "right_subject_id": item.right_subject_id,
                "value": item.value,
                "basis": item.basis,
            }
            for item in decision_input.dependence_inputs
        ],
    }


def _subject_signal_summary(
    decision_input: PortfolioDecisionInput,
    subject_id: str,
) -> dict[str, float | None]:
    signals = [
        signal
        for signal in decision_input.predictive_signals
        if signal.subject_id == subject_id
    ]
    if not signals:
        return {"value": None, "confidence": None}
    total_weight = sum(
        max(signal.confidence if signal.confidence is not None else 1.0, 0.0)
        for signal in signals
    )
    if total_weight <= 0.0:
        mean_value = 0.0
    else:
        mean_value = sum(
            signal.value
            * max(signal.confidence if signal.confidence is not None else 1.0, 0.0)
            for signal in signals
        ) / total_weight
    confidence_values = [
        signal.confidence
        for signal in signals
        if signal.confidence is not None
    ]
    mean_confidence = (
        None
        if not confidence_values
        else sum(confidence_values) / len(confidence_values)
    )
    return {
        "value": float(mean_value),
        "confidence": (
            None if mean_confidence is None else float(mean_confidence)
        ),
    }


def _named_subject_values(
    items: tuple[object, ...],
    subject_id: str,
    *,
    value_attr: str = "value",
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        item_subject_id = getattr(item, "subject_id", None)
        if item_subject_id != subject_id:
            continue
        name = getattr(item, "name", None)
        if not isinstance(name, str):
            name = getattr(item, "source_id", None)
        value = getattr(item, value_attr, None)
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            continue
        grouped.setdefault(name, []).append(float(value))
    return {
        name: float(sum(values) / len(values))
        for name, values in grouped.items()
    }


def _uncertainty_proxy_summary(
    uncertainty_inputs: tuple[object, ...],
    subject_id: str,
) -> dict[str, float]:
    return _proxy_component_summary(uncertainty_inputs, subject_id)


def _proxy_component_summary(
    inputs: tuple[object, ...],
    subject_id: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in inputs:
        item_subject_id = getattr(item, "subject_id", None)
        if item_subject_id != subject_id:
            continue
        proxy_components = getattr(item, "proxy_components", None)
        if not isinstance(proxy_components, dict):
            continue
        for name, value in proxy_components.items():
            if not isinstance(name, str) or not isinstance(value, (int, float)):
                continue
            grouped.setdefault(name, []).append(float(value))
    return {
        name: float(sum(values) / len(values))
        for name, values in grouped.items()
    }


def _latest_meta_prediction(
    store: EvaluationStore,
    *,
    subject_id: str,
    asset: str,
    target_id: str,
    aggregation_kind: str,
) -> MetaPredictionState | None:
    items = store.list_meta_predictions(
        subject_id=subject_id,
        asset=asset,
        target_id=target_id,
        aggregation_kind=aggregation_kind,
        limit=1,
    )
    if not items:
        items = store.list_meta_predictions(
            asset=asset,
            target_id=target_id,
            aggregation_kind=aggregation_kind,
            limit=1,
        )
    return items[0] if items else None


def _meta_metric(
    store: EvaluationStore,
    *,
    subject_id: str,
    asset: str,
    target_id: str,
    aggregation_kind: str,
) -> MetaPredictionMetricState | None:
    items = store.list_meta_prediction_metrics(
        subject_id=subject_id,
        asset=asset,
        target_id=target_id,
    )
    if not items:
        items = store.list_meta_prediction_metrics(
            asset=asset,
            target_id=target_id,
        )
    for item in items:
        if item.aggregation_kind == aggregation_kind:
            return item
    return None


def _realized_observation_volatility(
    store: EvaluationStore,
    *,
    subject_id: str,
    asset: str,
    target_id: str,
    window_size: int,
) -> float:
    rows = store.conn.execute(
        """
        SELECT value
        FROM observations
        WHERE subject_id = ? AND target_id = ?
        ORDER BY evaluation_id DESC
        LIMIT ?
        """,
        (subject_id, target_id, int(window_size)),
    ).fetchall()
    if not rows:
        rows = store.conn.execute(
            """
            SELECT value
            FROM observations
            WHERE asset = ? AND target_id = ?
            ORDER BY evaluation_id DESC
            LIMIT ?
            """,
            (asset, target_id, int(window_size)),
        ).fetchall()
    values = [float(row["value"]) for row in rows]
    return realized_observation_volatility(values)


def _observed_dependence_inputs(
    store: EvaluationStore,
    *,
    subject_set: SubjectSet,
    target_id: str,
    portfolio_state: PortfolioState,
    config: RuntimeDecisionBuildConfig,
) -> tuple[DependenceInput, ...]:
    subject_asset_map = subject_set.asset_by_subject
    observation_series_by_subject: dict[str, dict[str, float]] = {}
    decision_subject_ids = subject_set.subject_ids
    for observed_subject_id in {
        *decision_subject_ids,
        *(position.subject_id for position in portfolio_state.positions),
    }:
        runtime_subject_asset = subject_asset_map.get(observed_subject_id)
        if runtime_subject_asset is None:
            continue
        series = _observation_series_by_date(
            store,
            subject_id=observed_subject_id,
            asset=runtime_subject_asset,
            target_id=target_id,
            window_size=config.dependence_window,
        )
        if series:
            observation_series_by_subject[observed_subject_id] = series
    return build_runtime_observed_dependence_inputs(
        subject_ids=tuple(
            subject_id
            for subject_id in sorted(
                set(decision_subject_ids)
                | {
                    position.subject_id
                    for position in portfolio_state.positions
                    if position.subject_id in observation_series_by_subject
                }
            )
        ),
        observation_series_by_subject=observation_series_by_subject,
    )


def _resolved_subject_set(
    *,
    runtime_asset: str,
    subject_id: str,
    configured_subject_set: SubjectSet | None,
) -> SubjectSet:
    observation_specs: list[ObservationSpec] = [
        ObservationSpec(
            observation_spec_id="__runtime__",
            observable_id="daily_close",
            source_id="runtime",
        )
    ]
    if configured_subject_set is not None:
        subject_set_id = configured_subject_set.subject_set_id
        configured_bindings = configured_subject_set.bindings
        configured_instruments = configured_subject_set.instruments
        universe_policy = configured_subject_set.universe_policy
        configured_primary = next(
            (
                item
                for item in configured_bindings
                if item.subject_id == subject_id
            ),
            None,
        )
        for item in configured_subject_set.observation_specs:
            if item.observation_spec_id == "__runtime__":
                continue
            observation_specs.append(item)
    else:
        subject_set_id = None
        configured_bindings = ()
        configured_instruments = ()
        configured_primary = None
        from .portfolio_decision import UniversePolicySpec

        universe_policy = UniversePolicySpec()
    if configured_primary is None:
        ordered: list[SubjectObservationBinding] = [
            SubjectObservationBinding(
                subject_id=subject_id,
                asset=runtime_asset,
                observation_spec_id="__runtime__",
            )
        ]
    else:
        ordered = [configured_primary]
    seen_subject_ids = {subject_id}
    for item in configured_bindings:
        if item.subject_id in seen_subject_ids:
            continue
        ordered.append(item)
        seen_subject_ids.add(item.subject_id)
    resolved_subject_set = SubjectSet(
        subject_set_id=subject_set_id,
        instruments=tuple(configured_instruments),
        observation_specs=tuple(observation_specs),
        bindings=tuple(ordered),
        universe_policy=universe_policy,
    )
    validate_subject_set_universe_contract(resolved_subject_set)
    return resolved_subject_set


def _observation_series_by_date(
    store: EvaluationStore,
    *,
    subject_id: str,
    asset: str,
    target_id: str,
    window_size: int,
) -> dict[str, float]:
    rows = store.conn.execute(
        """
        SELECT evaluation_id, value
        FROM observations
        WHERE subject_id = ? AND target_id = ?
        ORDER BY evaluation_id DESC
        LIMIT ?
        """,
        (subject_id, target_id, int(window_size)),
    ).fetchall()
    if not rows:
        rows = store.conn.execute(
            """
            SELECT evaluation_id, value
            FROM observations
            WHERE asset = ? AND target_id = ?
            ORDER BY evaluation_id DESC
            LIMIT ?
            """,
            (asset, target_id, int(window_size)),
        ).fetchall()
    series: dict[str, float] = {}
    for row in rows:
        evaluation_id = str(row["evaluation_id"])
        date = evaluation_id.rsplit(":", 1)[-1]
        series[date] = float(row["value"])
    return series


def _historical_return_inputs(
    *,
    observation_series_by_subject: dict[str, dict[str, float]],
    subject_ids: tuple[str, ...],
) -> tuple[HistoricalReturnInput, ...]:
    inputs: list[HistoricalReturnInput] = []
    for subject_id in subject_ids:
        series = observation_series_by_subject.get(subject_id)
        if not series:
            continue
        inputs.append(
            HistoricalReturnInput(
                subject_id=subject_id,
                returns_by_date={
                    date: float(value)
                    for date, value in sorted(series.items())
                },
            )
        )
    return tuple(inputs)
