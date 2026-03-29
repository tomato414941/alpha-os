from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from statistics import pstdev

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .meta_aggregation_service import AGGREGATION_CORR_WEIGHTED_MEAN
from .portfolio_decision import (
    CostInput,
    DependenceInput,
    PortfolioDecisionInput,
    PortfolioDecisionOutput,
    PortfolioState,
    PredictiveSignalInput,
    RiskInput,
    UncertaintyInput,
)
from .portfolio_decision_policy import (
    RuleBasedPortfolioPolicy,
    apply_rule_based_policy,
)
from .store import EvaluationStore, MetaPredictionMetricState, MetaPredictionState


@dataclass(frozen=True)
class RuntimeDecisionBuildConfig:
    aggregation_kind: str = AGGREGATION_CORR_WEIGHTED_MEAN
    risk_window: int = 20


@dataclass(frozen=True)
class PortfolioDecisionAssumptions:
    risk_inputs: tuple[RiskInput, ...] = ()
    cost_inputs: tuple[CostInput, ...] = ()
    uncertainty_inputs: tuple[UncertaintyInput, ...] = ()
    dependence_inputs: tuple[DependenceInput, ...] = ()


def build_portfolio_decision_input(
    store: EvaluationStore,
    *,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    portfolio_id: str | None = None,
    subject_id: str | None = None,
    portfolio_state: PortfolioState | None = None,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
) -> PortfolioDecisionInput | None:
    config = config or RuntimeDecisionBuildConfig()
    assumptions = assumptions or PortfolioDecisionAssumptions()
    resolved_subject_id = subject_id or asset
    meta_prediction = _latest_meta_prediction(
        store,
        asset=asset,
        target_id=target_id,
        aggregation_kind=config.aggregation_kind,
    )
    if meta_prediction is None:
        return None

    metric = _meta_metric(
        store,
        asset=asset,
        target_id=target_id,
        aggregation_kind=config.aggregation_kind,
    )
    confidence = _signal_confidence(metric)
    uncertainty_value = _uncertainty_level(metric)
    realized_vol = _realized_observation_volatility(
        store,
        asset=asset,
        target_id=target_id,
        window_size=config.risk_window,
    )

    runtime_risk_inputs = (
        RiskInput(
            name=f"realized_vol_{config.risk_window}",
            subject_id=resolved_subject_id,
            value=realized_vol,
            horizon_days=config.risk_window,
            unit="vol",
        ),
    )
    runtime_uncertainty_inputs = (
        UncertaintyInput(
            name="small_sample_penalty",
            subject_id=resolved_subject_id,
            value=uncertainty_value,
            source_id=config.aggregation_kind,
            basis="per_signal",
        ),
    )

    return PortfolioDecisionInput(
        portfolio_id=portfolio_id,
        asset=asset,
        as_of=meta_prediction.updated_at,
        portfolio_state=portfolio_state
        or PortfolioState(portfolio_id=portfolio_id, asset=asset),
        predictive_signals=(
            PredictiveSignalInput(
                source_id=config.aggregation_kind,
                source_kind="meta_prediction",
                subject_id=resolved_subject_id,
                target_id=target_id,
                value=meta_prediction.value,
                confidence=confidence,
            ),
        ),
        risk_inputs=runtime_risk_inputs + assumptions.risk_inputs,
        cost_inputs=assumptions.cost_inputs,
        uncertainty_inputs=runtime_uncertainty_inputs + assumptions.uncertainty_inputs,
        dependence_inputs=assumptions.dependence_inputs,
    )


def build_portfolio_decision_output(
    store: EvaluationStore,
    *,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    portfolio_id: str | None = None,
    subject_id: str | None = None,
    portfolio_state: PortfolioState | None = None,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
    policy: RuleBasedPortfolioPolicy | None = None,
) -> PortfolioDecisionOutput | None:
    decision_input = build_portfolio_decision_input(
        store,
        asset=asset,
        target_id=target_id,
        portfolio_id=portfolio_id,
        subject_id=subject_id,
        portfolio_state=portfolio_state,
        config=config,
        assumptions=assumptions,
    )
    if decision_input is None:
        return None
    decision_output = apply_rule_based_policy(decision_input, policy=policy)
    return PortfolioDecisionOutput(
        portfolio_id=portfolio_id,
        asset=asset,
        as_of=decision_output.as_of,
        targets=decision_output.targets,
    )


def persist_portfolio_decision_output(
    store: EvaluationStore,
    *,
    decision_output: PortfolioDecisionOutput,
    target_id: str,
    aggregation_kind: str,
    config: RuntimeDecisionBuildConfig | None = None,
    assumptions: PortfolioDecisionAssumptions | None = None,
    recorded_at: str | None = None,
) -> None:
    config = config or RuntimeDecisionBuildConfig()
    assumptions = assumptions or PortfolioDecisionAssumptions()
    portfolio_id = decision_output.portfolio_id or "default"
    asset = decision_output.asset
    as_of = decision_output.as_of or ""
    details_json = json.dumps(
        {
            "config": asdict(config),
            "assumptions": {
                "risk_inputs": [asdict(item) for item in assumptions.risk_inputs],
                "cost_inputs": [asdict(item) for item in assumptions.cost_inputs],
                "uncertainty_inputs": [
                    asdict(item) for item in assumptions.uncertainty_inputs
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
            asset=asset,
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


def _latest_meta_prediction(
    store: EvaluationStore,
    *,
    asset: str,
    target_id: str,
    aggregation_kind: str,
) -> MetaPredictionState | None:
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
    asset: str,
    target_id: str,
    aggregation_kind: str,
) -> MetaPredictionMetricState | None:
    items = store.list_meta_prediction_metrics(asset=asset, target_id=target_id)
    for item in items:
        if item.aggregation_kind == aggregation_kind:
            return item
    return None


def _signal_confidence(metric: MetaPredictionMetricState | None) -> float | None:
    if metric is None:
        return None
    return float(min(max(metric.corr, 0.0), 1.0))


def _uncertainty_level(metric: MetaPredictionMetricState | None) -> float:
    if metric is None:
        return 1.0
    if metric.window_size <= 0:
        return 1.0
    coverage = min(float(metric.sample_count) / float(metric.window_size), 1.0)
    return float(max(0.0, 1.0 - coverage))


def _realized_observation_volatility(
    store: EvaluationStore,
    *,
    asset: str,
    target_id: str,
    window_size: int,
) -> float:
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
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))
