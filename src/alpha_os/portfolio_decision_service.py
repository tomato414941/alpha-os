from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .meta_aggregation_service import AGGREGATION_CORR_WEIGHTED_MEAN
from .portfolio_decision import (
    CostInput,
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
    turnover_penalty: float = 0.01
    expected_slippage_bps: float = 10.0
    no_trade_band: float = 0.0


def build_portfolio_decision_input(
    store: EvaluationStore,
    *,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    portfolio_state: PortfolioState | None = None,
    config: RuntimeDecisionBuildConfig | None = None,
) -> PortfolioDecisionInput | None:
    config = config or RuntimeDecisionBuildConfig()
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

    return PortfolioDecisionInput(
        asset=asset,
        as_of=meta_prediction.updated_at,
        portfolio_state=portfolio_state or PortfolioState(asset=asset),
        predictive_signals=(
            PredictiveSignalInput(
                source_id=config.aggregation_kind,
                source_kind="meta_prediction",
                subject_id=asset,
                target_id=target_id,
                value=meta_prediction.value,
                confidence=confidence,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name=f"realized_vol_{config.risk_window}",
                subject_id=asset,
                value=realized_vol,
                horizon_days=config.risk_window,
                unit="vol",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="turnover_penalty",
                subject_id=None,
                value=config.turnover_penalty,
                basis="per_turnover",
                unit="weight",
            ),
            CostInput(
                name="expected_slippage",
                subject_id=asset,
                value=config.expected_slippage_bps,
                basis="per_notional",
                unit="bps",
            ),
            CostInput(
                name="no_trade_band",
                subject_id=asset,
                value=config.no_trade_band,
                basis="per_delta_weight",
                unit="weight",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                name="small_sample_penalty",
                subject_id=asset,
                value=uncertainty_value,
                source_id=config.aggregation_kind,
                basis="per_signal",
            ),
        ),
    )


def build_portfolio_decision_output(
    store: EvaluationStore,
    *,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    portfolio_state: PortfolioState | None = None,
    config: RuntimeDecisionBuildConfig | None = None,
    policy: RuleBasedPortfolioPolicy | None = None,
) -> PortfolioDecisionOutput | None:
    decision_input = build_portfolio_decision_input(
        store,
        asset=asset,
        target_id=target_id,
        portfolio_state=portfolio_state,
        config=config,
    )
    if decision_input is None:
        return None
    return apply_rule_based_policy(decision_input, policy=policy)


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
