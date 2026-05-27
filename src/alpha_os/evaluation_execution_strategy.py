from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .data_repositories import FeaturePlaneRepository
from .evaluation_spec import EvaluationDateRange
from .strategy_backtest import run_strategy_backtest_from_store
from .evaluation_result import EvaluationResult
from .universe_contract import validate_subject_set_universe_contract


class EvaluationExecutionReadPort(Protocol):
    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_subject_set(self, subject_set_id: str):
        ...


@dataclass(frozen=True)
class EvaluationExecutionContext:
    store: EvaluationExecutionReadPort
    feature_plane_repository: FeaturePlaneRepository | None = None


def _required_strategy_field(
    *,
    value: str | None,
    strategy_id: str,
    label: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation strategy is missing {label}: {strategy_id}")
    return value


def _strategy_state_for_id(
    store: EvaluationExecutionReadPort,
    strategy_id: str,
):
    strategy_state = store.get_trading_strategy(strategy_id)
    if strategy_state is None:
        raise ValueError(f"evaluation strategy does not exist: {strategy_id}")
    return strategy_state


def run_strategy_evaluation(
    *,
    strategy_id: str,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
    context: EvaluationExecutionContext,
) -> EvaluationResult:
    store = context.store
    strategy_state = _strategy_state_for_id(store, strategy_id)
    trading_strategy = strategy_state.trading_strategy
    subject_set_id = _required_strategy_field(
        value=trading_strategy.subject_set_id,
        strategy_id=trading_strategy.strategy_id,
        label="subject_set",
    )
    target_id = _required_strategy_field(
        value=trading_strategy.target_id,
        strategy_id=trading_strategy.strategy_id,
        label="prediction target",
    )
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is not None:
        validate_subject_set_universe_contract(subject_set_state.definition)
    direct_evaluation = run_strategy_backtest_from_store(
        store=store,
        strategy_id=strategy_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        evaluation_date_ranges=evaluation_date_ranges,
        base_url=base_url,
        portfolio_construction=trading_strategy.portfolio_construction,
        trading_environment=trading_strategy.trading_environment,
        feature_plane_repository=context.feature_plane_repository,
    )
    direct_metric_group_results, direct_failure_finding_groups = direct_evaluation
    return EvaluationResult(
        strategy_id=strategy_id,
        metric_group_results=tuple(
            direct_metric_group_results[metric_group_name]
            for metric_group_name in metric_group_names
            if metric_group_name in direct_metric_group_results
        ),
        failure_finding_groups=direct_failure_finding_groups,
    )
