from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .evaluation_task import EvaluationTask
from .evaluation_spec import (
    EvaluationSpec,
    EvaluationDateRange,
)
from .strategy_engine import (
    StrategyEvaluationInputRefs,
    StrategyEvaluationContext,
    StrategyEvaluationRequest,
)

if TYPE_CHECKING:
    from .store import (
        StrategyCheckpointRecord,
        TradingStrategyState,
    )


class EvaluationPlanReadPort(Protocol):
    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None: ...

    def get_strategy_checkpoint(
        self,
        strategy_checkpoint_id: str,
    ) -> StrategyCheckpointRecord | None: ...

    def list_strategy_checkpoints(
        self,
        *,
        strategy_id: str | None = None,
        signal_discovery_id: str | None = None,
        fold_label: str | None = None,
        execution_start_date: str | None = None,
        execution_end_date: str | None = None,
        limit: int = 20,
    ) -> list[StrategyCheckpointRecord]: ...


@dataclass(frozen=True)
class EvaluationPlan:
    evaluation_spec_id: str
    metric_group_names: tuple[str, ...]
    execution_requests: tuple[StrategyEvaluationRequest, ...]


def _strategy_evaluation_request(
    *,
    evaluation_task_id: str,
    evaluation_spec_id: str,
    fold_label: str,
    strategy_id: str,
    strategy_checkpoint_id: str | None,
    target_id: str,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
) -> StrategyEvaluationRequest:
    input_refs = None
    if strategy_checkpoint_id is not None:
        input_refs = StrategyEvaluationInputRefs(
            strategy_checkpoint_id=strategy_checkpoint_id,
        )
    return StrategyEvaluationRequest(
        evaluation_task_id=evaluation_task_id,
        evaluation_spec_id=evaluation_spec_id,
        fold_label=fold_label,
        context=StrategyEvaluationContext(
            strategy_id=strategy_id,
            target_id=target_id,
            base_url=base_url,
        ),
        input_refs=input_refs,
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=metric_group_names,
    )


def _resolve_strategy_checkpoint_for_fold(
    store: EvaluationPlanReadPort,
    *,
    strategy_id: str,
    signal_discovery_id: str,
    fold,
) -> StrategyCheckpointRecord | None:
    strategy_checkpoints = store.list_strategy_checkpoints(
        strategy_id=strategy_id,
        signal_discovery_id=signal_discovery_id,
        fold_label=fold.label,
        execution_start_date=fold.execution_range.start_date,
        execution_end_date=fold.execution_range.end_date,
        limit=1,
    )
    if strategy_checkpoints:
        return strategy_checkpoints[0]
    strategy_checkpoints = store.list_strategy_checkpoints(
        strategy_id=strategy_id,
        signal_discovery_id=signal_discovery_id,
        execution_start_date=fold.execution_range.start_date,
        execution_end_date=fold.execution_range.end_date,
        limit=1,
    )
    if not strategy_checkpoints:
        return None
    return strategy_checkpoints[0]


def build_evaluation_plan(
    store: EvaluationPlanReadPort,
    *,
    evaluation_spec_id: str,
    evaluation_spec: EvaluationSpec,
    evaluation_tasks: tuple[EvaluationTask, ...] | None = None,
    default_target_id: str,
    base_url: str,
) -> EvaluationPlan:
    execution_requests: list[StrategyEvaluationRequest] = []
    if evaluation_tasks is None:
        raise ValueError("evaluation plan requires evaluation_tasks")
    for evaluation_task in evaluation_tasks:
        strategy_state = store.get_trading_strategy(evaluation_task.strategy_id)
        if strategy_state is None:
            raise ValueError(
                "evaluation task strategy does not exist: "
                f"{evaluation_task.strategy_id}"
            )
        trading_strategy = strategy_state.trading_strategy
        strategy_signal_discovery_id = trading_strategy.signal_discovery_id
        if strategy_signal_discovery_id is None:
            subject_set_id = trading_strategy.subject_set_id
            if not isinstance(subject_set_id, str) or not subject_set_id:
                raise ValueError(
                    "direct evaluation task requires strategy subject_set: "
                    f"{evaluation_task.evaluation_task_id}"
                )
            target_id = trading_strategy.target_id or default_target_id
            for fold in evaluation_spec.resolved_evaluation_folds:
                execution_requests.append(
                    _strategy_evaluation_request(
                        evaluation_task_id=evaluation_task.evaluation_task_id,
                        evaluation_spec_id=evaluation_spec_id,
                        fold_label=fold.label,
                        strategy_id=evaluation_task.strategy_id,
                        strategy_checkpoint_id=None,
                        target_id=target_id,
                        execution_range=fold.execution_range,
                        evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                        metric_group_names=evaluation_spec.metric_group_names,
                        base_url=base_url,
                    )
                )
            continue
        for fold in evaluation_spec.resolved_evaluation_folds:
            strategy_checkpoint_record = _resolve_strategy_checkpoint_for_fold(
                store,
                strategy_id=evaluation_task.strategy_id,
                signal_discovery_id=strategy_signal_discovery_id,
                fold=fold,
            )
            if strategy_checkpoint_record is not None:
                strategy_checkpoint = strategy_checkpoint_record.state
                strategy_checkpoint_id = (
                    strategy_checkpoint.strategy_checkpoint_id
                )
                target_id = strategy_checkpoint.target_id
            else:
                raise ValueError(
                    "checkpoint evaluation task requires a strategy checkpoint for "
                    f"{evaluation_task.evaluation_task_id} "
                    f"{fold.execution_range.start_date}->{fold.execution_range.end_date}"
                )
            execution_requests.append(
                _strategy_evaluation_request(
                    evaluation_task_id=evaluation_task.evaluation_task_id,
                    evaluation_spec_id=evaluation_spec_id,
                    fold_label=fold.label,
                    strategy_id=evaluation_task.strategy_id,
                    strategy_checkpoint_id=strategy_checkpoint_id,
                    target_id=target_id,
                    execution_range=fold.execution_range,
                    evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                    metric_group_names=evaluation_spec.metric_group_names,
                    base_url=base_url,
                )
            )
    return EvaluationPlan(
        evaluation_spec_id=evaluation_spec_id,
        metric_group_names=evaluation_spec.metric_group_names,
        execution_requests=tuple(execution_requests),
    )
