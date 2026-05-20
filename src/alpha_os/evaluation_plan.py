from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .evaluation_task import EvaluationTask
from .evaluation_spec import (
    EvaluationSpec,
    EvaluationDateRange,
)
from .strategy_engine import (
    StrategyEvaluationContext,
    StrategyEvaluationRequest,
)

if TYPE_CHECKING:
    from .store import (
        TradingStrategyState,
    )


class EvaluationRequestBuildReadPort(Protocol):
    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None: ...


DIRECT_STRATEGY_POSITION_RULE_IDS = frozenset(
    {
        "constant_hold",
        "dual_momentum_hold",
        "crypto_regime_momentum_hold",
    }
)


def _strategy_evaluation_request(
    *,
    evaluation_task_id: str,
    evaluation_spec_id: str,
    fold_label: str,
    strategy_id: str,
    target_id: str,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
) -> StrategyEvaluationRequest:
    return StrategyEvaluationRequest(
        evaluation_task_id=evaluation_task_id,
        evaluation_spec_id=evaluation_spec_id,
        fold_label=fold_label,
        context=StrategyEvaluationContext(
            strategy_id=strategy_id,
            target_id=target_id,
            base_url=base_url,
        ),
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=metric_group_names,
    )


def build_strategy_evaluation_requests(
    store: EvaluationRequestBuildReadPort,
    *,
    evaluation_spec_id: str,
    evaluation_spec: EvaluationSpec,
    evaluation_tasks: tuple[EvaluationTask, ...] | None = None,
    base_url: str,
) -> tuple[StrategyEvaluationRequest, ...]:
    execution_requests: list[StrategyEvaluationRequest] = []
    if evaluation_tasks is None:
        raise ValueError("strategy evaluation request builder requires evaluation_tasks")
    for evaluation_task in evaluation_tasks:
        strategy_state = store.get_trading_strategy(evaluation_task.strategy_id)
        if strategy_state is None:
            raise ValueError(
                "evaluation task strategy does not exist: "
                f"{evaluation_task.strategy_id}"
            )
        trading_strategy = strategy_state.trading_strategy
        subject_set_id = trading_strategy.subject_set_id
        target_id = trading_strategy.target_id
        if (
            isinstance(subject_set_id, str)
            and subject_set_id
            and isinstance(target_id, str)
            and target_id
            and trading_strategy.position_rule_id in DIRECT_STRATEGY_POSITION_RULE_IDS
        ):
            for fold in evaluation_spec.resolved_evaluation_folds:
                execution_requests.append(
                    _strategy_evaluation_request(
                        evaluation_task_id=evaluation_task.evaluation_task_id,
                        evaluation_spec_id=evaluation_spec_id,
                        fold_label=fold.label,
                        strategy_id=evaluation_task.strategy_id,
                        target_id=target_id,
                        execution_range=fold.execution_range,
                        evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                        metric_group_names=evaluation_spec.metric_group_names,
                        base_url=base_url,
                    )
                )
            continue
        if not isinstance(subject_set_id, str) or not subject_set_id:
            raise ValueError(
                "direct evaluation task requires strategy subject_set: "
                f"{evaluation_task.evaluation_task_id}"
            )
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(
                "direct evaluation task requires strategy prediction target: "
                f"{evaluation_task.evaluation_task_id}"
            )
        raise ValueError(
            "strategy evaluation request builder does not resolve checkpoints: "
            f"{evaluation_task.evaluation_task_id}"
        )
    return tuple(execution_requests)
