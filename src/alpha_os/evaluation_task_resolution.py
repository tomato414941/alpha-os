from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .evaluation_task import (
    EvaluationTask,
    build_evaluation_task_id,
)
from .store import EvaluationStore
from .strategy_variant import (
    derive_trading_strategy_from_signal_discovery,
    overridden_strategy_variant_config,
    strategy_variant_config_from_strategy,
)
from .trading_strategy import TradingStrategySpec


class EvaluationTaskResolutionReadPort(Protocol):
    def list_evaluation_tasks(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 100,
    ):
        ...

    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_signal_discovery_spec(self, signal_discovery_id: str):
        ...


class EvaluationTaskResolutionWritePort(Protocol):
    def upsert_trading_strategy(self, *, trading_strategy: TradingStrategySpec):
        ...

    def upsert_evaluation_task(self, *, task: EvaluationTask):
        ...


@dataclass(frozen=True)
class EvaluationTaskResolutionRequest:
    evaluation_spec_id: str
    sizing_method: str | None
    sizing_engine: str | None
    strategy_ids: tuple[str, ...] | None
    created_at: str
    direction_mode: str | None = None
    evaluation_task_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EvaluationTaskResolutionEntry:
    task: EvaluationTask
    source_task: EvaluationTask | None = None
    trading_strategy_to_persist: TradingStrategySpec | None = None
    task_to_persist: EvaluationTask | None = None
    resolution_action: str = "existing"
    reason: str = "existing"


@dataclass(frozen=True)
class EvaluationTaskResolutionPlan:
    entries: tuple[EvaluationTaskResolutionEntry, ...]

    @property
    def pending_write_count(self) -> int:
        return sum(
            int(entry.trading_strategy_to_persist is not None)
            + int(entry.task_to_persist is not None)
            for entry in self.entries
        )

    @property
    def has_pending_writes(self) -> bool:
        return self.pending_write_count > 0

    @property
    def tasks(self) -> tuple[EvaluationTask, ...]:
        unique_tasks: dict[str, EvaluationTask] = {}
        for entry in self.entries:
            unique_tasks.setdefault(entry.task.evaluation_task_id, entry.task)
        return tuple(
            sorted(
                unique_tasks.values(),
                key=lambda item: (
                    item.strategy_id,
                    item.evaluation_task_id,
                ),
            )
        )


def build_evaluation_task_resolution_plan(
    read_port: EvaluationTaskResolutionReadPort,
    request: EvaluationTaskResolutionRequest,
) -> EvaluationTaskResolutionPlan:
    evaluation_spec_id = request.evaluation_spec_id
    existing_tasks = tuple(
        state.task
        for state in read_port.list_evaluation_tasks(
            evaluation_spec_id=evaluation_spec_id,
            limit=10_000,
        )
    )
    if not existing_tasks:
        raise ValueError(
            "evaluation spec requires at least one evaluation task: "
            f"{evaluation_spec_id}"
        )
    if request.evaluation_task_ids is not None:
        allowed_task_ids = set(request.evaluation_task_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.evaluation_task_id in allowed_task_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested evaluation tasks: "
                f"{evaluation_spec_id}"
            )
    if request.strategy_ids:
        allowed_strategy_ids = set(request.strategy_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.strategy_id in allowed_strategy_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested strategies: "
                f"{evaluation_spec_id}"
            )
    has_strategy_override = (
        request.sizing_method is not None
        or request.sizing_engine is not None
        or request.direction_mode is not None
    )
    source_tasks = (
        _dedupe_tasks_by_signal_discovery(read_port, existing_tasks)
        if has_strategy_override
        else existing_tasks
    )
    entries: list[EvaluationTaskResolutionEntry] = []
    for source_task in source_tasks:
        source_strategy_state = read_port.get_trading_strategy(source_task.strategy_id)
        if source_strategy_state is None:
            raise ValueError(
                "evaluation task strategy does not exist: "
                f"{source_task.strategy_id}"
            )
        source_signal_discovery_id = source_strategy_state.trading_strategy.signal_discovery_id
        if has_strategy_override and source_signal_discovery_id is None:
            raise ValueError(
                "unknown signal discovery for evaluation task override: "
                f"{source_signal_discovery_id}"
            )
        source_config = strategy_variant_config_from_strategy(
            source_strategy_state.trading_strategy
        )
        resolved_config = (
            overridden_strategy_variant_config(
                source_config,
                sizing_method=request.sizing_method,
                sizing_engine=request.sizing_engine,
                direction_mode=request.direction_mode,
            )
            if has_strategy_override
            else source_config
        )
        if resolved_config == source_config:
            entries.append(
                EvaluationTaskResolutionEntry(
                    task=source_task,
                    source_task=source_task,
                    resolution_action="existing",
                    reason="existing",
                )
            )
            continue
        signal_discovery_state = read_port.get_signal_discovery_spec(
            source_signal_discovery_id
        )
        if signal_discovery_state is None:
            raise ValueError(
                "unknown signal discovery for evaluation task override: "
                f"{source_signal_discovery_id}"
            )
        trading_strategy = derive_trading_strategy_from_signal_discovery(
            signal_discovery=signal_discovery_state,
            variant_config=resolved_config,
            created_at=request.created_at,
        )
        derived_task = EvaluationTask(
            evaluation_task_id=build_evaluation_task_id(
                strategy_id=trading_strategy.strategy_id,
                evaluation_spec_id=evaluation_spec_id,
            ),
            strategy_id=trading_strategy.strategy_id,
            evaluation_spec_id=evaluation_spec_id,
        )
        entries.append(
            EvaluationTaskResolutionEntry(
                task=derived_task,
                source_task=source_task,
                trading_strategy_to_persist=trading_strategy,
                task_to_persist=derived_task,
                resolution_action="derived_override",
                reason="derived_override",
            )
        )
    return EvaluationTaskResolutionPlan(entries=tuple(entries))


def _dedupe_tasks_by_signal_discovery(
    read_port: EvaluationTaskResolutionReadPort,
    tasks: tuple[EvaluationTask, ...],
) -> tuple[EvaluationTask, ...]:
    unique_tasks_by_signal_discovery_id: dict[str | None, EvaluationTask] = {}
    for task in tasks:
        strategy_state = read_port.get_trading_strategy(task.strategy_id)
        if strategy_state is None:
            raise ValueError(
                "evaluation task strategy does not exist: "
                f"{task.strategy_id}"
            )
        signal_discovery_id = strategy_state.trading_strategy.signal_discovery_id
        unique_tasks_by_signal_discovery_id.setdefault(
            signal_discovery_id,
            task,
        )
    return tuple(unique_tasks_by_signal_discovery_id.values())


def persist_evaluation_task_resolution_plan(
    write_port: EvaluationTaskResolutionWritePort,
    plan: EvaluationTaskResolutionPlan,
) -> tuple[EvaluationTask, ...]:
    for entry in plan.entries:
        if entry.trading_strategy_to_persist is not None:
            write_port.upsert_trading_strategy(
                trading_strategy=entry.trading_strategy_to_persist
            )
        if entry.task_to_persist is not None:
            write_port.upsert_evaluation_task(task=entry.task_to_persist)
    return plan.tasks


def resolve_evaluation_tasks_for_spec(
    store: EvaluationStore,
    *,
    evaluation_spec_state,
    sizing_method: str | None,
    sizing_engine: str | None,
    strategy_ids: tuple[str, ...] | None,
    created_at: str,
    direction_mode: str | None = None,
    evaluation_task_ids: tuple[str, ...] | None = None,
) -> tuple[EvaluationTask, ...]:
    plan = build_evaluation_task_resolution_plan(
        store,
        EvaluationTaskResolutionRequest(
            evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
            sizing_method=sizing_method,
            sizing_engine=sizing_engine,
            strategy_ids=strategy_ids,
            created_at=created_at,
            direction_mode=direction_mode,
            evaluation_task_ids=evaluation_task_ids,
        ),
    )
    return persist_evaluation_task_resolution_plan(store, plan)
