from __future__ import annotations

from .strategy_checkpoint import StrategyCheckpoint
from .signal_discovery_run import SignalDiscoveryRun
from .subject_set_backfill_service import SubjectSetBackfillResult


def build_strategy_checkpoint_id(
    *,
    strategy_id: str,
    fold_label: str,
    start_date: str,
    end_date: str,
) -> str:
    return f"{strategy_id}:{fold_label}:{start_date}:{end_date}:initial-state"


def build_signal_discovery_run(
    *,
    signal_discovery_run_id: str,
    snapshot_set_id: str,
    signal_discovery_id: str,
    subject_set_id: str,
    target_id: str,
    start_date: str,
    end_date: str,
    screening_result_id: str,
    compressed_belief_id: str,
    workflow_runtime_s: float,
    backfill_result: SubjectSetBackfillResult,
    pruned_snapshot_count: int,
    created_at: str,
) -> SignalDiscoveryRun:
    return SignalDiscoveryRun(
        signal_discovery_run_id=signal_discovery_run_id,
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        execution_start_date=start_date,
        execution_end_date=end_date,
        snapshot_set_id=snapshot_set_id,
        screening_result_id=screening_result_id,
        compressed_belief_id=compressed_belief_id,
        workflow_runtime_s=float(workflow_runtime_s),
        total_executables=int(backfill_result.total_executables),
        pre_screen_selected=int(backfill_result.pre_screen_selected_executables),
        probe_selected=int(backfill_result.probe_selected_executables),
        survivor_selected=int(backfill_result.survivor_selected_executables),
        persisted_signals=int(backfill_result.selected_executables),
        evaluation_inputs=int(backfill_result.evaluation_inputs),
        pruned_snapshots=int(pruned_snapshot_count),
        created_at=created_at,
    )


def build_strategy_checkpoint(
    *,
    strategy_id: str,
    signal_train_id: str,
    signal_discovery_id: str | None,
    subject_set_id: str,
    target_id: str,
    fold_label: str,
    start_date: str,
    end_date: str,
    snapshot_set_id: str,
    screening_state,
    compressed_belief_state,
    created_at: str,
) -> StrategyCheckpoint:
    return StrategyCheckpoint(
        strategy_checkpoint_id=build_strategy_checkpoint_id(
            strategy_id=strategy_id,
            fold_label=fold_label,
            start_date=start_date,
            end_date=end_date,
        ),
        strategy_id=strategy_id,
        signal_train_id=signal_train_id,
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        fold_label=fold_label,
        execution_start_date=start_date,
        execution_end_date=end_date,
        snapshot_set_id=snapshot_set_id,
        screening_result_id=screening_state.screening_result_id,
        compressed_belief_id=compressed_belief_state.compressed_belief_id,
        survivor_signal_ids=tuple(item.signal_id for item in screening_state.result.survivors),
        created_at=created_at,
    )
