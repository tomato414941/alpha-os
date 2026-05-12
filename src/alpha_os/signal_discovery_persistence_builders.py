from __future__ import annotations

from .strategy_checkpoint import StrategyCheckpoint


def build_strategy_checkpoint_id(
    *,
    strategy_id: str,
    fold_label: str,
    start_date: str,
    end_date: str,
) -> str:
    return f"{strategy_id}:{fold_label}:{start_date}:{end_date}:initial-state"


def build_strategy_checkpoint(
    *,
    strategy_id: str,
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
