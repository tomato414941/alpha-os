from __future__ import annotations

from .portfolio_decision import SubjectSet
from .signal_client import build_signal_client
from .signal_discovery_compression_service import build_compressed_belief_from_screening_result
from .signal_discovery_persistence_builders import (
    build_strategy_checkpoint,
    build_strategy_checkpoint_id as build_strategy_checkpoint_id,
)
from .store import EvaluationStore, _utc_now
from .universe_contract import validate_subject_set_universe_contract


def ensure_subject_set_backend_available(
    subject_set: SubjectSet,
    *,
    base_url: str,
) -> None:
    validate_subject_set_universe_contract(subject_set)
    client = build_signal_client(base_url=base_url)
    if not client.health():
        return
    checks = []
    for binding in subject_set.bindings:
        metadata = client.metadata(
            asset=binding.asset,
            observable_id=binding.observation_spec.observable_id,
        )
        checks.append(
            {
                "subject_id": binding.subject_id,
                "observable_id": binding.observation_spec.observable_id,
                "source_id": binding.observation_spec.source_id,
                "available": metadata is not None,
            }
        )
    missing = [item for item in checks if not bool(item["available"])]
    if not missing:
        return
    joined = ", ".join(
        f"{item['subject_id']}->{item['observable_id']}@{item['source_id']}"
        for item in missing
    )
    raise ValueError(f"subject set contains unavailable backend observations: {joined}")


def compress_screening_result_state(
    store: EvaluationStore,
    *,
    screening_result_id: str,
    strategy_id: str | None = None,
):
    state = store.get_screening_result(screening_result_id)
    if state is None:
        raise ValueError(f"unknown screening result: {screening_result_id}")
    result = state.result
    survivors = result.survivors
    latest_snapshots = store.list_latest_evaluation_snapshots(
        signal_ids=[item.signal_id for item in survivors]
    )
    prediction_values_by_signal_id = {
        item.signal_id: item.prediction_value for item in latest_snapshots
    }
    strategy_adaptation_state_record = None
    adaptation_blend = 0.2
    if strategy_id is not None:
        strategy_state = store.get_trading_strategy(strategy_id)
        if strategy_state is None:
            raise ValueError(f"unknown strategy: {strategy_id}")
        trading_strategy = strategy_state.trading_strategy
        adaptation_blend = trading_strategy.adaptation_policy.adaptation_blend
        if not trading_strategy.adaptation_policy.enabled:
            strategy_id = None
        else:
            strategy_adaptation_state_record = store.get_strategy_adaptation_state(strategy_id)
            if strategy_adaptation_state_record is None:
                raise ValueError(
                    f"strategy adaptation state does not exist for strategy: {strategy_id}"
                )
    belief = build_compressed_belief_from_screening_result(
        screening_result=result,
        prediction_values_by_signal_id=prediction_values_by_signal_id,
        created_at=_utc_now(),
        strategy_adaptation_state=(
            None
            if strategy_adaptation_state_record is None
            else strategy_adaptation_state_record.state
        ),
        adaptation_blend=adaptation_blend,
    )
    return store.upsert_compressed_belief(belief=belief)


def prune_screened_snapshots(
    store: EvaluationStore,
    *,
    selected_signal_ids: tuple[str, ...],
    screening_state,
    snapshot_retention: str,
) -> int:
    survivor_ids = {item.signal_id for item in screening_state.result.survivors}
    non_survivor_ids = [
        signal_id for signal_id in selected_signal_ids if signal_id not in survivor_ids
    ]
    deleted = store.delete_evaluation_snapshots_for_signals(signal_ids=non_survivor_ids)
    if snapshot_retention != "latest_per_survivor" or not survivor_ids:
        return deleted
    return deleted + store.delete_non_latest_evaluation_snapshots_for_signals(
        signal_ids=sorted(survivor_ids)
    )


def persist_strategy_checkpoint(
    store: EvaluationStore,
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
    created_at: str | None = None,
):
    timestamp = _utc_now() if created_at is None else created_at
    strategy_checkpoint = build_strategy_checkpoint(
        strategy_id=strategy_id,
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        fold_label=fold_label,
        start_date=start_date,
        end_date=end_date,
        snapshot_set_id=snapshot_set_id,
        screening_state=screening_state,
        compressed_belief_state=compressed_belief_state,
        created_at=timestamp,
    )
    return store.upsert_strategy_checkpoint(state=strategy_checkpoint)
