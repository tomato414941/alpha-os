from __future__ import annotations

from .screening import ScreeningPolicy, screen_signals
from .store import EvaluationStore, _utc_now
from .universe_contract import validate_subject_set_universe_contract


def screen_signal_discovery(
    store: EvaluationStore,
    *,
    signal_discovery_id: str,
    min_sample_count: int | None = None,
    min_abs_corr: float | None = None,
    min_stability_score: float | None = None,
    max_family_survivors_per_subject: int | None = None,
):
    state = store.get_signal_discovery_spec(signal_discovery_id)
    if state is None:
        raise ValueError(f"unknown signal discovery spec: {signal_discovery_id}")
    signal_discovery = state.definition
    subject_set_state = store.get_subject_set(signal_discovery.subject_set_id)
    if subject_set_state is None:
        raise ValueError(
            f"unknown subject set for signal discovery: {signal_discovery.subject_set_id}"
        )
    validate_subject_set_universe_contract(subject_set_state.definition)
    subject_ids = {binding.subject_id for binding in subject_set_state.definition.bindings}
    available_specifications = [item.definition for item in store.list_signal_specs(limit=10_000)]
    resolved_specification_ids = tuple(
        signal_discovery.resolve_signal_spec_ids(available_specifications)
    )
    resolved_specification_id_set = set(resolved_specification_ids)
    family_resolutions = signal_discovery.family_resolutions_by_signal_spec_id(
        available_specifications
    )
    family_ids_by_signal_spec_id = {
        signal_spec_id: family.resolved_family_id
        for signal_spec_id, family in family_resolutions.items()
    }
    family_budgets_by_family_id = {
        family.resolved_family_id: family.survivor_budget
        for family in family_resolutions.values()
        if family.survivor_budget is not None
    }
    selected_signals = []
    for subject_id in sorted(subject_ids):
        selected_signals.extend(
            item
            for item in store.list_signals(
                subject_id=subject_id,
                target_id=signal_discovery.target_id,
            )
            if item.status == "active"
            and item.specification_signal_id in resolved_specification_id_set
        )
    selection_policy = signal_discovery.selection_policy
    metrics_by_id = {
        item.signal_id: item
        for item in store.list_signal_metrics(
            signal_ids=[item.signal_id for item in selected_signals]
        )
    }
    result = screen_signals(
        signals=selected_signals,
        metrics_by_id=metrics_by_id,
        signal_discovery_id=signal_discovery.signal_discovery_id,
        policy=ScreeningPolicy(
            min_sample_count=(
                selection_policy.min_sample_count
                if min_sample_count is None
                else int(min_sample_count)
            ),
            min_abs_corr=(
                selection_policy.min_abs_corr if min_abs_corr is None else float(min_abs_corr)
            ),
            min_stability_score=(
                selection_policy.min_stability_score
                if min_stability_score is None
                else float(min_stability_score)
            ),
            adaptive_family_budget=selection_policy.adaptive_family_budget,
            adaptive_budget_stability_scale=(selection_policy.adaptive_budget_stability_scale),
            max_family_survivors_per_subject=(
                selection_policy.max_family_survivors_per_subject
                if max_family_survivors_per_subject is None
                else int(max_family_survivors_per_subject)
            ),
        ),
        family_ids_by_signal_spec_id=family_ids_by_signal_spec_id,
        family_budgets_by_family_id=family_budgets_by_family_id,
        created_at=_utc_now(),
    )
    return store.upsert_screening_result(result=result)
