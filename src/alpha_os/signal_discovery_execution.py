from __future__ import annotations

from dataclasses import dataclass

from .signal_discovery import SignalDiscoverySpec
from .portfolio_decision import SubjectSet
from .universe_contract import validate_subject_set_universe_contract


@dataclass(frozen=True)
class SignalDiscoveryExecutionPlan:
    signal_discovery: SignalDiscoverySpec
    subject_set: SubjectSet
    target_id: str
    signal_spec_ids: tuple[str, ...]
    family_ids_by_signal_spec_id: dict[str, str]
    family_budgets_by_family_id: dict[str, int]

def build_signal_discovery_execution_plan(
    store,
    *,
    signal_discovery_id: str,
    default_target_id: str,
) -> SignalDiscoveryExecutionPlan:
    signal_discovery_state = store.get_signal_discovery_spec(signal_discovery_id)
    if signal_discovery_state is None:
        raise ValueError(f"unknown signal discovery spec: {signal_discovery_id}")
    signal_discovery = signal_discovery_state.definition
    subject_set_state = store.get_subject_set(signal_discovery.subject_set_id)
    if subject_set_state is None:
        raise ValueError(
            f"subject set for signal discovery is missing: {signal_discovery.subject_set_id}"
        )
    validate_subject_set_universe_contract(subject_set_state.definition)
    specification_states = store.list_signal_specs(limit=10000)
    specification_definitions = [item.definition for item in specification_states]
    signal_spec_ids = signal_discovery.resolve_signal_spec_ids(
        specification_definitions
    )
    family_resolutions = signal_discovery.family_resolutions_by_signal_spec_id(
        specification_definitions
    )
    if not signal_spec_ids:
        raise ValueError(
            f"signal discovery resolves to no specifications: {signal_discovery_id}"
        )
    return SignalDiscoveryExecutionPlan(
        signal_discovery=signal_discovery,
        subject_set=subject_set_state.definition,
        target_id=(
            signal_discovery.target_id
            if signal_discovery.target_id is not None
            else default_target_id
        ),
        signal_spec_ids=tuple(signal_spec_ids),
        family_ids_by_signal_spec_id={
            signal_spec_id: family.resolved_family_id
            for signal_spec_id, family in family_resolutions.items()
        },
        family_budgets_by_family_id={
            family.resolved_family_id: family.survivor_budget
            for family in family_resolutions.values()
            if family.survivor_budget is not None
        },
    )
