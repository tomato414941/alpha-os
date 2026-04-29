from __future__ import annotations

import argparse
from dataclasses import dataclass

from .cli_output import print_signal_competition_summary
from .data_repositories import EvaluationInputRepository, FeaturePlaneRepository
from .evaluation_generation import generate_evaluation_inputs_batch_from_feature_plane
from .evaluation_inputs import EvaluationInput
from .evaluation_runtime import apply_evaluations_batch
from .portfolio_decision import ObservationSpec, SubjectSet
from .pre_screening import CheapPreScreenPolicy, cheap_pre_screen_on_feature_plane
from .probe_screening import ProbeScreenPolicy, probe_screen_on_feature_plane
from .signal_registry import (
    SignalDefinition,
    build_signal_spec,
    build_subject_bound_signal_definition,
)
from .store import EvaluationStore
from .subject_set_feature_plane import SubjectPlaneKey, build_subject_set_feature_planes
from .survivor_screening import SurvivorScreenPolicy, survivor_screen_on_feature_plane
from .universe_contract import validate_subject_set_universe_contract


@dataclass(frozen=True)
class SubjectSetBackfillResult:
    latest_snapshot: object | None
    created_count: int
    existing_count: int
    total_executables: int
    pre_screen_selected_executables: int
    probe_selected_executables: int
    survivor_selected_executables: int
    selected_executables: int
    selected_signal_ids: tuple[str, ...]
    evaluation_inputs: int


def resolve_subject_set_for_build(
    store: EvaluationStore,
    args: argparse.Namespace,
    *,
    inline_subject_set: SubjectSet | None = None,
) -> SubjectSet | None:
    subject_set_id = (
        None if args.subject_set_id is None else str(args.subject_set_id).strip()
    )
    if subject_set_id and inline_subject_set is not None:
        raise ValueError(
            "subject-set-id and subject-binding cannot be used together"
        )
    if not subject_set_id:
        if inline_subject_set is not None:
            validate_subject_set_universe_contract(inline_subject_set)
        return inline_subject_set
    state = store.get_subject_set(subject_set_id)
    if state is None:
        raise ValueError(f"unknown subject set: {subject_set_id}")
    validate_subject_set_universe_contract(state.definition)
    return state.definition


def generate_backfill_inputs_for_subject_group(
    *,
    executable_definitions: list[SignalDefinition],
    plane,
    start_date: str,
    end_date: str,
    contract_multiplier: float | None = None,
    pre_screen_top_k_per_kind: int | None = None,
    pre_screen_min_abs_corr: float = 0.0,
    probe_max_dates: int | None = None,
    probe_min_sample_count: int = 0,
    probe_min_abs_corr: float = 0.0,
    probe_max_family_survivors_per_subject: int | None = None,
    survivor_min_sample_count: int = 0,
    survivor_min_abs_corr: float = 0.0,
    survivor_max_family_survivors_per_subject: int | None = None,
    family_ids_by_signal_id: dict[str, str] | None = None,
    evaluation_input_repository: EvaluationInputRepository | None = None,
    observation_spec: ObservationSpec | None = None,
    asset: str | None = None,
    base_url: str | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> tuple[list[SignalDefinition], list[EvaluationInput], int, int, int, int]:
    if not executable_definitions:
        return [], [], 0, 0, 0, 0
    selected_definitions = executable_definitions
    pre_screen_selected = len(executable_definitions)
    if pre_screen_top_k_per_kind is not None or pre_screen_min_abs_corr > 0.0:
        pre_screen_result = cheap_pre_screen_on_feature_plane(
            plane=plane,
            start_date=start_date,
            end_date=end_date,
            definitions=executable_definitions,
            policy=CheapPreScreenPolicy(
                min_abs_corr=pre_screen_min_abs_corr,
                top_k_per_kind=pre_screen_top_k_per_kind,
            ),
        )
        selected_definitions = list(pre_screen_result.selected_definitions)
        pre_screen_selected = len(selected_definitions)
    if not selected_definitions:
        return [], [], len(executable_definitions), 0, 0, 0
    probe_selected = len(selected_definitions)
    if (
        probe_max_dates is not None
        or probe_min_sample_count > 0
        or probe_min_abs_corr > 0.0
        or probe_max_family_survivors_per_subject is not None
    ):
        probe_result = probe_screen_on_feature_plane(
            plane=plane,
            start_date=start_date,
            end_date=end_date,
            definitions=selected_definitions,
            policy=ProbeScreenPolicy(
                max_dates=probe_max_dates,
                min_sample_count=probe_min_sample_count,
                min_abs_corr=probe_min_abs_corr,
                max_family_survivors_per_subject=probe_max_family_survivors_per_subject,
            ),
            family_ids_by_signal_id=family_ids_by_signal_id,
        )
        selected_definitions = list(probe_result.selected_definitions)
        probe_selected = len(selected_definitions)
    if not selected_definitions:
        return [], [], len(executable_definitions), pre_screen_selected, 0, 0
    survivor_selected = len(selected_definitions)
    if (
        survivor_min_sample_count > 0
        or survivor_min_abs_corr > 0.0
        or survivor_max_family_survivors_per_subject is not None
    ):
        survivor_result = survivor_screen_on_feature_plane(
            plane=plane,
            start_date=start_date,
            end_date=end_date,
            definitions=selected_definitions,
            policy=SurvivorScreenPolicy(
                min_sample_count=survivor_min_sample_count,
                min_abs_corr=survivor_min_abs_corr,
                max_family_survivors_per_subject=survivor_max_family_survivors_per_subject,
            ),
            family_ids_by_signal_id=family_ids_by_signal_id,
        )
        selected_definitions = list(survivor_result.selected_definitions)
        survivor_selected = len(selected_definitions)
    if not selected_definitions:
        return [], [], len(executable_definitions), pre_screen_selected, probe_selected, 0
    if (
        evaluation_input_repository is not None
        and observation_spec is not None
        and asset is not None
        and base_url is not None
    ):
        evaluation_inputs = evaluation_input_repository.load_inputs_for_range(
            plane=plane,
            definitions=selected_definitions,
            start_date=start_date,
            end_date=end_date,
            observation_spec=observation_spec,
            asset=asset,
            base_url=base_url,
            contract_multiplier=contract_multiplier,
            contract_family=contract_family,
            quote_ccy=quote_ccy,
            collateral_ccy=collateral_ccy,
            roll_rule=roll_rule,
        )
    else:
        evaluation_inputs = generate_evaluation_inputs_batch_from_feature_plane(
            plane=plane,
            start_date=start_date,
            end_date=end_date,
            definitions=selected_definitions,
            observation_spec=observation_spec,
            contract_multiplier=contract_multiplier,
            contract_family=contract_family,
            quote_ccy=quote_ccy,
            collateral_ccy=collateral_ccy,
            roll_rule=roll_rule,
        )
    return (
        selected_definitions,
        evaluation_inputs,
        len(executable_definitions),
        pre_screen_selected,
        probe_selected,
        survivor_selected,
    )


def planned_subject_bound_signals(
    store: EvaluationStore,
    *,
    subject_set: SubjectSet,
    signal_spec_ids: list[str],
    target_id: str,
) -> list[tuple[str, SignalDefinition]]:
    planned_definitions: list[tuple[str, SignalDefinition]] = []
    specifications = {
        signal_spec_id: store.get_signal_spec(signal_spec_id)
        for signal_spec_id in signal_spec_ids
    }
    missing_specification_ids = [
        signal_spec_id
        for signal_spec_id, state in specifications.items()
        if state is None
    ]
    if missing_specification_ids:
        joined = ", ".join(sorted(missing_specification_ids))
        raise ValueError(
            f"signal specs must be registered first: {joined}"
        )
    for binding in subject_set.bindings:
        observation_spec = subject_set.observation_spec_for_subject(
            binding.subject_id
        )
        for signal_spec_id in signal_spec_ids:
            specification_state = specifications[signal_spec_id]
            assert specification_state is not None
            specification_definition = (
                specification_state.definition
                if specification_state.target_id == target_id
                else build_signal_spec(
                    base_signal_id=specification_state.signal_id,
                    signal_id=specification_state.signal_id,
                    target_id=target_id,
                )
            )
            definition = build_subject_bound_signal_definition(
                specification=specification_definition,
                subject_id=binding.subject_id,
                asset=binding.asset,
                observation_spec=observation_spec,
            )
            planned_definitions.append((specification_state.signal_id, definition))
    return planned_definitions


def register_subject_bound_signals(
    store: EvaluationStore,
    *,
    planned_definitions: list[tuple[str, SignalDefinition]],
    target_id: str,
) -> list[str]:
    executable_signal_ids: list[str] = []
    for signal_spec_id, definition in planned_definitions:
        store.register_signal(
            definition.signal_id,
            asset=definition.asset,
            target_id=target_id,
            definition=definition,
            specification_signal_id=signal_spec_id,
        )
        executable_signal_ids.append(definition.signal_id)
    return executable_signal_ids


def run_subject_set_backfill(
    store: EvaluationStore,
    *,
    subject_set: SubjectSet,
    subject_set_id: str,
    signal_spec_ids: list[str],
    target_id: str,
    start_date: str,
    end_date: str,
    base_url: str,
    pre_screen_top_k_per_kind: int | None,
    pre_screen_min_abs_corr: float,
    probe_max_dates: int | None = None,
    probe_min_sample_count: int = 0,
    probe_min_abs_corr: float = 0.0,
    probe_max_family_survivors_per_subject: int | None = None,
    survivor_min_sample_count: int = 0,
    survivor_min_abs_corr: float = 0.0,
    survivor_max_family_survivors_per_subject: int | None = None,
    family_ids_by_signal_spec_id: dict[str, str] | None = None,
    signal_discovery_id: str | None = None,
    feature_plane_repository: FeaturePlaneRepository | None = None,
    evaluation_input_repository: EvaluationInputRepository | None = None,
) -> SubjectSetBackfillResult:
    validate_subject_set_universe_contract(subject_set)
    planned_definitions = planned_subject_bound_signals(
        store,
        subject_set=subject_set,
        signal_spec_ids=signal_spec_ids,
        target_id=target_id,
    )
    persist_survivors_only = signal_discovery_id is not None
    all_evaluation_inputs: list[EvaluationInput] = []
    total_executables = 0
    pre_screen_selected_executables = 0
    probe_selected_executables = 0
    survivor_selected_executables = 0
    selected_signal_ids: list[str] = []
    executable_groups: dict[
        tuple[str, ObservationSpec], list[tuple[str, SignalDefinition]]
    ] = {}
    for signal_spec_id, definition in planned_definitions:
        if definition.observation_spec is None:
            raise ValueError(
                f"executable signal is missing observation spec: {definition.signal_id}"
            )
        group_key = (definition.asset, definition.observation_spec)
        executable_groups.setdefault(group_key, []).append(
            (signal_spec_id, definition)
        )

    if not persist_survivors_only:
        selected_signal_ids = register_subject_bound_signals(
            store,
            planned_definitions=planned_definitions,
            target_id=target_id,
        )

    subject_planes = build_subject_set_feature_planes(
        subject_set=subject_set,
        executable_definitions=[definition for _, definition in planned_definitions],
        base_url=base_url,
        feature_plane_repository=feature_plane_repository,
    )

    for grouped_definitions in executable_groups.values():
        executable_definitions = [definition for _, definition in grouped_definitions]
        first_definition = executable_definitions[0]
        if first_definition.observation_spec is None:
            raise ValueError(
                "executable signal is missing observation spec: "
                f"{first_definition.signal_id}"
            )
        plane_key = SubjectPlaneKey(
            asset=first_definition.asset,
            observation_spec_id=first_definition.observation_spec.observation_spec_id,
        )
        plane = subject_planes.get(plane_key)
        if plane is None:
            raise ValueError(
                "subject-set feature plane is missing: "
                f"{first_definition.asset}/{first_definition.observation_spec.observation_spec_id}"
            )
        family_ids_by_signal_id = {
            definition.signal_id: family_ids_by_signal_spec_id[
                signal_spec_id
            ]
            for signal_spec_id, definition in grouped_definitions
            if family_ids_by_signal_spec_id is not None
            and signal_spec_id in family_ids_by_signal_spec_id
        }
        contract_multiplier = None
        contract_family = None
        quote_ccy = None
        collateral_ccy = None
        roll_rule = None
        binding = next(
            (
                item
                for item in subject_set.bindings
                if item.asset == first_definition.asset
                and item.observation_spec_id
                == first_definition.observation_spec.observation_spec_id
            ),
            None,
        )
        if binding is not None:
            instrument = subject_set.instrument_for_subject(binding.subject_id)
            if instrument is not None:
                if instrument.multiplier is not None:
                    contract_multiplier = float(instrument.multiplier)
                contract_family = instrument.contract_family
                quote_ccy = instrument.quote_ccy
                collateral_ccy = instrument.collateral_ccy
                roll_rule = instrument.roll_rule
        (
            selected_definitions,
            group_evaluation_inputs,
            group_total_executables,
            group_pre_screen_selected,
            group_probe_selected,
            group_survivor_selected,
        ) = generate_backfill_inputs_for_subject_group(
            executable_definitions=executable_definitions,
            plane=plane,
            start_date=start_date,
            end_date=end_date,
            contract_multiplier=contract_multiplier,
            pre_screen_top_k_per_kind=pre_screen_top_k_per_kind,
            pre_screen_min_abs_corr=pre_screen_min_abs_corr,
            probe_max_dates=probe_max_dates,
            probe_min_sample_count=probe_min_sample_count,
            probe_min_abs_corr=probe_min_abs_corr,
            probe_max_family_survivors_per_subject=probe_max_family_survivors_per_subject,
            survivor_min_sample_count=survivor_min_sample_count,
            survivor_min_abs_corr=survivor_min_abs_corr,
            survivor_max_family_survivors_per_subject=survivor_max_family_survivors_per_subject,
            family_ids_by_signal_id=family_ids_by_signal_id,
            evaluation_input_repository=evaluation_input_repository,
            observation_spec=first_definition.observation_spec,
            asset=first_definition.asset,
            base_url=base_url,
            contract_family=contract_family,
            quote_ccy=quote_ccy,
            collateral_ccy=collateral_ccy,
            roll_rule=roll_rule,
        )
        if persist_survivors_only and selected_definitions:
            specification_signal_id_by_signal_id = {
                definition.signal_id: specification_signal_id
                for specification_signal_id, definition in grouped_definitions
            }
            selected_signal_ids.extend(
                register_subject_bound_signals(
                    store,
                    planned_definitions=[
                        (
                            specification_signal_id_by_signal_id[
                                definition.signal_id
                            ],
                            definition,
                        )
                        for definition in selected_definitions
                    ],
                    target_id=target_id,
                )
            )
        all_evaluation_inputs.extend(group_evaluation_inputs)
        total_executables += group_total_executables
        pre_screen_selected_executables += group_pre_screen_selected
        probe_selected_executables += group_probe_selected
        survivor_selected_executables += group_survivor_selected

    latest_snapshot, created_count, existing_count = apply_evaluations_batch(
        store,
        evaluation_inputs=all_evaluation_inputs,
        input_source="subject_set_backfill",
        input_range_start=start_date,
        input_range_end=end_date,
        refresh_meta_predictions=not persist_survivors_only,
    )
    print(
        "Batch complete: "
        f"subject_set={subject_set_id} "
        f"{'' if signal_discovery_id is None else f'signal_discovery={signal_discovery_id} '}"
        f"specifications={len(signal_spec_ids)} "
        f"executables={len(selected_signal_ids)} "
        f"pre_screen_selected={pre_screen_selected_executables}/{total_executables} "
        f"probe_selected={probe_selected_executables}/{pre_screen_selected_executables} "
        f"survivor_selected={survivor_selected_executables}/{probe_selected_executables} "
        f"evaluations={len(all_evaluation_inputs)} "
        f"created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(
            f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.signal_id}"
        )
    print_signal_competition_summary(
        store,
        signal_ids=selected_signal_ids,
    )
    return SubjectSetBackfillResult(
        latest_snapshot=latest_snapshot,
        created_count=created_count,
        existing_count=existing_count,
        total_executables=total_executables,
        pre_screen_selected_executables=pre_screen_selected_executables,
        probe_selected_executables=probe_selected_executables,
        survivor_selected_executables=survivor_selected_executables,
        selected_executables=survivor_selected_executables,
        selected_signal_ids=tuple(selected_signal_ids),
        evaluation_inputs=len(all_evaluation_inputs),
    )
