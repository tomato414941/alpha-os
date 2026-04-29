from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from .signal_operator_registry import find_signal_operator_definition
from .signal_registry import SignalSpec
from .signal_discovery import (
    SignalFamily,
    SignalParameterSpace,
    SignalDiscoverySpec,
    SignalDiscoverySelectionPolicy,
)
from .observables import ObservableDefinition, find_observable_definition
from .targets import get_target_definition


@dataclass(frozen=True)
class SignalDiscoveryGenerationConstraint:
    max_families_total: int | None = None
    max_families_per_operator: int | None = None
    max_families_per_primary_observable: int | None = None
    max_families_per_family_group: int | None = None
    allowed_family_groups: tuple[str, ...] = ()
    require_distinct_observables: bool = True
    novelty_key: str = "observable_combo"

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "require_distinct_observables": self.require_distinct_observables,
            "novelty_key": self.novelty_key,
        }
        if self.max_families_total is not None:
            document["max_families_total"] = self.max_families_total
        if self.max_families_per_operator is not None:
            document["max_families_per_operator"] = self.max_families_per_operator
        if self.max_families_per_primary_observable is not None:
            document["max_families_per_primary_observable"] = (
                self.max_families_per_primary_observable
            )
        if self.max_families_per_family_group is not None:
            document["max_families_per_family_group"] = (
                self.max_families_per_family_group
            )
        if self.allowed_family_groups:
            document["allowed_family_groups"] = list(self.allowed_family_groups)
        return document

    @classmethod
    def from_document(cls, document: object) -> "SignalDiscoveryGenerationConstraint":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("generation constraint must be an object")
        max_families_total = document.get("max_families_total")
        max_families_per_operator = document.get("max_families_per_operator")
        max_families_per_primary_observable = document.get(
            "max_families_per_primary_observable"
        )
        max_families_per_family_group = document.get(
            "max_families_per_family_group"
        )
        allowed_family_groups = document.get("allowed_family_groups", [])
        require_distinct_observables = document.get("require_distinct_observables", True)
        novelty_key = document.get("novelty_key", "observable_combo")
        if max_families_total is not None and (
            not isinstance(max_families_total, int) or max_families_total < 1
        ):
            raise ValueError("generation constraint max_families_total must be >= 1")
        if max_families_per_operator is not None and (
            not isinstance(max_families_per_operator, int) or max_families_per_operator < 1
        ):
            raise ValueError(
                "generation constraint max_families_per_operator must be >= 1"
            )
        if max_families_per_primary_observable is not None and (
            not isinstance(max_families_per_primary_observable, int)
            or max_families_per_primary_observable < 1
        ):
            raise ValueError(
                "generation constraint max_families_per_primary_observable must be >= 1"
            )
        if max_families_per_family_group is not None and (
            not isinstance(max_families_per_family_group, int)
            or max_families_per_family_group < 1
        ):
            raise ValueError(
                "generation constraint max_families_per_family_group must be >= 1"
            )
        if not isinstance(allowed_family_groups, list) or any(
            not isinstance(item, str) or not item
            for item in allowed_family_groups
        ):
            raise ValueError(
                "generation constraint allowed_family_groups must be a list of strings"
            )
        if not isinstance(require_distinct_observables, bool):
            raise ValueError(
                "generation constraint require_distinct_observables must be boolean"
            )
        if novelty_key not in {
            "observable_combo",
            "family_group_primary",
            "operator_primary",
        }:
            raise ValueError(
                "generation constraint novelty_key must be observable_combo, "
                "family_group_primary, or operator_primary"
            )
        return cls(
            max_families_total=max_families_total,
            max_families_per_operator=max_families_per_operator,
            max_families_per_primary_observable=max_families_per_primary_observable,
            max_families_per_family_group=max_families_per_family_group,
            allowed_family_groups=tuple(allowed_family_groups),
            require_distinct_observables=require_distinct_observables,
            novelty_key=novelty_key,
        )


@dataclass(frozen=True)
class SignalDiscoveryGenerationSpec:
    signal_discovery_id: str
    subject_set_id: str
    operator_ids: tuple[str, ...]
    primary_observable_ids: tuple[str, ...]
    parameter_space: SignalParameterSpace
    target_id: str | None = None
    secondary_observable_ids: tuple[str, ...] = ()
    conditioning_observable_ids: tuple[str, ...] = ()
    applicable_subject_kinds: tuple[str, ...] = ()
    selection_policy: SignalDiscoverySelectionPolicy = SignalDiscoverySelectionPolicy()
    constraint: SignalDiscoveryGenerationConstraint = SignalDiscoveryGenerationConstraint()

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "signal_discovery_id": self.signal_discovery_id,
            "subject_set_id": self.subject_set_id,
            "operator_ids": list(self.operator_ids),
            "primary_observable_ids": list(self.primary_observable_ids),
            "parameter_space": self.parameter_space.to_document(),
            "selection_policy": self.selection_policy.to_document(),
            "constraint": self.constraint.to_document(),
        }
        if self.target_id is not None:
            document["target_id"] = self.target_id
        if self.secondary_observable_ids:
            document["secondary_observable_ids"] = list(self.secondary_observable_ids)
        if self.conditioning_observable_ids:
            document["conditioning_observable_ids"] = list(
                self.conditioning_observable_ids
            )
        if self.applicable_subject_kinds:
            document["applicable_subject_kinds"] = list(self.applicable_subject_kinds)
        return document

    @classmethod
    def from_document(cls, document: dict[str, object]) -> "SignalDiscoveryGenerationSpec":
        signal_discovery_id = document.get("signal_discovery_id")
        subject_set_id = document.get("subject_set_id")
        operator_ids = document.get("operator_ids", [])
        primary_observable_ids = document.get("primary_observable_ids", [])
        parameter_space = document.get("parameter_space")
        target_id = document.get("target_id")
        secondary_observable_ids = document.get("secondary_observable_ids", [])
        conditioning_observable_ids = document.get("conditioning_observable_ids", [])
        applicable_subject_kinds = document.get("applicable_subject_kinds", [])
        if not isinstance(signal_discovery_id, str) or not signal_discovery_id:
            raise ValueError("generated signal discovery spec is missing signal_discovery_id")
        if not isinstance(subject_set_id, str) or not subject_set_id:
            raise ValueError("generated signal discovery spec is missing subject_set_id")
        for field_name, values in (
            ("operator_ids", operator_ids),
            ("primary_observable_ids", primary_observable_ids),
            ("secondary_observable_ids", secondary_observable_ids),
            ("conditioning_observable_ids", conditioning_observable_ids),
            ("applicable_subject_kinds", applicable_subject_kinds),
        ):
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item
                for item in values
            ):
                raise ValueError(
                    f"generated signal discovery spec {field_name} must be a list of strings"
                )
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise ValueError("generated signal discovery spec target_id is invalid")
        return cls(
            signal_discovery_id=signal_discovery_id,
            subject_set_id=subject_set_id,
            operator_ids=tuple(operator_ids),
            primary_observable_ids=tuple(primary_observable_ids),
            parameter_space=SignalParameterSpace.from_document(parameter_space),
            target_id=None if target_id is None else target_id,
            secondary_observable_ids=tuple(secondary_observable_ids),
            conditioning_observable_ids=tuple(conditioning_observable_ids),
            applicable_subject_kinds=tuple(applicable_subject_kinds),
            selection_policy=SignalDiscoverySelectionPolicy.from_document(
                document.get("selection_policy")
            ),
            constraint=SignalDiscoveryGenerationConstraint.from_document(
                document.get("constraint")
            ),
        )


def _observable_matches_family(
    observable: ObservableDefinition,
    *,
    allowed_families: tuple[str, ...],
) -> bool:
    return observable.family in allowed_families


def _resolve_observables(
    observable_ids: tuple[str, ...],
) -> tuple[ObservableDefinition, ...]:
    resolved: list[ObservableDefinition] = []
    for observable_id in observable_ids:
        definition = find_observable_definition(observable_id)
        if definition is None:
            raise ValueError(f"unknown observable for generator: {observable_id}")
        resolved.append(definition)
    return tuple(resolved)


def _novelty_signature(
    *,
    novelty_key: str,
    operator_id: str,
    operator_family_group: str,
    primary_observable_id: str,
    secondary_observable_id: str | None,
    conditioning_observable_id: str | None,
) -> tuple[str, ...]:
    if novelty_key == "operator_primary":
        return (operator_id, primary_observable_id)
    if novelty_key == "family_group_primary":
        return (operator_family_group, primary_observable_id)
    return (
        operator_family_group,
        primary_observable_id,
        "-" if secondary_observable_id is None else secondary_observable_id,
        "-" if conditioning_observable_id is None else conditioning_observable_id,
    )


def generate_signal_discovery(
    spec: SignalDiscoveryGenerationSpec,
) -> SignalDiscoverySpec:
    primary_observables = _resolve_observables(spec.primary_observable_ids)
    secondary_observables = _resolve_observables(spec.secondary_observable_ids)
    conditioning_observables = _resolve_observables(spec.conditioning_observable_ids)
    families: list[SignalFamily] = []
    families_per_operator: dict[str, int] = {}
    families_per_primary_observable: dict[str, int] = {}
    families_per_family_group: dict[str, int] = {}
    novelty_signatures: set[tuple[str, ...]] = set()

    for operator_id in spec.operator_ids:
        operator = find_signal_operator_definition(operator_id)
        if operator is None:
            raise ValueError(f"unknown operator in generator: {operator_id}")
        if (
            spec.constraint.allowed_family_groups
            and operator.family_group not in spec.constraint.allowed_family_groups
        ):
            continue
        matching_primaries = [
            item
            for item in primary_observables
            if _observable_matches_family(
                item,
                allowed_families=operator.primary_observable_families,
            )
        ]
        if not matching_primaries:
            continue
        matching_secondaries = [
            item
            for item in secondary_observables
            if _observable_matches_family(
                item,
                allowed_families=operator.secondary_observable_families,
            )
        ]
        matching_conditionings = [
            item
            for item in conditioning_observables
            if _observable_matches_family(
                item,
                allowed_families=operator.conditioning_observable_families,
            )
        ]
        secondary_choices: tuple[ObservableDefinition | None, ...]
        conditioning_choices: tuple[ObservableDefinition | None, ...]
        if operator.requires_secondary:
            secondary_choices = tuple(matching_secondaries)
        else:
            secondary_choices = (None, *matching_secondaries)
        if operator.requires_conditioning:
            conditioning_choices = tuple(matching_conditionings)
        else:
            conditioning_choices = (None, *matching_conditionings)
        if not secondary_choices or not conditioning_choices:
            continue
        for primary, secondary, conditioning in product(
            matching_primaries,
            secondary_choices,
            conditioning_choices,
        ):
            observable_ids = [
                primary.observable_id,
                *( [] if secondary is None else [secondary.observable_id] ),
                *( [] if conditioning is None else [conditioning.observable_id] ),
            ]
            if spec.constraint.require_distinct_observables and (
                len(observable_ids) != len(set(observable_ids))
            ):
                continue
            novelty_signature = _novelty_signature(
                novelty_key=spec.constraint.novelty_key,
                operator_id=operator.operator_id,
                operator_family_group=operator.family_group,
                primary_observable_id=primary.observable_id,
                secondary_observable_id=(
                    None if secondary is None else secondary.observable_id
                ),
                conditioning_observable_id=(
                    None if conditioning is None else conditioning.observable_id
                ),
            )
            if novelty_signature in novelty_signatures:
                continue
            applicable_subject_kinds = spec.applicable_subject_kinds or (
                operator.applicable_subject_kinds or primary.applicable_subject_kinds
            )
            family = SignalFamily(
                family_id="__".join(
                    [
                        operator.operator_id,
                        primary.observable_id,
                        *( [] if secondary is None else [secondary.observable_id] ),
                        *( [] if conditioning is None else [conditioning.observable_id] ),
                    ]
                ),
                kind=operator.generated_kind,
                parameter_space=spec.parameter_space,
                required_observable_id=primary.observable_id,
                target_id=spec.target_id,
                family_group=operator.family_group,
                secondary_observable_ids=(
                    ()
                    if secondary is None
                    else (secondary.observable_id,)
                ),
                conditioning_observable_ids=(
                    ()
                    if conditioning is None
                    else (conditioning.observable_id,)
                ),
                applicable_subject_kinds=tuple(applicable_subject_kinds),
                thesis=operator.thesis,
            )
            kept_count = families_per_operator.get(operator.operator_id, 0)
            if (
                spec.constraint.max_families_per_operator is not None
                and kept_count >= spec.constraint.max_families_per_operator
            ):
                continue
            kept_primary_count = families_per_primary_observable.get(
                primary.observable_id,
                0,
            )
            if (
                spec.constraint.max_families_per_primary_observable is not None
                and kept_primary_count
                >= spec.constraint.max_families_per_primary_observable
            ):
                continue
            kept_group_count = families_per_family_group.get(operator.family_group, 0)
            if (
                spec.constraint.max_families_per_family_group is not None
                and kept_group_count >= spec.constraint.max_families_per_family_group
            ):
                continue
            families.append(family)
            families_per_operator[operator.operator_id] = kept_count + 1
            families_per_primary_observable[primary.observable_id] = (
                kept_primary_count + 1
            )
            families_per_family_group[operator.family_group] = kept_group_count + 1
            novelty_signatures.add(novelty_signature)
            if (
                spec.constraint.max_families_total is not None
                and len(families) >= spec.constraint.max_families_total
            ):
                break
        if (
            spec.constraint.max_families_total is not None
            and len(families) >= spec.constraint.max_families_total
        ):
            break

    if not families:
        raise ValueError(
            f"generated signal discovery spec resolves no legal families: {spec.signal_discovery_id}"
        )
    return SignalDiscoverySpec(
        signal_discovery_id=spec.signal_discovery_id,
        subject_set_id=spec.subject_set_id,
        families=tuple(families),
        target_id=spec.target_id,
        selection_policy=spec.selection_policy,
    )


def materialize_signal_specs(
    signal_discovery: SignalDiscoverySpec,
) -> tuple[SignalSpec, ...]:
    specification_definitions: list[SignalSpec] = []
    seen_ids: set[str] = set()
    for family in signal_discovery.families:
        lookbacks = family.parameter_space.axis_values("lookback")
        if not lookbacks:
            raise ValueError(
                "generated family parameter_space must include lookback to materialize specifications: "
                f"{family.family_id or family.kind}"
            )
        target_id = family.target_id or "residual_return_3d"
        target_definition = get_target_definition(target_id)
        for lookback in lookbacks:
            if not isinstance(lookback, int):
                raise ValueError(
                    "generated family lookback values must be integers to materialize specifications: "
                    f"{family.family_id or family.kind}"
                )
            signal_id = f"{family.resolved_family_id}__lookback_{lookback}"
            if signal_id in seen_ids:
                continue
            seen_ids.add(signal_id)
            specification_definitions.append(
                SignalSpec(
                    signal_id=signal_id,
                    kind=family.kind,
                    lookback=lookback,
                    required_observable_id=family.required_observable_id,
                    target=target_definition,
                )
            )
    return tuple(specification_definitions)
