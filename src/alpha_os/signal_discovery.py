from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


_PARAMETER_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class SignalParameterAxis:
    name: str
    values: tuple[str | int | float | bool, ...]

    def to_document(self) -> list[str | int | float | bool]:
        return list(self.values)

    @classmethod
    def from_document(
        cls,
        *,
        name: str,
        document: object,
    ) -> "SignalParameterAxis":
        if not isinstance(name, str) or not name:
            raise ValueError("signal parameter axis is missing name")
        if not isinstance(document, list) or not document:
            raise ValueError(f"signal parameter axis is missing values: {name}")
        normalized_values: list[str | int | float | bool] = []
        for item in document:
            if not isinstance(item, _PARAMETER_SCALAR_TYPES):
                raise ValueError(
                    f"signal parameter axis values must be scalar: {name}"
                )
            normalized_values.append(item)
        return cls(name=name, values=tuple(normalized_values))


@dataclass(frozen=True)
class SignalParameterSpace:
    axes: tuple[SignalParameterAxis, ...]

    def to_document(self) -> dict[str, list[str | int | float | bool]]:
        return {
            axis.name: axis.to_document()
            for axis in self.axes
        }

    @classmethod
    def from_document(
        cls,
        document: object,
    ) -> "SignalParameterSpace":
        if not isinstance(document, dict) or not document:
            raise ValueError("signal parameter space must be a non-empty object")
        axes = []
        for name, values in document.items():
            axes.append(
                SignalParameterAxis.from_document(
                    name=str(name),
                    document=values,
                )
            )
        return cls(axes=tuple(axes))

    def matches(self, params: dict[str, object]) -> bool:
        for axis in self.axes:
            if axis.name not in params:
                return False
            if params[axis.name] not in axis.values:
                return False
        return True

    def axis_values(self, name: str) -> tuple[str | int | float | bool, ...]:
        for axis in self.axes:
            if axis.name == name:
                return axis.values
        return ()


@dataclass(frozen=True)
class SignalFamily:
    kind: str
    parameter_space: SignalParameterSpace
    required_observable_id: str = "daily_close"
    target_id: str | None = None
    family_id: str | None = None
    survivor_budget: int | None = None
    family_group: str = "price"
    secondary_observable_ids: tuple[str, ...] = ()
    conditioning_observable_ids: tuple[str, ...] = ()
    applicable_subject_kinds: tuple[str, ...] = ()
    thesis: str | None = None

    @property
    def lookbacks(self) -> tuple[int, ...]:
        values = self.parameter_space.axis_values("lookback")
        return tuple(value for value in values if isinstance(value, int))

    @property
    def resolved_family_id(self) -> str:
        if self.family_id is not None:
            return self.family_id
        payload = {
            "kind": self.kind,
            "parameter_space": self.parameter_space.to_document(),
            "required_observable_id": self.required_observable_id,
            "target_id": self.target_id,
            "family_group": self.family_group,
            "secondary_observable_ids": list(self.secondary_observable_ids),
            "conditioning_observable_ids": list(self.conditioning_observable_ids),
            "applicable_subject_kinds": list(self.applicable_subject_kinds),
        }
        return json.dumps(payload, sort_keys=True)

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "kind": self.kind,
            "parameter_space": self.parameter_space.to_document(),
            "required_observable_id": self.required_observable_id,
        }
        if self.family_id is not None:
            document["family_id"] = self.family_id
        if self.target_id is not None:
            document["target_id"] = self.target_id
        if self.survivor_budget is not None:
            document["survivor_budget"] = self.survivor_budget
        if self.family_group != "price":
            document["family_group"] = self.family_group
        if self.secondary_observable_ids:
            document["secondary_observable_ids"] = list(self.secondary_observable_ids)
        if self.conditioning_observable_ids:
            document["conditioning_observable_ids"] = list(
                self.conditioning_observable_ids
            )
        if self.applicable_subject_kinds:
            document["applicable_subject_kinds"] = list(
                self.applicable_subject_kinds
            )
        if self.thesis is not None:
            document["thesis"] = self.thesis
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SignalFamily":
        family_id = document.get("family_id")
        kind = document.get("kind")
        parameter_space_document = document.get("parameter_space")
        required_observable_id = document.get("required_observable_id", "daily_close")
        target_id = document.get("target_id")
        survivor_budget = document.get("survivor_budget")
        family_group = document.get("family_group", "price")
        secondary_observable_ids = document.get("secondary_observable_ids", [])
        conditioning_observable_ids = document.get("conditioning_observable_ids", [])
        applicable_subject_kinds = document.get("applicable_subject_kinds", [])
        thesis = document.get("thesis")
        if parameter_space_document is None and "lookbacks" in document:
            parameter_space_document = {"lookback": document.get("lookbacks")}
        if family_id is not None and (not isinstance(family_id, str) or not family_id):
            raise ValueError(f"signal family family_id is invalid: {kind}")
        if not isinstance(kind, str) or not kind:
            raise ValueError("signal family is missing kind")
        parameter_space = SignalParameterSpace.from_document(
            parameter_space_document
        )
        if (
            not isinstance(required_observable_id, str)
            or not required_observable_id
        ):
            raise ValueError(
                f"signal family required_observable_id is invalid: {kind}"
            )
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise ValueError(f"signal family target_id is invalid: {kind}")
        if survivor_budget is not None and (
            not isinstance(survivor_budget, int) or survivor_budget < 1
        ):
            raise ValueError(f"signal family survivor_budget is invalid: {kind}")
        if not isinstance(family_group, str) or not family_group:
            raise ValueError(f"signal family family_group is invalid: {kind}")
        if not isinstance(secondary_observable_ids, list) or any(
            not isinstance(item, str) or not item
            for item in secondary_observable_ids
        ):
            raise ValueError(
                f"signal family secondary_observable_ids are invalid: {kind}"
            )
        if not isinstance(conditioning_observable_ids, list) or any(
            not isinstance(item, str) or not item
            for item in conditioning_observable_ids
        ):
            raise ValueError(
                f"signal family conditioning_observable_ids are invalid: {kind}"
            )
        if not isinstance(applicable_subject_kinds, list) or any(
            not isinstance(item, str) or not item
            for item in applicable_subject_kinds
        ):
            raise ValueError(
                f"signal family applicable_subject_kinds are invalid: {kind}"
            )
        if thesis is not None and (not isinstance(thesis, str) or not thesis):
            raise ValueError(f"signal family thesis is invalid: {kind}")
        return cls(
            family_id=None if family_id is None else family_id,
            kind=kind,
            parameter_space=parameter_space,
            required_observable_id=required_observable_id,
            target_id=None if target_id is None else target_id,
            survivor_budget=survivor_budget,
            family_group=family_group,
            secondary_observable_ids=tuple(secondary_observable_ids),
            conditioning_observable_ids=tuple(conditioning_observable_ids),
            applicable_subject_kinds=tuple(applicable_subject_kinds),
            thesis=None if thesis is None else thesis,
        )

    def matches_specification(self, specification) -> bool:
        if specification.kind != self.kind:
            return False
        if not self.parameter_space.matches(specification.params):
            return False
        if specification.required_observable_id != self.required_observable_id:
            return False
        if self.target_id is not None and specification.target_id != self.target_id:
            return False
        return True


@dataclass(frozen=True)
class SignalDiscoverySelectionPolicy:
    min_sample_count: int = 10
    min_abs_corr: float = 0.05
    min_stability_score: float = 0.0
    pre_screen_top_k_per_kind: int | None = None
    pre_screen_min_abs_corr: float = 0.0
    probe_max_dates: int | None = None
    probe_min_sample_count: int = 0
    probe_min_abs_corr: float = 0.0
    probe_max_family_survivors_per_subject: int | None = None
    survivor_min_sample_count: int = 0
    survivor_min_abs_corr: float = 0.0
    survivor_max_family_survivors_per_subject: int | None = None
    snapshot_retention: str = "latest_per_survivor"
    adaptive_family_budget: bool = True
    adaptive_budget_stability_scale: float = 2.0
    max_family_survivors_per_subject: int = 2

    def to_document(self) -> dict[str, Any]:
        return {
            "min_sample_count": self.min_sample_count,
            "min_abs_corr": self.min_abs_corr,
            "min_stability_score": self.min_stability_score,
            "pre_screen_top_k_per_kind": self.pre_screen_top_k_per_kind,
            "pre_screen_min_abs_corr": self.pre_screen_min_abs_corr,
            "probe_max_dates": self.probe_max_dates,
            "probe_min_sample_count": self.probe_min_sample_count,
            "probe_min_abs_corr": self.probe_min_abs_corr,
            "probe_max_family_survivors_per_subject": self.probe_max_family_survivors_per_subject,
            "survivor_min_sample_count": self.survivor_min_sample_count,
            "survivor_min_abs_corr": self.survivor_min_abs_corr,
            "survivor_max_family_survivors_per_subject": self.survivor_max_family_survivors_per_subject,
            "snapshot_retention": self.snapshot_retention,
            "adaptive_family_budget": self.adaptive_family_budget,
            "adaptive_budget_stability_scale": self.adaptive_budget_stability_scale,
            "max_family_survivors_per_subject": self.max_family_survivors_per_subject,
        }

    @classmethod
    def from_document(cls, document: object) -> "SignalDiscoverySelectionPolicy":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("signal discovery selection_policy must be an object")
        min_sample_count = document.get("min_sample_count", 10)
        min_abs_corr = document.get("min_abs_corr", 0.05)
        min_stability_score = document.get("min_stability_score", 0.0)
        pre_screen_top_k_per_kind = document.get("pre_screen_top_k_per_kind")
        pre_screen_min_abs_corr = document.get("pre_screen_min_abs_corr", 0.0)
        probe_max_dates = document.get("probe_max_dates")
        probe_min_sample_count = document.get("probe_min_sample_count", 0)
        probe_min_abs_corr = document.get("probe_min_abs_corr", 0.0)
        probe_max_family_survivors_per_subject = document.get(
            "probe_max_family_survivors_per_subject"
        )
        survivor_min_sample_count = document.get("survivor_min_sample_count", 0)
        survivor_min_abs_corr = document.get("survivor_min_abs_corr", 0.0)
        survivor_max_family_survivors_per_subject = document.get(
            "survivor_max_family_survivors_per_subject"
        )
        snapshot_retention = document.get("snapshot_retention", "latest_per_survivor")
        adaptive_family_budget = document.get("adaptive_family_budget", True)
        adaptive_budget_stability_scale = document.get(
            "adaptive_budget_stability_scale",
            2.0,
        )
        max_family_survivors_per_subject = document.get(
            "max_family_survivors_per_subject",
            2,
        )
        if not isinstance(min_sample_count, int) or min_sample_count < 0:
            raise ValueError("signal discovery min_sample_count must be >= 0")
        if not isinstance(min_abs_corr, (int, float)) or float(min_abs_corr) < 0.0:
            raise ValueError("signal discovery min_abs_corr must be >= 0")
        if (
            not isinstance(min_stability_score, (int, float))
            or float(min_stability_score) < 0.0
        ):
            raise ValueError("signal discovery min_stability_score must be >= 0")
        if pre_screen_top_k_per_kind is not None and (
            not isinstance(pre_screen_top_k_per_kind, int)
            or pre_screen_top_k_per_kind < 1
        ):
            raise ValueError("signal discovery pre_screen_top_k_per_kind must be >= 1")
        if (
            not isinstance(pre_screen_min_abs_corr, (int, float))
            or float(pre_screen_min_abs_corr) < 0.0
        ):
            raise ValueError("signal discovery pre_screen_min_abs_corr must be >= 0")
        if probe_max_dates is not None and (
            not isinstance(probe_max_dates, int) or probe_max_dates < 1
        ):
            raise ValueError("signal discovery probe_max_dates must be >= 1")
        if (
            not isinstance(probe_min_sample_count, int)
            or probe_min_sample_count < 0
        ):
            raise ValueError("signal discovery probe_min_sample_count must be >= 0")
        if (
            not isinstance(probe_min_abs_corr, (int, float))
            or float(probe_min_abs_corr) < 0.0
        ):
            raise ValueError("signal discovery probe_min_abs_corr must be >= 0")
        if probe_max_family_survivors_per_subject is not None and (
            not isinstance(probe_max_family_survivors_per_subject, int)
            or probe_max_family_survivors_per_subject < 1
        ):
            raise ValueError(
                "signal discovery probe_max_family_survivors_per_subject must be >= 1"
            )
        if (
            not isinstance(survivor_min_sample_count, int)
            or survivor_min_sample_count < 0
        ):
            raise ValueError("signal discovery survivor_min_sample_count must be >= 0")
        if (
            not isinstance(survivor_min_abs_corr, (int, float))
            or float(survivor_min_abs_corr) < 0.0
        ):
            raise ValueError("signal discovery survivor_min_abs_corr must be >= 0")
        if survivor_max_family_survivors_per_subject is not None and (
            not isinstance(survivor_max_family_survivors_per_subject, int)
            or survivor_max_family_survivors_per_subject < 1
        ):
            raise ValueError(
                "signal discovery survivor_max_family_survivors_per_subject must be >= 1"
            )
        if snapshot_retention not in {"survivors_only", "latest_per_survivor"}:
            raise ValueError(
                "signal discovery snapshot_retention must be survivors_only or "
                "latest_per_survivor"
            )
        if not isinstance(adaptive_family_budget, bool):
            raise ValueError("signal discovery adaptive_family_budget must be boolean")
        if (
            not isinstance(adaptive_budget_stability_scale, (int, float))
            or float(adaptive_budget_stability_scale) <= 0.0
        ):
            raise ValueError(
                "signal discovery adaptive_budget_stability_scale must be > 0"
            )
        if (
            not isinstance(max_family_survivors_per_subject, int)
            or max_family_survivors_per_subject < 1
        ):
            raise ValueError("signal discovery max_family_survivors_per_subject must be >= 1")
        return cls(
            min_sample_count=min_sample_count,
            min_abs_corr=float(min_abs_corr),
            min_stability_score=float(min_stability_score),
            pre_screen_top_k_per_kind=(
                None
                if pre_screen_top_k_per_kind is None
                else int(pre_screen_top_k_per_kind)
            ),
            pre_screen_min_abs_corr=float(pre_screen_min_abs_corr),
            probe_max_dates=(
                None if probe_max_dates is None else int(probe_max_dates)
            ),
            probe_min_sample_count=int(probe_min_sample_count),
            probe_min_abs_corr=float(probe_min_abs_corr),
            probe_max_family_survivors_per_subject=(
                None
                if probe_max_family_survivors_per_subject is None
                else int(probe_max_family_survivors_per_subject)
            ),
            survivor_min_sample_count=int(survivor_min_sample_count),
            survivor_min_abs_corr=float(survivor_min_abs_corr),
            survivor_max_family_survivors_per_subject=(
                None
                if survivor_max_family_survivors_per_subject is None
                else int(survivor_max_family_survivors_per_subject)
            ),
            snapshot_retention=str(snapshot_retention),
            adaptive_family_budget=adaptive_family_budget,
            adaptive_budget_stability_scale=float(adaptive_budget_stability_scale),
            max_family_survivors_per_subject=max_family_survivors_per_subject,
        )


@dataclass(frozen=True, init=False)
class SignalDiscoverySpec:
    signal_discovery_id: str
    subject_set_id: str
    signal_spec_ids: tuple[str, ...] = ()
    families: tuple[SignalFamily, ...] = ()
    target_id: str | None = None
    selection_policy: SignalDiscoverySelectionPolicy = SignalDiscoverySelectionPolicy()

    def __init__(
        self,
        *,
        signal_discovery_id: str,
        subject_set_id: str,
        signal_spec_ids: tuple[str, ...] | None = None,
        families: tuple[SignalFamily, ...] = (),
        target_id: str | None = None,
        selection_policy: SignalDiscoverySelectionPolicy = SignalDiscoverySelectionPolicy(),
    ) -> None:
        resolved_signal_spec_ids = (
            signal_spec_ids
            if signal_spec_ids is not None
            else ()
        )
        object.__setattr__(self, "signal_discovery_id", signal_discovery_id)
        object.__setattr__(self, "subject_set_id", subject_set_id)
        object.__setattr__(
            self,
            "signal_spec_ids",
            tuple(resolved_signal_spec_ids),
        )
        object.__setattr__(self, "families", tuple(families))
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "selection_policy", selection_policy)

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "subject_set_id": self.subject_set_id,
        }
        if self.signal_spec_ids:
            document["signal_spec_ids"] = list(self.signal_spec_ids)
        if self.families:
            document["families"] = [item.to_document() for item in self.families]
        if self.target_id is not None:
            document["target_id"] = self.target_id
        document["selection_policy"] = self.selection_policy.to_document()
        return document

    def resolve_signal_spec_ids(
        self,
        specifications,
    ) -> tuple[str, ...]:
        selected_ids = list(self.signal_spec_ids)
        selected_id_set = set(selected_ids)
        for family in self.families:
            for specification in specifications:
                if family.matches_specification(specification):
                    if specification.signal_id not in selected_id_set:
                        selected_ids.append(specification.signal_id)
                        selected_id_set.add(specification.signal_id)
        return tuple(selected_ids)

    def family_resolutions_by_signal_spec_id(
        self,
        specifications,
    ) -> dict[str, SignalFamily]:
        resolutions: dict[str, SignalFamily] = {}
        for family in self.families:
            for specification in specifications:
                if family.matches_specification(specification):
                    resolutions[specification.signal_id] = family
        return resolutions

    @classmethod
    def from_document(
        cls,
        *,
        signal_discovery_id: str,
        document: dict[str, Any],
    ) -> "SignalDiscoverySpec":
        subject_set_id = document.get("subject_set_id")
        signal_spec_ids = document.get("signal_spec_ids", [])
        families = document.get("families", [])
        target_id = document.get("target_id")
        selection_policy = SignalDiscoverySelectionPolicy.from_document(
            document.get("selection_policy")
        )
        if not isinstance(subject_set_id, str) or not subject_set_id:
            raise ValueError(
                f"signal discovery spec is missing subject_set_id: {signal_discovery_id}"
            )
        if not isinstance(signal_spec_ids, list):
            raise ValueError(
                "signal discovery spec signal_spec_ids must be a list: "
                f"{signal_discovery_id}"
            )
        if not isinstance(families, list):
            raise ValueError(
                f"signal discovery spec families must be a list: {signal_discovery_id}"
            )
        normalized_specification_ids = []
        for item in signal_spec_ids:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    "signal discovery spec signal_spec_ids must be "
                    f"strings: {signal_discovery_id}"
                )
            normalized_specification_ids.append(item)
        normalized_families = []
        for item in families:
            if not isinstance(item, dict):
                raise ValueError(
                    f"signal discovery spec families must contain objects: {signal_discovery_id}"
                )
            normalized_families.append(SignalFamily.from_document(item))
        if not normalized_specification_ids and not normalized_families:
            raise ValueError(
                "signal discovery spec must contain signal_spec_ids or "
                f"families: {signal_discovery_id}"
            )
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise ValueError(
                f"signal discovery spec target_id is invalid: {signal_discovery_id}"
            )
        return cls(
            signal_discovery_id=signal_discovery_id,
            subject_set_id=subject_set_id,
            signal_spec_ids=tuple(normalized_specification_ids),
            families=tuple(normalized_families),
            target_id=None if target_id is None else target_id,
            selection_policy=selection_policy,
        )
