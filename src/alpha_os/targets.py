from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    family: str
    observation_kind: str
    subject_kind: str
    output_kind: str
    scoring_kind: str
    params: dict[str, Any]

    @property
    def horizon_days(self) -> int | None:
        value = self.params.get("horizon_days")
        return value if isinstance(value, int) else None

    def to_document(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "family": self.family,
            "observation_kind": self.observation_kind,
            "subject_kind": self.subject_kind,
            "output_kind": self.output_kind,
            "scoring_kind": self.scoring_kind,
            "params": dict(self.params),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TargetDefinition":
        target_id = document.get("target_id")
        family = document.get("family")
        observation_kind = document.get("observation_kind")
        subject_kind = document.get("subject_kind")
        output_kind = document.get("output_kind")
        scoring_kind = document.get("scoring_kind")
        params = document.get("params")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target document is missing target_id")
        if not isinstance(family, str) or not family:
            raise ValueError("target document is missing family")
        if not isinstance(observation_kind, str) or not observation_kind:
            raise ValueError("target document is missing observation_kind")
        if not isinstance(subject_kind, str) or not subject_kind:
            raise ValueError("target document is missing subject_kind")
        if not isinstance(output_kind, str) or not output_kind:
            raise ValueError("target document is missing output_kind")
        if not isinstance(scoring_kind, str) or not scoring_kind:
            raise ValueError("target document is missing scoring_kind")
        if not isinstance(params, dict):
            raise ValueError("target document is missing params")
        return cls(
            target_id=target_id,
            family=family,
            observation_kind=observation_kind,
            subject_kind=subject_kind,
            output_kind=output_kind,
            scoring_kind=scoring_kind,
            params=dict(params),
        )


def residual_return_target_definition(horizon_days: int) -> TargetDefinition:
    return TargetDefinition(
        target_id=f"residual_return_{horizon_days}d",
        family="residual_return",
        observation_kind="fixed_horizon",
        subject_kind="asset",
        output_kind="real_value",
        scoring_kind="corr_mmc",
        params={"horizon_days": horizon_days},
    )


_TARGET_DEFINITIONS = {
    definition.target_id: definition
    for definition in (
        residual_return_target_definition(1),
        residual_return_target_definition(3),
        residual_return_target_definition(5),
    )
}


def get_target_definition(target_id: str) -> TargetDefinition:
    try:
        return _TARGET_DEFINITIONS[target_id]
    except KeyError as exc:
        available = ", ".join(sorted(_TARGET_DEFINITIONS))
        raise ValueError(
            f"unknown target definition: {target_id} (available: {available})"
        ) from exc


def find_target_definition(target_id: str) -> TargetDefinition | None:
    return _TARGET_DEFINITIONS.get(target_id)


def list_target_definitions() -> list[TargetDefinition]:
    return [definition for _, definition in sorted(_TARGET_DEFINITIONS.items())]
