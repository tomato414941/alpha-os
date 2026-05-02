from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _tuple_of_strings(document: dict[str, Any], field_name: str) -> tuple[str, ...]:
    raw_value = document.get(field_name, ())
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list | tuple):
        raise ValueError(f"{field_name} must be a list of strings")
    values = tuple(str(item) for item in raw_value if str(item))
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _normalize_optional(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    return value


@dataclass(frozen=True)
class StrategySleeveSubjectFilterSpec:
    subject_ids: tuple[str, ...] = ()
    instrument_types: tuple[str, ...] = ()
    asset_classes: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    clusters: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {}
        if self.subject_ids:
            document["subject_ids"] = list(self.subject_ids)
        if self.instrument_types:
            document["instrument_types"] = list(self.instrument_types)
        if self.asset_classes:
            document["asset_classes"] = list(self.asset_classes)
        if self.regions:
            document["regions"] = list(self.regions)
        if self.clusters:
            document["clusters"] = list(self.clusters)
        return document

    @classmethod
    def from_document(
        cls,
        document: dict[str, Any] | None,
    ) -> "StrategySleeveSubjectFilterSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("sleeve subject_filter must be an object")
        return cls(
            subject_ids=_tuple_of_strings(document, "subject_ids"),
            instrument_types=_tuple_of_strings(document, "instrument_types"),
            asset_classes=_tuple_of_strings(document, "asset_classes"),
            regions=_tuple_of_strings(document, "regions"),
            clusters=_tuple_of_strings(document, "clusters"),
        )


@dataclass(frozen=True)
class StrategySleeveSpec:
    sleeve_id: str
    sleeve_kind: str
    risk_budget: float
    signal_source_kind: str | None = None
    signal_discovery_id: str | None = None
    family_mix: str | None = None
    subject_filter: StrategySleeveSubjectFilterSpec = field(
        default_factory=StrategySleeveSubjectFilterSpec
    )
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.sleeve_id, str) or not self.sleeve_id:
            raise ValueError("strategy sleeve is missing sleeve_id")
        if not isinstance(self.sleeve_kind, str) or not self.sleeve_kind:
            raise ValueError(f"strategy sleeve is missing sleeve_kind: {self.sleeve_id}")
        if not isinstance(self.risk_budget, int | float):
            raise ValueError(f"strategy sleeve risk_budget must be numeric: {self.sleeve_id}")
        if self.enabled and float(self.risk_budget) <= 0.0:
            raise ValueError(
                f"enabled strategy sleeve risk_budget must be > 0: {self.sleeve_id}"
            )
        object.__setattr__(self, "risk_budget", float(self.risk_budget))
        object.__setattr__(
            self,
            "signal_source_kind",
            _normalize_optional(self.signal_source_kind),
        )
        object.__setattr__(
            self,
            "signal_discovery_id",
            _normalize_optional(self.signal_discovery_id),
        )
        object.__setattr__(self, "family_mix", _normalize_optional(self.family_mix))

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "sleeve_id": self.sleeve_id,
            "sleeve_kind": self.sleeve_kind,
            "risk_budget": self.risk_budget,
            "enabled": self.enabled,
        }
        if self.signal_source_kind is not None:
            document["signal_source_kind"] = self.signal_source_kind
        if self.signal_discovery_id is not None:
            document["signal_discovery_id"] = self.signal_discovery_id
        if self.family_mix is not None:
            document["family_mix"] = self.family_mix
        subject_filter = self.subject_filter.to_document()
        if subject_filter:
            document["subject_filter"] = subject_filter
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "StrategySleeveSpec":
        if not isinstance(document, dict):
            raise ValueError("strategy sleeve must be an object")
        return cls(
            sleeve_id=str(document.get("sleeve_id", "")),
            sleeve_kind=str(document.get("sleeve_kind", "")),
            risk_budget=float(document.get("risk_budget", 0.0)),
            signal_source_kind=(
                None
                if document.get("signal_source_kind") is None
                else str(document["signal_source_kind"])
            ),
            signal_discovery_id=(
                None
                if document.get("signal_discovery_id") is None
                else str(document["signal_discovery_id"])
            ),
            family_mix=(
                None if document.get("family_mix") is None else str(document["family_mix"])
            ),
            subject_filter=StrategySleeveSubjectFilterSpec.from_document(
                None
                if document.get("subject_filter") is None
                else dict(document["subject_filter"])
            ),
            enabled=bool(document.get("enabled", True)),
        )


@dataclass(frozen=True)
class StrategySleeveCompositionSpec:
    sleeves: tuple[StrategySleeveSpec, ...]
    combination_method: str = "risk_budgeted_signal_blend"
    normalize_risk_budgets: bool = True

    def __post_init__(self) -> None:
        if self.combination_method != "risk_budgeted_signal_blend":
            raise ValueError(
                "strategy sleeve combination_method must be "
                "risk_budgeted_signal_blend"
            )
        sleeve_ids = [item.sleeve_id for item in self.sleeves]
        if len(sleeve_ids) != len(set(sleeve_ids)):
            raise ValueError("strategy sleeve ids must be unique")
        enabled_budget = sum(item.risk_budget for item in self.sleeves if item.enabled)
        if self.sleeves and enabled_budget <= 0.0:
            raise ValueError("enabled strategy sleeve risk budgets must sum to > 0")

    @property
    def enabled_sleeves(self) -> tuple[StrategySleeveSpec, ...]:
        return tuple(item for item in self.sleeves if item.enabled)

    def to_document(self) -> dict[str, Any]:
        return {
            "combination_method": self.combination_method,
            "normalize_risk_budgets": self.normalize_risk_budgets,
            "sleeves": [item.to_document() for item in self.sleeves],
        }

    def stable_payload(self) -> str:
        return json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_document(
        cls,
        document: dict[str, Any] | None,
    ) -> "StrategySleeveCompositionSpec | None":
        if document is None:
            return None
        if not isinstance(document, dict):
            raise ValueError("sleeve_composition must be an object")
        sleeves = document.get("sleeves", [])
        if not isinstance(sleeves, list):
            raise ValueError("sleeve_composition.sleeves must be a list")
        return cls(
            sleeves=tuple(
                StrategySleeveSpec.from_document(item)
                for item in sleeves
                if isinstance(item, dict)
            ),
            combination_method=str(
                document.get("combination_method", "risk_budgeted_signal_blend")
            ),
            normalize_risk_budgets=bool(document.get("normalize_risk_budgets", True)),
        )


@dataclass(frozen=True)
class SleeveAttributionSummary:
    sleeve_id: str
    sleeve_kind: str
    risk_budget: float
    subject_count: int
    mean_signal: float = 0.0
    mean_abs_signal: float = 0.0
    mean_gross_notional_exposure: float = 0.0
    mean_net_notional_exposure: float = 0.0
    mean_long_notional_exposure: float = 0.0
    mean_short_notional_exposure: float = 0.0
    total_cost_notional: float = 0.0
    total_funding_cost_notional: float = 0.0
    total_borrow_cost_notional: float = 0.0
    total_roll_cost_notional: float = 0.0

    def to_document(self) -> dict[str, Any]:
        return {
            "sleeve_id": self.sleeve_id,
            "sleeve_kind": self.sleeve_kind,
            "risk_budget": self.risk_budget,
            "subject_count": self.subject_count,
            "mean_signal": self.mean_signal,
            "mean_abs_signal": self.mean_abs_signal,
            "mean_gross_notional_exposure": self.mean_gross_notional_exposure,
            "mean_net_notional_exposure": self.mean_net_notional_exposure,
            "mean_long_notional_exposure": self.mean_long_notional_exposure,
            "mean_short_notional_exposure": self.mean_short_notional_exposure,
            "total_cost_notional": self.total_cost_notional,
            "total_funding_cost_notional": self.total_funding_cost_notional,
            "total_borrow_cost_notional": self.total_borrow_cost_notional,
            "total_roll_cost_notional": self.total_roll_cost_notional,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SleeveAttributionSummary":
        if not isinstance(document, dict):
            raise ValueError("sleeve attribution summary must be an object")
        return cls(
            sleeve_id=str(document.get("sleeve_id", "")),
            sleeve_kind=str(document.get("sleeve_kind", "")),
            risk_budget=float(document.get("risk_budget", 0.0)),
            subject_count=int(document.get("subject_count", 0)),
            mean_signal=float(document.get("mean_signal", 0.0)),
            mean_abs_signal=float(document.get("mean_abs_signal", 0.0)),
            mean_gross_notional_exposure=float(
                document.get("mean_gross_notional_exposure", 0.0)
            ),
            mean_net_notional_exposure=float(
                document.get("mean_net_notional_exposure", 0.0)
            ),
            mean_long_notional_exposure=float(
                document.get("mean_long_notional_exposure", 0.0)
            ),
            mean_short_notional_exposure=float(
                document.get("mean_short_notional_exposure", 0.0)
            ),
            total_cost_notional=float(document.get("total_cost_notional", 0.0)),
            total_funding_cost_notional=float(
                document.get("total_funding_cost_notional", 0.0)
            ),
            total_borrow_cost_notional=float(
                document.get("total_borrow_cost_notional", 0.0)
            ),
            total_roll_cost_notional=float(
                document.get("total_roll_cost_notional", 0.0)
            ),
        )


def sleeve_normalized_budget(
    sleeve: StrategySleeveSpec,
    composition: StrategySleeveCompositionSpec,
) -> float:
    if not sleeve.enabled:
        return 0.0
    if not composition.normalize_risk_budgets:
        return float(sleeve.risk_budget)
    total = sum(item.risk_budget for item in composition.enabled_sleeves)
    if total <= 0.0:
        return 0.0
    return float(sleeve.risk_budget / total)
