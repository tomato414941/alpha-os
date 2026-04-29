from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _field(item: object, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name)


def _float_field(item: object, name: str, *, default: float = 0.0) -> float:
    value = _field(item, name)
    if value is None:
        return float(default)
    return float(value)


@dataclass(frozen=True)
class ValidationSignalSummary:
    signal_id: str
    conditions: int
    positive_corr: int
    mean_corr: float
    mean_mmc: float | None

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "conditions": self.conditions,
            "positive_corr": self.positive_corr,
            "mean_corr": self.mean_corr,
            "mean_mmc": self.mean_mmc,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ValidationSignalSummary":
        return cls(
            signal_id=str(document["signal_id"]),
            conditions=int(document["conditions"]),
            positive_corr=int(document["positive_corr"]),
            mean_corr=float(document["mean_corr"]),
            mean_mmc=(
                None if document.get("mean_mmc") is None else float(document["mean_mmc"])
            ),
        )


@dataclass(frozen=True)
class ValidationMetaSummary:
    aggregation_kind: str
    conditions: int
    wins: int
    mean_corr: float

    def to_document(self) -> dict[str, Any]:
        return {
            "aggregation_kind": self.aggregation_kind,
            "conditions": self.conditions,
            "wins": self.wins,
            "mean_corr": self.mean_corr,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ValidationMetaSummary":
        return cls(
            aggregation_kind=str(document["aggregation_kind"]),
            conditions=int(document["conditions"]),
            wins=int(document["wins"]),
            mean_corr=float(document["mean_corr"]),
        )


@dataclass(frozen=True)
class ValidationDecisionSummary:
    subject_set_id: str | None
    aggregation_kind: str
    conditions: int
    wins: int
    negative_conditions: int
    mean_net: float
    worst_net: float
    mean_drawdown: float
    mean_gross_notional: float
    mean_net_notional: float
    mean_long_notional: float
    mean_short_notional: float
    mean_traded_notional: float
    total_cost_notional: float
    total_funding_cost_notional: float
    total_borrow_cost_notional: float
    total_roll_cost_notional: float
    subject_set_contract_groups: tuple[str, ...] = ()
    universe_policy_fields: dict[str, str | None] = field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        return {
            "subject_set_id": self.subject_set_id,
            "subject_set_contract_groups": list(self.subject_set_contract_groups),
            "universe_policy_fields": dict(self.universe_policy_fields),
            "aggregation_kind": self.aggregation_kind,
            "conditions": self.conditions,
            "wins": self.wins,
            "negative_conditions": self.negative_conditions,
            "mean_net": self.mean_net,
            "worst_net": self.worst_net,
            "mean_drawdown": self.mean_drawdown,
            "mean_gross_notional": self.mean_gross_notional,
            "mean_net_notional": self.mean_net_notional,
            "mean_long_notional": self.mean_long_notional,
            "mean_short_notional": self.mean_short_notional,
            "mean_traded_notional": self.mean_traded_notional,
            "total_cost_notional": self.total_cost_notional,
            "total_funding_cost_notional": self.total_funding_cost_notional,
            "total_borrow_cost_notional": self.total_borrow_cost_notional,
            "total_roll_cost_notional": self.total_roll_cost_notional,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ValidationDecisionSummary":
        subject_set_id = document.get("subject_set_id")
        subject_set_contract_groups = document.get("subject_set_contract_groups", [])
        universe_policy_fields = document.get("universe_policy_fields", {})
        if not isinstance(subject_set_contract_groups, list):
            raise ValueError(
                "validation decision summary subject_set_contract_groups are invalid"
            )
        if not isinstance(universe_policy_fields, dict):
            raise ValueError(
                "validation decision summary universe_policy_fields are invalid"
            )
        return cls(
            subject_set_id=None if subject_set_id in {None, ""} else str(subject_set_id),
            subject_set_contract_groups=tuple(
                str(item) for item in subject_set_contract_groups
            ),
            universe_policy_fields={
                str(key): (
                    None if value is None else str(value)
                )
                for key, value in universe_policy_fields.items()
            },
            aggregation_kind=str(document["aggregation_kind"]),
            conditions=int(document["conditions"]),
            wins=int(document["wins"]),
            negative_conditions=int(document["negative_conditions"]),
            mean_net=float(document["mean_net"]),
            worst_net=float(document["worst_net"]),
            mean_drawdown=float(document["mean_drawdown"]),
            mean_gross_notional=float(document["mean_gross_notional"]),
            mean_net_notional=float(document.get("mean_net_notional", 0.0)),
            mean_long_notional=float(document.get("mean_long_notional", 0.0)),
            mean_short_notional=float(document.get("mean_short_notional", 0.0)),
            mean_traded_notional=float(document["mean_traded_notional"]),
            total_cost_notional=float(document["total_cost_notional"]),
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


@dataclass(frozen=True)
class ValidationResultSet:
    signal_summaries: tuple[ValidationSignalSummary, ...]
    meta_summaries: tuple[ValidationMetaSummary, ...]
    decision_summaries: tuple[ValidationDecisionSummary, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_summaries": [item.to_document() for item in self.signal_summaries],
            "meta_summaries": [item.to_document() for item in self.meta_summaries],
            "decision_summaries": [item.to_document() for item in self.decision_summaries],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ValidationResultSet":
        signal_summaries = document.get("signal_summaries", [])
        meta_summaries = document.get("meta_summaries", [])
        decision_summaries = document.get("decision_summaries", [])
        if not isinstance(signal_summaries, list):
            raise ValueError("validation result set signal_summaries are invalid")
        if not isinstance(meta_summaries, list):
            raise ValueError("validation result set meta_summaries are invalid")
        if not isinstance(decision_summaries, list):
            raise ValueError("validation result set decision_summaries are invalid")
        return cls(
            signal_summaries=tuple(
                ValidationSignalSummary.from_document(item)
                for item in signal_summaries
                if isinstance(item, dict)
            ),
            meta_summaries=tuple(
                ValidationMetaSummary.from_document(item)
                for item in meta_summaries
                if isinstance(item, dict)
            ),
            decision_summaries=tuple(
                ValidationDecisionSummary.from_document(item)
                for item in decision_summaries
                if isinstance(item, dict)
            ),
        )


def build_validation_result_set(
    signal_results: list[object],
    meta_results: list[object],
    decision_results: list[object],
    *,
    subject_set_contract_groups_by_id: dict[str, tuple[str, ...]] | None = None,
    universe_policy_by_subject_set_id: dict[str, dict[str, str | None]] | None = None,
) -> ValidationResultSet:
    grouped_hypotheses: dict[str, list[object]] = {}
    for item in signal_results:
        grouped_hypotheses.setdefault(str(_field(item, "signal_id")), []).append(item)
    signal_summaries = tuple(
        ValidationSignalSummary(
            signal_id=signal_id,
            conditions=len(items),
            positive_corr=sum(1 for item in items if float(_field(item, "corr")) > 0.0),
            mean_corr=sum(float(_field(item, "corr")) for item in items) / len(items),
            mean_mmc=(
                None
                if not [item for item in items if _field(item, "mmc") is not None]
                else sum(
                    float(_field(item, "mmc"))
                    for item in items
                    if _field(item, "mmc") is not None
                )
                / len([item for item in items if _field(item, "mmc") is not None])
            ),
        )
        for signal_id, items in sorted(grouped_hypotheses.items())
    )

    grouped_meta: dict[str, list[object]] = {}
    by_condition: dict[tuple[str, str, int], list[object]] = {}
    for item in meta_results:
        aggregation_kind = str(_field(item, "aggregation_kind"))
        grouped_meta.setdefault(aggregation_kind, []).append(item)
        condition_key = (
            str(_field(item, "date_range_label")),
            str(_field(item, "target_id")),
            int(_field(item, "window_size")),
        )
        by_condition.setdefault(condition_key, []).append(item)
    wins: dict[str, int] = {}
    for items in by_condition.values():
        ordered = sorted(
            items,
            key=lambda item: (-float(_field(item, "corr")), str(_field(item, "aggregation_kind"))),
        )
        if ordered:
            winner = str(_field(ordered[0], "aggregation_kind"))
            wins[winner] = wins.get(winner, 0) + 1
    meta_summaries = tuple(
        ValidationMetaSummary(
            aggregation_kind=aggregation_kind,
            conditions=len(items),
            wins=wins.get(aggregation_kind, 0),
            mean_corr=sum(float(_field(item, "corr")) for item in items) / len(items),
        )
        for aggregation_kind, items in sorted(grouped_meta.items())
    )

    grouped_decisions: dict[tuple[str | None, str], list[object]] = {}
    decision_by_condition: dict[tuple[str, str, str | None, int], list[object]] = {}
    for item in decision_results:
        subject_set_id = _field(item, "subject_set_id")
        aggregation_kind = str(_field(item, "aggregation_kind"))
        group_key = (None if subject_set_id in {None, ""} else str(subject_set_id), aggregation_kind)
        grouped_decisions.setdefault(group_key, []).append(item)
        condition_key = (
            str(_field(item, "date_range_label")),
            str(_field(item, "target_id")),
            None if subject_set_id in {None, ""} else str(subject_set_id),
            int(_field(item, "window_size")),
        )
        decision_by_condition.setdefault(condition_key, []).append(item)
    decision_wins: dict[tuple[str | None, str], int] = {}
    for items in decision_by_condition.values():
        ordered = sorted(
            items,
            key=lambda item: (
                -float(_field(item, "net_return_total")),
                float(_field(item, "max_drawdown")),
                str(_field(item, "aggregation_kind")),
            ),
        )
        if ordered:
            winner_subject_set_id = _field(ordered[0], "subject_set_id")
            winner_key = (
                None if winner_subject_set_id in {None, ""} else str(winner_subject_set_id),
                str(_field(ordered[0], "aggregation_kind")),
            )
            decision_wins[winner_key] = decision_wins.get(winner_key, 0) + 1
    decision_summaries = tuple(
        ValidationDecisionSummary(
            subject_set_id=subject_set_id,
            subject_set_contract_groups=(
                ()
                if subject_set_id is None or subject_set_contract_groups_by_id is None
                else tuple(subject_set_contract_groups_by_id.get(subject_set_id, ()))
            ),
            universe_policy_fields=(
                {}
                if subject_set_id is None or universe_policy_by_subject_set_id is None
                else dict(universe_policy_by_subject_set_id.get(subject_set_id, {}))
            ),
            aggregation_kind=aggregation_kind,
            conditions=len(items),
            wins=decision_wins.get((subject_set_id, aggregation_kind), 0),
            negative_conditions=sum(
                1 for item in items if float(_field(item, "net_return_total")) <= 0.0
            ),
            mean_net=sum(_float_field(item, "net_return_total") for item in items) / len(items),
            worst_net=min(_float_field(item, "net_return_total") for item in items),
            mean_drawdown=sum(_float_field(item, "max_drawdown") for item in items) / len(items),
            mean_gross_notional=(
                sum(_float_field(item, "mean_gross_notional_exposure") for item in items)
                / len(items)
            ),
            mean_net_notional=(
                sum(_float_field(item, "mean_net_notional_exposure") for item in items)
                / len(items)
            ),
            mean_long_notional=(
                sum(_float_field(item, "mean_long_notional_exposure") for item in items)
                / len(items)
            ),
            mean_short_notional=(
                sum(_float_field(item, "mean_short_notional_exposure") for item in items)
                / len(items)
            ),
            mean_traded_notional=(
                sum(_float_field(item, "mean_traded_notional") for item in items) / len(items)
            ),
            total_cost_notional=sum(_float_field(item, "cost_notional_total") for item in items),
            total_funding_cost_notional=sum(
                _float_field(item, "funding_cost_notional_total") for item in items
            ),
            total_borrow_cost_notional=sum(
                _float_field(item, "borrow_cost_notional_total") for item in items
            ),
            total_roll_cost_notional=sum(
                _float_field(item, "roll_cost_notional_total") for item in items
            ),
        )
        for (subject_set_id, aggregation_kind), items in sorted(grouped_decisions.items())
    )

    return ValidationResultSet(
        signal_summaries=signal_summaries,
        meta_summaries=meta_summaries,
        decision_summaries=decision_summaries,
    )
