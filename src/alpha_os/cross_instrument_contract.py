from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossInstrumentReportUnit:
    unit_id: str
    fields: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "fields": list(self.fields),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentReportUnit":
        fields = document.get("fields", [])
        if not isinstance(fields, list):
            raise ValueError("cross-instrument report unit fields are invalid")
        return cls(
            unit_id=str(document["unit_id"]),
            fields=tuple(str(item) for item in fields if str(item)),
        )

    def format_summary(self) -> str:
        fields = "+".join(self.fields) if self.fields else "-"
        return f"{self.unit_id}={fields}"


@dataclass(frozen=True)
class CrossInstrumentMetricContract:
    outcome_kind: str
    metric_group_name: str
    metric_fields: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "outcome_kind": self.outcome_kind,
            "metric_group_name": self.metric_group_name,
            "metric_fields": list(self.metric_fields),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentMetricContract":
        metric_fields = document.get("metric_fields", [])
        if "dimension_name" in document:
            raise ValueError(
                "cross-instrument metric contract dimension_name field is no longer "
                "supported; use metric_group_name"
            )
        metric_group_name = document.get("metric_group_name")
        if not isinstance(metric_fields, list):
            raise ValueError("cross-instrument metric contract metric_fields are invalid")
        if not isinstance(metric_group_name, str) or not metric_group_name:
            raise ValueError("cross-instrument metric contract metric_group_name is invalid")
        return cls(
            outcome_kind=str(document["outcome_kind"]),
            metric_group_name=metric_group_name,
            metric_fields=tuple(str(item) for item in metric_fields if str(item)),
        )

    def format_summary(self) -> str:
        metric_fields = "+".join(self.metric_fields) if self.metric_fields else "-"
        return f"{self.outcome_kind}:{self.metric_group_name}={metric_fields}"


@dataclass(frozen=True)
class CrossInstrumentReportContract:
    contract_fields: tuple[str, ...]
    outcome_fields: tuple[str, ...]
    report_units: tuple[CrossInstrumentReportUnit, ...] = ()
    metric_contracts: tuple[CrossInstrumentMetricContract, ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "contract_fields": list(self.contract_fields),
            "outcome_fields": list(self.outcome_fields),
            "report_units": [item.to_document() for item in self.report_units],
            "metric_contracts": [item.to_document() for item in self.metric_contracts],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentReportContract":
        contract_fields = document.get("contract_fields", [])
        outcome_fields = document.get("outcome_fields", [])
        report_units = document.get("report_units", [])
        metric_contracts = document.get("metric_contracts", [])
        if "comparison_units" in document:
            raise ValueError(
                "cross-instrument contract comparison_units field is no longer "
                "supported; use report_units"
            )
        if not isinstance(contract_fields, list):
            raise ValueError("cross-instrument contract contract_fields are invalid")
        if not isinstance(outcome_fields, list):
            raise ValueError("cross-instrument contract outcome_fields are invalid")
        if not isinstance(report_units, list):
            raise ValueError("cross-instrument contract report_units are invalid")
        if not isinstance(metric_contracts, list):
            raise ValueError("cross-instrument contract metric_contracts are invalid")
        return cls(
            contract_fields=tuple(str(item) for item in contract_fields if str(item)),
            outcome_fields=tuple(str(item) for item in outcome_fields if str(item)),
            report_units=tuple(
                CrossInstrumentReportUnit.from_document(item)
                for item in report_units
                if isinstance(item, dict)
            ),
            metric_contracts=tuple(
                CrossInstrumentMetricContract.from_document(item)
                for item in metric_contracts
                if isinstance(item, dict)
            ),
        )

    def format_summary(self) -> str:
        contract = ",".join(self.contract_fields) if self.contract_fields else "-"
        outcomes = ",".join(self.outcome_fields) if self.outcome_fields else "-"
        return f"contract={contract} outcomes={outcomes}"

    def format_report_units(self) -> str:
        if not self.report_units:
            return "-"
        return ", ".join(item.format_summary() for item in self.report_units)

    def format_metric_contracts(self) -> str:
        if not self.metric_contracts:
            return "-"
        return ", ".join(item.format_summary() for item in self.metric_contracts)


def default_validation_result_set_cross_instrument_contract() -> CrossInstrumentReportContract:
    return CrossInstrumentReportContract(
        contract_fields=(
            "subject_set",
            "universe_policy",
            "instrument_mix",
            "aggregation_kind",
        ),
        outcome_fields=(
            "mean_net",
            "mean_drawdown",
            "mean_net_notional",
            "mean_long_notional",
            "mean_short_notional",
            "mean_traded_notional",
            "total_cost_notional",
            "total_funding_cost_notional",
            "total_borrow_cost_notional",
            "total_roll_cost_notional",
        ),
        report_units=(
            CrossInstrumentReportUnit(
                unit_id="signal_level",
                fields=("signal_id",),
            ),
            CrossInstrumentReportUnit(
                unit_id="meta_aggregation",
                fields=("aggregation_kind",),
            ),
            CrossInstrumentReportUnit(
                unit_id="decision_aggregation",
                fields=("subject_set_id", "aggregation_kind"),
            ),
        ),
    )
