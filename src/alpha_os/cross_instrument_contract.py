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


def default_evaluation_report_cross_instrument_contract() -> CrossInstrumentReportContract:
    return CrossInstrumentReportContract(
        contract_fields=(
            "strategy",
            "subject_set",
            "universe_policy",
            "instrument_mix",
            "selection",
            "sizing",
            "rebalance",
            "risk_caps",
            "costs",
        ),
        outcome_fields=(
            "metric_group_outcomes",
            "failure_finding_outcomes",
        ),
        report_units=(
            CrossInstrumentReportUnit(
                unit_id="task_result",
                fields=("strategy_id", "evaluation_task_id"),
            ),
            CrossInstrumentReportUnit(
                unit_id="metric_group_outcome",
                fields=("evaluation_task_id", "metric_group_name", "source"),
            ),
            CrossInstrumentReportUnit(
                unit_id="failure_finding_outcome",
                fields=("evaluation_task_id", "metric_group_name", "source"),
            ),
        ),
        metric_contracts=(
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="signed_belief_quality",
                metric_fields=(
                    "mean_survivor_corr",
                    "mean_survivor_stability_score",
                    "mean_component_confidence",
                    "mean_range_signed_belief_corr",
                    "best_range_signed_belief_corr",
                    "worst_range_signed_belief_corr",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="prediction_diagnostics",
                metric_fields=(
                    "mean_signal_forward_corr",
                    "mean_signal_hit_rate",
                    "mean_long_short_forward_spread",
                    "mean_long_bucket_return",
                    "mean_short_bucket_return",
                    "mean_prediction_coverage",
                    "positive_group_fraction",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="portfolio_target_return_alignment",
                metric_fields=(
                    "mean_range_portfolio_target_return_corr",
                    "best_range_portfolio_target_return_corr",
                    "worst_range_portfolio_target_return_corr",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="decision_quality",
                metric_fields=(
                    "mean_decision_net_return",
                    "best_decision_net_return",
                    "mean_decision_drawdown",
                    "mean_decision_turnover",
                    "mean_decision_gross_leverage_exposure",
                    "mean_decision_net_leverage_exposure",
                    "mean_decision_long_leverage_exposure",
                    "mean_decision_short_leverage_exposure",
                    "mean_decision_gross_notional_exposure",
                    "mean_decision_net_notional_exposure",
                    "mean_decision_long_notional_exposure",
                    "mean_decision_short_notional_exposure",
                    "mean_decision_traded_notional",
                    "total_decision_cost_notional",
                    "total_decision_funding_cost_notional",
                    "total_decision_borrow_cost_notional",
                    "total_decision_roll_cost_notional",
                    "mean_decision_step_count",
                    "total_decision_step_count",
                    "mean_step_net_return",
                    "step_net_return_std",
                    "pooled_step_max_drawdown",
                    "annualized_step_sharpe",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="portfolio_concentration",
                metric_fields=(
                    "mean_effective_n",
                    "mean_active_position_count",
                    "mean_top1_gross_share",
                    "mean_top3_gross_share",
                    "mean_top5_gross_share",
                    "mean_top_intent_gross_share",
                    "max_subject_gross_share",
                    "max_cluster_gross_share",
                    "effective_n_floor",
                    "top_gross_share_cap_n",
                    "top_gross_share_cap",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="portfolio_risk_budget",
                metric_fields=(
                    "risk_normalization_mode",
                    "allow_releverage",
                    "target_gross_exposure",
                    "mean_gross_budget_utilization",
                    "mean_gross_budget_error",
                    "mean_decision_gross_leverage_exposure",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="portfolio_construction_trace",
                metric_fields=(
                    "risk_budget_stage_mean_gross_delta",
                    "target_vol_stage_mean_gross_delta",
                    "net_target_stage_mean_net_delta",
                    "top_k_stage_mean_active_count_delta",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="execution_trace",
                metric_fields=(
                    "mean_desired_turnover",
                    "mean_executed_turnover",
                    "mean_turnover_suppression",
                    "mean_skipped_trade_count",
                    "mean_expected_execution_cost",
                    "mean_trade_utility",
                    "negative_utility_trade_count",
                    "negative_utility_trade_fraction",
                    "utility_rejected_turnover",
                    "priority_filled_turnover",
                    "partial_fill_count",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="cost_drag",
                metric_fields=(
                    "cost_to_gross_pnl",
                    "execution_cost_to_gross_pnl",
                    "total_execution_cost_notional",
                    "top_cost_subjects",
                    "top_cost_clusters",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="signal_churn",
                metric_fields=(
                    "mean_signal_abs_change",
                    "mean_signal_sign_flip_rate",
                    "mean_desired_weight_change",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="sizing_policy_quality",
                metric_fields=(
                    "selected_sizing_method",
                    "selected_sizing_engine",
                    "mean_equal_weight_decision_net_return",
                    "mean_equal_weight_daily_decision_net_return",
                    "mean_selected_vs_equal_weight_decision_net_return_edge",
                    "best_selected_vs_equal_weight_decision_net_return_edge",
                    "worst_selected_vs_equal_weight_decision_net_return_edge",
                    "mean_daily_signal_weighted_vs_equal_weight_decision_net_return_edge",
                    "mean_selected_vs_equal_weight_drawdown_edge",
                    "mean_selected_vs_equal_weight_turnover_edge",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="rebalance_policy_quality",
                metric_fields=(
                    "selected_rebalance_interval_steps",
                    "mean_selected_vs_daily_rebalance_net_return_edge",
                    "best_selected_vs_daily_rebalance_net_return_edge",
                    "worst_selected_vs_daily_rebalance_net_return_edge",
                    "mean_selected_vs_daily_rebalance_turnover_savings",
                    "mean_equal_weight_vs_daily_rebalance_net_return_edge",
                    "mean_equal_weight_vs_daily_rebalance_turnover_savings",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="robustness",
                metric_fields=(
                    "range_count",
                    "signed_belief_corr_std",
                    "portfolio_target_return_corr_std",
                    "decision_net_return_std",
                    "decision_negative_fraction",
                    "worst_decision_net_return",
                ),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="decision_quality",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="portfolio_concentration",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="portfolio_risk_budget",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="sizing_policy_quality",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="rebalance_policy_quality",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="signed_belief_quality",
                metric_fields=("finding_count", "max_severity"),
            ),
            CrossInstrumentMetricContract(
                outcome_kind="failure_finding_outcome",
                metric_group_name="portfolio_target_return_alignment",
                metric_fields=("finding_count", "max_severity"),
            ),
        ),
    )


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
