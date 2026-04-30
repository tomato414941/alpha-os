from __future__ import annotations

from collections import defaultdict

from .cross_instrument_contract import (
    default_evaluation_report_cross_instrument_contract,
    default_validation_result_set_cross_instrument_contract,
)
from .evaluation_report_service import format_report_strategy_contract_fields
from .portfolio_construction_config import inferred_sizing_family
from .subject_set_facts import format_subject_set_facts
from .validation_result_set import build_validation_result_set


def _format_universe_policy_fields(
    universe_policy_fields: dict[str, str | None],
) -> str | None:
    parts = [f"{key}={value}" for key, value in universe_policy_fields.items() if value is not None]
    if not parts:
        return None
    return " ".join(parts)


def _metric_group_result_metric(task_result, dimension: str, metric: str) -> float | None:
    for metric_group_result in task_result.metric_group_results:
        if metric_group_result.metric_group_name != dimension:
            continue
        value = metric_group_result.metrics.get(metric)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        return None
    return None


def _format_optional_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _format_case_metric_facts(report) -> list[str]:
    if not report.task_results:
        return []
    lines = ["  CaseMetricFacts:"]
    for task_result in report.task_results:
        net = _metric_group_result_metric(task_result, "decision_quality", "mean_decision_net_return")
        drawdown = _metric_group_result_metric(
            task_result,
            "decision_quality",
            "mean_decision_drawdown",
        )
        turnover = _metric_group_result_metric(
            task_result,
            "decision_quality",
            "mean_decision_turnover",
        )
        cost_drag = _metric_group_result_metric(
            task_result,
            "cost_drag",
            "execution_cost_to_gross_pnl",
        )
        top3_share = _metric_group_result_metric(
            task_result,
            "portfolio_concentration",
            "mean_top3_gross_share",
        )
        target_return_corr = _metric_group_result_metric(
            task_result,
            "portfolio_target_return_alignment",
            "mean_range_portfolio_target_return_corr",
        )
        rejected_turnover = _metric_group_result_metric(
            task_result,
            "execution_trace",
            "utility_rejected_turnover",
        )
        lines.append(
            "    Task: "
            f"{task_result.evaluation_task_id} "
            f"net={_format_optional_metric(net)} "
            f"drawdown={_format_optional_metric(drawdown)} "
            f"turnover={_format_optional_metric(turnover)} "
            f"cost_drag={_format_optional_metric(cost_drag)} "
            f"top3_share={_format_optional_metric(top3_share)} "
            f"target_return_corr={_format_optional_metric(target_return_corr)} "
            f"utility_rejected_turnover={_format_optional_metric(rejected_turnover)}"
        )
    return lines


def format_snapshot_replay_artifacts(snapshot) -> str | None:
    parts: list[str] = []
    if getattr(snapshot, "funding_cost_bps", None) is not None:
        parts.append(f"funding_bps={snapshot.funding_cost_bps:.6f}")
    if getattr(snapshot, "borrow_fee_bps", None) is not None:
        parts.append(f"borrow_bps={snapshot.borrow_fee_bps:.6f}")
    if getattr(snapshot, "roll_cost_bps", None) is not None:
        parts.append(f"roll_bps={snapshot.roll_cost_bps:.6f}")
    if getattr(snapshot, "contract_multiplier", None) is not None:
        parts.append(f"multiplier={snapshot.contract_multiplier:.6f}")
    if not parts:
        return None
    return " ".join(parts)


def _format_sleeve_attribution(summary) -> str:
    return (
        f"sleeve={summary.sleeve_id} "
        f"kind={summary.sleeve_kind} "
        f"risk_budget={summary.risk_budget:.6f} "
        f"subjects={summary.subject_count} "
        f"signal={summary.mean_signal:.6f} "
        f"abs_signal={summary.mean_abs_signal:.6f} "
        f"gross={summary.mean_gross_notional_exposure:.6f} "
        f"net={summary.mean_net_notional_exposure:.6f} "
        f"long={summary.mean_long_notional_exposure:.6f} "
        f"short={summary.mean_short_notional_exposure:.6f} "
        f"cost={summary.total_cost_notional:.6f} "
        f"funding={summary.total_funding_cost_notional:.6f} "
        f"borrow={summary.total_borrow_cost_notional:.6f} "
        f"roll={summary.total_roll_cost_notional:.6f}"
    )


def _format_tail_risk_row(row) -> str:
    return (
        f"{row.label} "
        f"net_pnl={row.net_pnl_notional:.6f} "
        f"gross_pnl={row.gross_pnl_notional:.6f} "
        f"cost={row.cost_notional:.6f} "
        f"wrong_way={row.wrong_way_pnl_notional:.6f} "
        f"avg_weight={row.average_weight:.6f} "
        f"avg_signal={row.average_signal:.6f} "
        f"avg_gross={row.average_gross_exposure:.6f}"
    )


def _format_baseline_contribution_row(row) -> str:
    return (
        f"{row.label} "
        f"steps={row.subject_step_count} "
        f"net_pnl={row.net_pnl_notional:.6f} "
        f"gross_pnl={row.gross_pnl_notional:.6f} "
        f"cost={row.cost_notional:.6f} "
        f"wrong_way={row.wrong_way_pnl_notional:.6f} "
        f"avg_weight={row.average_weight:.6f} "
        f"avg_signal={row.average_signal:.6f} "
        f"avg_gross={row.average_gross_exposure:.6f}"
    )


def _format_ablation_contribution_row(row) -> str:
    return (
        f"{row.label} "
        f"steps={row.subject_step_count} "
        f"net_pnl={row.net_pnl_notional:.6f} "
        f"gross_pnl={row.gross_pnl_notional:.6f} "
        f"cost={row.cost_notional:.6f} "
        f"avg_weight={row.average_weight:.6f} "
        f"avg_gross={row.average_gross_exposure:.6f}"
    )


def _append_exposure(lines: list[str], exposure, *, indent: str = "    ") -> None:
    subject_label = exposure.max_subject_concentration_label or "-"
    cluster_label = exposure.max_cluster_concentration_label or "-"
    lines.append(
        f"{indent}exposure "
        f"avg_gross={exposure.average_gross_exposure:.6f} "
        f"avg_net={exposure.average_net_exposure:.6f} "
        f"avg_long={exposure.average_long_exposure:.6f} "
        f"avg_short={exposure.average_short_exposure:.6f} "
        f"avg_risk_scale={exposure.average_risk_scale:.6f} "
        f"max_subject={subject_label}:{exposure.max_subject_concentration:.6f} "
        f"max_cluster={cluster_label}:{exposure.max_cluster_concentration:.6f}"
    )


def format_evaluation_diagnostics(report) -> str:
    lines = [
        "alpha-os evaluation diagnostics",
        f"  Report:   {report.evaluation_report_id}",
        f"  Lane:     {report.evaluation_lane}",
        f"  Variant:  {report.variant}",
        f"  Ranges:   {len(report.ranges)}",
        "  Note:     trace contribution ablation; does not refit allocation or replay dropped-leg turnover",
    ]
    for range_result in report.ranges:
        lines.append(f"  Range: {range_result.range_label}")
        baseline = range_result.baseline
        lines.append("    baseline:")
        direction = baseline.direction
        lines.append(
            "      direction "
            f"subject_steps={direction.subject_step_count} "
            f"active_steps={direction.active_subject_step_count} "
            f"hit_rate={direction.hit_rate:.6f} "
            f"signed_edge={direction.signed_edge:.6f}"
        )
        for row in direction.rows:
            lines.append(
                f"        {row.direction} "
                f"steps={row.subject_step_count} "
                f"hit_rate={row.hit_rate:.6f} "
                f"hits={row.hit_count} "
                f"misses={row.miss_count} "
                f"neutral={row.neutral_count} "
                f"signed_edge={row.signed_edge:.6f} "
                f"net_pnl={row.net_pnl_notional:.6f} "
                f"gross_pnl={row.gross_pnl_notional:.6f} "
                f"wrong_way={row.wrong_way_pnl_notional:.6f}"
            )
        cost = baseline.cost_turnover
        lines.append(
            "      cost_turnover "
            f"gross_return={cost.gross_return:.6f} "
            f"net_return={cost.net_return:.6f} "
            f"return_cost_drag={cost.return_cost_drag:.6f} "
            f"gross_pnl={cost.gross_pnl_notional:.6f} "
            f"net_pnl={cost.net_pnl_notional:.6f} "
            f"cost={cost.cost_notional:.6f} "
            f"execution={cost.execution_cost_notional:.6f} "
            f"funding={cost.funding_cost_notional:.6f} "
            f"borrow={cost.borrow_cost_notional:.6f} "
            f"roll={cost.roll_cost_notional:.6f} "
            f"turnover={cost.total_turnover:.6f} "
            f"avg_turnover={cost.average_turnover:.6f} "
            f"traded={cost.traded_notional:.6f} "
            f"cost_per_traded={cost.cost_per_traded_notional:.6f} "
            f"cost_to_abs_gross_pnl={cost.cost_to_abs_gross_pnl:.6f}"
        )
        _append_exposure(lines, baseline.exposure, indent="      ")
        lines.append("      cluster_contribution:")
        for row in baseline.contribution.cluster_rows:
            lines.append("        " + _format_baseline_contribution_row(row))
        lines.append("      asset_class_contribution:")
        for row in baseline.contribution.asset_class_rows:
            lines.append("        " + _format_baseline_contribution_row(row))
        lines.append("      subject_contribution:")
        for row in baseline.contribution.subject_rows:
            lines.append("        " + _format_baseline_contribution_row(row))
        lines.append("      direction_contribution:")
        for row in baseline.contribution.direction_rows:
            lines.append("        " + _format_baseline_contribution_row(row))
        tail_risk = range_result.tail_risk
        lines.append("    tail_risk:")
        lines.append(
            "      summary "
            f"steps={tail_risk.step_count} "
            f"subject_steps={tail_risk.subject_step_count} "
            f"net_return={tail_risk.net_return:.6f} "
            f"gross_return={tail_risk.gross_return:.6f} "
            f"cost={tail_risk.cost_notional:.6f} "
            f"funding={tail_risk.funding_cost_notional:.6f} "
            f"borrow={tail_risk.borrow_cost_notional:.6f} "
            f"roll={tail_risk.roll_cost_notional:.6f} "
            f"worst_day={tail_risk.worst_day or '-'} "
            f"worst_day_net={tail_risk.worst_day_net_return:.6f} "
            f"max_drawdown={tail_risk.max_drawdown:.6f}"
        )
        lines.append("      cluster_losers:")
        for row in tail_risk.cluster_losers:
            lines.append("        " + _format_tail_risk_row(row))
        lines.append("      asset_class_losers:")
        for row in tail_risk.asset_class_losers:
            lines.append("        " + _format_tail_risk_row(row))
        lines.append("      subject_losers:")
        for row in tail_risk.subject_losers:
            lines.append("        " + _format_tail_risk_row(row))
        lines.append("      direction:")
        for row in tail_risk.direction_rows:
            lines.append(
                f"        {row.direction} "
                f"net_pnl={row.net_pnl_notional:.6f} "
                f"gross_pnl={row.gross_pnl_notional:.6f} "
                f"cost={row.cost_notional:.6f} "
                f"wrong_way={row.wrong_way_pnl_notional:.6f}"
            )
        _append_exposure(lines, tail_risk.exposure, indent="      ")
        lines.append("    direction_ablation:")
        for mode_result in range_result.direction_ablation.modes:
            lines.append(
                f"      mode={mode_result.mode} "
                f"steps={mode_result.step_count} "
                f"subject_steps={mode_result.subject_step_count} "
                f"gross_return={mode_result.gross_return:.6f} "
                f"net_return={mode_result.net_return:.6f} "
                f"return_cost_drag={mode_result.return_cost_drag:.6f} "
                f"gross_pnl={mode_result.gross_pnl_notional:.6f} "
                f"net_pnl={mode_result.net_pnl_notional:.6f} "
                f"cost={mode_result.cost_notional:.6f} "
                f"turnover={mode_result.total_turnover:.6f} "
                f"avg_turnover={mode_result.average_turnover:.6f} "
                f"avg_gross={mode_result.average_gross_exposure:.6f} "
                f"avg_net={mode_result.average_net_exposure:.6f} "
                f"avg_long={mode_result.average_long_exposure:.6f} "
                f"avg_short={mode_result.average_short_exposure:.6f}"
            )
            lines.append(f"        asset_class_contribution mode={mode_result.mode}:")
            for row in mode_result.asset_class_rows:
                lines.append("          " + _format_ablation_contribution_row(row))
            lines.append(f"        cluster_contribution mode={mode_result.mode}:")
            for row in mode_result.cluster_rows:
                lines.append("          " + _format_ablation_contribution_row(row))
            lines.append(f"        subject_contribution mode={mode_result.mode}:")
            for row in mode_result.subject_rows:
                lines.append("          " + _format_ablation_contribution_row(row))
    return "\n".join(lines)


def print_evaluation_diagnostics(report) -> None:
    print(format_evaluation_diagnostics(report))


def print_signal_details(signal) -> None:
    if getattr(signal, "signal_spec_id", None) is not None:
        print(f"  Specification: {signal.signal_spec_id}")
    print(f"  Asset:    {signal.asset}")
    print(f"  Target:   {signal.target_id}")
    if signal.kind is not None:
        print(f"  Kind:     {signal.kind}")
    if getattr(signal, "observation_text", None) is not None:
        print(f"  Observe:  {signal.observation_text}")
    if signal.lookback is not None:
        print(f"  Lookback: {signal.lookback}")
    if signal.horizon_days is not None:
        print(f"  Horizon:  {signal.horizon_days}d")
    print(f"  Status:   {signal.status}")
    print(f"  Evals:    {signal.observation_count}")


def print_evaluation_snapshot(snapshot, *, created: bool) -> None:
    outcome = "created" if created else "existing"
    print(f"Evaluation [{outcome}] {snapshot.evaluation_id}")
    print(f"  Asset:    {snapshot.asset}")
    print(f"  Target:   {snapshot.target_id}")
    print(f"  Signal:   {snapshot.signal_id}")
    print(
        f"  Signal:   pred={snapshot.prediction_value:.6f} "
        f"obs={snapshot.observation_value:.6f} edge={snapshot.signed_edge:.6f}"
    )
    print(f"  Error:    abs={snapshot.absolute_error:.6f}")
    replay_artifacts = format_snapshot_replay_artifacts(snapshot)
    if replay_artifacts is not None:
        print(f"  Replay:   {replay_artifacts}")


def print_signal_metric(metric) -> None:
    if metric is None:
        print("  Metrics:  corr=0.000000 mmc=n/a evals=0 mmc_evals=0 peers=0 baseline=-")
        return
    mmc_text = "n/a" if metric.mmc is None else f"{metric.mmc:.6f}"
    baseline_text = "-" if metric.mmc_baseline_type is None else metric.mmc_baseline_type
    print(
        "  Metrics:  "
        f"corr={metric.corr:.6f} "
        f"mmc={mmc_text} "
        f"evals={metric.sample_count} "
        f"mmc_evals={metric.mmc_sample_count} "
        f"peers={metric.mmc_peer_count} "
        f"baseline={baseline_text}"
    )


def print_signal_competition_summary(
    store,
    *,
    signal_ids: list[str],
) -> None:
    selected = set(signal_ids)
    signals = {
        item.signal_id: item
        for item in (store.get_signal(signal_id) for signal_id in signal_ids)
        if item is not None and item.signal_id in selected
    }
    metrics = {item.signal_id: item for item in store.list_signal_metrics(signal_ids=signal_ids)}
    print("alpha-os signal competition")
    print(f"  Count:    {len(signals)}")
    for signal_id in signal_ids:
        signal = signals.get(signal_id)
        if signal is None:
            continue
        metric = metrics.get(signal_id)
        kind = signal.kind or "-"
        observation_text = getattr(signal, "observation_text", None) or "-"
        lookback = "-" if signal.lookback is None else str(signal.lookback)
        horizon = "-" if signal.horizon_days is None else f"{signal.horizon_days}d"
        mmc_text = "n/a" if metric is None or metric.mmc is None else f"{metric.mmc:.6f}"
        baseline_text = (
            "-" if metric is None or metric.mmc_baseline_type is None else metric.mmc_baseline_type
        )
        print(
            f"  {signal.signal_id} "
            f"kind={kind} observation={observation_text} lookback={lookback} horizon={horizon} "
            f"status={signal.status} "
            f"corr={0.0 if metric is None else metric.corr:.6f} "
            f"mmc={mmc_text} "
            f"evals={signal.observation_count if metric is None else metric.sample_count} "
            f"mmc_evals={0 if metric is None else metric.mmc_sample_count} "
            f"peers={0 if metric is None else metric.mmc_peer_count} "
            f"baseline={baseline_text}"
        )


def print_target_summaries(signals, metrics_by_id) -> None:
    grouped = defaultdict(list)
    for signal in signals:
        grouped[signal.target_id].append(signal)

    print("  Targets:")
    for target_id, target_signals in sorted(grouped.items()):
        active = sum(1 for item in target_signals if item.status == "active")
        inactive = sum(1 for item in target_signals if item.status == "inactive")
        target_metrics = [
            metrics_by_id[item.signal_id]
            for item in target_signals
            if item.signal_id in metrics_by_id
        ]
        tracked = len(target_metrics)
        mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in target_metrics) / tracked
        target_mmcs = [item.mmc for item in target_metrics if item.mmc is not None]
        mean_mmc_text = "n/a" if not target_mmcs else f"{sum(target_mmcs) / len(target_mmcs):.6f}"
        print(
            f"    {target_id}: total={len(target_signals)} "
            f"active={active} inactive={inactive} "
            f"tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}"
        )


def print_meta_predictions(meta_predictions) -> None:
    print("alpha-os meta predictions")
    print(f"  Count:    {len(meta_predictions)}")
    for item in meta_predictions:
        print(
            f"  {item.evaluation_id} "
            f"asset={item.asset} "
            f"target={item.target_id} "
            f"kind={item.aggregation_kind} "
            f"value={item.value:.6f} "
            f"contributors={item.contributor_count}"
        )


def print_meta_prediction_metrics(metrics) -> None:
    print("alpha-os meta metrics")
    print(f"  Count:    {len(metrics)}")
    for item in metrics:
        print(
            f"  asset={item.asset} "
            f"target={item.target_id} "
            f"kind={item.aggregation_kind} "
            f"corr={item.corr:.6f} "
            f"evals={item.sample_count}"
        )


def print_meta_aggregation_comparison(metrics) -> None:
    grouped = defaultdict(list)
    for item in metrics:
        grouped[(item.asset, item.target_id)].append(item)

    print("alpha-os meta aggregation comparison")
    print(f"  Targets:  {len(grouped)}")
    for (asset, target_id), items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda item: (-item.corr, -item.sample_count, item.aggregation_kind),
        )
        print(f"  {asset} / {target_id}")
        for rank, item in enumerate(ordered, start=1):
            print(
                f"    {rank}. kind={item.aggregation_kind} "
                f"corr={item.corr:.6f} evals={item.sample_count}"
            )


def print_subject_sets(subject_sets) -> None:
    print("alpha-os subject sets")
    print(f"  Count:    {len(subject_sets)}")
    for item in sorted(
        subject_sets,
        key=lambda item: (
            -len(item.definition.instruments),
            -len(item.definition.bindings),
            item.subject_set_id,
        ),
    ):
        definition = item.definition
        instruments = ", ".join(
            f"{instrument.instrument_id}={instrument.instrument_type}={instrument.asset}"
            + ("" if instrument.venue is None else f"@{instrument.venue}")
            + (
                ""
                if all(
                    value is None
                    for value in (
                        instrument.asset_class,
                        instrument.region,
                        instrument.liquidity_tier,
                        instrument.cluster,
                    )
                )
                else "["
                + ",".join(
                    part
                    for part in (
                        None
                        if instrument.asset_class is None
                        else f"class={instrument.asset_class}",
                        None if instrument.region is None else f"region={instrument.region}",
                        None
                        if instrument.liquidity_tier is None
                        else f"liq={instrument.liquidity_tier}",
                        None if instrument.cluster is None else f"cluster={instrument.cluster}",
                    )
                    if part is not None
                )
                + "]"
            )
            for instrument in definition.instruments
        )
        specs = ", ".join(
            f"{spec.observation_spec_id}={spec.observable_id}@{spec.adapter_kind}"
            + (
                ""
                if not spec.provided_observable_ids
                else "+provides(" + ",".join(spec.provided_observable_ids) + ")"
            )
            for spec in definition.observation_specs
        )
        bindings = ", ".join(
            f"{binding.subject_id}={binding.subject_kind}="
            f"{binding.asset}={binding.observation_spec_id}"
            + ("" if binding.instrument_id is None else f"#{binding.instrument_id}")
            for binding in definition.bindings
        )
        summary_text = format_subject_set_facts(definition)
        instrument_text = "" if not instruments else f" instruments=[{instruments}]"
        print(
            f"  {item.subject_set_id} summary=[{summary_text}]"
            f"{instrument_text} "
            f"observation_specs=[{specs}] subjects=[{bindings}]"
        )


def print_signal_discovery_specs(discoverys) -> None:
    print("alpha-os signal discovery specs")
    print(f"  Count:    {len(discoverys)}")
    for item in discoverys:
        definition = item.definition
        target_text = "-" if definition.target_id is None else definition.target_id
        specifications = ", ".join(definition.signal_spec_ids) or "-"
        family_text = (
            ", ".join(
                (
                    f"{family.resolved_family_id}["
                    + ", ".join(
                        f"{axis.name}={','.join(str(value) for value in axis.values)}"
                        for axis in family.parameter_space.axes
                    )
                    + "]"
                    f"@{family.required_observable_id}"
                    f"{'' if not family.secondary_observable_ids else '+secondary(' + ','.join(family.secondary_observable_ids) + ')'}"
                    f"{'' if not family.conditioning_observable_ids else '+cond(' + ','.join(family.conditioning_observable_ids) + ')'}"
                    f"{'' if family.family_group == 'price' else '#group=' + family.family_group}"
                    f"{'' if not family.applicable_subject_kinds else '#subjects=' + ','.join(family.applicable_subject_kinds)}"
                    f"{'' if family.target_id is None else ':' + family.target_id}"
                    f"{'' if family.survivor_budget is None else '/budget=' + str(family.survivor_budget)}"
                )
                for family in definition.families
            )
            or "-"
        )
        print(
            f"  {item.signal_discovery_id} "
            f"subject_set={definition.subject_set_id} "
            f"target={target_text} "
            f"specifications=[{specifications}] "
            f"families=[{family_text}] "
            f"selection_policy=min_samples={definition.selection_policy.min_sample_count},"
            f"min_abs_corr={definition.selection_policy.min_abs_corr:.6f},"
            f"min_stability={definition.selection_policy.min_stability_score:.6f},"
            "pre_screen_top_k="
            f"{'-' if definition.selection_policy.pre_screen_top_k_per_kind is None else definition.selection_policy.pre_screen_top_k_per_kind},"
            f"pre_screen_min_abs_corr={definition.selection_policy.pre_screen_min_abs_corr:.6f},"
            "probe_max_dates="
            f"{'-' if definition.selection_policy.probe_max_dates is None else definition.selection_policy.probe_max_dates},"
            f"probe_min_samples={definition.selection_policy.probe_min_sample_count},"
            f"probe_min_abs_corr={definition.selection_policy.probe_min_abs_corr:.6f},"
            "probe_max_family_survivors="
            f"{'-' if definition.selection_policy.probe_max_family_survivors_per_subject is None else definition.selection_policy.probe_max_family_survivors_per_subject},"
            f"survivor_min_samples={definition.selection_policy.survivor_min_sample_count},"
            f"survivor_min_abs_corr={definition.selection_policy.survivor_min_abs_corr:.6f},"
            "survivor_max_family_survivors="
            f"{'-' if definition.selection_policy.survivor_max_family_survivors_per_subject is None else definition.selection_policy.survivor_max_family_survivors_per_subject},"
            f"snapshot_retention={definition.selection_policy.snapshot_retention},"
            f"adaptive_budget={str(definition.selection_policy.adaptive_family_budget).lower()},"
            "adaptive_scale="
            f"{definition.selection_policy.adaptive_budget_stability_scale:.6f},"
            "max_family_survivors="
            f"{definition.selection_policy.max_family_survivors_per_subject}"
        )


def print_strategy_adaptation_states(states) -> None:
    print("alpha-os strategy adaptation states")
    print(f"  Count:    {len(states)}")
    for item in states:
        state = item.state
        print(
            f"  {state.strategy_id} "
            f"signal_train={state.signal_train_id} "
            f"signal_discovery={state.signal_discovery_id} "
            f"report={state.source_evaluation_report_id} "
            f"screening={state.source_screening_result_id} "
            f"signals={len(state.signal_reputations)} "
            f"families={len(state.family_reputations)} "
            f"updated={state.created_at}"
        )
        for family in state.family_reputations:
            print(
                f"    family={family.family_id} "
                f"edge={family.mean_edge_score:.6f} "
                f"confidence={family.mean_confidence:.6f} "
                f"stability={family.mean_stability_score:.6f} "
                f"subjects={family.subject_coverage} "
                f"members={family.member_count} "
                f"updates={family.update_count}"
            )


def print_strategy_specs(strategy_specs) -> None:
    print("alpha-os strategy specs")
    print(f"  Count:    {len(strategy_specs)}")
    for item in strategy_specs:
        trading_strategy = item.trading_strategy if hasattr(item, "trading_strategy") else item
        scope = trading_strategy.scope
        signal_definition = trading_strategy.signal_policy.definition_policy
        portfolio_policy = trading_strategy.portfolio.to_portfolio_policy()
        selection = portfolio_policy.selection_policy
        sizing = portfolio_policy.sizing_policy
        rebalance = portfolio_policy.rebalance_policy
        risk = portfolio_policy.risk_policy
        friction = trading_strategy.portfolio.rebalance_friction_policy
        adaptation = trading_strategy.adaptation_policy
        execution = trading_strategy.portfolio.execution_policy
        holding = trading_strategy.portfolio.holding_cost_policy
        is_hold_baseline_strategy = (
            signal_definition.signal_kind == "constant_hold"
            and selection.selection_kind == "all_assets"
            and sizing.sizing_method == "equal_weight"
            and risk.long_only is True
        )
        print(
            f"  {trading_strategy.strategy_id} "
            f"label={trading_strategy.label} "
            f"subject_set={scope.subject_set_id or '-'} "
            f"target={scope.target_id or '-'} "
            f"signal={signal_definition.signal_kind} "
            f"signal_discovery={signal_definition.signal_discovery_id or '-'} "
            f"selection={selection.selection_kind} "
            f"top_k={'-' if selection.top_k is None else selection.top_k} "
            f"sizing={sizing.sizing_method or '-'} "
            f"{'' if is_hold_baseline_strategy else 'family=' + ('-' if sizing.sizing_method is None else inferred_sizing_family(sizing.sizing_method)) + ' '}"
            f"{'holding_style=equal_weight_hold ' if is_hold_baseline_strategy else ''}"
            f"rebalance={rebalance.rebalance or '-'} "
            f"long_only={'-' if risk.long_only is None else str(risk.long_only).lower()} "
            f"direction_mode={risk.direction_mode or '-'} "
            "gross_exposure_cap="
            f"{'-' if risk.gross_exposure_cap is None else risk.gross_exposure_cap} "
            f"target_vol={'-' if risk.target_vol is None else risk.target_vol} "
            "gross_leverage_cap="
            f"{'-' if risk.gross_leverage_cap is None else risk.gross_leverage_cap} "
            "net_exposure_target="
            f"{'-' if risk.net_exposure_target is None else risk.net_exposure_target} "
            "turnover_friction="
            f"{friction.turnover_friction if friction.turnover_friction is not None else '-'} "
            f"no_trade_band={friction.no_trade_band if friction.no_trade_band is not None else '-'} "
            "execution_mode="
            f"{getattr(friction, 'execution_mode', None) or 'utility_priority'} "
            "turnover_budget="
            f"{getattr(friction, 'turnover_budget', None) if getattr(friction, 'turnover_budget', None) is not None else '-'} "
            f"benefit_scale={getattr(friction, 'benefit_scale', 1.0)} "
            f"min_trade_utility={getattr(friction, 'min_trade_utility', 0.0)} "
            f"uncertainty_aversion={getattr(friction, 'uncertainty_aversion', 1.0)} "
            f"risk_aversion={getattr(friction, 'risk_aversion', 0.0)} "
            "partial_fill_enabled="
            f"{str(getattr(friction, 'partial_fill_enabled', True)).lower()} "
            f"market_impact_bps={execution.market_impact_bps if execution.market_impact_bps is not None else '-'} "
            f"fee_bps={execution.fee_bps if execution.fee_bps is not None else '-'} "
            "bid_ask_spread_bps="
            f"{execution.bid_ask_spread_bps if execution.bid_ask_spread_bps is not None else '-'} "
            "funding_bps_per_step="
            f"{holding.funding_bps_per_step if holding.funding_bps_per_step is not None else '-'} "
            "borrow_fee_bps_per_step="
            f"{holding.borrow_fee_bps_per_step if holding.borrow_fee_bps_per_step is not None else '-'} "
            f"adaptation={'on' if adaptation.enabled else 'off'} "
            f"adaptation_blend={adaptation.adaptation_blend:.2f}"
        )


def print_signal_specs(specifications) -> None:
    print("alpha-os signal specs")
    print(f"  Count:    {len(specifications)}")
    for item in specifications:
        horizon = "-" if item.horizon_days is None else f"{item.horizon_days}d"
        print(
            f"  {item.signal_id} "
            f"target={item.target_id} "
            f"kind={item.kind} "
            f"observable={item.definition.required_observable_id} "
            f"lookback={item.lookback} "
            f"horizon={horizon}"
        )


def print_observables(observables) -> None:
    print("alpha-os observables")
    print(f"  Count:    {len(observables)}")
    for item in observables:
        definition = item.definition
        extra = ""
        if definition.input_observable_ids:
            extra += f" inputs={','.join(definition.input_observable_ids)}"
        if definition.applicable_subject_kinds:
            extra += " subject_kinds=" + ",".join(definition.applicable_subject_kinds)
        print(
            f"  {item.observable_id} "
            f"family={definition.family} "
            f"value_kind={definition.value_kind} "
            f"resolution={definition.default_resolution}"
            f"{extra}"
        )


def print_evaluation_specs(evaluation_specs) -> None:
    print("alpha-os evaluation specs")
    print(f"  Count:    {len(evaluation_specs)}")
    for item in evaluation_specs:
        definition = item.definition
        print(
            f"  {item.evaluation_spec_id} "
            f"metric_groups={','.join(definition.metric_group_names)} "
            f"folds={len(definition.resolved_evaluation_folds)} "
            f"execution={definition.execution_range.label}:"
            f"{definition.execution_range.start_date}->{definition.execution_range.end_date}"
        )


print_evaluation_specs = print_evaluation_specs


def print_evaluation_tasks(cases) -> None:
    print("alpha-os evaluation tasks")
    print(f"  Count:    {len(cases)}")
    for item in cases:
        case = item.task if hasattr(item, "task") else item
        print(
            f"  {case.evaluation_task_id} "
            f"evaluation_spec={case.evaluation_spec_id} "
            f"strategy={case.strategy_id} "
        )



def print_evaluation_report(
    report_state,
    *,
    strategy_subject_set_context: dict[str, str] | None = None,
) -> None:
    report = report_state.report if hasattr(report_state, "report") else report_state
    contract = getattr(
        report,
        "cross_instrument_contract",
        default_evaluation_report_cross_instrument_contract(),
    )
    print("alpha-os evaluation report")
    print(f"  Report:    {report.evaluation_report_id}")
    print(f"  Evaluation spec:  {report.evaluation_spec_id}")
    print(f"  Lane:      {report.evaluation_lane}")
    print(f"  Created:   {report.created_at}")
    if getattr(report, "oos_contract_summary", None):
        summary = report.oos_contract_summary
        print(
            "  OOS contract: "
            f"rigor_level={summary.get('rigor_level', '-')} "
            f"enforcement={summary.get('enforcement', '-')} "
            f"date_parse={summary.get('date_parse', '-')} "
            f"range_non_overlap={summary.get('range_non_overlap', '-')} "
            "evaluation_after_execution="
            f"{summary.get('evaluation_after_execution', '-')} "
            f"frozen_state_required={summary.get('frozen_state_required', '-')}"
        )
    print("  CrossInstrumentReportContract: " + contract.format_summary())
    if getattr(contract, "report_units", ()):
        print("  ReportUnits: " + contract.format_report_units())
    if getattr(contract, "metric_contracts", ()):
        print("  MetricContracts: " + contract.format_metric_contracts())
    print(f"  TaskResults: {len(report.task_results)}")
    for line in _format_case_metric_facts(report):
        print(line)
    def _print_task_result(task_result) -> None:
        line = (
            f"  Task: {task_result.evaluation_task_id} "
            f"construction={task_result.construction_kind} "
            f"strategy={task_result.strategy_id}"
            f"{'' if task_result.signal_discovery_id is None else ' signal_discovery=' + task_result.signal_discovery_id}"
        )
        if task_result.strategy_contract_fields:
            line += " " + format_report_strategy_contract_fields(
                task_result.strategy_contract_fields,
                subject_set_facts=task_result.subject_set_facts,
            )
        elif (
            strategy_subject_set_context is not None
            and task_result.strategy_id in strategy_subject_set_context
        ):
            line += f" {strategy_subject_set_context[task_result.strategy_id]}"
        print(line)
        if task_result.subject_set_contract_groups:
            print(
                "    subject_set_contract_groups=" + ",".join(task_result.subject_set_contract_groups)
            )
        universe_policy_text = _format_universe_policy_fields(task_result.universe_policy_fields)
        if universe_policy_text is not None:
            print("    universe_policy=" + universe_policy_text)
        if task_result.constraint_stages:
            print("    constraint_stages=" + ";".join(task_result.constraint_stages))
        for sleeve_summary in task_result.sleeve_attribution_summaries:
            print("    " + _format_sleeve_attribution(sleeve_summary))
        outcome = task_result.cross_instrument_outcome
        metric_group_outcomes = task_result.metric_group_results if outcome is None else outcome.metric_group_outcomes
        for metric_group_result in metric_group_outcomes:
            metrics_text = " ".join(
                f"{key}={value}" for key, value in sorted(metric_group_result.metrics.items())
            )
            print(
                f"    metric_group_name={metric_group_result.metric_group_name} "
                f"source={metric_group_result.source} {metrics_text}"
            )
        failure_finding_outcomes = task_result.failure_finding_groups if outcome is None else outcome.failure_finding_outcomes
        for failure_result in failure_finding_outcomes:
            max_severity = getattr(failure_result, "max_severity", None)
            print(
                f"    failure_metric_group={failure_result.metric_group_name} "
                f"source={failure_result.source} "
                f"findings={getattr(failure_result, 'finding_count', len(getattr(failure_result, 'findings', ())))}"
                + ("" if max_severity is None else f" max_severity={float(max_severity):.6f}")
            )
            for case in getattr(failure_result, "findings", ()):
                metrics_text = " ".join(
                    f"{key}={value}" for key, value in sorted(case.metrics.items())
                )
                print(f"      label={case.label} severity={case.severity:.6f} {metrics_text}")
        for name, values in sorted(task_result.artifact_refs.items()):
            print(f"    {name}={','.join(values) if values else '-'}")

    if report.task_results:
        print("  TaskResultDetails:")
        for task_result in report.task_results:
            _print_task_result(task_result)


def print_subject_set_backend_checks(
    subject_set_id: str,
    checks,
    *,
    base_url: str,
) -> None:
    print("alpha-os subject-set backend check")
    print(f"  SubjectSet: {subject_set_id}")
    print(f"  BaseURL:    {base_url}")
    print(f"  Count:      {len(checks)}")
    for item in checks:
        status = "ok" if item["available"] else "missing"
        line = (
            f"  {item['subject_id']} kind={item['subject_kind']} "
            f"asset={item['asset']} observable={item['observable_id']} "
            f"source={item['source_id']} resolution={item['resolution']} status={status}"
        )
        if item["available"]:
            line += (
                f" category={item['category']} "
                f"type={item['signal_type']} "
                f"updated={item['last_updated'] or '-'}"
            )
        print(line)


def _format_portfolio_decision_strategy(details: dict[str, object]) -> str | None:
    strategy = details.get("strategy")
    if not isinstance(strategy, dict):
        return None
    strategy_id = strategy.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        return None
    parts = [strategy_id]
    selection_kind = strategy.get("selection_kind")
    if isinstance(selection_kind, str) and selection_kind:
        parts.append(f"selection={selection_kind}")
    sizing_method = strategy.get("sizing_method")
    if isinstance(sizing_method, str) and sizing_method:
        parts.append(f"sizing={sizing_method}")
    rebalance = strategy.get("rebalance")
    if isinstance(rebalance, str) and rebalance:
        parts.append(f"rebalance={rebalance}")
    top_k = strategy.get("top_k")
    if top_k is not None:
        parts.append(f"top_k={top_k}")
    long_only = strategy.get("long_only")
    if isinstance(long_only, bool):
        parts.append(f"long_only={str(long_only).lower()}")
    direction_mode = strategy.get("direction_mode")
    if isinstance(direction_mode, str) and direction_mode:
        parts.append(f"direction_mode={direction_mode}")
    gross_exposure_cap = strategy.get("gross_exposure_cap")
    if gross_exposure_cap is not None:
        parts.append(f"gross_exposure_cap={gross_exposure_cap}")
    target_vol = strategy.get("target_vol")
    if target_vol is not None:
        parts.append(f"target_vol={target_vol}")
    gross_leverage_cap = strategy.get("gross_leverage_cap")
    if gross_leverage_cap is not None:
        parts.append(f"gross_leverage_cap={gross_leverage_cap}")
    net_exposure_target = strategy.get("net_exposure_target")
    if net_exposure_target is not None:
        parts.append(f"net_exposure_target={net_exposure_target}")
    return " ".join(parts)


def print_portfolio_decisions(decisions, *, show_details: bool = False) -> None:
    print("alpha-os portfolio decisions")
    print(f"  Count:    {len(decisions)}")
    for item in decisions:
        print(
            f"  {item.as_of} "
            f"portfolio={item.portfolio_id} "
            f"subject={item.subject_id} "
            f"target={item.target_id} "
            f"kind={item.aggregation_kind} "
            f"weight={item.target_weight:.6f} "
            f"delta={item.position_delta:.6f} "
            f"entry={str(item.entry_allowed).lower()} "
            f"risk_scale={item.risk_scale:.6f}"
        )
        if not show_details:
            continue
        details = item.details
        if not isinstance(details, dict):
            continue
        sizing_method = details.get("sizing_method")
        if isinstance(sizing_method, str):
            sizing_engine = details.get("sizing_engine")
            if isinstance(sizing_engine, str):
                print(f"    sizing={sizing_method} engine={sizing_engine}")
            else:
                print(f"    sizing={sizing_method}")
        strategy_summary = _format_portfolio_decision_strategy(details)
        if strategy_summary is not None:
            print(f"    strategy={strategy_summary}")
        summary = details.get("input_summary")
        if not isinstance(summary, dict):
            continue
        subjects = summary.get("subjects")
        if not isinstance(subjects, dict):
            continue
        subject_summary = subjects.get(item.subject_id)
        if not isinstance(subject_summary, dict):
            continue
        predictive_signal = subject_summary.get("predictive_signal")
        if isinstance(predictive_signal, dict):
            print(f"    signal={predictive_signal}")
        cost_inputs = subject_summary.get("cost_inputs")
        if isinstance(cost_inputs, dict) and cost_inputs:
            print(f"    cost={cost_inputs}")
        uncertainty_inputs = subject_summary.get("uncertainty_inputs")
        if isinstance(uncertainty_inputs, dict) and uncertainty_inputs:
            print(f"    uncertainty={uncertainty_inputs}")
        model_uncertainty_inputs = subject_summary.get("model_uncertainty_inputs")
        if isinstance(model_uncertainty_inputs, dict) and model_uncertainty_inputs:
            print(f"    model_uncertainty={model_uncertainty_inputs}")
        structural_uncertainty_inputs = subject_summary.get("structural_uncertainty_inputs")
        if isinstance(structural_uncertainty_inputs, dict) and structural_uncertainty_inputs:
            print(f"    structural_uncertainty={structural_uncertainty_inputs}")
        risk_inputs = subject_summary.get("risk_inputs")
        if isinstance(risk_inputs, dict) and risk_inputs:
            print(f"    risk={risk_inputs}")


def print_validation_results(
    run,
    signal_results,
    meta_results,
    decision_results,
    *,
    subject_set_facts_by_id: dict[str, str] | None = None,
) -> None:
    print("alpha-os validation")
    print(f"  Run:      {run.run_id}")
    print(f"  Signal:   {len(signal_results)}")
    print(f"  Meta:     {len(meta_results)}")
    print(f"  Decision: {len(decision_results)}")
    print("  Signal Results:")
    for item in signal_results:
        mmc_text = "n/a" if item.mmc is None else f"{item.mmc:.6f}"
        baseline_text = "-" if item.mmc_baseline_type is None else item.mmc_baseline_type
        print(
            f"    {item.date_range_label} target={item.target_id} "
            f"window={item.window_size} hyp={item.signal_id} "
            f"corr={item.corr:.6f} mmc={mmc_text} "
            f"evals={item.sample_count} mmc_evals={item.mmc_sample_count} "
            f"peers={item.mmc_peer_count} baseline={baseline_text}"
        )
    print("  Meta Results:")
    for item in meta_results:
        print(
            f"    {item.date_range_label} target={item.target_id} "
            f"window={item.window_size} kind={item.aggregation_kind} "
            f"corr={item.corr:.6f} evals={item.sample_count}"
        )
    print("  Decision Results:")
    for item in decision_results:
        line = (
            f"    {item.date_range_label} target={item.target_id} "
            f"subject_set={item.subject_set_id or '-'} "
            f"window={item.window_size} kind={item.aggregation_kind} "
            f"gross={item.gross_return_total:.6f} "
            f"net={item.net_return_total:.6f} "
            f"drawdown={item.max_drawdown:.6f} "
            f"turnover={item.mean_turnover:.6f} "
            f"gross_notional={item.mean_gross_notional_exposure:.6f} "
            f"net_notional={item.mean_net_notional_exposure:.6f} "
            f"long_notional={item.mean_long_notional_exposure:.6f} "
            f"short_notional={item.mean_short_notional_exposure:.6f} "
            f"traded_notional={item.mean_traded_notional:.6f} "
            f"cost_notional={item.cost_notional_total:.6f} "
            f"funding_cost_notional={item.funding_cost_notional_total:.6f} "
            f"borrow_cost_notional={item.borrow_cost_notional_total:.6f} "
            f"roll_cost_notional={item.roll_cost_notional_total:.6f} "
            f"steps={item.step_count}"
        )
        if (
            subject_set_facts_by_id is not None
            and item.subject_set_id is not None
            and item.subject_set_id in subject_set_facts_by_id
        ):
            line += f" summary=[{subject_set_facts_by_id[item.subject_set_id]}]"
        print(line)


def print_validation_result_set(
    run,
    signal_results,
    meta_results,
    decision_results,
    *,
    subject_set_facts_by_id: dict[str, str] | None = None,
) -> None:
    contract = getattr(
        run,
        "cross_instrument_contract",
        default_validation_result_set_cross_instrument_contract(),
    )
    print("alpha-os validation summary")
    print(f"  Run:      {run.run_id}")
    print("  CrossInstrumentReportContract: " + contract.format_summary())
    if getattr(contract, "report_units", ()):
        print("  ReportUnits: " + contract.format_report_units())
    result_set = getattr(run, "validation_result_set", None)
    if result_set is None:
        result_set = build_validation_result_set(
            signal_results=signal_results,
            meta_results=meta_results,
            decision_results=decision_results,
        )
    print("  Signals:")
    for item in result_set.signal_summaries:
        mean_mmc_text = "n/a" if item.mean_mmc is None else f"{item.mean_mmc:.6f}"
        print(
            f"    {item.signal_id} conditions={item.conditions} "
            f"positive_corr={item.positive_corr} "
            f"mean_corr={item.mean_corr:.6f} mean_mmc={mean_mmc_text}"
        )
    print("  Meta Aggregations:")
    for item in result_set.meta_summaries:
        print(
            f"    {item.aggregation_kind} conditions={item.conditions} "
            f"wins={item.wins} mean_corr={item.mean_corr:.6f}"
        )
    print("  Decision Aggregations:")
    for item in result_set.decision_summaries:
        subject_set_id = item.subject_set_id
        line = (
            f"    subject_set={subject_set_id or '-'} "
            f"kind={item.aggregation_kind} conditions={item.conditions} "
            f"wins={item.wins} "
            f"negative_conditions={item.negative_conditions} "
            f"mean_net={item.mean_net:.6f} "
            f"worst_net={item.worst_net:.6f} "
            f"mean_drawdown={item.mean_drawdown:.6f} "
            f"mean_gross_notional={item.mean_gross_notional:.6f} "
            f"mean_net_notional={item.mean_net_notional:.6f} "
            f"mean_long_notional={item.mean_long_notional:.6f} "
            f"mean_short_notional={item.mean_short_notional:.6f} "
            f"mean_traded_notional={item.mean_traded_notional:.6f} "
            f"total_cost_notional={item.total_cost_notional:.6f} "
            f"total_funding_cost_notional={item.total_funding_cost_notional:.6f} "
            f"total_borrow_cost_notional={item.total_borrow_cost_notional:.6f} "
            f"total_roll_cost_notional={item.total_roll_cost_notional:.6f}"
        )
        if (
            subject_set_facts_by_id is not None
            and subject_set_id is not None
            and subject_set_id in subject_set_facts_by_id
        ):
            line += f" summary=[{subject_set_facts_by_id[subject_set_id]}]"
        if item.subject_set_contract_groups:
            line += " subject_set_contract_groups=" + ",".join(item.subject_set_contract_groups)
        universe_policy_text = _format_universe_policy_fields(item.universe_policy_fields)
        if universe_policy_text is not None:
            line += " universe_policy=" + universe_policy_text
        print(line)


def print_screening_result(screening_result) -> None:
    result = screening_result.result if hasattr(screening_result, "result") else screening_result
    survivors = result.survivors
    print("alpha-os screening")
    print(f"  SignalDiscovery: {result.signal_discovery_id}")
    print(f"  Result:      {result.screening_result_id}")
    print(f"  Created:     {result.created_at}")
    print(
        "  Policy:      "
        f"min_samples={result.policy.min_sample_count} "
        f"min_abs_corr={result.policy.min_abs_corr:.6f} "
        f"min_stability={result.policy.min_stability_score:.6f} "
        f"adaptive_budget={str(result.policy.adaptive_family_budget).lower()} "
        f"adaptive_scale={result.policy.adaptive_budget_stability_scale:.6f} "
        f"max_family_survivors={result.policy.max_family_survivors_per_subject}"
    )
    print(f"  Candidates:  total={len(result.candidates)} survivors={len(survivors)}")
    for item in survivors:
        corr_text = "n/a" if item.corr is None else f"{item.corr:.6f}"
        print(
            f"    keep hyp={item.signal_id} "
            f"family={item.family_id or '-'} "
            f"subject={item.subject_id} "
            f"kind={item.kind or '-'} "
            f"lookback={item.lookback if item.lookback is not None else '-'} "
            f"score={item.score:.6f} "
            f"corr={corr_text} "
            f"stability={item.stability_score:.6f} "
            f"samples={item.sample_count} "
            f"rank={item.family_rank if item.family_rank is not None else '-'}"
        )


def print_compressed_belief(compressed_belief) -> None:
    belief = compressed_belief.belief if hasattr(compressed_belief, "belief") else compressed_belief
    print("alpha-os compressed belief")
    print(f"  SignalDiscovery: {belief.signal_discovery_id}")
    print(f"  Screening:   {belief.screening_result_id}")
    print(f"  Belief:      {belief.compressed_belief_id}")
    print(f"  Created:     {belief.created_at}")
    print(f"  Components:  {len(belief.components)}")
    for item in belief.components:
        family_text = ",".join(item.family_ids) if item.family_ids else "-"
        representative_text = (
            ",".join(item.representative_signal_ids) if item.representative_signal_ids else "-"
        )
        regime_text = ",".join(item.regime_tags) if item.regime_tags else "-"
        print(
            f"    subject={item.subject_id} "
            f"target={item.target_id} "
            f"belief={item.belief_value:.6f} "
            f"confidence={item.confidence:.6f} "
            f"signal_contributions={item.signal_contribution_count} "
            f"families={family_text} "
            f"family_count={item.family_count} "
            f"cluster_count={item.cluster_count} "
            f"effective_beliefs={item.effective_belief_count:.6f} "
            f"diversity={item.diversity_score:.6f} "
            f"mean_marginal_signal_contribution={item.mean_marginal_signal_contribution:.6f} "
            f"regimes={regime_text} "
            f"representatives={representative_text}"
        )
