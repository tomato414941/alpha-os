from __future__ import annotations

from .subject_set_facts import format_subject_set_facts


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
