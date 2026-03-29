from __future__ import annotations

from collections import defaultdict


def print_hypothesis_details(hypothesis) -> None:
    print(f"  Asset:    {hypothesis.asset}")
    print(f"  Target:   {hypothesis.target_id}")
    if hypothesis.kind is not None:
        print(f"  Kind:     {hypothesis.kind}")
    if hypothesis.signal_name is not None:
        print(f"  Signal:   {hypothesis.signal_name}")
    if hypothesis.lookback is not None:
        print(f"  Lookback: {hypothesis.lookback}")
    if hypothesis.horizon_days is not None:
        print(f"  Horizon:  {hypothesis.horizon_days}d")
    print(f"  Status:   {hypothesis.status}")
    print(f"  Evals:    {hypothesis.observation_count}")


def print_evaluation_snapshot(snapshot, *, created: bool) -> None:
    outcome = "created" if created else "existing"
    print(f"Evaluation [{outcome}] {snapshot.evaluation_id}")
    print(f"  Asset:    {snapshot.asset}")
    print(f"  Target:   {snapshot.target_id}")
    print(f"  Hyp:      {snapshot.hypothesis_id}")
    print(
        f"  Signal:   pred={snapshot.prediction_value:.6f} "
        f"obs={snapshot.observation_value:.6f} edge={snapshot.signed_edge:.6f}"
    )
    print(f"  Error:    abs={snapshot.absolute_error:.6f}")


def print_hypothesis_metric(metric) -> None:
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


def print_hypothesis_competition_summary(
    store,
    *,
    hypothesis_ids: list[str],
) -> None:
    selected = set(hypothesis_ids)
    hypotheses = {
        item.hypothesis_id: item
        for item in (store.get_hypothesis(hypothesis_id) for hypothesis_id in hypothesis_ids)
        if item is not None and item.hypothesis_id in selected
    }
    metrics = {
        item.hypothesis_id: item
        for item in store.list_hypothesis_metrics(hypothesis_ids=hypothesis_ids)
    }
    print("alpha-os hypothesis competition")
    print(f"  Count:    {len(hypotheses)}")
    for hypothesis_id in hypothesis_ids:
        hypothesis = hypotheses.get(hypothesis_id)
        if hypothesis is None:
            continue
        metric = metrics.get(hypothesis_id)
        kind = hypothesis.kind or "-"
        signal_name = hypothesis.signal_name or "-"
        lookback = "-" if hypothesis.lookback is None else str(hypothesis.lookback)
        horizon = "-" if hypothesis.horizon_days is None else f"{hypothesis.horizon_days}d"
        mmc_text = "n/a" if metric is None or metric.mmc is None else f"{metric.mmc:.6f}"
        baseline_text = (
            "-"
            if metric is None or metric.mmc_baseline_type is None
            else metric.mmc_baseline_type
        )
        print(
            f"  {hypothesis.hypothesis_id} "
            f"kind={kind} signal={signal_name} lookback={lookback} horizon={horizon} "
            f"status={hypothesis.status} "
            f"corr={0.0 if metric is None else metric.corr:.6f} "
            f"mmc={mmc_text} "
            f"evals={hypothesis.observation_count if metric is None else metric.sample_count} "
            f"mmc_evals={0 if metric is None else metric.mmc_sample_count} "
            f"peers={0 if metric is None else metric.mmc_peer_count} "
            f"baseline={baseline_text}"
        )


def print_target_summaries(hypotheses, metrics_by_id) -> None:
    grouped = defaultdict(list)
    for hypothesis in hypotheses:
        grouped[hypothesis.target_id].append(hypothesis)

    print("  Targets:")
    for target_id, target_hypotheses in sorted(grouped.items()):
        active = sum(1 for item in target_hypotheses if item.status == "active")
        inactive = sum(1 for item in target_hypotheses if item.status == "inactive")
        target_metrics = [
            metrics_by_id[item.hypothesis_id]
            for item in target_hypotheses
            if item.hypothesis_id in metrics_by_id
        ]
        tracked = len(target_metrics)
        mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in target_metrics) / tracked
        target_mmcs = [item.mmc for item in target_metrics if item.mmc is not None]
        mean_mmc_text = (
            "n/a"
            if not target_mmcs
            else f"{sum(target_mmcs) / len(target_mmcs):.6f}"
        )
        print(
            f"    {target_id}: total={len(target_hypotheses)} "
            f"active={active} inactive={inactive} "
            f"tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}"
        )


def print_meta_predictions(meta_predictions) -> None:
    print("alpha-os meta predictions")
    print(f"  Count:    {len(meta_predictions)}")
    for item in meta_predictions:
        print(
            f"  {item.evaluation_id} "
            f"kind={item.aggregation_kind} "
            f"value={item.value:.6f} "
            f"contributors={item.contributor_count}"
        )


def print_meta_prediction_metrics(metrics) -> None:
    print("alpha-os meta metrics")
    print(f"  Count:    {len(metrics)}")
    for item in metrics:
        print(
            f"  {item.target_id} "
            f"kind={item.aggregation_kind} "
            f"corr={item.corr:.6f} "
            f"evals={item.sample_count}"
        )


def print_meta_aggregation_comparison(metrics) -> None:
    grouped = defaultdict(list)
    for item in metrics:
        grouped[item.target_id].append(item)

    print("alpha-os meta aggregation comparison")
    print(f"  Targets:  {len(grouped)}")
    for target_id, items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda item: (-item.corr, -item.sample_count, item.aggregation_kind),
        )
        print(f"  {target_id}")
        for rank, item in enumerate(ordered, start=1):
            print(
                f"    {rank}. kind={item.aggregation_kind} "
                f"corr={item.corr:.6f} evals={item.sample_count}"
            )


def print_portfolio_decisions(decisions) -> None:
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


def print_validation_results(run, hypothesis_results, meta_results, decision_results) -> None:
    print("alpha-os validation")
    print(f"  Run:      {run.run_id}")
    print(f"  Hyp:      {len(hypothesis_results)}")
    print(f"  Meta:     {len(meta_results)}")
    print(f"  Decision: {len(decision_results)}")
    print("  Hypothesis Results:")
    for item in hypothesis_results:
        mmc_text = "n/a" if item.mmc is None else f"{item.mmc:.6f}"
        baseline_text = "-" if item.mmc_baseline_type is None else item.mmc_baseline_type
        print(
            f"    {item.date_range_label} target={item.target_id} "
            f"window={item.window_size} hyp={item.hypothesis_id} "
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
        print(
            f"    {item.date_range_label} target={item.target_id} "
            f"window={item.window_size} kind={item.aggregation_kind} "
            f"gross={item.gross_return_total:.6f} "
            f"net={item.net_return_total:.6f} "
            f"drawdown={item.max_drawdown:.6f} "
            f"turnover={item.mean_turnover:.6f} "
            f"steps={item.step_count}"
        )


def print_validation_summary(run, hypothesis_results, meta_results, decision_results) -> None:
    print("alpha-os validation summary")
    print(f"  Run:      {run.run_id}")
    grouped_hypotheses = defaultdict(list)
    for item in hypothesis_results:
        grouped_hypotheses[item.hypothesis_id].append(item)
    print("  Hypotheses:")
    for hypothesis_id, items in sorted(grouped_hypotheses.items()):
        mean_corr = sum(item.corr for item in items) / len(items)
        positive = sum(1 for item in items if item.corr > 0.0)
        mean_mmcs = [item.mmc for item in items if item.mmc is not None]
        mean_mmc_text = (
            "n/a"
            if not mean_mmcs
            else f"{sum(mean_mmcs) / len(mean_mmcs):.6f}"
        )
        print(
            f"    {hypothesis_id} conditions={len(items)} "
            f"positive_corr={positive} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}"
        )
    grouped_meta = defaultdict(list)
    by_condition = defaultdict(list)
    for item in meta_results:
        grouped_meta[item.aggregation_kind].append(item)
        by_condition[(item.date_range_label, item.target_id, item.window_size)].append(item)
    wins = defaultdict(int)
    for items in by_condition.values():
        ordered = sorted(items, key=lambda item: (-item.corr, item.aggregation_kind))
        if ordered:
            wins[ordered[0].aggregation_kind] += 1
    print("  Meta Aggregations:")
    for aggregation_kind, items in sorted(grouped_meta.items()):
        mean_corr = sum(item.corr for item in items) / len(items)
        print(
            f"    {aggregation_kind} conditions={len(items)} "
            f"wins={wins[aggregation_kind]} mean_corr={mean_corr:.6f}"
        )
    grouped_decisions = defaultdict(list)
    decision_wins = defaultdict(int)
    decision_by_condition = defaultdict(list)
    for item in decision_results:
        grouped_decisions[item.aggregation_kind].append(item)
        decision_by_condition[
            (item.date_range_label, item.target_id, item.window_size)
        ].append(item)
    for items in decision_by_condition.values():
        ordered = sorted(
            items,
            key=lambda item: (-item.net_return_total, item.max_drawdown, item.aggregation_kind),
        )
        if ordered:
            decision_wins[ordered[0].aggregation_kind] += 1
    print("  Decision Aggregations:")
    for aggregation_kind, items in sorted(grouped_decisions.items()):
        mean_net = sum(item.net_return_total for item in items) / len(items)
        mean_drawdown = sum(item.max_drawdown for item in items) / len(items)
        print(
            f"    {aggregation_kind} conditions={len(items)} "
            f"wins={decision_wins[aggregation_kind]} "
            f"mean_net={mean_net:.6f} mean_drawdown={mean_drawdown:.6f}"
        )
