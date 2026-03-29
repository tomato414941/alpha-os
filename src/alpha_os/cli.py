from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
from typing import Iterator

from .evaluation_runtime import apply_evaluation, update_evaluation_state
from .metrics_service import refresh_target_metrics
from .meta_aggregation_service import refresh_target_meta_predictions
from .meta_metrics_service import refresh_target_meta_prediction_metrics
from .evaluation_generation import (
    generate_evaluation_input_from_signal_noise,
    generate_evaluation_inputs_from_signal_noise,
    write_evaluation_input,
    write_evaluation_inputs,
)
from .config import (
    DEFAULT_ASSET,
    DEFAULT_PRICE_SIGNAL,
    DEFAULT_SIGNAL_NOISE_BASE_URL,
    DEFAULT_TARGET,
    load_runtime_config,
)
from .hypothesis_registry import HypothesisDefinition
from .evaluation_inputs import (
    EvaluationInput,
    load_evaluation_input,
    load_evaluation_inputs,
)
from .meta_aggregation_service import AGGREGATION_CORR_WEIGHTED_MEAN
from .portfolio_decision import (
    CostInput,
    PortfolioPositionState,
    PortfolioState,
    RiskInput,
)
from .portfolio_decision_service import (
    PortfolioDecisionAssumptions,
    RuntimeDecisionBuildConfig,
    build_portfolio_decision_output,
    persist_portfolio_decision_output,
)
from .store import EvaluationStore
from .validation_service import run_validation
from .validation_spec import default_validation_spec, load_validation_spec, write_validation_spec


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-os",
        description="alpha-os evaluation engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Initialize the runtime database")
    init_db.add_argument("--db", type=str, default=None)

    register = sub.add_parser(
        "register-hypothesis",
        help="Register one hypothesis before recording predictions",
    )
    register.add_argument("--db", type=str, default=None)
    register.add_argument("--hypothesis-id", type=str, required=True)
    register.add_argument("--target-id", type=str, default=None)

    deactivate = sub.add_parser(
        "deactivate-hypothesis",
        help="Deactivate one active hypothesis",
    )
    deactivate.add_argument("--db", type=str, default=None)
    deactivate.add_argument("--hypothesis-id", type=str, required=True)

    activate = sub.add_parser(
        "activate-hypothesis",
        help="Activate one inactive hypothesis",
    )
    activate.add_argument("--db", type=str, default=None)
    activate.add_argument("--hypothesis-id", type=str, required=True)

    record = sub.add_parser(
        "record-prediction",
        help="Low-level: record one prediction before apply/update",
    )
    record.add_argument("--db", type=str, default=None)
    record.add_argument("--date", type=str, required=True)
    record.add_argument("--hypothesis-id", type=str, required=True)
    record.add_argument("--prediction", type=float, required=True)
    record.add_argument("--evaluation-id", type=str, default=None)
    record.add_argument("--target-id", type=str, default=None)

    finalize = sub.add_parser(
        "finalize-observation",
        help="Low-level: finalize one observation before apply/update",
    )
    finalize.add_argument("--db", type=str, default=None)
    finalize.add_argument("--date", type=str, required=True)
    finalize.add_argument("--observation", type=float, required=True)
    finalize.add_argument("--evaluation-id", type=str, default=None)
    finalize.add_argument("--target-id", type=str, default=None)

    update = sub.add_parser(
        "update-state",
        help="Low-level: write one evaluation snapshot from recorded prediction and observation",
    )
    update.add_argument("--db", type=str, default=None)
    update.add_argument("--date", type=str, required=True)
    update.add_argument("--hypothesis-id", type=str, required=True)
    update.add_argument("--evaluation-id", type=str, default=None)
    update.add_argument("--target-id", type=str, default=None)

    generate_input = sub.add_parser(
        "generate-evaluation-input",
        help="Generate one deterministic evaluation-input JSON from signal-noise daily closes",
    )
    generate_input.add_argument("--db", type=str, default=None)
    generate_input.add_argument("--date", type=str, required=True)
    generate_input.add_argument("--hypothesis-id", type=str, required=True)
    generate_input.add_argument("--out", type=str, required=True)
    generate_input.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    generate_input.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    generate_inputs = sub.add_parser(
        "generate-evaluation-inputs",
        help="Generate deterministic evaluation-input JSON for a date range from signal-noise daily closes",
    )
    generate_inputs.add_argument("--db", type=str, default=None)
    generate_inputs.add_argument("--start-date", type=str, required=True)
    generate_inputs.add_argument("--end-date", type=str, required=True)
    generate_inputs.add_argument("--hypothesis-id", type=str, required=True)
    generate_inputs.add_argument("--out", type=str, required=True)
    generate_inputs.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    generate_inputs.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    run = sub.add_parser(
        "apply-evaluation",
        help="Apply one evaluation input through record-prediction -> finalize-observation -> update-state",
    )
    run.add_argument("--db", type=str, default=None)
    run.add_argument("--date", type=str, default=None)
    run.add_argument("--hypothesis-id", type=str, default=None)
    run.add_argument("--prediction", type=float, default=None)
    run.add_argument("--observation", type=float, default=None)
    run.add_argument("--evaluation-id", type=str, default=None)
    run.add_argument("--target-id", type=str, default=None)
    run.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON object with date, hypothesis_id, prediction, observation, target_id",
    )

    batch = sub.add_parser(
        "apply-evaluations",
        help="Apply a deterministic batch of evaluation inputs through the bounded runtime",
    )
    batch.add_argument("--db", type=str, default=None)
    batch.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON array of evaluation input objects",
    )

    backfill = sub.add_parser(
        "apply-backfill",
        help="Generate deterministic evaluation inputs for a date range and apply them through the bounded runtime",
    )
    backfill.add_argument("--db", type=str, default=None)
    backfill.add_argument("--start-date", type=str, required=True)
    backfill.add_argument("--end-date", type=str, required=True)
    backfill.add_argument("--hypothesis-id", type=str, required=True)
    backfill.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    backfill.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)
    backfill.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the generated evaluation-input JSON array",
    )

    backfill_many = sub.add_parser(
        "apply-hypotheses-backfill",
        help="Generate and apply deterministic evaluation inputs for multiple active hypotheses over one date range",
    )
    backfill_many.add_argument("--db", type=str, default=None)
    backfill_many.add_argument("--start-date", type=str, required=True)
    backfill_many.add_argument("--end-date", type=str, required=True)
    backfill_many.add_argument(
        "--hypothesis-id",
        type=str,
        action="append",
        required=True,
        help="Repeat to include multiple active hypotheses",
    )
    backfill_many.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    backfill_many.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    status = sub.add_parser("status", help="Show runtime status across targets")
    status.add_argument("--db", type=str, default=None)

    show = sub.add_parser(
        "show-evaluations",
        help="Show recent evaluation snapshots with provenance",
    )
    show.add_argument("--db", type=str, default=None)
    show.add_argument("--limit", type=int, default=10)

    meta = sub.add_parser(
        "show-meta-predictions",
        help="Show recent meta predictions by aggregation kind",
    )
    meta.add_argument("--db", type=str, default=None)
    meta.add_argument("--limit", type=int, default=10)

    compare_meta = sub.add_parser(
        "compare-meta-aggregations",
        help="Compare meta aggregation kinds by target-level corr",
    )
    compare_meta.add_argument("--db", type=str, default=None)
    compare_meta.add_argument("--target-id", type=str, default=None)

    build_decision = sub.add_parser(
        "build-portfolio-decision",
        help="Build and persist one portfolio decision from latest meta predictions",
    )
    build_decision.add_argument("--db", type=str, default=None)
    build_decision.add_argument("--portfolio-id", type=str, default="default")
    build_decision.add_argument("--target-id", type=str, default=None)
    build_decision.add_argument("--subject-id", type=str, default=None)
    build_decision.add_argument(
        "--aggregation-kind",
        type=str,
        default=AGGREGATION_CORR_WEIGHTED_MEAN,
    )
    build_decision.add_argument("--risk-window", type=int, default=20)
    build_decision.add_argument("--current-weight", type=float, default=0.0)
    build_decision.add_argument("--gross-exposure-cap", type=float, default=None)
    build_decision.add_argument("--turnover-penalty", type=float, default=None)
    build_decision.add_argument("--expected-slippage-bps", type=float, default=None)
    build_decision.add_argument("--no-trade-band", type=float, default=None)

    show_decisions = sub.add_parser(
        "show-portfolio-decisions",
        help="Show recent persisted portfolio decisions",
    )
    show_decisions.add_argument("--db", type=str, default=None)
    show_decisions.add_argument("--portfolio-id", type=str, default=None)
    show_decisions.add_argument("--target-id", type=str, default=None)
    show_decisions.add_argument("--aggregation-kind", type=str, default=None)
    show_decisions.add_argument("--limit", type=int, default=10)

    write_validation = sub.add_parser(
        "write-validation-spec",
        help="Write a default validation spec JSON",
    )
    write_validation.add_argument("--out", type=str, required=True)
    write_validation.add_argument("--base-url", type=str, default=None)

    run_validation_cmd = sub.add_parser(
        "run-validation",
        help="Run a validation spec across targets, ranges, and metric windows",
    )
    run_validation_cmd.add_argument("--db", type=str, default=None)
    run_validation_cmd.add_argument("--spec", type=str, required=True)
    run_validation_cmd.add_argument("--base-url", type=str, default=None)

    show_validation = sub.add_parser(
        "show-validation",
        help="Show raw validation results for one validation run",
    )
    show_validation.add_argument("--db", type=str, default=None)
    show_validation.add_argument("--run-id", type=str, default=None)

    summarize_validation = sub.add_parser(
        "summarize-validation",
        help="Summarize validation stability across conditions",
    )
    summarize_validation.add_argument("--db", type=str, default=None)
    summarize_validation.add_argument("--run-id", type=str, default=None)

    return parser


def _default_evaluation_id(*, asset: str, target_id: str, date: str) -> str:
    return f"{asset}:{target_id}:{date}"


@contextmanager
def _runtime_store(db_path: str | None) -> Iterator[tuple[object, EvaluationStore]]:
    cfg = load_runtime_config(db_path=db_path)
    store = EvaluationStore(cfg.db_path)
    try:
        yield cfg, store
    finally:
        store.close()


def _active_hypothesis_definition(
    store: EvaluationStore,
    *,
    hypothesis_id: str,
) -> HypothesisDefinition:
    hypothesis = store.get_hypothesis(hypothesis_id)
    if hypothesis is None:
        raise ValueError(f"hypothesis must exist before generation: {hypothesis_id}")
    if hypothesis.status != "active":
        raise ValueError(
            f"hypothesis must be active before generation: {hypothesis_id}"
        )
    if hypothesis.definition is None:
        raise ValueError(
            "active hypothesis does not define an executable generation rule: "
            f"{hypothesis_id}"
        )
    return HypothesisDefinition.from_document(
        hypothesis_id=hypothesis.hypothesis_id,
        document=hypothesis.definition,
        asset=hypothesis.asset,
    )

def _generate_backfill_inputs_for_hypothesis(
    store: EvaluationStore,
    *,
    hypothesis_id: str,
    start_date: str,
    end_date: str,
    base_url: str,
    signal_name: str,
) -> list[EvaluationInput]:
    definition = _active_hypothesis_definition(store, hypothesis_id=hypothesis_id)
    return generate_evaluation_inputs_from_signal_noise(
        start_date=start_date,
        end_date=end_date,
        hypothesis_id=hypothesis_id,
        base_url=base_url,
        signal_name=signal_name,
        definition=definition,
    )


def _print_hypothesis_details(hypothesis) -> None:
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


def _print_evaluation_snapshot(snapshot, *, created: bool) -> None:
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


def _print_hypothesis_metric(metric) -> None:
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


def _unique_hypothesis_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _print_hypothesis_competition_summary(
    store: EvaluationStore,
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
        baseline_text = "-" if metric is None or metric.mmc_baseline_type is None else metric.mmc_baseline_type
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


def _print_target_summaries(hypotheses, metrics_by_id) -> None:
    grouped = defaultdict(list)
    for hypothesis in hypotheses:
        grouped[hypothesis.target_id].append(hypothesis)

    print("  Targets:")
    for target, target_hypotheses in sorted(grouped.items()):
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
            f"    {target}: total={len(target_hypotheses)} "
            f"active={active} inactive={inactive} "
            f"tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}"
        )


def _print_meta_predictions(meta_predictions) -> None:
    print("alpha-os meta predictions")
    print(f"  Count:    {len(meta_predictions)}")
    for item in meta_predictions:
        print(
            f"  {item.evaluation_id} "
            f"kind={item.aggregation_kind} "
            f"value={item.value:.6f} "
            f"contributors={item.contributor_count}"
        )


def _print_meta_prediction_metrics(metrics) -> None:
    print("alpha-os meta metrics")
    print(f"  Count:    {len(metrics)}")
    for item in metrics:
        print(
            f"  {item.target_id} "
            f"kind={item.aggregation_kind} "
            f"corr={item.corr:.6f} "
            f"evals={item.sample_count}"
        )


def _print_meta_aggregation_comparison(metrics) -> None:
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


def _portfolio_decision_assumptions_from_args(
    args: argparse.Namespace,
    *,
    subject_id: str,
) -> PortfolioDecisionAssumptions:
    risk_inputs: list[RiskInput] = []
    if args.gross_exposure_cap is not None:
        risk_inputs.append(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=float(args.gross_exposure_cap),
                unit="weight",
            )
        )
    cost_inputs: list[CostInput] = []
    if args.turnover_penalty is not None:
        cost_inputs.append(
            CostInput(
                name="turnover_penalty",
                subject_id=None,
                value=float(args.turnover_penalty),
                basis="per_turnover",
                unit="weight",
            )
        )
    if args.expected_slippage_bps is not None:
        cost_inputs.append(
            CostInput(
                name="expected_slippage",
                subject_id=subject_id,
                value=float(args.expected_slippage_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if args.no_trade_band is not None:
        cost_inputs.append(
            CostInput(
                name="no_trade_band",
                subject_id=subject_id,
                value=float(args.no_trade_band),
                basis="per_delta_weight",
                unit="weight",
            )
        )
    return PortfolioDecisionAssumptions(
        risk_inputs=tuple(risk_inputs),
        cost_inputs=tuple(cost_inputs),
    )


def _print_portfolio_decisions(decisions) -> None:
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


def _resolve_validation_run(store: EvaluationStore, run_id: str | None):
    run = (
        store.get_latest_validation_run()
        if run_id is None
        else store.get_validation_run(str(run_id))
    )
    if run is None:
        raise ValueError("validation run does not exist")
    return run


def _print_validation_results(run, hypothesis_results, meta_results) -> None:
    print("alpha-os validation")
    print(f"  Run:      {run.run_id}")
    print(f"  Hyp:      {len(hypothesis_results)}")
    print(f"  Meta:     {len(meta_results)}")
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


def _print_validation_summary(run, hypothesis_results, meta_results) -> None:
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
    for condition, items in by_condition.items():
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


def _resolve_evaluation_input(
    args: argparse.Namespace,
    *,
    default_target_id: str,
) -> EvaluationInput:
    if args.input:
        evaluation_input = load_evaluation_input(args.input)
        if args.evaluation_id is not None or args.target_id is not None:
            evaluation_input = EvaluationInput(
                date=evaluation_input.date,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction=evaluation_input.prediction,
                observation=evaluation_input.observation,
                evaluation_id=(
                    evaluation_input.evaluation_id
                    if args.evaluation_id is None
                    else str(args.evaluation_id)
                ),
                asset=evaluation_input.asset,
                target_id=(
                    evaluation_input.target_id
                    if args.target_id is None
                    else str(args.target_id)
                ),
            )
        return evaluation_input

    required = {
        "date": args.date,
        "hypothesis_id": args.hypothesis_id,
        "prediction": args.prediction,
        "observation": args.observation,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"apply-evaluation requires --input or manual values for: {joined}")
    return EvaluationInput(
        date=str(args.date),
        hypothesis_id=str(args.hypothesis_id),
        prediction=float(args.prediction),
        observation=float(args.observation),
        evaluation_id=None if args.evaluation_id is None else str(args.evaluation_id),
        target_id=default_target_id if args.target_id is None else str(args.target_id),
    )


def cmd_init_db(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
    print(f"Initialized runtime db: {cfg.db_path}")
    print(f"  Asset:    {cfg.asset}")
    print(f"  Target:   {cfg.target_id}")
    return 0


def cmd_register_hypothesis(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        hypothesis, created = store.register_hypothesis(
            args.hypothesis_id,
            target_id=cfg.target_id if args.target_id is None else str(args.target_id),
        )
    outcome = "created" if created else "existing"
    print(f"Hypothesis [{outcome}] {hypothesis.hypothesis_id}")
    _print_hypothesis_details(hypothesis)
    return 0


def _cmd_change_hypothesis_status(
    args: argparse.Namespace,
    *,
    action: str,
    verb: str,
) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        hypothesis = store.set_hypothesis_status(
            args.hypothesis_id,
            action=action,
        )
        refresh_target_metrics(
            store,
            asset=hypothesis.asset,
            target_id=hypothesis.target_id,
        )
        refresh_target_meta_predictions(
            store,
            asset=hypothesis.asset,
            target_id=hypothesis.target_id,
        )
        refresh_target_meta_prediction_metrics(
            store,
            asset=hypothesis.asset,
            target_id=hypothesis.target_id,
        )
    print(f"Hypothesis [{verb}] {hypothesis.hypothesis_id}")
    _print_hypothesis_details(hypothesis)
    return 0


def cmd_deactivate_hypothesis(args: argparse.Namespace) -> int:
    return _cmd_change_hypothesis_status(
        args,
        action="deactivate",
        verb="deactivated",
    )


def cmd_activate_hypothesis(args: argparse.Namespace) -> int:
    return _cmd_change_hypothesis_status(
        args,
        action="activate",
        verb="activated",
    )


def cmd_record_prediction(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target_id=target_id,
            date=args.date,
        )
        prediction, created = store.record_prediction(
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
            prediction_value=args.prediction,
            target_id=target_id,
        )
    outcome = "created" if created else "existing"
    print(f"Prediction [{outcome}] {prediction.evaluation_id}")
    print(f"  Asset:    {prediction.asset}")
    print(f"  Target:   {prediction.target_id}")
    print(f"  Hyp:      {prediction.hypothesis_id}")
    print(f"  Value:    {prediction.value:.6f}")
    return 0


def cmd_finalize_observation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target_id=target_id,
            date=args.date,
        )
        observation, created = store.finalize_observation(
            evaluation_id=evaluation_id,
            observation_value=args.observation,
            target_id=target_id,
        )
    outcome = "created" if created else "existing"
    print(f"Observation [{outcome}] {observation.evaluation_id}")
    print(f"  Asset:    {observation.asset}")
    print(f"  Target:   {observation.target_id}")
    print(f"  Value:    {observation.value:.6f}")
    return 0


def cmd_update_state(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target_id=target_id,
            date=args.date,
        )
        snapshot, created = update_evaluation_state(
            store,
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
            target_id=target_id,
        )
        metric = store.get_hypothesis_metric(args.hypothesis_id)
    _print_evaluation_snapshot(snapshot, created=created)
    _print_hypothesis_metric(metric)
    return 0


def cmd_generate_evaluation_input(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        definition = _active_hypothesis_definition(
            store,
            hypothesis_id=args.hypothesis_id,
        )
    evaluation_input = generate_evaluation_input_from_signal_noise(
        date=args.date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
        definition=definition,
    )
    output_path = write_evaluation_input(args.out, evaluation_input)
    print(f"Generated evaluation input: {output_path}")
    print(f"  Asset:    {evaluation_input.asset}")
    print(f"  Target:   {evaluation_input.target_id}")
    print(f"  Date:     {evaluation_input.date}")
    print(f"  Hyp:      {evaluation_input.hypothesis_id}")
    print(
        f"  Signal:   pred={evaluation_input.prediction:.6f} "
        f"obs={evaluation_input.observation:.6f}"
    )
    return 0


def cmd_generate_evaluation_inputs(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        definition = _active_hypothesis_definition(
            store,
            hypothesis_id=args.hypothesis_id,
        )
    evaluation_inputs = generate_evaluation_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
        definition=definition,
    )
    output_path = write_evaluation_inputs(args.out, evaluation_inputs)
    print(f"Generated evaluation inputs: {output_path}")
    print(f"  Count:    {len(evaluation_inputs)}")
    if evaluation_inputs:
        print(f"  Asset:    {evaluation_inputs[0].asset}")
        print(f"  Target:   {evaluation_inputs[0].target_id}")
        print(f"  Range:    {evaluation_inputs[0].date} -> {evaluation_inputs[-1].date}")
    else:
        print(f"  Asset:    {DEFAULT_ASSET}")
        print(f"  Target:   {DEFAULT_TARGET}")
    return 0


def cmd_apply_evaluation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        evaluation_input = _resolve_evaluation_input(
            args,
            default_target_id=cfg.target_id,
        )
        input_source = "json_file" if args.input else "manual"
        evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
            asset=evaluation_input.asset,
            target_id=evaluation_input.target_id,
            date=evaluation_input.date,
        )
        snapshot, created = apply_evaluation(
            store,
            evaluation_id=evaluation_id,
            hypothesis_id=evaluation_input.hypothesis_id,
            prediction_value=evaluation_input.prediction,
            observation_value=evaluation_input.observation,
            asset=evaluation_input.asset,
            target_id=evaluation_input.target_id,
            input_source=input_source,
        )
        metric = store.get_hypothesis_metric(evaluation_input.hypothesis_id)
    _print_evaluation_snapshot(snapshot, created=created)
    _print_hypothesis_metric(metric)
    return 0


def cmd_apply_evaluations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    evaluation_inputs = load_evaluation_inputs(args.input)
    return _apply_evaluation_inputs(
        cfg.db_path,
        evaluation_inputs,
        input_source="json_batch",
    )


def _apply_evaluation_inputs(
    db_path: Path,
    evaluation_inputs: list[EvaluationInput],
    *,
    input_source: str,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    signal_name: str | None = None,
) -> int:
    with _runtime_store(str(db_path)) as (_cfg, store):
        created_count = 0
        existing_count = 0
        latest_snapshot = None
        latest_metric = None
        touched_targets: set[tuple[str, str]] = set()
        for evaluation_input in evaluation_inputs:
            evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
                asset=evaluation_input.asset,
                target_id=evaluation_input.target_id,
                date=evaluation_input.date,
            )
            latest_snapshot, created = apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction_value=evaluation_input.prediction,
                observation_value=evaluation_input.observation,
                asset=evaluation_input.asset,
                target_id=evaluation_input.target_id,
                input_source=input_source,
                input_range_start=input_range_start,
                input_range_end=input_range_end,
                signal_name=signal_name,
                refresh_metrics=False,
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
            touched_targets.add((evaluation_input.asset, evaluation_input.target_id))
        for asset, target_id in sorted(touched_targets):
            refresh_target_metrics(
                store,
                asset=asset,
                target_id=target_id,
            )
            refresh_target_meta_predictions(
                store,
                asset=asset,
                target_id=target_id,
            )
            refresh_target_meta_prediction_metrics(
                store,
                asset=asset,
                target_id=target_id,
            )
        if evaluation_inputs:
            latest_metric = store.get_hypothesis_metric(evaluation_inputs[-1].hypothesis_id)

    print(
        "Batch complete: "
        f"evaluations={len(evaluation_inputs)} created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.hypothesis_id}")
        _print_hypothesis_metric(latest_metric)
    return 0


def cmd_apply_backfill(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        evaluation_inputs = _generate_backfill_inputs_for_hypothesis(
            store,
            hypothesis_id=args.hypothesis_id,
            start_date=args.start_date,
            end_date=args.end_date,
            base_url=args.base_url,
            signal_name=args.signal_name,
        )
    if args.out is not None:
        output_path = write_evaluation_inputs(args.out, evaluation_inputs)
        print(f"Wrote evaluation inputs: {output_path}")
    return _apply_evaluation_inputs(
        cfg.db_path,
        evaluation_inputs,
        input_source="signal_noise_backfill",
        input_range_start=args.start_date,
        input_range_end=args.end_date,
        signal_name=args.signal_name,
    )


def cmd_apply_hypotheses_backfill(args: argparse.Namespace) -> int:
    hypothesis_ids = _unique_hypothesis_ids(args.hypothesis_id)

    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        all_evaluation_inputs: list[EvaluationInput] = []
        for hypothesis_id in hypothesis_ids:
            evaluation_inputs = _generate_backfill_inputs_for_hypothesis(
                store,
                hypothesis_id=hypothesis_id,
                start_date=args.start_date,
                end_date=args.end_date,
                base_url=args.base_url,
                signal_name=args.signal_name,
            )
            all_evaluation_inputs.extend(evaluation_inputs)
        created_count = 0
        existing_count = 0
        latest_snapshot = None
        touched_targets: set[tuple[str, str]] = set()
        for evaluation_input in all_evaluation_inputs:
            evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
                asset=evaluation_input.asset,
                target_id=evaluation_input.target_id,
                date=evaluation_input.date,
            )
            latest_snapshot, created = apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction_value=evaluation_input.prediction,
                observation_value=evaluation_input.observation,
                asset=evaluation_input.asset,
                target_id=evaluation_input.target_id,
                input_source="signal_noise_backfill",
                input_range_start=args.start_date,
                input_range_end=args.end_date,
                signal_name=args.signal_name,
                refresh_metrics=False,
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
            touched_targets.add((evaluation_input.asset, evaluation_input.target_id))
        for asset, target_id in sorted(touched_targets):
            refresh_target_metrics(
                store,
                asset=asset,
                target_id=target_id,
            )
            refresh_target_meta_predictions(
                store,
                asset=asset,
                target_id=target_id,
            )
            refresh_target_meta_prediction_metrics(
                store,
                asset=asset,
                target_id=target_id,
            )
        print(
            "Batch complete: "
            f"hypotheses={len(hypothesis_ids)} "
            f"evaluations={len(all_evaluation_inputs)} "
            f"created={created_count} existing={existing_count}"
        )
        if latest_snapshot is not None:
            print(
                f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.hypothesis_id}"
            )
        _print_hypothesis_competition_summary(
            store,
            hypothesis_ids=hypothesis_ids,
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        hypotheses = store.list_hypotheses(asset=cfg.asset, target_id=None)
        metrics = (
            []
            if not hypotheses
            else store.list_hypothesis_metrics(
                hypothesis_ids=[item.hypothesis_id for item in hypotheses]
            )
        )
        snapshots = store.list_evaluation_snapshots(limit=1)
        latest = snapshots[0] if snapshots else None
    finally:
        store.close()

    print("alpha-os status")
    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Asset:    {cfg.asset}")
    print("  Targets:  all")
    if latest is None and not hypotheses:
        print("  Latest:   no evaluations recorded")
        return 0
    if latest is not None:
        print(
            f"  Latest:   {latest.evaluation_id} / {latest.hypothesis_id}"
        )
    else:
        print("  Latest:   no evaluations recorded")
    total = len(hypotheses)
    active = sum(1 for item in hypotheses if item.status == "active")
    inactive = sum(1 for item in hypotheses if item.status == "inactive")
    print(
        "  Hyp:      "
        f"total={total} active={active} inactive={inactive}"
    )
    tracked = len(metrics)
    mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in metrics) / tracked
    mmcs = [item.mmc for item in metrics if item.mmc is not None]
    mean_mmc_text = "n/a" if not mmcs else f"{sum(mmcs) / len(mmcs):.6f}"
    print(f"  Metrics:  tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}")
    _print_target_summaries(
        hypotheses,
        {item.hypothesis_id: item for item in metrics},
    )
    return 0


def cmd_show_evaluations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        snapshots = store.list_evaluation_snapshots(limit=args.limit)
    finally:
        store.close()

    print("alpha-os evaluations")
    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Count:    {len(snapshots)}")
    for snapshot in snapshots:
        range_text = "-"
        if snapshot.input_range_start or snapshot.input_range_end:
            start = snapshot.input_range_start or "-"
            end = snapshot.input_range_end or "-"
            range_text = f"{start}->{end}"
        print(
            f"  {snapshot.evaluation_id} "
            f"hyp={snapshot.hypothesis_id} "
            f"source={snapshot.input_source or '-'} "
            f"signal={snapshot.signal_name or '-'} "
            f"range={range_text} "
            f"pred={snapshot.prediction_value:.6f} "
            f"obs={snapshot.observation_value:.6f} "
            f"edge={snapshot.signed_edge:.6f}"
        )
    return 0


def cmd_show_meta_predictions(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        meta_predictions = store.list_meta_predictions(asset=cfg.asset, limit=args.limit)
        meta_metrics = store.list_meta_prediction_metrics(asset=cfg.asset)
    finally:
        store.close()

    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Asset:    {cfg.asset}")
    _print_meta_predictions(meta_predictions)
    _print_meta_prediction_metrics(meta_metrics)
    return 0


def cmd_compare_meta_aggregations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        metrics = store.list_meta_prediction_metrics(
            asset=cfg.asset,
            target_id=None if args.target_id is None else str(args.target_id),
        )
    finally:
        store.close()

    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Asset:    {cfg.asset}")
    _print_meta_aggregation_comparison(metrics)
    return 0


def cmd_build_portfolio_decision(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        subject_id = cfg.asset if args.subject_id is None else str(args.subject_id)
        assumptions = _portfolio_decision_assumptions_from_args(
            args,
            subject_id=subject_id,
        )
        portfolio_state = PortfolioState(
            portfolio_id=str(args.portfolio_id),
            asset=cfg.asset,
            positions=(
                PortfolioPositionState(
                    subject_id=subject_id,
                    weight=float(args.current_weight),
                ),
            ),
        )
        config = RuntimeDecisionBuildConfig(
            aggregation_kind=str(args.aggregation_kind),
            risk_window=int(args.risk_window),
        )
        decision_output = build_portfolio_decision_output(
            store,
            asset=cfg.asset,
            target_id=target_id,
            portfolio_id=str(args.portfolio_id),
            subject_id=subject_id,
            portfolio_state=portfolio_state,
            config=config,
            assumptions=assumptions,
        )
        if decision_output is None:
            raise ValueError("portfolio decision could not be built from current runtime state")
        persist_portfolio_decision_output(
            store,
            decision_output=decision_output,
            target_id=target_id,
            aggregation_kind=config.aggregation_kind,
            config=config,
            assumptions=assumptions,
        )
        decisions = store.list_portfolio_decisions(
            portfolio_id=str(args.portfolio_id),
            target_id=target_id,
            aggregation_kind=config.aggregation_kind,
            limit=10,
        )
    print(f"  DB:       {cfg.db_path}")
    print(f"  Asset:    {cfg.asset}")
    _print_portfolio_decisions(decisions[: len(decision_output.targets)])
    return 0


def cmd_show_portfolio_decisions(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        decisions = store.list_portfolio_decisions(
            portfolio_id=None if args.portfolio_id is None else str(args.portfolio_id),
            target_id=None if args.target_id is None else str(args.target_id),
            aggregation_kind=(
                None if args.aggregation_kind is None else str(args.aggregation_kind)
            ),
            limit=int(args.limit),
        )
    finally:
        store.close()
    print(f"  DB:       {Path(cfg.db_path)}")
    _print_portfolio_decisions(decisions)
    return 0


def cmd_write_validation_spec(args: argparse.Namespace) -> int:
    spec = default_validation_spec()
    if args.base_url is not None:
        spec = spec.__class__(
            hypothesis_ids=spec.hypothesis_ids,
            target_ids=spec.target_ids,
            date_ranges=spec.date_ranges,
            metric_windows=spec.metric_windows,
            aggregation_kinds=spec.aggregation_kinds,
            asset=spec.asset,
            base_url=str(args.base_url),
        )
    path = write_validation_spec(args.out, spec)
    print(f"Wrote validation spec: {path}")
    return 0


def cmd_run_validation(args: argparse.Namespace) -> int:
    spec = load_validation_spec(args.spec)
    if args.base_url is not None:
        spec = spec.__class__(
            hypothesis_ids=spec.hypothesis_ids,
            target_ids=spec.target_ids,
            date_ranges=spec.date_ranges,
            metric_windows=spec.metric_windows,
            aggregation_kinds=spec.aggregation_kinds,
            asset=spec.asset,
            base_url=str(args.base_url),
        )
    with _runtime_store(args.db) as (_cfg, store):
        result = run_validation(store, spec=spec)
    print("Validation complete")
    print(f"  Run:      {result.run_id}")
    print(f"  Hyp:      {result.hypothesis_result_count}")
    print(f"  Meta:     {result.meta_result_count}")
    return 0


def cmd_show_validation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        run = _resolve_validation_run(store, args.run_id)
        hypothesis_results = store.list_validation_hypothesis_results(run_id=run.run_id)
        meta_results = store.list_validation_meta_results(run_id=run.run_id)
    _print_validation_results(run, hypothesis_results, meta_results)
    return 0


def cmd_summarize_validation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        run = _resolve_validation_run(store, args.run_id)
        hypothesis_results = store.list_validation_hypothesis_results(run_id=run.run_id)
        meta_results = store.list_validation_meta_results(run_id=run.run_id)
    _print_validation_summary(run, hypothesis_results, meta_results)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-db":
            return cmd_init_db(args)
        if args.command == "register-hypothesis":
            return cmd_register_hypothesis(args)
        if args.command == "deactivate-hypothesis":
            return cmd_deactivate_hypothesis(args)
        if args.command == "activate-hypothesis":
            return cmd_activate_hypothesis(args)
        if args.command == "record-prediction":
            return cmd_record_prediction(args)
        if args.command == "finalize-observation":
            return cmd_finalize_observation(args)
        if args.command == "update-state":
            return cmd_update_state(args)
        if args.command == "generate-evaluation-input":
            return cmd_generate_evaluation_input(args)
        if args.command == "generate-evaluation-inputs":
            return cmd_generate_evaluation_inputs(args)
        if args.command == "apply-evaluation":
            return cmd_apply_evaluation(args)
        if args.command == "apply-evaluations":
            return cmd_apply_evaluations(args)
        if args.command == "apply-backfill":
            return cmd_apply_backfill(args)
        if args.command == "apply-hypotheses-backfill":
            return cmd_apply_hypotheses_backfill(args)
        if args.command == "status":
            return cmd_status(args)
        if args.command == "show-evaluations":
            return cmd_show_evaluations(args)
        if args.command == "show-meta-predictions":
            return cmd_show_meta_predictions(args)
        if args.command == "compare-meta-aggregations":
            return cmd_compare_meta_aggregations(args)
        if args.command == "build-portfolio-decision":
            return cmd_build_portfolio_decision(args)
        if args.command == "show-portfolio-decisions":
            return cmd_show_portfolio_decisions(args)
        if args.command == "write-validation-spec":
            return cmd_write_validation_spec(args)
        if args.command == "run-validation":
            return cmd_run_validation(args)
        if args.command == "show-validation":
            return cmd_show_validation(args)
        if args.command == "summarize-validation":
            return cmd_summarize_validation(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2
