from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
    build_config,
)
from .hypothesis_registry import HypothesisDefinition
from .evaluation_inputs import (
    EvaluationInput,
    load_evaluation_input,
    load_evaluation_inputs,
)
from .store import EvaluationStore


def _build_parser() -> argparse.ArgumentParser:
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

    pause = sub.add_parser("pause-hypothesis", help="Pause one registered hypothesis")
    pause.add_argument("--db", type=str, default=None)
    pause.add_argument("--hypothesis-id", type=str, required=True)

    resume = sub.add_parser("resume-hypothesis", help="Resume one paused hypothesis")
    resume.add_argument("--db", type=str, default=None)
    resume.add_argument("--hypothesis-id", type=str, required=True)

    retire = sub.add_parser("retire-hypothesis", help="Retire one registered or paused hypothesis")
    retire.add_argument("--db", type=str, default=None)
    retire.add_argument("--hypothesis-id", type=str, required=True)

    record = sub.add_parser(
        "record-prediction",
        help="Record one prediction before observation finalization",
    )
    record.add_argument("--db", type=str, default=None)
    record.add_argument("--date", type=str, required=True)
    record.add_argument("--hypothesis-id", type=str, required=True)
    record.add_argument("--prediction", type=float, required=True)
    record.add_argument("--evaluation-id", type=str, default=None)

    finalize = sub.add_parser(
        "finalize-observation",
        help="Finalize one observation before state update",
    )
    finalize.add_argument("--db", type=str, default=None)
    finalize.add_argument("--date", type=str, required=True)
    finalize.add_argument("--observation", type=float, required=True)
    finalize.add_argument("--evaluation-id", type=str, default=None)

    update = sub.add_parser(
        "update-state",
        help="Update state from recorded prediction and finalized observation",
    )
    update.add_argument("--db", type=str, default=None)
    update.add_argument("--date", type=str, required=True)
    update.add_argument("--hypothesis-id", type=str, required=True)
    update.add_argument("--evaluation-id", type=str, default=None)

    build = sub.add_parser(
        "generate-evaluation-input",
        help="Generate one deterministic evaluation-input JSON from signal-noise daily closes",
    )
    build.add_argument("--db", type=str, default=None)
    build.add_argument("--date", type=str, required=True)
    build.add_argument("--hypothesis-id", type=str, required=True)
    build.add_argument("--out", type=str, required=True)
    build.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    build.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    builds = sub.add_parser(
        "generate-evaluation-inputs",
        help="Generate deterministic evaluation-input JSON for a date range from signal-noise daily closes",
    )
    builds.add_argument("--db", type=str, default=None)
    builds.add_argument("--start-date", type=str, required=True)
    builds.add_argument("--end-date", type=str, required=True)
    builds.add_argument("--hypothesis-id", type=str, required=True)
    builds.add_argument("--out", type=str, required=True)
    builds.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    builds.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

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
    run.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON object with date, hypothesis_id, prediction, observation",
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
        help="Generate and apply deterministic evaluation inputs for multiple registered hypotheses over one date range",
    )
    backfill_many.add_argument("--db", type=str, default=None)
    backfill_many.add_argument("--start-date", type=str, required=True)
    backfill_many.add_argument("--end-date", type=str, required=True)
    backfill_many.add_argument(
        "--hypothesis-id",
        type=str,
        action="append",
        required=True,
        help="Repeat to include multiple registered hypotheses",
    )
    backfill_many.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    backfill_many.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    status = sub.add_parser("status", help="Show the latest BTC evaluation state")
    status.add_argument("--db", type=str, default=None)

    show = sub.add_parser(
        "show-evaluations",
        help="Show recent evaluation snapshots with provenance",
    )
    show.add_argument("--db", type=str, default=None)
    show.add_argument("--limit", type=int, default=10)

    return parser


def _default_evaluation_id(*, asset: str, target: str, date: str) -> str:
    return f"{asset}:{target}:{date}"


@contextmanager
def _runtime_store(db_path: str | None) -> Iterator[tuple[object, EvaluationStore]]:
    cfg = build_config(db_path=db_path)
    store = EvaluationStore(cfg.db_path)
    try:
        yield cfg, store
    finally:
        store.close()


def _registered_hypothesis_definition(
    store: EvaluationStore,
    *,
    hypothesis_id: str,
) -> HypothesisDefinition:
    hypothesis = store.get_hypothesis(hypothesis_id)
    if hypothesis is None:
        raise ValueError(f"hypothesis must be registered before generation: {hypothesis_id}")
    if hypothesis.kind is None or hypothesis.signal_name is None or hypothesis.lookback is None:
        raise ValueError(
            "registered hypothesis does not define an executable generation rule: "
            f"{hypothesis_id}"
        )
    return HypothesisDefinition(
        hypothesis_id=hypothesis.hypothesis_id,
        kind=hypothesis.kind,
        signal_name=hypothesis.signal_name,
        lookback=hypothesis.lookback,
        asset=hypothesis.asset,
        target=hypothesis.target,
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
    definition = _registered_hypothesis_definition(store, hypothesis_id=hypothesis_id)
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
    print(f"  Target:   {hypothesis.target}")
    if hypothesis.kind is not None:
        print(f"  Kind:     {hypothesis.kind}")
    if hypothesis.signal_name is not None:
        print(f"  Signal:   {hypothesis.signal_name}")
    if hypothesis.lookback is not None:
        print(f"  Lookback: {hypothesis.lookback}")
    print(f"  Status:   {hypothesis.status}")
    print(f"  Evals:    {hypothesis.observation_count}")


def _print_evaluation_snapshot(snapshot, *, created: bool) -> None:
    outcome = "created" if created else "existing"
    print(f"Evaluation [{outcome}] {snapshot.evaluation_id}")
    print(f"  Asset:    {snapshot.asset}")
    print(f"  Target:   {snapshot.target}")
    print(f"  Hyp:      {snapshot.hypothesis_id}")
    print(
        f"  Signal:   pred={snapshot.prediction_value:.6f} "
        f"obs={snapshot.observation_value:.6f} edge={snapshot.signed_edge:.6f}"
    )
    print(f"  Error:    abs={snapshot.absolute_error:.6f}")


def _print_hypothesis_metric(metric) -> None:
    if metric is None:
        print("  Metrics:  corr=0.000000 mmc=0.000000 evals=0")
        return
    print(
        "  Metrics:  "
        f"corr={metric.corr:.6f} "
        f"mmc={metric.mmc:.6f} "
        f"evals={metric.sample_count}"
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
    asset: str,
    target: str,
    hypothesis_ids: list[str],
) -> None:
    selected = set(hypothesis_ids)
    hypotheses = {
        item.hypothesis_id: item
        for item in store.list_hypotheses(asset=asset, target=target)
        if item.hypothesis_id in selected
    }
    metrics = {item.hypothesis_id: item for item in store.list_hypothesis_metrics(hypothesis_ids=hypothesis_ids)}
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
        print(
            f"  {hypothesis.hypothesis_id} "
            f"kind={kind} signal={signal_name} lookback={lookback} "
            f"status={hypothesis.status} "
            f"corr={0.0 if metric is None else metric.corr:.6f} "
            f"mmc={0.0 if metric is None else metric.mmc:.6f} "
            f"evals={hypothesis.observation_count if metric is None else metric.sample_count}"
        )


def _resolve_evaluation_input(args: argparse.Namespace) -> EvaluationInput:
    if args.input:
        evaluation_input = load_evaluation_input(args.input)
        if args.evaluation_id is not None:
            evaluation_input = EvaluationInput(
                date=evaluation_input.date,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction=evaluation_input.prediction,
                observation=evaluation_input.observation,
                evaluation_id=args.evaluation_id,
                asset=evaluation_input.asset,
                target=evaluation_input.target,
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
    )


def cmd_init_db(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
    print(f"Initialized runtime db: {cfg.db_path}")
    print(f"  Asset:    {cfg.asset}")
    print(f"  Target:   {cfg.target}")
    return 0


def cmd_register_hypothesis(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        hypothesis, created = store.register_hypothesis(args.hypothesis_id)
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
    print(f"Hypothesis [{verb}] {hypothesis.hypothesis_id}")
    _print_hypothesis_details(hypothesis)
    return 0


def cmd_pause_hypothesis(args: argparse.Namespace) -> int:
    return _cmd_change_hypothesis_status(
        args,
        action="pause",
        verb="paused",
    )


def cmd_resume_hypothesis(args: argparse.Namespace) -> int:
    return _cmd_change_hypothesis_status(
        args,
        action="resume",
        verb="resumed",
    )


def cmd_retire_hypothesis(args: argparse.Namespace) -> int:
    return _cmd_change_hypothesis_status(
        args,
        action="retire",
        verb="retired",
    )


def cmd_record_prediction(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target=cfg.target,
            date=args.date,
        )
        prediction, created = store.record_prediction(
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
            prediction_value=args.prediction,
        )
    outcome = "created" if created else "existing"
    print(f"Prediction [{outcome}] {prediction.evaluation_id}")
    print(f"  Asset:    {prediction.asset}")
    print(f"  Target:   {prediction.target}")
    print(f"  Hyp:      {prediction.hypothesis_id}")
    print(f"  Value:    {prediction.value:.6f}")
    return 0


def cmd_finalize_observation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target=cfg.target,
            date=args.date,
        )
        observation, created = store.finalize_observation(
            evaluation_id=evaluation_id,
            observation_value=args.observation,
        )
    outcome = "created" if created else "existing"
    print(f"Observation [{outcome}] {observation.evaluation_id}")
    print(f"  Asset:    {observation.asset}")
    print(f"  Target:   {observation.target}")
    print(f"  Value:    {observation.value:.6f}")
    return 0


def cmd_update_state(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target=cfg.target,
            date=args.date,
        )
        snapshot, created = store.update_state(
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
        )
        metric = store.get_hypothesis_metric(args.hypothesis_id)
    _print_evaluation_snapshot(snapshot, created=created)
    _print_hypothesis_metric(metric)
    return 0


def cmd_generate_evaluation_input(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        definition = _registered_hypothesis_definition(
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
    print(f"  Target:   {evaluation_input.target}")
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
        definition = _registered_hypothesis_definition(
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
    print(f"  Asset:    {DEFAULT_ASSET}")
    print(f"  Target:   {DEFAULT_TARGET}")
    if evaluation_inputs:
        print(f"  Range:    {evaluation_inputs[0].date} -> {evaluation_inputs[-1].date}")
    return 0


def cmd_run_cycle(args: argparse.Namespace) -> int:
    evaluation_input = _resolve_evaluation_input(args)
    input_source = "json_file" if args.input else "manual"
    with _runtime_store(args.db) as (cfg, store):
        evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
            asset=cfg.asset,
            target=cfg.target,
            date=evaluation_input.date,
        )
        snapshot, created = store.run_cycle(
            evaluation_id=evaluation_id,
            hypothesis_id=evaluation_input.hypothesis_id,
            prediction_value=evaluation_input.prediction,
            observation_value=evaluation_input.observation,
            input_source=input_source,
        )
        metric = store.get_hypothesis_metric(evaluation_input.hypothesis_id)
    _print_evaluation_snapshot(snapshot, created=created)
    _print_hypothesis_metric(metric)
    return 0


def cmd_run_cycles(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
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
    with _runtime_store(str(db_path)) as (cfg, store):
        created_count = 0
        existing_count = 0
        latest_snapshot = None
        latest_metric = None
        for evaluation_input in evaluation_inputs:
            evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
                asset=cfg.asset,
                target=cfg.target,
                date=evaluation_input.date,
            )
            latest_snapshot, created = store.run_cycle(
                evaluation_id=evaluation_id,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction_value=evaluation_input.prediction,
                observation_value=evaluation_input.observation,
                input_source=input_source,
                input_range_start=input_range_start,
                input_range_end=input_range_end,
                signal_name=signal_name,
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
            latest_metric = store.get_hypothesis_metric(evaluation_input.hypothesis_id)

    print(
        "Batch complete: "
        f"evaluations={len(evaluation_inputs)} created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.hypothesis_id}")
        _print_hypothesis_metric(latest_metric)
    return 0


def cmd_run_backfill(args: argparse.Namespace) -> int:
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


def cmd_run_hypotheses_backfill(args: argparse.Namespace) -> int:
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
        for evaluation_input in all_evaluation_inputs:
            evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
                asset=cfg.asset,
                target=cfg.target,
                date=evaluation_input.date,
            )
            latest_snapshot, created = store.run_cycle(
                evaluation_id=evaluation_id,
                hypothesis_id=evaluation_input.hypothesis_id,
                prediction_value=evaluation_input.prediction,
                observation_value=evaluation_input.observation,
                input_source="signal_noise_backfill",
                input_range_start=args.start_date,
                input_range_end=args.end_date,
                signal_name=args.signal_name,
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
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
            asset=cfg.asset,
            target=cfg.target,
            hypothesis_ids=hypothesis_ids,
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        hypotheses = store.list_hypotheses(asset=cfg.asset, target=cfg.target)
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
    print(f"  Target:   {cfg.target}")
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
    registered = sum(1 for item in hypotheses if item.status == "registered")
    paused = sum(1 for item in hypotheses if item.status == "paused")
    retired = sum(1 for item in hypotheses if item.status == "retired")
    print(
        "  Hyp:      "
        f"total={total} registered={registered} paused={paused} retired={retired}"
    )
    tracked = len(metrics)
    mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in metrics) / tracked
    mean_mmc = 0.0 if tracked == 0 else sum(item.mmc for item in metrics) / tracked
    print(f"  Metrics:  tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc:.6f}")
    return 0


def cmd_show_evaluations(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-db":
            return cmd_init_db(args)
        if args.command == "register-hypothesis":
            return cmd_register_hypothesis(args)
        if args.command == "pause-hypothesis":
            return cmd_pause_hypothesis(args)
        if args.command == "resume-hypothesis":
            return cmd_resume_hypothesis(args)
        if args.command == "retire-hypothesis":
            return cmd_retire_hypothesis(args)
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
            return cmd_run_cycle(args)
        if args.command == "apply-evaluations":
            return cmd_run_cycles(args)
        if args.command == "apply-backfill":
            return cmd_run_backfill(args)
        if args.command == "apply-hypotheses-backfill":
            return cmd_run_hypotheses_backfill(args)
        if args.command == "status":
            return cmd_status(args)
        if args.command == "show-evaluations":
            return cmd_show_evaluations(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2
