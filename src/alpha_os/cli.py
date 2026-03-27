from __future__ import annotations

import argparse
from pathlib import Path

from .build import (
    build_cycle_input_from_signal_noise,
    build_cycle_inputs_from_signal_noise,
    write_cycle_input,
    write_cycle_inputs,
)
from .config import (
    DEFAULT_ASSET,
    DEFAULT_PRICE_SIGNAL,
    DEFAULT_SIGNAL_NOISE_BASE_URL,
    DEFAULT_TARGET,
    build_config,
)
from .hypothesis_registry import HypothesisDefinition
from .inputs import CycleInput, load_cycle_input, load_cycle_inputs
from .store import V1Store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-os",
        description="alpha-os trust engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Initialize the v1 runtime database")
    init_db.add_argument("--db", type=str, default=None)

    register = sub.add_parser(
        "register-hypothesis",
        help="Register one v2 hypothesis before recording predictions",
    )
    register.add_argument("--db", type=str, default=None)
    register.add_argument("--hypothesis-id", type=str, required=True)

    pause = sub.add_parser("pause-hypothesis", help="Pause one allocation-eligible hypothesis")
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
        help="Record one v2 prediction before observation finalization",
    )
    record.add_argument("--db", type=str, default=None)
    record.add_argument("--date", type=str, required=True)
    record.add_argument("--hypothesis-id", type=str, required=True)
    record.add_argument("--prediction", type=float, required=True)
    record.add_argument("--evaluation-id", type=str, default=None)

    finalize = sub.add_parser(
        "finalize-observation",
        help="Finalize one v2 observation before state update",
    )
    finalize.add_argument("--db", type=str, default=None)
    finalize.add_argument("--date", type=str, required=True)
    finalize.add_argument("--observation", type=float, required=True)
    finalize.add_argument("--evaluation-id", type=str, default=None)

    update = sub.add_parser(
        "update-state",
        help="Update v2 state from recorded prediction and finalized observation",
    )
    update.add_argument("--db", type=str, default=None)
    update.add_argument("--date", type=str, required=True)
    update.add_argument("--hypothesis-id", type=str, required=True)
    update.add_argument("--evaluation-id", type=str, default=None)

    build = sub.add_parser(
        "generate-cycle-input",
        help="Generate one deterministic evaluation-input JSON from signal-noise daily closes",
    )
    build.add_argument("--db", type=str, default=None)
    build.add_argument("--date", type=str, required=True)
    build.add_argument("--hypothesis-id", type=str, required=True)
    build.add_argument("--out", type=str, required=True)
    build.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    build.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    builds = sub.add_parser(
        "generate-cycle-inputs",
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
        "apply-cycle",
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
        "apply-cycles",
        help="Apply a deterministic batch of evaluation inputs through the bounded runtime",
    )
    batch.add_argument("--db", type=str, default=None)
    batch.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON array of cycle input objects",
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
        help="Optional path to write the generated cycle-input JSON array",
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

    status = sub.add_parser("status", help="Show the latest BTC sleeve state")
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


def _runtime_hypothesis_definition(*, db_path: str | None, hypothesis_id: str) -> HypothesisDefinition:
    cfg = build_config(db_path=db_path)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        hypothesis = store.get_hypothesis(hypothesis_id)
    finally:
        store.close()
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
    print(
        "  Quality:  "
        f"{snapshot.quality_before:.6f} -> {snapshot.quality_after:.6f} "
        f"(delta={snapshot.quality_delta:+.6f})"
    )
    print(
        "  Trust:    "
        f"{snapshot.allocation_trust_before:.6f} -> {snapshot.allocation_trust_after:.6f} "
        f"(delta={snapshot.allocation_trust_delta:+.6f})"
    )
    print(f"  Weight:   {snapshot.generated_weight:.6f}")


def _unique_hypothesis_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _print_hypothesis_competition_summary(
    store: V1Store,
    *,
    asset: str,
    target: str,
    hypothesis_ids: list[str],
) -> None:
    selected = set(hypothesis_ids)
    hypotheses = [
        item
        for item in store.list_hypotheses(asset=asset, target=target)
        if item.hypothesis_id in selected
    ]
    print("alpha-os v1 hypothesis competition")
    print(f"  Count:    {len(hypotheses)}")
    for hypothesis in hypotheses:
        live_label = "yes" if hypothesis.status == "registered" and hypothesis.allocation_trust > 0.0 else "no"
        kind = hypothesis.kind or "-"
        signal_name = hypothesis.signal_name or "-"
        lookback = "-" if hypothesis.lookback is None else str(hypothesis.lookback)
        print(
            f"  {hypothesis.hypothesis_id} "
            f"kind={kind} signal={signal_name} lookback={lookback} "
            f"status={hypothesis.status} live={live_label} "
            f"q={hypothesis.quality:.6f} "
            f"t={hypothesis.allocation_trust:.6f} "
            f"evals={hypothesis.observation_count}"
        )


def _resolve_cycle_input(args: argparse.Namespace) -> CycleInput:
    if args.input:
        cycle_input = load_cycle_input(args.input)
        if args.evaluation_id is not None:
            cycle_input = CycleInput(
                date=cycle_input.date,
                hypothesis_id=cycle_input.hypothesis_id,
                prediction=cycle_input.prediction,
                observation=cycle_input.observation,
                evaluation_id=args.evaluation_id,
                asset=cycle_input.asset,
                target=cycle_input.target,
            )
        return cycle_input

    required = {
        "date": args.date,
        "hypothesis_id": args.hypothesis_id,
        "prediction": args.prediction,
        "observation": args.observation,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"apply-cycle requires --input or manual values for: {joined}")
    return CycleInput(
        date=str(args.date),
        hypothesis_id=str(args.hypothesis_id),
        prediction=float(args.prediction),
        observation=float(args.observation),
        evaluation_id=None if args.evaluation_id is None else str(args.evaluation_id),
    )


def cmd_init_db(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
    finally:
        store.close()
    print(f"Initialized v1 db: {cfg.db_path}")
    print(f"  Asset:    {cfg.asset}")
    print(f"  Target:   {cfg.target}")
    return 0


def cmd_register_hypothesis(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        hypothesis, created = store.register_hypothesis(args.hypothesis_id)
    finally:
        store.close()
    outcome = "created" if created else "existing"
    print(f"Hypothesis [{outcome}] {hypothesis.hypothesis_id}")
    print(f"  Asset:    {hypothesis.asset}")
    print(f"  Target:   {hypothesis.target}")
    if hypothesis.kind is not None:
        print(f"  Kind:     {hypothesis.kind}")
    if hypothesis.signal_name is not None:
        print(f"  Signal:   {hypothesis.signal_name}")
    if hypothesis.lookback is not None:
        print(f"  Lookback: {hypothesis.lookback}")
    print(f"  Status:   {hypothesis.status}")
    print(f"  Quality:  {hypothesis.quality:.6f}")
    print(f"  Trust:    {hypothesis.allocation_trust:.6f}")
    return 0


def _cmd_change_hypothesis_status(
    args: argparse.Namespace,
    *,
    action: str,
    verb: str,
) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        hypothesis = store.set_hypothesis_status(
            args.hypothesis_id,
            action=action,
        )
    finally:
        store.close()
    print(f"Hypothesis [{verb}] {hypothesis.hypothesis_id}")
    print(f"  Asset:    {hypothesis.asset}")
    print(f"  Target:   {hypothesis.target}")
    if hypothesis.kind is not None:
        print(f"  Kind:     {hypothesis.kind}")
    if hypothesis.signal_name is not None:
        print(f"  Signal:   {hypothesis.signal_name}")
    if hypothesis.lookback is not None:
        print(f"  Lookback: {hypothesis.lookback}")
    print(f"  Status:   {hypothesis.status}")
    print(f"  Quality:  {hypothesis.quality:.6f}")
    print(f"  Trust:    {hypothesis.allocation_trust:.6f}")
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
    cfg = build_config(db_path=args.db)
    evaluation_id = args.evaluation_id or _default_evaluation_id(
        asset=cfg.asset,
        target=cfg.target,
        date=args.date,
    )
    store = V1Store(cfg.db_path)
    try:
        prediction, created = store.record_prediction(
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
            prediction_value=args.prediction,
        )
    finally:
        store.close()
    outcome = "created" if created else "existing"
    print(f"Prediction [{outcome}] {prediction.evaluation_id}")
    print(f"  Asset:    {prediction.asset}")
    print(f"  Target:   {prediction.target}")
    print(f"  Hyp:      {prediction.hypothesis_id}")
    print(f"  Value:    {prediction.value:.6f}")
    return 0


def cmd_finalize_observation(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    evaluation_id = args.evaluation_id or _default_evaluation_id(
        asset=cfg.asset,
        target=cfg.target,
        date=args.date,
    )
    store = V1Store(cfg.db_path)
    try:
        observation, created = store.finalize_observation(
            evaluation_id=evaluation_id,
            observation_value=args.observation,
        )
    finally:
        store.close()
    outcome = "created" if created else "existing"
    print(f"Observation [{outcome}] {observation.evaluation_id}")
    print(f"  Asset:    {observation.asset}")
    print(f"  Target:   {observation.target}")
    print(f"  Value:    {observation.value:.6f}")
    return 0


def cmd_update_state(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    evaluation_id = args.evaluation_id or _default_evaluation_id(
        asset=cfg.asset,
        target=cfg.target,
        date=args.date,
    )
    store = V1Store(cfg.db_path)
    try:
        snapshot, created = store.update_state(
            evaluation_id=evaluation_id,
            hypothesis_id=args.hypothesis_id,
        )
    finally:
        store.close()
    _print_evaluation_snapshot(snapshot, created=created)
    return 0


def cmd_build_cycle_input(args: argparse.Namespace) -> int:
    definition = _runtime_hypothesis_definition(
        db_path=args.db,
        hypothesis_id=args.hypothesis_id,
    )
    cycle_input = build_cycle_input_from_signal_noise(
        date=args.date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
        definition=definition,
    )
    output_path = write_cycle_input(args.out, cycle_input)
    print(f"Generated evaluation input: {output_path}")
    print(f"  Asset:    {cycle_input.asset}")
    print(f"  Target:   {cycle_input.target}")
    print(f"  Date:     {cycle_input.date}")
    print(f"  Hyp:      {cycle_input.hypothesis_id}")
    print(f"  Signal:   pred={cycle_input.prediction:.6f} obs={cycle_input.observation:.6f}")
    return 0


def cmd_build_cycle_inputs(args: argparse.Namespace) -> int:
    definition = _runtime_hypothesis_definition(
        db_path=args.db,
        hypothesis_id=args.hypothesis_id,
    )
    cycle_inputs = build_cycle_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
        definition=definition,
    )
    output_path = write_cycle_inputs(args.out, cycle_inputs)
    print(f"Generated evaluation inputs: {output_path}")
    print(f"  Count:    {len(cycle_inputs)}")
    print(f"  Asset:    {DEFAULT_ASSET}")
    print(f"  Target:   {DEFAULT_TARGET}")
    if cycle_inputs:
        print(f"  Range:    {cycle_inputs[0].date} -> {cycle_inputs[-1].date}")
    return 0


def cmd_run_cycle(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_input = _resolve_cycle_input(args)
    evaluation_id = cycle_input.evaluation_id or _default_evaluation_id(
        asset=cfg.asset,
        target=cfg.target,
        date=cycle_input.date,
    )
    input_source = "json_file" if args.input else "manual"
    store = V1Store(cfg.db_path)
    try:
        snapshot, created = store.run_cycle(
            evaluation_id=evaluation_id,
            hypothesis_id=cycle_input.hypothesis_id,
            prediction_value=cycle_input.prediction,
            observation_value=cycle_input.observation,
            input_source=input_source,
        )
    finally:
        store.close()
    _print_evaluation_snapshot(snapshot, created=created)
    return 0


def cmd_run_cycles(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_inputs = load_cycle_inputs(args.input)
    return _run_cycle_inputs(
        cfg.db_path,
        cycle_inputs,
        input_source="json_batch",
    )


def _run_cycle_inputs(
    db_path: Path,
    cycle_inputs: list[CycleInput],
    *,
    input_source: str,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    signal_name: str | None = None,
) -> int:
    cfg = build_config(db_path=str(db_path))
    store = V1Store(cfg.db_path)
    try:
        created_count = 0
        existing_count = 0
        latest_snapshot = None
        for cycle_input in cycle_inputs:
            evaluation_id = cycle_input.evaluation_id or _default_evaluation_id(
                asset=cfg.asset,
                target=cfg.target,
                date=cycle_input.date,
            )
            latest_snapshot, created = store.run_cycle(
                evaluation_id=evaluation_id,
                hypothesis_id=cycle_input.hypothesis_id,
                prediction_value=cycle_input.prediction,
                observation_value=cycle_input.observation,
                input_source=input_source,
                input_range_start=input_range_start,
                input_range_end=input_range_end,
                signal_name=signal_name,
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
    finally:
        store.close()

    print(
        "Batch complete: "
        f"evaluations={len(cycle_inputs)} created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.hypothesis_id}")
        print(f"  Quality:  {latest_snapshot.quality_after:.6f}")
        print(f"  Trust:    {latest_snapshot.allocation_trust_after:.6f}")
    return 0


def cmd_run_backfill(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    definition = _runtime_hypothesis_definition(
        db_path=args.db,
        hypothesis_id=args.hypothesis_id,
    )
    cycle_inputs = build_cycle_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
        definition=definition,
    )
    if args.out is not None:
        output_path = write_cycle_inputs(args.out, cycle_inputs)
        print(f"Wrote evaluation inputs: {output_path}")
    return _run_cycle_inputs(
        cfg.db_path,
        cycle_inputs,
        input_source="signal_noise_backfill",
        input_range_start=args.start_date,
        input_range_end=args.end_date,
        signal_name=args.signal_name,
    )


def cmd_run_hypotheses_backfill(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    hypothesis_ids = _unique_hypothesis_ids(args.hypothesis_id)
    total_evaluations = 0
    created_count = 0
    existing_count = 0
    latest_snapshot = None

    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        for hypothesis_id in hypothesis_ids:
            definition = _runtime_hypothesis_definition(
                db_path=args.db,
                hypothesis_id=hypothesis_id,
            )
            cycle_inputs = build_cycle_inputs_from_signal_noise(
                start_date=args.start_date,
                end_date=args.end_date,
                hypothesis_id=hypothesis_id,
                base_url=args.base_url,
                signal_name=args.signal_name,
                definition=definition,
            )
            total_evaluations += len(cycle_inputs)
            for cycle_input in cycle_inputs:
                evaluation_id = cycle_input.evaluation_id or _default_evaluation_id(
                    asset=cfg.asset,
                    target=cfg.target,
                    date=cycle_input.date,
                )
                latest_snapshot, created = store.run_cycle(
                    evaluation_id=evaluation_id,
                    hypothesis_id=cycle_input.hypothesis_id,
                    prediction_value=cycle_input.prediction,
                    observation_value=cycle_input.observation,
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
            f"evaluations={total_evaluations} "
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
    finally:
        store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        sleeve_state = store.get_sleeve_state(asset=cfg.asset, target=cfg.target)
        latest = (
            None
            if sleeve_state is None
            or sleeve_state.latest_evaluation_id is None
            or sleeve_state.latest_hypothesis_id is None
            else store.get_evaluation_snapshot(
                sleeve_state.latest_evaluation_id,
                sleeve_state.latest_hypothesis_id,
            )
        )
    finally:
        store.close()

    print("alpha-os v1 status")
    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Asset:    {cfg.asset}")
    print(f"  Target:   {cfg.target}")
    if sleeve_state is None:
        print("  Latest:   no evaluations recorded")
        return 0
    print(
        f"  Latest:   {sleeve_state.latest_evaluation_id} / {sleeve_state.latest_hypothesis_id}"
    )
    print(f"  Live:     {sleeve_state.live_hypothesis_count}")
    print(f"  Quality:  mean={sleeve_state.mean_quality:.6f}")
    print(f"  Trust:    total={sleeve_state.total_allocation_trust:.6f}")
    if latest is not None:
        print(f"  Weight:   latest={latest.generated_weight:.6f}")
    return 0


def cmd_show_evaluations(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        snapshots = store.list_evaluation_snapshots(limit=args.limit)
    finally:
        store.close()

    print("alpha-os v1 evaluations")
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
            f"q={snapshot.quality_after:.6f} "
            f"t={snapshot.allocation_trust_after:.6f}"
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
        if args.command == "generate-cycle-input":
            return cmd_build_cycle_input(args)
        if args.command == "generate-cycle-inputs":
            return cmd_build_cycle_inputs(args)
        if args.command == "apply-cycle":
            return cmd_run_cycle(args)
        if args.command == "apply-cycles":
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
