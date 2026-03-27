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

    pause = sub.add_parser("pause-hypothesis", help="Pause one live hypothesis")
    pause.add_argument("--db", type=str, default=None)
    pause.add_argument("--hypothesis-id", type=str, required=True)

    resume = sub.add_parser("resume-hypothesis", help="Resume one paused hypothesis")
    resume.add_argument("--db", type=str, default=None)
    resume.add_argument("--hypothesis-id", type=str, required=True)

    retire = sub.add_parser("retire-hypothesis", help="Retire one active or paused hypothesis")
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
    record.add_argument("--cycle-id", type=str, default=None)

    finalize = sub.add_parser(
        "finalize-observation",
        help="Finalize one v2 observation before state update",
    )
    finalize.add_argument("--db", type=str, default=None)
    finalize.add_argument("--date", type=str, required=True)
    finalize.add_argument("--observation", type=float, required=True)
    finalize.add_argument("--cycle-id", type=str, default=None)

    update = sub.add_parser(
        "update-state",
        help="Update v2 state from recorded prediction and finalized observation",
    )
    update.add_argument("--db", type=str, default=None)
    update.add_argument("--date", type=str, required=True)
    update.add_argument("--hypothesis-id", type=str, required=True)
    update.add_argument("--cycle-id", type=str, default=None)

    build = sub.add_parser(
        "build-cycle-input",
        help="Build one deterministic v1 cycle-input JSON from signal-noise daily closes",
    )
    build.add_argument("--date", type=str, required=True)
    build.add_argument("--hypothesis-id", type=str, required=True)
    build.add_argument("--out", type=str, required=True)
    build.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    build.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    builds = sub.add_parser(
        "build-cycle-inputs",
        help="Build deterministic v1 cycle-input JSON for a date range from signal-noise daily closes",
    )
    builds.add_argument("--start-date", type=str, required=True)
    builds.add_argument("--end-date", type=str, required=True)
    builds.add_argument("--hypothesis-id", type=str, required=True)
    builds.add_argument("--out", type=str, required=True)
    builds.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    builds.add_argument("--signal-name", type=str, default=DEFAULT_PRICE_SIGNAL)

    run = sub.add_parser(
        "run-cycle",
        help="Convenience wrapper for record-prediction -> finalize-observation -> update-state",
    )
    run.add_argument("--db", type=str, default=None)
    run.add_argument("--date", type=str, default=None)
    run.add_argument("--hypothesis-id", type=str, default=None)
    run.add_argument("--prediction", type=float, default=None)
    run.add_argument("--observation", type=float, default=None)
    run.add_argument("--cycle-id", type=str, default=None)
    run.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON object with date, hypothesis_id, prediction, observation",
    )

    batch = sub.add_parser(
        "run-cycles",
        help="Run a deterministic batch through the convenience cycle wrapper",
    )
    batch.add_argument("--db", type=str, default=None)
    batch.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON array of cycle input objects",
    )

    backfill = sub.add_parser(
        "run-backfill",
        help="Build deterministic cycle inputs and apply them through the convenience cycle wrapper",
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

    status = sub.add_parser("status", help="Show the latest BTC sleeve state")
    status.add_argument("--db", type=str, default=None)

    show = sub.add_parser("show-cycles", help="Show recent cycle snapshots with provenance")
    show.add_argument("--db", type=str, default=None)
    show.add_argument("--limit", type=int, default=10)

    return parser


def _print_cycle_snapshot(snapshot, *, created: bool) -> None:
    outcome = "created" if created else "existing"
    print(f"Cycle [{outcome}] {snapshot.cycle_id}")
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


def _resolve_cycle_input(args: argparse.Namespace) -> CycleInput:
    if args.input:
        cycle_input = load_cycle_input(args.input)
        if args.cycle_id is not None:
            cycle_input = CycleInput(
                date=cycle_input.date,
                hypothesis_id=cycle_input.hypothesis_id,
                prediction=cycle_input.prediction,
                observation=cycle_input.observation,
                cycle_id=args.cycle_id,
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
        raise ValueError(f"run-cycle requires --input or manual values for: {joined}")
    return CycleInput(
        date=str(args.date),
        hypothesis_id=str(args.hypothesis_id),
        prediction=float(args.prediction),
        observation=float(args.observation),
        cycle_id=None if args.cycle_id is None else str(args.cycle_id),
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
    cycle_id = args.cycle_id or f"{cfg.asset}:{cfg.target}:{args.date}"
    store = V1Store(cfg.db_path)
    try:
        prediction, created = store.record_prediction(
            cycle_id=cycle_id,
            hypothesis_id=args.hypothesis_id,
            prediction_value=args.prediction,
        )
    finally:
        store.close()
    outcome = "created" if created else "existing"
    print(f"Prediction [{outcome}] {prediction.cycle_id}")
    print(f"  Asset:    {prediction.asset}")
    print(f"  Target:   {prediction.target}")
    print(f"  Hyp:      {prediction.hypothesis_id}")
    print(f"  Value:    {prediction.value:.6f}")
    return 0


def cmd_finalize_observation(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_id = args.cycle_id or f"{cfg.asset}:{cfg.target}:{args.date}"
    store = V1Store(cfg.db_path)
    try:
        observation, created = store.finalize_observation(
            cycle_id=cycle_id,
            observation_value=args.observation,
        )
    finally:
        store.close()
    outcome = "created" if created else "existing"
    print(f"Observation [{outcome}] {observation.cycle_id}")
    print(f"  Asset:    {observation.asset}")
    print(f"  Target:   {observation.target}")
    print(f"  Value:    {observation.value:.6f}")
    return 0


def cmd_update_state(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_id = args.cycle_id or f"{cfg.asset}:{cfg.target}:{args.date}"
    store = V1Store(cfg.db_path)
    try:
        snapshot, created = store.update_state(
            cycle_id=cycle_id,
            hypothesis_id=args.hypothesis_id,
        )
    finally:
        store.close()
    _print_cycle_snapshot(snapshot, created=created)
    return 0


def cmd_build_cycle_input(args: argparse.Namespace) -> int:
    cycle_input = build_cycle_input_from_signal_noise(
        date=args.date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
    )
    output_path = write_cycle_input(args.out, cycle_input)
    print(f"Built cycle input: {output_path}")
    print(f"  Asset:    {cycle_input.asset}")
    print(f"  Target:   {cycle_input.target}")
    print(f"  Date:     {cycle_input.date}")
    print(f"  Hyp:      {cycle_input.hypothesis_id}")
    print(f"  Signal:   pred={cycle_input.prediction:.6f} obs={cycle_input.observation:.6f}")
    return 0


def cmd_build_cycle_inputs(args: argparse.Namespace) -> int:
    cycle_inputs = build_cycle_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
    )
    output_path = write_cycle_inputs(args.out, cycle_inputs)
    print(f"Built cycle inputs: {output_path}")
    print(f"  Count:    {len(cycle_inputs)}")
    print(f"  Asset:    {DEFAULT_ASSET}")
    print(f"  Target:   {DEFAULT_TARGET}")
    if cycle_inputs:
        print(f"  Range:    {cycle_inputs[0].date} -> {cycle_inputs[-1].date}")
    return 0


def cmd_run_cycle(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_input = _resolve_cycle_input(args)
    cycle_id = cycle_input.cycle_id or f"{cfg.asset}:{cfg.target}:{cycle_input.date}"
    input_source = "json_file" if args.input else "manual"
    store = V1Store(cfg.db_path)
    try:
        snapshot, created = store.run_cycle(
            cycle_id=cycle_id,
            hypothesis_id=cycle_input.hypothesis_id,
            prediction_value=cycle_input.prediction,
            observation_value=cycle_input.observation,
            input_source=input_source,
        )
    finally:
        store.close()
    _print_cycle_snapshot(snapshot, created=created)
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
            cycle_id = cycle_input.cycle_id or f"{cfg.asset}:{cfg.target}:{cycle_input.date}"
            latest_snapshot, created = store.run_cycle(
                cycle_id=cycle_id,
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
        f"Batch complete: cycles={len(cycle_inputs)} created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(f"  Latest:   {latest_snapshot.cycle_id}")
        print(f"  Quality:  {latest_snapshot.quality_after:.6f}")
        print(f"  Trust:    {latest_snapshot.allocation_trust_after:.6f}")
    return 0


def cmd_run_backfill(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    cycle_inputs = build_cycle_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
        base_url=args.base_url,
        signal_name=args.signal_name,
    )
    if args.out is not None:
        output_path = write_cycle_inputs(args.out, cycle_inputs)
        print(f"Wrote cycle inputs: {output_path}")
    return _run_cycle_inputs(
        cfg.db_path,
        cycle_inputs,
        input_source="signal_noise_backfill",
        input_range_start=args.start_date,
        input_range_end=args.end_date,
        signal_name=args.signal_name,
    )


def cmd_status(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        sleeve_state = store.get_sleeve_state(asset=cfg.asset, target=cfg.target)
        latest = None if sleeve_state is None or sleeve_state.latest_cycle_id is None else (
            store.get_cycle_snapshot(sleeve_state.latest_cycle_id)
        )
    finally:
        store.close()

    print("alpha-os v1 status")
    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Asset:    {cfg.asset}")
    print(f"  Target:   {cfg.target}")
    if sleeve_state is None:
        print("  Latest:   no cycles recorded")
        return 0
    print(f"  Latest:   {sleeve_state.latest_cycle_id}")
    print(f"  Live:     {sleeve_state.live_hypothesis_count}")
    print(f"  Quality:  mean={sleeve_state.mean_quality:.6f}")
    print(f"  Trust:    total={sleeve_state.total_allocation_trust:.6f}")
    if latest is not None:
        print(f"  Weight:   latest={latest.generated_weight:.6f}")
    return 0


def cmd_show_cycles(args: argparse.Namespace) -> int:
    cfg = build_config(db_path=args.db)
    store = V1Store(cfg.db_path)
    try:
        store.ensure_schema()
        snapshots = store.list_cycle_snapshots(limit=args.limit)
    finally:
        store.close()

    print("alpha-os v1 cycles")
    print(f"  DB:       {Path(cfg.db_path)}")
    print(f"  Count:    {len(snapshots)}")
    for snapshot in snapshots:
        range_text = "-"
        if snapshot.input_range_start or snapshot.input_range_end:
            start = snapshot.input_range_start or "-"
            end = snapshot.input_range_end or "-"
            range_text = f"{start}->{end}"
        print(
            f"  {snapshot.cycle_id} "
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
        if args.command == "build-cycle-input":
            return cmd_build_cycle_input(args)
        if args.command == "build-cycle-inputs":
            return cmd_build_cycle_inputs(args)
        if args.command == "run-cycle":
            return cmd_run_cycle(args)
        if args.command == "run-cycles":
            return cmd_run_cycles(args)
        if args.command == "run-backfill":
            return cmd_run_backfill(args)
        if args.command == "status":
            return cmd_status(args)
        if args.command == "show-cycles":
            return cmd_show_cycles(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2
