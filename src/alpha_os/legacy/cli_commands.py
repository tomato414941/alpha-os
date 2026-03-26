from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import Config


def add_hidden_root_legacy_commands(sub) -> None:
    rex = sub.add_parser("replay-experiment", help=argparse.SUPPRESS)
    _add_replay_experiment_arguments(rex)

    rmx = sub.add_parser("replay-matrix", help=argparse.SUPPRESS)
    _add_replay_matrix_arguments(rmx)


def add_legacy_subcommands(legacy_sub) -> None:
    ladm = legacy_sub.add_parser(
        "admission-daemon",
        help="Run the legacy candidate admission daemon",
    )
    ladm.add_argument("--asset", type=str, default="BTC")
    ladm.add_argument("--config", type=str, default=None)

    lpsc = legacy_sub.add_parser(
        "prune-stale-candidates",
        help="Reject stale pending candidates outside the active discovery/manual sources",
    )
    lpsc.add_argument("--asset", type=str, default="BTC")
    lpsc.add_argument("--max-age-days", type=int, default=7)
    lpsc.add_argument("--dry-run", action="store_true")


def add_legacy_research_subcommands(research_sub) -> None:
    rpr = research_sub.add_parser(
        "paper-replay",
        help="Run the legacy historical paper replay path",
    )
    rpr.add_argument("--start", type=str, required=True, help="Start date for replay (ISO format)")
    rpr.add_argument("--end", type=str, required=True, help="End date for replay (ISO format)")
    rpr.add_argument(
        "--sizing-mode",
        type=str,
        default="runtime",
        choices=["runtime", "raw_mean", "compare"],
        help="Legacy replay sizing mode",
    )
    rpr.add_argument("--asset", type=str, default="BTC")
    rpr.add_argument("--config", type=str, default=None)

    rrex = research_sub.add_parser(
        "replay-experiment",
        help="Run a legacy replay experiment and persist the artifact",
    )
    _add_replay_experiment_arguments(rrex)

    rrmx = research_sub.add_parser(
        "replay-matrix",
        help="Run a TOML-defined replay experiment matrix",
    )
    _add_replay_matrix_arguments(rrmx)


def cmd_paper_replay(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    _cmd_paper_replay(args, cfg)


def cmd_replay_experiment(args: argparse.Namespace) -> None:
    from .replay_experiment import (
        ReplayExperimentSpec,
        parse_override_assignment,
        run_replay_experiment,
    )

    overrides = dict(parse_override_assignment(raw) for raw in args.set)
    run = run_replay_experiment(
        ReplayExperimentSpec(
            name=args.name,
            asset=args.asset,
            start_date=args.start,
            end_date=args.end,
            config_path=Path(args.config) if args.config else None,
            managed_alpha_mode=args.managed_alpha_mode,
            admission_source=args.source,
            fail_state=args.fail_state,
            deployment_mode=args.deployment_mode,
            sizing_mode=args.sizing_mode,
            overrides=overrides,
            notes=args.notes,
        )
    )

    result = run.payload["result"]
    profile = run.payload.get("runtime_profile", {})
    profile_id = profile.get("profile_id", "")
    profile_commit = profile.get("git_commit", "")
    print(f"Replay experiment: {run.experiment_id}")
    print(f"  Detail:   {run.detail_path}")
    print(f"  Index:    {run.index_path}")
    if profile_id:
        suffix = f" ({profile_commit[:8]})" if profile_commit else ""
        print(f"  Profile:  {profile_id[:12]}{suffix}")
    print(f"  Deployment: {run.payload['deployment']['mode']}")
    print(f"  Final:    ${result['final_value']:,.2f}")
    print(f"  Return:   {result['total_return']:+.2%}")
    print(f"  Sharpe:   {result['sharpe']:.3f}")
    print(f"  Max DD:   {result['max_drawdown']:.2%}")
    print(f"  Trades:   {result['total_trades']}")


def cmd_replay_matrix(args: argparse.Namespace) -> None:
    from ..experiments.matrix import load_replay_matrix, run_replay_matrix

    matrix = load_replay_matrix(Path(args.manifest))
    runs = run_replay_matrix(matrix, max_workers=args.max_workers)

    print(f"Replay matrix: {args.manifest}")
    profile_ids: set[str] = set()
    for run in runs:
        result = run.payload["result"]
        profile_id = run.payload.get("runtime_profile", {}).get("profile_id", "")
        if profile_id:
            profile_ids.add(profile_id)
        profile_text = f" profile={profile_id[:12]}" if profile_id else ""
        print(
            f"  - {run.payload['name']}: "
            f"return={result['total_return']:+.2%} "
            f"sharpe={result['sharpe']:.3f} "
            f"dd={result['max_drawdown']:.2%} "
            f"trades={result['total_trades']} "
            f"{profile_text} "
            f"detail={run.detail_path}"
        )
    print(f"  Profiles: {len(profile_ids)} unique across {len(runs)} runs")


def cmd_admission_daemon(args: argparse.Namespace) -> None:
    from .admission import AdmissionDaemon

    cfg = _load_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    daemon = AdmissionDaemon(asset=args.asset, config=cfg)
    daemon.run()


def cmd_prune_stale_candidates(args: argparse.Namespace) -> None:
    from .admission_queue import prune_stale_pending_candidates

    stats = prune_stale_pending_candidates(
        args.asset,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )
    print(f"Stale candidates: asset={stats.asset}")
    print(f"  Max age days: {stats.max_age_days}")
    print(f"  Selected:     {stats.selected_count}")
    print(f"  Pruned:       {stats.pruned_count}")
    if args.dry_run:
        print("  Mode:         dry-run")


def _add_replay_experiment_arguments(parser) -> None:
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start", required=True, help="Replay start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Replay end date (YYYY-MM-DD)")
    parser.add_argument(
        "--managed-alpha-mode",
        choices=["current", "admission"],
        default="current",
        help="Use the current managed-alpha set as-is or rebuild it from admission rules first",
    )
    parser.add_argument(
        "--source",
        choices=["alphas", "candidates"],
        default="candidates",
        help="Admission replay source when --managed-alpha-mode=admission",
    )
    parser.add_argument(
        "--fail-state",
        choices=["rejected", "dormant"],
        default="rejected",
        help="Fallback state for records that fail admission replay",
    )
    parser.add_argument(
        "--deployment-mode",
        choices=["current", "refresh"],
        default="current",
        help="Use the current deployed alpha set or refresh it inside the experiment",
    )
    parser.add_argument(
        "--sizing-mode",
        type=str,
        default="runtime",
        choices=["runtime", "raw_mean"],
        help="Replay sizing mode",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override merged config via dotted path, e.g. lifecycle.candidate_quality_min=1.10",
    )
    parser.add_argument("--notes", default="", help="Optional experiment notes")


def _add_replay_matrix_arguments(parser) -> None:
    parser.add_argument("--manifest", required=True, help="Path to TOML matrix manifest")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for historical replay runs",
    )


def _cmd_paper_replay(args: argparse.Namespace, cfg: Config) -> None:
    from .replay_simulator import run_replay

    if not args.start or not args.end:
        print("Error: --replay requires --start and --end dates")
        sys.exit(1)

    print(f"Running legacy historical replay: {args.start} to {args.end} ({args.asset})")

    def _print_replay_result(label: str, result) -> None:
        print(f"\nLegacy Historical Replay [{label}]: {args.start} to {args.end}")
        print(f"  Days:       {result.n_days}")
        print(f"  {'─'*36}")
        print(f"  Initial:    ${result.initial_capital:,.2f}")
        print(f"  Final:      ${result.final_value:,.2f}")
        print(f"  Return:     {result.total_return:+.2%}")
        print(f"  Sharpe:     {result.sharpe:.3f}")
        print(f"  Max DD:     {result.max_drawdown:.2%}")
        print(f"  Trades:     {result.total_trades}")
        print(f"  Win Rate:   {result.win_rate:.1%}")
        print(f"  {'─'*36}")
        if result.best_day[0]:
            print(f"  Best Day:   {result.best_day[1]:+.2%} ({result.best_day[0]})")
        if result.worst_day[0]:
            print(f"  Worst Day:  {result.worst_day[1]:+.2%} ({result.worst_day[0]})")

    if args.sizing_mode == "compare":
        runtime_result = run_replay(
            asset=args.asset,
            config=cfg,
            start_date=args.start,
            end_date=args.end,
            sizing_mode="runtime",
        )
        raw_result = run_replay(
            asset=args.asset,
            config=cfg,
            start_date=args.start,
            end_date=args.end,
            sizing_mode="raw_mean",
        )
        _print_replay_result("runtime", runtime_result)
        _print_replay_result("raw_mean", raw_result)
        print("\nDelta (runtime - raw_mean)")
        print(f"  Final:      ${runtime_result.final_value - raw_result.final_value:+,.2f}")
        print(f"  Return:     {runtime_result.total_return - raw_result.total_return:+.2%}")
        print(f"  Sharpe:     {runtime_result.sharpe - raw_result.sharpe:+.3f}")
        print(f"  Max DD:     {runtime_result.max_drawdown - raw_result.max_drawdown:+.2%}")
        print(f"  Trades:     {runtime_result.total_trades - raw_result.total_trades:+d}")
        return

    result = run_replay(
        asset=args.asset,
        config=cfg,
        start_date=args.start,
        end_date=args.end,
        sizing_mode=args.sizing_mode,
    )
    _print_replay_result(args.sizing_mode, result)


def _load_config(config_path: str | None) -> Config:
    return Config.load(Path(config_path)) if config_path else Config.load()
