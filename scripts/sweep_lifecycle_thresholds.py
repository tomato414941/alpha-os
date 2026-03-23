#!/usr/bin/env python3
"""Sweep lifecycle thresholds with admission replay before historical replay."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class SweepRow:
    source_name: str
    source_rows: int
    candidate_quality_min: float
    active_quality_min: float
    starting_active: int
    result: object
    elapsed: float


def _parse_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _print_rows(rows: list[SweepRow]) -> None:
    print(
        "rank  source      cand   active  start   final      return   sharpe   max_dd   trades  days  elapsed"
    )
    print(
        "----  ----------  -----  ------  -----  ---------  -------  -------  -------  ------  ----  -------"
    )
    for idx, row in enumerate(rows, start=1):
        result = row.result
        print(
            f"{idx:>4}  "
            f"{row.source_name:>10}  "
            f"{row.candidate_quality_min:>5.2f}  "
            f"{row.active_quality_min:>6.2f}  "
            f"{row.starting_active:>5}  "
            f"${result.final_value:>8,.2f}  "
            f"{_format_pct(result.total_return):>7}  "
            f"{result.sharpe:>7.3f}  "
            f"{result.max_drawdown:>7.2%}  "
            f"{result.total_trades:>6}  "
            f"{result.n_days:>4}  "
            f"{row.elapsed:>6.1f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run historical replay sweeps over lifecycle threshold combinations.",
    )
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--config", required=True, help="Base TOML config path")
    parser.add_argument("--start", required=True, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Backfill end date (YYYY-MM-DD)")
    parser.add_argument(
        "--candidate-values",
        default="0.05,0.10,0.15",
        help="Comma-separated candidate_quality_min values",
    )
    parser.add_argument(
        "--active-values",
        default="0.00,0.05,0.10",
        help="Comma-separated active_quality_min values",
    )
    parser.add_argument(
        "--source",
        default="alphas",
        choices=["alphas", "candidates"],
        help="Replay source: current registry alphas or validated candidate history",
    )
    args = parser.parse_args()

    candidate_values = _parse_values(args.candidate_values)
    active_values = _parse_values(args.active_values)

    from alpha_os.legacy.admission_replay import (
        apply_registry_snapshot,
        load_source_records,
        materialize_admission_snapshot,
    )
    from alpha_os.config import Config
    from alpha_os.config import asset_data_dir
    from alpha_os.paper.simulator import run_replay

    registry_db = asset_data_dir(args.asset) / "alpha_registry.db"
    source_records = load_source_records(registry_db, args.source)
    rows: list[SweepRow] = []
    total = len(candidate_values) * len(active_values)
    run_no = 0

    print(
        f"Sweeping {total} combinations for {args.asset}: "
        f"{args.start} -> {args.end}"
    )
    print(
        f"source={args.source} rows={len(source_records)} "
        f"candidate_quality_min={candidate_values} "
        f"active_quality_min={active_values}"
    )
    print()

    for candidate_quality_min in candidate_values:
        for active_quality_min in active_values:
            run_no += 1
            cfg = Config.load(Path(args.config))
            cfg.lifecycle.candidate_quality_min = candidate_quality_min
            cfg.lifecycle.active_quality_min = active_quality_min

            print(
                f"[{run_no}/{total}] "
                f"source={args.source} "
                f"candidate={candidate_quality_min:.2f} "
                f"active={active_quality_min:.2f}",
                flush=True,
            )
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="alpha_os_replay_") as tmp:
                replay_db = Path(tmp) / "registry.db"
                snapshot, counts = materialize_admission_snapshot(
                    source_records,
                    cfg.to_lifecycle_config(),
                )
                apply_registry_snapshot(replay_db, snapshot)
                starting_active = counts["active"]
                result = run_replay(
                    asset=args.asset,
                    config=cfg,
                    start_date=args.start,
                    end_date=args.end,
                    registry_db=replay_db,
                )
            elapsed = time.perf_counter() - t0
            rows.append(
                SweepRow(
                    source_name=args.source,
                    source_rows=len(source_records),
                    candidate_quality_min=candidate_quality_min,
                    active_quality_min=active_quality_min,
                    starting_active=starting_active,
                    result=result,
                    elapsed=elapsed,
                )
            )
            print(
                f"      start_active={starting_active} "
                f"final=${result.final_value:,.2f} "
                f"return={_format_pct(result.total_return)} "
                f"sharpe={result.sharpe:.3f} "
                f"max_dd={result.max_drawdown:.2%} "
                f"trades={result.total_trades} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            row.result.final_value,
            row.result.sharpe,
            -row.result.max_drawdown,
            -row.result.total_trades,
        ),
        reverse=True,
    )

    print()
    print("Sorted by final_value, sharpe, max_drawdown, trades")
    _print_rows(rows)

    best = rows[0]
    print()
    print(
        "Best combination: "
        f"source={best.source_name}, "
        f"candidate_quality_min={best.candidate_quality_min:.2f}, "
        f"active_quality_min={best.active_quality_min:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
