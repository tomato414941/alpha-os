#!/usr/bin/env python3
"""Sweep lifecycle thresholds with admission replay before backfill."""
from __future__ import annotations

import argparse
import sqlite3
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


def _load_registry_records(asset: str) -> list[object]:
    from alpha_os.alpha.registry import AlphaRegistry
    from alpha_os.config import asset_data_dir

    registry = AlphaRegistry(asset_data_dir(asset) / "alpha_registry.db")
    try:
        return registry.top(n=registry.count(), metric="sharpe")
    finally:
        registry.close()


def _load_candidate_records(asset: str) -> list[object]:
    from alpha_os.alpha.registry import AlphaRecord, AlphaState
    from alpha_os.config import asset_data_dir

    db_path = asset_data_dir(asset) / "alpha_registry.db"
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT candidate_id, expression, fitness, oos_sharpe, pbo, dsr_pvalue, validated_at
            FROM candidates
            WHERE oos_sharpe IS NOT NULL
              AND pbo IS NOT NULL
              AND dsr_pvalue IS NOT NULL
            ORDER BY validated_at ASC, candidate_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    records = []
    for cid, expression, fitness, oos_sharpe, pbo, dsr_pvalue, validated_at in rows:
        records.append(
            AlphaRecord(
                alpha_id=f"cand_{cid}",
                expression=expression,
                state=AlphaState.CANDIDATE,
                fitness=float(fitness or 0.0),
                oos_sharpe=float(oos_sharpe or 0.0),
                pbo=float(pbo or 1.0),
                dsr_pvalue=float(dsr_pvalue or 1.0),
                created_at=float(validated_at or 0.0),
                updated_at=float(validated_at or 0.0),
            )
        )
    return records


def _load_source_records(asset: str, source: str) -> list[object]:
    if source == "alphas":
        return _load_registry_records(asset)
    if source == "candidates":
        return _load_candidate_records(asset)
    raise ValueError(f"unsupported source: {source}")


def _materialize_replay_registry(records: list[object], cfg, db_path: Path) -> int:
    from alpha_os.alpha.lifecycle import passes_candidate_gate
    from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState

    replay_registry = AlphaRegistry(db_path)
    lifecycle_cfg = cfg.to_lifecycle_config()
    admitted = 0

    try:
        for record in records:
            state = (
                AlphaState.ACTIVE
                if passes_candidate_gate(record, lifecycle_cfg)
                else AlphaState.REJECTED
            )
            replay_registry.register(
                AlphaRecord(
                    alpha_id=record.alpha_id,
                    expression=record.expression,
                    state=state,
                    fitness=record.fitness,
                    oos_sharpe=record.oos_sharpe,
                    oos_log_growth=getattr(record, "oos_log_growth", 0.0),
                    pbo=record.pbo,
                    dsr_pvalue=record.dsr_pvalue,
                    turnover=getattr(record, "turnover", 0.0),
                    correlation_avg=getattr(record, "correlation_avg", 0.0),
                    created_at=getattr(record, "created_at", 0.0),
                    updated_at=getattr(record, "updated_at", 0.0),
                    metadata=getattr(record, "metadata", {}),
                )
            )
            if state == AlphaState.ACTIVE:
                admitted += 1
    finally:
        replay_registry.close()

    return admitted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run backfill sweeps over lifecycle threshold combinations.",
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

    from alpha_os.config import Config
    from alpha_os.paper.simulator import run_backfill

    source_records = _load_source_records(args.asset, args.source)
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
                starting_active = _materialize_replay_registry(
                    source_records,
                    cfg,
                    replay_db,
                )
                result = run_backfill(
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
