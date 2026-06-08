from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXECUTION_COSTS_BPS = {
    "taker_round_trip": 8.0,
    "low_fee_round_trip": 2.0,
    "maker_or_internalized": 0.5,
    "zero_cost_diagnostic": 0.0,
}


@dataclass(frozen=True)
class BookDepthExecutionCostSweepRow:
    feature: str
    bucket: str
    action: str
    train_count: str
    test_count: str
    train_mean_bps: str
    test_gross_bps: str
    test_hit_rate: str
    execution_mode: str
    cost_bps: float
    test_net_bps: float
    viability_score: float
    viability_status: str
    next_step: str


def build_book_depth_execution_cost_sweep(
    *,
    walk_forward_path: Path = ROOT / "book_depth_walk_forward_check.csv",
) -> tuple[BookDepthExecutionCostSweepRow, ...]:
    rows = [
        _build_row(row=row, execution_mode=execution_mode, cost_bps=cost_bps)
        for row in _read_rows(walk_forward_path)
        if _float(row.get("test_gross_bps")) > 0.0
        for execution_mode, cost_bps in EXECUTION_COSTS_BPS.items()
    ]
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_book_depth_execution_cost_sweep_csv(
    rows: tuple[BookDepthExecutionCostSweepRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "feature",
                "bucket",
                "action",
                "train_count",
                "test_count",
                "train_mean_bps",
                "test_gross_bps",
                "test_hit_rate",
                "execution_mode",
                "cost_bps",
                "test_net_bps",
                "viability_score",
                "viability_status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.feature,
                    row.bucket,
                    row.action,
                    row.train_count,
                    row.test_count,
                    row.train_mean_bps,
                    row.test_gross_bps,
                    row.test_hit_rate,
                    row.execution_mode,
                    f"{row.cost_bps:.8f}",
                    f"{row.test_net_bps:.8f}",
                    f"{row.viability_score:.8f}",
                    row.viability_status,
                    row.next_step,
                )
            )
    return output_path


def write_book_depth_execution_cost_sweep_md(
    rows: tuple[BookDepthExecutionCostSweepRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Book Depth Execution Cost Sweep\n\n")
        handle.write(
            "This re-prices book-depth walk-forward rows across execution-cost assumptions. "
            "It separates signals that are dead for taker execution from signals that might only survive "
            "with low-fee, maker, or internalized execution. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| feature | bucket | action | mode | gross bps | cost bps | net bps | hit | status | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.feature} | "
                f"{row.bucket} | "
                f"{row.action} | "
                f"{row.execution_mode} | "
                f"{row.test_gross_bps} | "
                f"{row.cost_bps:.2f} | "
                f"{row.test_net_bps:.4f} | "
                f"{row.test_hit_rate} | "
                f"{row.viability_status} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary_text(rows))
    return output_path


def _build_row(*, row: dict[str, str], execution_mode: str, cost_bps: float) -> BookDepthExecutionCostSweepRow:
    gross_bps = _float(row.get("test_gross_bps"))
    net_bps = gross_bps - cost_bps
    status = _viability_status(execution_mode=execution_mode, net_bps=net_bps, hit_rate=_float(row.get("test_hit_rate")))
    return BookDepthExecutionCostSweepRow(
        feature=row.get("feature", ""),
        bucket=row.get("bucket", ""),
        action=row.get("action", ""),
        train_count=row.get("train_count", ""),
        test_count=row.get("test_count", ""),
        train_mean_bps=row.get("train_mean_bps", ""),
        test_gross_bps=row.get("test_gross_bps", ""),
        test_hit_rate=row.get("test_hit_rate", ""),
        execution_mode=execution_mode,
        cost_bps=cost_bps,
        test_net_bps=net_bps,
        viability_score=_viability_score(status=status, net_bps=net_bps),
        viability_status=status,
        next_step=_next_step(status),
    )


def _viability_status(*, execution_mode: str, net_bps: float, hit_rate: float) -> str:
    if net_bps <= 0.0:
        return "not_viable_after_cost"
    if execution_mode == "zero_cost_diagnostic":
        return "zero_cost_only_signal"
    if execution_mode == "maker_or_internalized":
        if hit_rate < 0.5:
            return "maker_only_low_hit_rate"
        return "maker_or_internalized_candidate"
    if execution_mode == "low_fee_round_trip":
        return "low_fee_candidate"
    return "taker_viable_candidate"


def _next_step(status: str) -> str:
    if status == "maker_or_internalized_candidate":
        return "test maker fill probability, queue position, adverse selection, and cancellation rules"
    if status == "maker_only_low_hit_rate":
        return "do not trade directionally; inspect whether payoff tail or queue selection explains the weak hit rate"
    if status == "low_fee_candidate":
        return "validate under the actual fee tier and add spread, latency, and fill-risk controls"
    if status == "zero_cost_only_signal":
        return "keep as representation-learning feature only; it is not execution-ready"
    if status == "taker_viable_candidate":
        return "open a tiny paper label only after confirming spread and venue depth"
    return "reject for current execution mode; only revisit if execution cost or horizon changes"


def _viability_score(*, status: str, net_bps: float) -> float:
    status_rank = {
        "taker_viable_candidate": 500.0,
        "low_fee_candidate": 400.0,
        "maker_or_internalized_candidate": 300.0,
        "maker_only_low_hit_rate": 200.0,
        "zero_cost_only_signal": 100.0,
        "not_viable_after_cost": 0.0,
    }.get(status, 0.0)
    return status_rank + net_bps


def _summary_text(rows: tuple[BookDepthExecutionCostSweepRow, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.viability_status] = counts.get(row.viability_status, 0) + 1
    lines = [f"- {status}: {count}" for status, count in sorted(counts.items())]
    best_viability = max(rows, key=lambda row: row.viability_score, default=None)
    best_raw_net = max(rows, key=lambda row: row.test_net_bps, default=None)
    if best_viability:
        lines.append(
            "- best viability: "
            f"{best_viability.feature}/{best_viability.bucket}/{best_viability.action}/"
            f"{best_viability.execution_mode} gross={best_viability.test_gross_bps}bps "
            f"net={best_viability.test_net_bps:.8f}bps status={best_viability.viability_status}"
        )
    if best_raw_net:
        lines.append(
            "- best raw net diagnostic: "
            f"{best_raw_net.feature}/{best_raw_net.bucket}/{best_raw_net.action}/{best_raw_net.execution_mode} "
            f"gross={best_raw_net.test_gross_bps}bps net={best_raw_net.test_net_bps:.8f}bps"
        )
    if not lines:
        lines.append("- no positive-gross book-depth rows to sweep")
    return "\n".join(lines) + "\n"


def _sort_key(row: BookDepthExecutionCostSweepRow) -> tuple[float, float]:
    return row.viability_score, row.test_net_bps


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "book_depth_execution_cost_sweep.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "book_depth_execution_cost_sweep.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_book_depth_execution_cost_sweep()
    write_book_depth_execution_cost_sweep_csv(rows, output_path=args.output_path)
    write_book_depth_execution_cost_sweep_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.feature, row.bucket, row.execution_mode, row.viability_status, f"{row.test_net_bps:.4f}")


if __name__ == "__main__":
    main()
