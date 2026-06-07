from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SymbolClusterLabelQueueRow:
    symbol: str
    queue_action: str
    priority_score: float
    cluster_status: str
    dominant_bias: str
    source_count: int
    candidate_count: int
    long_count: int
    short_count: int
    relative_value_count: int
    yield_count: int
    risk_or_avoid_count: int
    top_opportunities: str
    reason: str
    next_step: str


def build_symbol_cluster_label_queue(
    *,
    conflict_path: Path = ROOT / "current_symbol_cluster_conflicts.csv",
) -> tuple[SymbolClusterLabelQueueRow, ...]:
    rows = tuple(_build_queue_row(row) for row in _read_rows(conflict_path))
    return tuple(sorted(rows, key=lambda row: row.priority_score, reverse=True))


def write_symbol_cluster_label_queue_csv(
    rows: tuple[SymbolClusterLabelQueueRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "queue_action",
                "priority_score",
                "cluster_status",
                "dominant_bias",
                "source_count",
                "candidate_count",
                "long_count",
                "short_count",
                "relative_value_count",
                "yield_count",
                "risk_or_avoid_count",
                "top_opportunities",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.queue_action,
                    f"{row.priority_score:.8f}",
                    row.cluster_status,
                    row.dominant_bias,
                    row.source_count,
                    row.candidate_count,
                    row.long_count,
                    row.short_count,
                    row.relative_value_count,
                    row.yield_count,
                    row.risk_or_avoid_count,
                    row.top_opportunities,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_symbol_cluster_label_queue_md(
    rows: tuple[SymbolClusterLabelQueueRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Symbol Cluster Label Queue\n\n")
        handle.write(
            "This turns symbol-level clusters into concrete next labeling work. "
            "It is a workflow queue for alpha discovery, not a trade list.\n\n"
        )
        handle.write(
            "| symbol | action | priority | status | bias | sources | candidates | counts | top opportunities | next step |\n"
        )
        handle.write("| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            counts = (
                f"L={row.long_count}, S={row.short_count}, RV={row.relative_value_count}, "
                f"Y={row.yield_count}, R={row.risk_or_avoid_count}"
            )
            handle.write(
                f"| {row.symbol} | "
                f"{row.queue_action} | "
                f"{row.priority_score:.4f} | "
                f"{row.cluster_status} | "
                f"{row.dominant_bias} | "
                f"{row.source_count} | "
                f"{row.candidate_count} | "
                f"{counts} | "
                f"{_escape(row.top_opportunities)} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`split_lane_forward_label` rows are usually the most important because "
            "multiple sources point at the same symbol but disagree about direction or structure. "
            "`confirmed_direction_forward_label` rows are cleaner candidates for direct paper labels.\n"
        )
    return output_path


def _build_queue_row(row: dict[str, str]) -> SymbolClusterLabelQueueRow:
    cluster_status = row.get("status", "")
    queue_action = _queue_action(cluster_status)
    priority_score = _priority_score(row, queue_action=queue_action)
    return SymbolClusterLabelQueueRow(
        symbol=row.get("symbol", ""),
        queue_action=queue_action,
        priority_score=priority_score,
        cluster_status=cluster_status,
        dominant_bias=row.get("dominant_bias", ""),
        source_count=_int(row.get("source_count")),
        candidate_count=_int(row.get("candidate_count")),
        long_count=_int(row.get("long_count")),
        short_count=_int(row.get("short_count")),
        relative_value_count=_int(row.get("relative_value_count")),
        yield_count=_int(row.get("yield_count")),
        risk_or_avoid_count=_int(row.get("risk_or_avoid_count")),
        top_opportunities=row.get("top_opportunities", ""),
        reason=_reason(row, queue_action=queue_action),
        next_step=_next_step(row, queue_action=queue_action),
    )


def _queue_action(cluster_status: str) -> str:
    if cluster_status in {"mixed_direction_conflict", "mixed_structure_conflict"}:
        return "split_lane_forward_label"
    if cluster_status.startswith("confirmed_"):
        return "confirmed_direction_forward_label"
    if cluster_status in {"relative_value_cluster", "yield_cluster", "risk_resolution_cluster"}:
        return "mechanics_and_unwind_check"
    if cluster_status == "single_candidate_watch":
        return "repeat_single_candidate_observation"
    return "collect_cluster_observations"


def _priority_score(row: dict[str, str], *, queue_action: str) -> float:
    action_bonus = {
        "split_lane_forward_label": 12.0,
        "confirmed_direction_forward_label": 8.0,
        "mechanics_and_unwind_check": 5.0,
        "collect_cluster_observations": 3.0,
        "repeat_single_candidate_observation": 2.0,
    }.get(queue_action, 0.0)
    return (
        _float(row.get("cluster_score"))
        + action_bonus
        + min(_int(row.get("source_count")) * 1.5, 9.0)
        + min(_int(row.get("candidate_count")) * 0.5, 5.0)
    )


def _reason(row: dict[str, str], *, queue_action: str) -> str:
    if queue_action == "split_lane_forward_label":
        return "symbol has multiple active ideas that cannot be collapsed into one directional trade"
    if queue_action == "confirmed_direction_forward_label":
        return f"symbol cluster points mostly {row.get('dominant_bias', '')}"
    if queue_action == "mechanics_and_unwind_check":
        return "edge depends on structure, venue access, carry mechanics, or unwind path"
    if queue_action == "repeat_single_candidate_observation":
        return "candidate is promising but still one-lane or one-candidate"
    return "cluster needs more observations before promotion"


def _next_step(row: dict[str, str], *, queue_action: str) -> str:
    symbol = row.get("symbol", "")
    if queue_action == "split_lane_forward_label":
        return f"label {symbol} per lane and compare direction, costs, depth, and failure regime separately"
    if queue_action == "confirmed_direction_forward_label":
        return f"label {symbol} {row.get('dominant_bias', '')} setup over short and medium horizons with costs"
    if queue_action == "mechanics_and_unwind_check":
        return f"validate {symbol} mechanics, access, liquidity, fees, and unwind path before any return label"
    if queue_action == "repeat_single_candidate_observation":
        return f"repeat {symbol} observation and require a second independent signal or fresh positive label"
    return f"collect more {symbol} observations and then rerun cluster conflict review"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _int(value: str | None) -> int:
    return int(value) if value else 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conflict-path",
        type=Path,
        default=ROOT / "current_symbol_cluster_conflicts.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_symbol_cluster_label_queue.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_symbol_cluster_label_queue.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_symbol_cluster_label_queue(conflict_path=args.conflict_path)
    write_symbol_cluster_label_queue_csv(rows, output_path=args.output_path)
    write_symbol_cluster_label_queue_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.queue_action,
            row.symbol,
            f"priority={row.priority_score:.4f}",
            f"status={row.cluster_status}",
        )


if __name__ == "__main__":
    main()
