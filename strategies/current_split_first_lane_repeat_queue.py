from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP = 20


@dataclass(frozen=True)
class SplitFirstLaneRepeatQueueRow:
    priority: float
    queue_id: str
    action: str
    cluster_id: str
    asset: str
    cluster_decision: str
    lane_opportunity: str
    lane_bias: str
    lane_side: str
    lane_status: str
    resolution_action: str
    resolution_score: str
    required_record: str
    next_step: str


def build_split_first_lane_repeat_queue(
    *,
    lane_plan_path: Path = ROOT / "current_split_first_cluster_lane_plan.csv",
    top: int = DEFAULT_TOP,
) -> tuple[SplitFirstLaneRepeatQueueRow, ...]:
    rows = tuple(
        _queue_row(row)
        for row in _read_rows(lane_plan_path)
        if row.get("resolution_action") in {"keep_for_lane_repeat", "label_before_lane_repeat"}
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True)[:top])


def write_split_first_lane_repeat_queue_csv(
    rows: tuple[SplitFirstLaneRepeatQueueRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "priority",
                "queue_id",
                "action",
                "cluster_id",
                "asset",
                "cluster_decision",
                "lane_opportunity",
                "lane_bias",
                "lane_side",
                "lane_status",
                "resolution_action",
                "resolution_score",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    f"{row.priority:.8f}",
                    row.queue_id,
                    row.action,
                    row.cluster_id,
                    row.asset,
                    row.cluster_decision,
                    row.lane_opportunity,
                    row.lane_bias,
                    row.lane_side,
                    row.lane_status,
                    row.resolution_action,
                    row.resolution_score,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_split_first_lane_repeat_queue_md(
    rows: tuple[SplitFirstLaneRepeatQueueRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Split First Lane Repeat Queue\n\n")
        handle.write(
            "This queues lane-level paper work after mixed clusters are split. "
            "It is not a live trade instruction and does not collapse lanes back into a symbol-level action.\n\n"
        )
        handle.write("| priority | action | cluster | lane | side | status | next step |\n")
        handle.write("| ---: | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.priority:.4f} | "
                f"{row.action} | "
                f"{row.cluster_id} | "
                f"{row.lane_opportunity} | "
                f"{row.lane_side} | "
                f"{row.lane_status} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _queue_row(row: dict[str, str]) -> SplitFirstLaneRepeatQueueRow:
    resolution_action = row.get("resolution_action", "")
    action = "open_lane_repeat_probe" if resolution_action == "keep_for_lane_repeat" else "open_lane_label"
    priority = _float(row.get("resolution_score"))
    if action == "open_lane_repeat_probe":
        priority += 25.0
    return SplitFirstLaneRepeatQueueRow(
        priority=priority,
        queue_id=f"split-first-queue-{row.get('plan_id', '')}",
        action=action,
        cluster_id=row.get("cluster_id", ""),
        asset=row.get("asset", ""),
        cluster_decision=row.get("cluster_decision", ""),
        lane_opportunity=row.get("lane_opportunity", ""),
        lane_bias=row.get("lane_bias", ""),
        lane_side=row.get("lane_side", ""),
        lane_status=row.get("lane_status", ""),
        resolution_action=resolution_action,
        resolution_score=row.get("resolution_score", ""),
        required_record=row.get("required_record", ""),
        next_step=row.get("next_step", ""),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_split_first_lane_repeat_queue.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_split_first_lane_repeat_queue.md")
    args = parser.parse_args()

    rows = build_split_first_lane_repeat_queue(top=args.top)
    write_split_first_lane_repeat_queue_csv(rows, output_path=args.output_path)
    write_split_first_lane_repeat_queue_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.action, row.cluster_id, row.lane_opportunity, f"{row.priority:.4f}")


if __name__ == "__main__":
    main()
