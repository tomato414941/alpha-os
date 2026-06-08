from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SplitFirstLaneLabelProgressRow:
    priority: float
    queue_id: str
    cluster_id: str
    asset: str
    lane_opportunity: str
    lane_status: str
    progress_status: str
    required_record: str
    evidence: str
    next_step: str


def build_split_first_lane_label_progress(
    *,
    queue_path: Path = ROOT / "current_split_first_lane_repeat_queue.csv",
    plan_path: Path = ROOT / "current_split_first_cluster_lane_plan.csv",
) -> tuple[SplitFirstLaneLabelProgressRow, ...]:
    plan_rows = {
        row.get("plan_id", ""): row
        for row in _read_rows(plan_path)
    }
    rows = []
    for row in _read_rows(queue_path):
        if row.get("action") != "open_lane_label":
            continue
        plan = plan_rows.get(row.get("queue_id", "").removeprefix("split-first-queue-"), {})
        rows.append(_progress_row(row, plan=plan))
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_split_first_lane_label_progress_csv(
    rows: tuple[SplitFirstLaneLabelProgressRow, ...],
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
                "cluster_id",
                "asset",
                "lane_opportunity",
                "lane_status",
                "progress_status",
                "required_record",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    f"{row.priority:.8f}",
                    row.queue_id,
                    row.cluster_id,
                    row.asset,
                    row.lane_opportunity,
                    row.lane_status,
                    row.progress_status,
                    row.required_record,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_split_first_lane_label_progress_md(
    rows: tuple[SplitFirstLaneLabelProgressRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Split First Lane Label Progress\n\n")
        handle.write(
            "This tracks split-first lane queue rows that need forward labels before any repeat probe. "
            "These rows are not trade instructions and are intentionally separate from repeat tickets.\n\n"
        )
        handle.write("| priority | asset | cluster | lane | lane status | progress | next step |\n")
        handle.write("| ---: | --- | --- | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.priority:.4f} | "
                f"{row.asset} | "
                f"{row.cluster_id} | "
                f"{row.lane_opportunity} | "
                f"{row.lane_status} | "
                f"{row.progress_status} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _progress_row(
    row: dict[str, str],
    *,
    plan: dict[str, str],
) -> SplitFirstLaneLabelProgressRow:
    lane_status = row.get("lane_status", "")
    return SplitFirstLaneLabelProgressRow(
        priority=_float(row.get("priority")),
        queue_id=row.get("queue_id", ""),
        cluster_id=row.get("cluster_id", ""),
        asset=row.get("asset", ""),
        lane_opportunity=row.get("lane_opportunity", ""),
        lane_status=lane_status,
        progress_status=_progress_status(lane_status),
        required_record=row.get("required_record", ""),
        evidence=plan.get("evidence", ""),
        next_step=_next_step(row, lane_status=lane_status),
    )


def _progress_status(lane_status: str) -> str:
    if lane_status == "pending_label":
        return "forward_label_pending"
    if lane_status == "unlabeled":
        return "forward_label_not_started"
    if "candidate" in lane_status or lane_status.endswith("_watch"):
        return "source_context_needs_forward_label"
    if "supported" in lane_status:
        return "support_exists_but_lane_still_split"
    return "lane_label_needs_review"


def _next_step(row: dict[str, str], *, lane_status: str) -> str:
    if lane_status == "pending_label":
        return (
            f"refresh the pending forward mark for {row.get('asset', '')}/"
            f"{row.get('lane_opportunity', '')} and keep it separate from cluster repeats"
        )
    return (
        f"record a lane-only forward label for {row.get('asset', '')}/"
        f"{row.get('lane_opportunity', '')} with source timestamp, duplicate-source check, "
        "cost/depth context, and failure regime"
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
    parser.add_argument("--queue-path", type=Path, default=ROOT / "current_split_first_lane_repeat_queue.csv")
    parser.add_argument("--plan-path", type=Path, default=ROOT / "current_split_first_cluster_lane_plan.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_split_first_lane_label_progress.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_split_first_lane_label_progress.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_split_first_lane_label_progress(queue_path=args.queue_path, plan_path=args.plan_path)
    write_split_first_lane_label_progress_csv(rows, output_path=args.output_path)
    write_split_first_lane_label_progress_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.progress_status, row.asset, row.lane_opportunity, f"{row.priority:.4f}")


if __name__ == "__main__":
    main()
