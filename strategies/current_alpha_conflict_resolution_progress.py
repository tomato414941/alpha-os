from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaConflictResolutionProgressRow:
    work_id: str
    work_kind: str
    cluster_id: str
    asset: str
    decision: str
    status: str
    progress_score: float
    cluster_action: str
    duplicate_pressure: float
    lane_plan_count: int
    queued_lane_count: int
    lane_repeat_count: int
    lane_win_count: int
    lane_flat_count: int
    lane_loss_count: int
    blocker: str
    evidence: str
    next_step: str


def build_alpha_conflict_resolution_progress(
    *,
    worklist_path: Path = ROOT / "current_alpha_promotion_worklist.csv",
    cost_survival_path: Path = ROOT / "current_cost_survival_cross_section.csv",
    cluster_repeat_plan_path: Path = ROOT / "current_cost_adjusted_cluster_repeat_plan.csv",
    lane_plan_path: Path = ROOT / "current_split_first_cluster_lane_plan.csv",
    lane_queue_path: Path = ROOT / "current_split_first_lane_repeat_queue.csv",
    lane_outcomes_path: Path = ROOT / "current_split_first_lane_repeat_outcomes.csv",
) -> tuple[AlphaConflictResolutionProgressRow, ...]:
    cost_rows = {row.get("cluster_id", ""): row for row in _read_rows(cost_survival_path)}
    cluster_actions = {row.get("cluster_id", ""): row for row in _read_rows(cluster_repeat_plan_path)}
    lane_plans = _group_by_cluster(_read_rows(lane_plan_path), key="cluster_id")
    lane_queue = _group_by_cluster(_read_rows(lane_queue_path), key="cluster_id")
    lane_outcomes = _group_by_cluster(_read_rows(lane_outcomes_path), key="cluster_id")
    rows = []
    for work in _read_rows(worklist_path):
        if work.get("work_kind") not in {"dedupe_cluster", "split_conflicting_sources"}:
            continue
        cluster_id = _cluster_id_from_frontier(work.get("source_frontier_id", ""))
        rows.append(
            _progress_row(
                work=work,
                cost=cost_rows.get(cluster_id, {}),
                cluster_action=cluster_actions.get(cluster_id, {}),
                lane_plans=lane_plans.get(cluster_id, ()),
                lane_queue=lane_queue.get(cluster_id, ()),
                lane_outcomes=lane_outcomes.get(cluster_id, ()),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.progress_score, reverse=True))


def write_alpha_conflict_resolution_progress_csv(
    rows: tuple[AlphaConflictResolutionProgressRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "work_id",
                "work_kind",
                "cluster_id",
                "asset",
                "decision",
                "status",
                "progress_score",
                "cluster_action",
                "duplicate_pressure",
                "lane_plan_count",
                "queued_lane_count",
                "lane_repeat_count",
                "lane_win_count",
                "lane_flat_count",
                "lane_loss_count",
                "blocker",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.work_id,
                    row.work_kind,
                    row.cluster_id,
                    row.asset,
                    row.decision,
                    row.status,
                    f"{row.progress_score:.8f}",
                    row.cluster_action,
                    f"{row.duplicate_pressure:.8f}",
                    row.lane_plan_count,
                    row.queued_lane_count,
                    row.lane_repeat_count,
                    row.lane_win_count,
                    row.lane_flat_count,
                    row.lane_loss_count,
                    row.blocker,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_conflict_resolution_progress_md(
    rows: tuple[AlphaConflictResolutionProgressRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Conflict Resolution Progress\n\n")
        handle.write(
            "This checks whether promotion-worklist dedupe and source-split items have actually been resolved. "
            "It prevents duplicate or conflicting clusters from being promoted as one trade.\n\n"
        )
        handle.write(
            "| work | asset | status | score | action | dup | plans | queued | repeats | blocker | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.work_id} | "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.progress_score:.4f} | "
                f"{row.cluster_action} | "
                f"{row.duplicate_pressure:.4f} | "
                f"{row.lane_plan_count} | "
                f"{row.queued_lane_count} | "
                f"{row.lane_repeat_count} | "
                f"{_escape(row.blocker)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _progress_row(
    *,
    work: dict[str, str],
    cost: dict[str, str],
    cluster_action: dict[str, str],
    lane_plans: tuple[dict[str, str], ...],
    lane_queue: tuple[dict[str, str], ...],
    lane_outcomes: tuple[dict[str, str], ...],
) -> AlphaConflictResolutionProgressRow:
    work_kind = work.get("work_kind", "")
    cluster_id = _cluster_id_from_frontier(work.get("source_frontier_id", ""))
    action = cluster_action.get("action", "")
    duplicate_pressure = _float(cost.get("duplicate_pressure"))
    outcome_counts = _outcome_counts(lane_outcomes)
    status = _status(
        work_kind=work_kind,
        cluster_action=action,
        duplicate_pressure=duplicate_pressure,
        lane_plan_count=len(lane_plans),
        queued_lane_count=len(lane_queue),
        lane_repeat_count=sum(outcome_counts.values()),
        lane_flat_count=outcome_counts.get("paper_mark_flat", 0),
        lane_loss_count=outcome_counts.get("paper_mark_loss", 0),
    )
    return AlphaConflictResolutionProgressRow(
        work_id=work.get("work_id", ""),
        work_kind=work_kind,
        cluster_id=cluster_id,
        asset=work.get("asset", ""),
        decision=work.get("action", ""),
        status=status,
        progress_score=_score(status=status, lane_plan_count=len(lane_plans), queued_lane_count=len(lane_queue)),
        cluster_action=action,
        duplicate_pressure=duplicate_pressure,
        lane_plan_count=len(lane_plans),
        queued_lane_count=len(lane_queue),
        lane_repeat_count=sum(outcome_counts.values()),
        lane_win_count=outcome_counts.get("paper_mark_win", 0),
        lane_flat_count=outcome_counts.get("paper_mark_flat", 0),
        lane_loss_count=outcome_counts.get("paper_mark_loss", 0),
        blocker=_blocker(status),
        evidence=(
            f"cost_status={cost.get('status', '')}; "
            f"cluster_action={action}; "
            f"duplicate_pressure={duplicate_pressure:.4f}; "
            f"plans={len(lane_plans)}; "
            f"queued={len(lane_queue)}; "
            f"outcomes={sum(outcome_counts.values())}; "
            f"wins={outcome_counts.get('paper_mark_win', 0)}; "
            f"flats={outcome_counts.get('paper_mark_flat', 0)}; "
            f"losses={outcome_counts.get('paper_mark_loss', 0)}"
        ),
        next_step=_next_step(status=status, asset=work.get("asset", ""), decision=work.get("action", "")),
    )


def _status(
    *,
    work_kind: str,
    cluster_action: str,
    duplicate_pressure: float,
    lane_plan_count: int,
    queued_lane_count: int,
    lane_repeat_count: int,
    lane_flat_count: int,
    lane_loss_count: int,
) -> str:
    if work_kind == "dedupe_cluster" and cluster_action == "open_consolidated_repeat_probe":
        return "dedupe_conflicts_with_consolidated_repeat"
    if work_kind == "dedupe_cluster" and duplicate_pressure >= 0.50:
        return "dedupe_not_resolved"
    if lane_repeat_count > 0 and lane_loss_count > 0:
        return "split_repeat_has_loss"
    if lane_repeat_count > 0 and lane_flat_count > 0:
        return "split_repeat_started_flat"
    if queued_lane_count > 0:
        return "split_queue_ready"
    if lane_plan_count > 0:
        return "split_plan_ready_not_queued"
    return "split_not_started"


def _score(*, status: str, lane_plan_count: int, queued_lane_count: int) -> float:
    base = {
        "split_repeat_started_flat": 520.0,
        "split_repeat_has_loss": 480.0,
        "split_queue_ready": 430.0,
        "split_plan_ready_not_queued": 330.0,
        "dedupe_conflicts_with_consolidated_repeat": 260.0,
        "dedupe_not_resolved": 220.0,
        "split_not_started": 100.0,
    }.get(status, 100.0)
    return base + min(lane_plan_count * 5.0, 100.0) + min(queued_lane_count * 8.0, 120.0)


def _blocker(status: str) -> str:
    if status == "dedupe_conflicts_with_consolidated_repeat":
        return "dedupe work says do not reuse the same move, but cluster plan still opens one consolidated repeat"
    if status == "dedupe_not_resolved":
        return "duplicate pressure remains high"
    if status == "split_repeat_started_flat":
        return "split repeat has started, but latest lane mark is flat"
    if status == "split_repeat_has_loss":
        return "split repeat has at least one losing lane"
    if status == "split_queue_ready":
        return "lane split is queued but not yet proven by repeat outcomes"
    if status == "split_plan_ready_not_queued":
        return "lane split plan exists but is not in the active top queue"
    return "cluster conflict has no active split plan yet"


def _next_step(*, status: str, asset: str, decision: str) -> str:
    if status == "dedupe_conflicts_with_consolidated_repeat":
        return f"rewrite {asset} {decision} repeat work as unique-opportunity dedupe before any consolidated repeat"
    if status == "dedupe_not_resolved":
        return f"choose one independent {asset} {decision} opportunity and suppress duplicates"
    if status == "split_repeat_started_flat":
        return f"wait for a non-flat lane mark or reject the flat {asset} split lane"
    if status == "split_repeat_has_loss":
        return f"mark the losing {asset} lane as a failure regime before any cluster action"
    if status == "split_queue_ready":
        return f"run or refresh queued lane labels for {asset} before cluster promotion"
    if status == "split_plan_ready_not_queued":
        return f"promote the highest-value {asset} split lane into the active queue or lower its priority"
    return f"build lane split plan for {asset} {decision}"


def _outcome_counts(rows: tuple[dict[str, str], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        outcome = row.get("outcome", "")
        if outcome:
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _group_by_cluster(rows: tuple[dict[str, str], ...], *, key: str) -> dict[str, tuple[dict[str, str], ...]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        cluster_id = row.get(key, "")
        if cluster_id:
            grouped.setdefault(cluster_id, []).append(row)
    return {cluster_id: tuple(value) for cluster_id, value in grouped.items()}


def _cluster_id_from_frontier(frontier_id: str) -> str:
    prefix = "cost_cluster:"
    if frontier_id.startswith(prefix):
        return frontier_id[len(prefix) :]
    return frontier_id


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_conflict_resolution_progress.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_conflict_resolution_progress.md")
    args = parser.parse_args()

    rows = build_alpha_conflict_resolution_progress()
    write_alpha_conflict_resolution_progress_csv(rows, output_path=args.output_path)
    write_alpha_conflict_resolution_progress_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.status, row.asset, f"{row.progress_score:.4f}", row.next_step)


if __name__ == "__main__":
    main()
