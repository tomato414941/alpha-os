from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SplitFirstClusterLanePlan:
    plan_id: str
    cluster_id: str
    asset: str
    cluster_decision: str
    lane_opportunity: str
    lane_bias: str
    lane_side: str
    lane_status: str
    lane_priority_score: float
    conflict_role: str
    support_state: str
    lane_action: str
    resolution_action: str
    resolution_score: float
    source_lanes: str
    cluster_score: str
    evidence: str
    required_record: str
    next_step: str


def build_split_first_cluster_lane_plan(
    *,
    repeat_plan_path: Path = ROOT / "current_cost_adjusted_cluster_repeat_plan.csv",
    split_review_path: Path = ROOT / "current_symbol_lane_split_review.csv",
) -> tuple[SplitFirstClusterLanePlan, ...]:
    split_clusters = {
        row.get("asset", ""): row
        for row in _read_rows(repeat_plan_path)
        if row.get("action") == "split_lanes_before_repeat_probe"
    }
    rows = []
    for lane in _read_rows(split_review_path):
        cluster = split_clusters.get(lane.get("symbol", ""))
        if cluster is None:
            continue
        rows.append(_plan_row(cluster=cluster, lane=lane))
    return tuple(sorted(rows, key=lambda row: row.resolution_score, reverse=True))


def write_split_first_cluster_lane_plan_csv(
    rows: tuple[SplitFirstClusterLanePlan, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "plan_id",
                "cluster_id",
                "asset",
                "cluster_decision",
                "lane_opportunity",
                "lane_bias",
                "lane_side",
                "lane_status",
                "lane_priority_score",
                "conflict_role",
                "support_state",
                "lane_action",
                "resolution_action",
                "resolution_score",
                "source_lanes",
                "cluster_score",
                "evidence",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.plan_id,
                    row.cluster_id,
                    row.asset,
                    row.cluster_decision,
                    row.lane_opportunity,
                    row.lane_bias,
                    row.lane_side,
                    row.lane_status,
                    f"{row.lane_priority_score:.8f}",
                    row.conflict_role,
                    row.support_state,
                    row.lane_action,
                    row.resolution_action,
                    f"{row.resolution_score:.8f}",
                    row.source_lanes,
                    row.cluster_score,
                    row.evidence,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_split_first_cluster_lane_plan_md(
    rows: tuple[SplitFirstClusterLanePlan, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Split First Cluster Lane Plan\n\n")
        handle.write(
            "This resolves mixed cost-adjusted clusters into lane-level work before any cluster-level repeat. "
            "It prevents a strong symbol-level paper edge from hiding opposite or non-directional theses.\n\n"
        )
        handle.write(
            "| action | cluster | lane | bias | status | support | score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | ---: | --- |\n")
        for row in rows[:50]:
            handle.write(
                "| "
                f"{row.resolution_action} | "
                f"{row.cluster_id} | "
                f"{row.lane_opportunity} | "
                f"{row.lane_bias} | "
                f"{row.lane_status} | "
                f"{row.support_state} | "
                f"{row.resolution_score:.4f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _plan_row(*, cluster: dict[str, str], lane: dict[str, str]) -> SplitFirstClusterLanePlan:
    lane_bias = lane.get("lane_bias", "")
    lane_side = _lane_side(lane)
    cluster_side = _cluster_side(cluster.get("decision", ""))
    resolution_action = _resolution_action(
        lane_bias=lane_bias,
        lane_side=lane_side,
        cluster_side=cluster_side,
        conflict_role=lane.get("conflict_role", ""),
        support_state=lane.get("support_state", ""),
    )
    lane_priority = _float(lane.get("priority_score"))
    resolution_score = _resolution_score(
        cluster_score=_float(cluster.get("cluster_score")),
        lane_priority=lane_priority,
        resolution_action=resolution_action,
        support_state=lane.get("support_state", ""),
    )
    asset = cluster.get("asset", "")
    opportunity = lane.get("opportunity", "")
    return SplitFirstClusterLanePlan(
        plan_id=f"split-first-{asset.lower()}-{_slug(opportunity)}",
        cluster_id=cluster.get("cluster_id", ""),
        asset=asset,
        cluster_decision=cluster.get("decision", ""),
        lane_opportunity=opportunity,
        lane_bias=lane_bias,
        lane_side=lane_side,
        lane_status=lane.get("status", ""),
        lane_priority_score=lane_priority,
        conflict_role=lane.get("conflict_role", ""),
        support_state=lane.get("support_state", ""),
        lane_action=lane.get("lane_action", ""),
        resolution_action=resolution_action,
        resolution_score=resolution_score,
        source_lanes=cluster.get("source_lanes", ""),
        cluster_score=cluster.get("cluster_score", ""),
        evidence=lane.get("evidence", ""),
        required_record=_required_record(resolution_action),
        next_step=_next_step(action=resolution_action, asset=asset, opportunity=opportunity, lane_side=lane_side),
    )


def _resolution_action(
    *,
    lane_bias: str,
    lane_side: str,
    cluster_side: str,
    conflict_role: str,
    support_state: str,
) -> str:
    if lane_side and cluster_side and lane_side != cluster_side:
        return "isolate_opposite_lane"
    if lane_bias in {"neutral", "relative_value"}:
        return "separate_non_directional_lane"
    if conflict_role == "dominant_lane" and support_state == "paper_execution_gated":
        return "keep_for_lane_repeat"
    if conflict_role == "dominant_lane":
        return "label_before_lane_repeat"
    return "use_as_conflict_control"


def _resolution_score(
    *,
    cluster_score: float,
    lane_priority: float,
    resolution_action: str,
    support_state: str,
) -> float:
    action_bonus = {
        "keep_for_lane_repeat": 80.0,
        "label_before_lane_repeat": 45.0,
        "separate_non_directional_lane": 20.0,
        "isolate_opposite_lane": 15.0,
        "use_as_conflict_control": 5.0,
    }.get(resolution_action, 0.0)
    support_bonus = {
        "paper_execution_gated": 40.0,
        "paper_repeat_cost_adjusted_probe": 30.0,
        "unlabeled": 5.0,
    }.get(support_state, 0.0)
    return cluster_score * 0.1 + lane_priority + action_bonus + support_bonus


def _required_record(action: str) -> str:
    if action == "keep_for_lane_repeat":
        return "lane-only entry, fill assumption, funding, spread/depth, stop, adverse excursion, 15m/1h mark"
    if action == "label_before_lane_repeat":
        return "lane-only forward label, source timestamp, duplicate-source check, cost and depth context"
    if action == "isolate_opposite_lane":
        return "opposite-lane source quality, causal timestamp, failure reason for cluster direction, separate label"
    if action == "separate_non_directional_lane":
        return "relative-value or neutral mechanics, hedge leg, unwind path, costs, margin, and mark source"
    return "negative-control label and reason this lane must not be merged into the cluster action"


def _next_step(*, action: str, asset: str, opportunity: str, lane_side: str) -> str:
    if action == "keep_for_lane_repeat":
        return f"open one {asset} {lane_side} lane-repeat for {opportunity} with explicit fill and stop notes"
    if action == "label_before_lane_repeat":
        return f"label {asset}/{opportunity} alone before any repeat"
    if action == "isolate_opposite_lane":
        return f"keep {asset}/{opportunity} outside the cluster repeat and use it as an opposite-direction control"
    if action == "separate_non_directional_lane":
        return f"model {asset}/{opportunity} as a separate non-directional setup, not as a cluster repeat"
    return f"use {asset}/{opportunity} as a conflict-control lane"


def _cluster_side(decision: str) -> str:
    if decision == "paper_long":
        return "long"
    if decision == "paper_short":
        return "short"
    return ""


def _lane_side(row: dict[str, str]) -> str:
    side = row.get("side", "")
    if side in {"long", "short"}:
        return side
    bias = row.get("lane_bias", "")
    if bias in {"long", "short"}:
        return bias
    if bias.startswith("long_"):
        return "long"
    if bias.startswith("short_"):
        return "short"
    return ""


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


def _slug(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_split_first_cluster_lane_plan.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_split_first_cluster_lane_plan.md")
    args = parser.parse_args()

    rows = build_split_first_cluster_lane_plan()
    write_split_first_cluster_lane_plan_csv(rows, output_path=args.output_path)
    write_split_first_cluster_lane_plan_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.resolution_action, row.cluster_id, row.lane_opportunity, f"{row.resolution_score:.4f}")


if __name__ == "__main__":
    main()
