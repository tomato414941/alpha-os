from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://doi.org/10.1016/j.irfa.2024.103244"


@dataclass(frozen=True)
class CostSurvivalCrossSectionRow:
    cluster_id: str
    asset: str
    decision: str
    status: str
    survival_score: float
    best_net_after_cost_bps: float
    mean_net_after_cost_bps: float
    candidate_count: int
    opportunity_count: int
    source_lane_count: int
    split_lane_count: int
    repeat_win_count: int
    repeat_loss_count: int
    capacity_gated_count: int
    max_visible_depth_usage: float
    duplicate_pressure: float
    evidence: str
    missing_work: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_cost_survival_cross_section_rows(
    *,
    clusters_path: Path = ROOT / "current_cost_adjusted_alpha_clusters.csv",
    split_plan_path: Path = ROOT / "current_split_first_cluster_lane_plan.csv",
    repeat_outcome_paths: tuple[Path, ...] = (
        ROOT / "current_promoted_ticket_repeat_outcomes.csv",
        ROOT / "current_second_promoted_ticket_repeat_outcomes.csv",
        ROOT / "current_symbol_lane_promoted_repeat_outcomes.csv",
        ROOT / "current_split_first_lane_repeat_outcomes.csv",
    ),
) -> tuple[CostSurvivalCrossSectionRow, ...]:
    split_counts = _split_counts(split_plan_path)
    repeat_counts = _repeat_counts(repeat_outcome_paths)
    rows = tuple(
        _row_from_cluster(
            cluster=cluster,
            split_lane_count=split_counts.get(cluster.get("cluster_id", ""), 0),
            repeat_count=repeat_counts.get((cluster.get("asset", ""), cluster.get("decision", "")), (0, 0)),
        )
        for cluster in _read_rows(clusters_path)
    )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_cost_survival_cross_section_csv(
    rows: tuple[CostSurvivalCrossSectionRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "cluster_id",
                "asset",
                "decision",
                "status",
                "survival_score",
                "best_net_after_cost_bps",
                "mean_net_after_cost_bps",
                "candidate_count",
                "opportunity_count",
                "source_lane_count",
                "split_lane_count",
                "repeat_win_count",
                "repeat_loss_count",
                "capacity_gated_count",
                "max_visible_depth_usage",
                "duplicate_pressure",
                "evidence",
                "missing_work",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.cluster_id,
                    row.asset,
                    row.decision,
                    row.status,
                    f"{row.survival_score:.8f}",
                    f"{row.best_net_after_cost_bps:.8f}",
                    f"{row.mean_net_after_cost_bps:.8f}",
                    row.candidate_count,
                    row.opportunity_count,
                    row.source_lane_count,
                    row.split_lane_count,
                    row.repeat_win_count,
                    row.repeat_loss_count,
                    row.capacity_gated_count,
                    f"{row.max_visible_depth_usage:.8f}",
                    f"{row.duplicate_pressure:.8f}",
                    row.evidence,
                    row.missing_work,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_cost_survival_cross_section_md(
    rows: tuple[CostSurvivalCrossSectionRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cost Survival Cross Section\n\n")
        handle.write(
            "This ranks cost-adjusted alpha clusters by whether they survive repeat outcomes, "
            "source-lane separation, depth usage, and duplicate-pressure checks. "
            "It is a cross-sectional filter, not a live trade instruction.\n\n"
        )
        handle.write(
            "| cluster | status | score | best net | mean net | lanes | split lanes | wins | losses | capacity gated | dup pressure | next probe |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:30]:
            handle.write(
                "| "
                f"{row.cluster_id} | "
                f"{row.status} | "
                f"{row.survival_score:.4f} | "
                f"{row.best_net_after_cost_bps:.4f} | "
                f"{row.mean_net_after_cost_bps:.4f} | "
                f"{row.source_lane_count} | "
                f"{row.split_lane_count} | "
                f"{row.repeat_win_count} | "
                f"{row.repeat_loss_count} | "
                f"{row.capacity_gated_count} | "
                f"{row.duplicate_pressure:.4f} | "
                f"{_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High rows are not automatically good strategies. They are the rows where paper edge, repeat evidence, "
            "and cost/depth assumptions are least contradictory under the current logs.\n"
        )
    return output_path


def _row_from_cluster(
    *,
    cluster: dict[str, str],
    split_lane_count: int,
    repeat_count: tuple[int, int],
) -> CostSurvivalCrossSectionRow:
    cluster_id = cluster.get("cluster_id", "")
    asset = cluster.get("asset", "")
    decision = cluster.get("decision", "")
    candidate_count = _int(cluster.get("candidate_count"))
    opportunity_count = _int(cluster.get("opportunity_count"))
    source_lane_count = _int(cluster.get("source_lane_count"))
    repeat_supported_count = _int(cluster.get("repeat_supported_count"))
    capacity_gated_count = _int(cluster.get("capacity_gated_count"))
    best_net = _float(cluster.get("best_net_after_cost_bps"))
    mean_net = _float(cluster.get("mean_net_after_cost_bps"))
    max_usage = _float(cluster.get("max_visible_depth_usage"))
    repeat_win_count, repeat_loss_count = repeat_count
    duplicate_pressure = _duplicate_pressure(candidate_count=candidate_count, opportunity_count=opportunity_count)
    survival_score = _survival_score(
        best_net=best_net,
        mean_net=mean_net,
        source_lane_count=source_lane_count,
        split_lane_count=split_lane_count,
        repeat_supported_count=repeat_supported_count,
        repeat_win_count=repeat_win_count,
        repeat_loss_count=repeat_loss_count,
        capacity_gated_count=capacity_gated_count,
        max_usage=max_usage,
        duplicate_pressure=duplicate_pressure,
    )
    status = _status(
        survival_score=survival_score,
        repeat_win_count=repeat_win_count,
        repeat_loss_count=repeat_loss_count,
        capacity_gated_count=capacity_gated_count,
        duplicate_pressure=duplicate_pressure,
    )
    evidence = (
        f"cluster_score={cluster.get('cluster_score', '')}; "
        f"sources={cluster.get('source_lanes', '')}; "
        f"top_opportunities={cluster.get('top_opportunities', '')}; "
        f"repeat_supported={repeat_supported_count}; "
        f"repeat_wins={repeat_win_count}; "
        f"repeat_losses={repeat_loss_count}; "
        f"max_usage={max_usage:.4f}; "
        f"duplicate_pressure={duplicate_pressure:.4f}"
    )
    return CostSurvivalCrossSectionRow(
        cluster_id=cluster_id,
        asset=asset,
        decision=decision,
        status=status,
        survival_score=survival_score,
        best_net_after_cost_bps=best_net,
        mean_net_after_cost_bps=mean_net,
        candidate_count=candidate_count,
        opportunity_count=opportunity_count,
        source_lane_count=source_lane_count,
        split_lane_count=split_lane_count,
        repeat_win_count=repeat_win_count,
        repeat_loss_count=repeat_loss_count,
        capacity_gated_count=capacity_gated_count,
        max_visible_depth_usage=max_usage,
        duplicate_pressure=duplicate_pressure,
        evidence=evidence,
        missing_work=_missing_work(status),
        next_probe=_next_probe(status=status, asset=asset, decision=decision),
    )


def _survival_score(
    *,
    best_net: float,
    mean_net: float,
    source_lane_count: int,
    split_lane_count: int,
    repeat_supported_count: int,
    repeat_win_count: int,
    repeat_loss_count: int,
    capacity_gated_count: int,
    max_usage: float,
    duplicate_pressure: float,
) -> float:
    net_component = min(best_net, 300.0) * 0.45 + min(mean_net, 250.0) * 0.55
    evidence_component = (
        min(source_lane_count, 4) * 28.0
        + min(split_lane_count, 6) * 14.0
        + min(repeat_supported_count, 8) * 18.0
        + min(repeat_win_count, 12) * 22.0
    )
    penalties = (
        repeat_loss_count * 70.0
        + capacity_gated_count * 120.0
        + max(max_usage - 0.05, 0.0) * 400.0
        + duplicate_pressure * 90.0
    )
    return net_component + evidence_component - penalties


def _status(
    *,
    survival_score: float,
    repeat_win_count: int,
    repeat_loss_count: int,
    capacity_gated_count: int,
    duplicate_pressure: float,
) -> str:
    if capacity_gated_count > 0:
        return "capacity_blocks_cost_survival"
    if repeat_loss_count > 0:
        return "repeat_outcome_conflicted"
    if duplicate_pressure >= 0.55:
        return "duplicate_pressure_control_required"
    if repeat_win_count == 0:
        return "cost_adjusted_but_unrepeated"
    if survival_score >= 420.0:
        return "cost_surviving_cross_section_leader"
    return "cost_surviving_watchlist"


def _missing_work(status: str) -> str:
    if status == "capacity_blocks_cost_survival":
        return "paper edge is still blocked by visible-depth or size assumptions"
    if status == "repeat_outcome_conflicted":
        return "repeat evidence is not one-sided enough for promotion"
    if status == "duplicate_pressure_control_required":
        return "many rows may be reusing the same underlying price move or opportunity"
    if status == "cost_adjusted_but_unrepeated":
        return "cost-adjusted paper edge has not survived repeat outcomes"
    return "still needs realized fill, stop, adverse-excursion, and independent timestamp evidence"


def _next_probe(*, status: str, asset: str, decision: str) -> str:
    if status == "capacity_blocks_cost_survival":
        return f"do not scale {asset} {decision}; rerun only with smaller size or deeper book"
    if status == "duplicate_pressure_control_required":
        return f"dedupe {asset} {decision} opportunities before any new paper ticket"
    if status == "cost_adjusted_but_unrepeated":
        return f"open one repeat probe for {asset} {decision} before ranking it against leaders"
    if status == "repeat_outcome_conflicted":
        return f"split {asset} {decision} by source and label winners and losers separately"
    return f"paper-check {asset} {decision} with realized fill, stop, and adverse-excursion notes"


def _split_counts(path: Path) -> dict[str, int]:
    counts: dict[str, set[str]] = {}
    for row in _read_rows(path):
        cluster_id = row.get("cluster_id", "")
        opportunity = row.get("lane_opportunity", "")
        if cluster_id and opportunity:
            counts.setdefault(cluster_id, set()).add(opportunity)
    return {cluster_id: len(opportunities) for cluster_id, opportunities in counts.items()}


def _repeat_counts(paths: tuple[Path, ...]) -> dict[tuple[str, str], tuple[int, int]]:
    counts: dict[tuple[str, str], list[int]] = {}
    for path in paths:
        for row in _read_rows(path):
            key = (row.get("asset", ""), row.get("decision", ""))
            if not all(key):
                continue
            outcome = row.get("outcome", "")
            wins, losses = counts.setdefault(key, [0, 0])
            if outcome == "paper_mark_win":
                wins += 1
            elif outcome in {"paper_mark_loss", "paper_mark_flat"}:
                losses += 1
            counts[key] = [wins, losses]
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _duplicate_pressure(*, candidate_count: int, opportunity_count: int) -> float:
    if candidate_count <= 0:
        return 0.0
    return max(1.0 - (opportunity_count / candidate_count), 0.0)


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


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cost_survival_cross_section.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cost_survival_cross_section.md")
    args = parser.parse_args()

    rows = build_cost_survival_cross_section_rows()
    write_cost_survival_cross_section_csv(rows, output_path=args.output_path)
    write_cost_survival_cross_section_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.cluster_id, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
