from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CostAdjustedAlphaCluster:
    cluster_id: str
    asset: str
    decision: str
    candidate_count: int
    source_lane_count: int
    source_lanes: str
    opportunity_count: int
    top_opportunities: str
    best_net_after_cost_bps: float
    mean_net_after_cost_bps: float
    best_priority_score: float
    cluster_score: float
    repeat_supported_count: int
    capacity_gated_count: int
    max_visible_depth_usage: float
    status: str
    evidence: str
    missing_work: str
    next_step: str


def build_cost_adjusted_alpha_clusters(
    *,
    candidates_path: Path = ROOT / "current_cost_adjusted_alpha_candidates.csv",
) -> tuple[CostAdjustedAlphaCluster, ...]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_rows(candidates_path):
        asset = row.get("asset", "")
        decision = row.get("decision", "")
        if not asset or not decision:
            continue
        groups.setdefault((asset, decision), []).append(row)
    rows = tuple(_cluster_from_rows(asset=asset, decision=decision, rows=group) for (asset, decision), group in groups.items())
    return tuple(sorted(rows, key=lambda row: row.cluster_score, reverse=True))


def write_cost_adjusted_alpha_clusters_csv(
    rows: tuple[CostAdjustedAlphaCluster, ...],
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
                "candidate_count",
                "source_lane_count",
                "source_lanes",
                "opportunity_count",
                "top_opportunities",
                "best_net_after_cost_bps",
                "mean_net_after_cost_bps",
                "best_priority_score",
                "cluster_score",
                "repeat_supported_count",
                "capacity_gated_count",
                "max_visible_depth_usage",
                "status",
                "evidence",
                "missing_work",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.cluster_id,
                    row.asset,
                    row.decision,
                    row.candidate_count,
                    row.source_lane_count,
                    row.source_lanes,
                    row.opportunity_count,
                    row.top_opportunities,
                    f"{row.best_net_after_cost_bps:.8f}",
                    f"{row.mean_net_after_cost_bps:.8f}",
                    f"{row.best_priority_score:.8f}",
                    f"{row.cluster_score:.8f}",
                    row.repeat_supported_count,
                    row.capacity_gated_count,
                    f"{row.max_visible_depth_usage:.8f}",
                    row.status,
                    row.evidence,
                    row.missing_work,
                    row.next_step,
                )
            )
    return output_path


def write_cost_adjusted_alpha_clusters_md(
    rows: tuple[CostAdjustedAlphaCluster, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cost Adjusted Alpha Clusters\n\n")
        handle.write(
            "This groups cost-adjusted candidates by asset and direction so repeated paper ideas do not hide "
            "which symbols have broad support.\n\n"
        )
        handle.write(
            "| cluster | candidates | lanes | opportunities | best net | mean net | repeat support | capacity gated | score | status | next step |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.cluster_id} | "
                f"{row.candidate_count} | "
                f"{row.source_lane_count} | "
                f"{row.opportunity_count} | "
                f"{row.best_net_after_cost_bps:.4f} | "
                f"{row.mean_net_after_cost_bps:.4f} | "
                f"{row.repeat_supported_count} | "
                f"{row.capacity_gated_count} | "
                f"{row.cluster_score:.4f} | "
                f"{row.status} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _cluster_from_rows(*, asset: str, decision: str, rows: list[dict[str, str]]) -> CostAdjustedAlphaCluster:
    source_lanes = tuple(sorted({row.get("source_lane", "") for row in rows if row.get("source_lane", "")}))
    opportunities = tuple(sorted({row.get("opportunity", "") for row in rows if row.get("opportunity", "")}))
    nets = tuple(_float(row.get("estimated_net_after_cost_bps")) for row in rows)
    priorities = tuple(_float(row.get("priority_score")) for row in rows)
    repeat_supported_count = sum(1 for row in rows if "repeat" in row.get("status", ""))
    capacity_gated_count = sum(1 for row in rows if row.get("status") == "capacity_gated_alpha_candidate")
    max_visible_depth_usage = max((_float(row.get("visible_depth_usage")) for row in rows), default=0.0)
    best_net = max(nets, default=0.0)
    mean_net = sum(nets) / len(nets) if nets else 0.0
    best_priority = max(priorities, default=0.0)
    cluster_score = _cluster_score(
        best_net=best_net,
        mean_net=mean_net,
        source_lane_count=len(source_lanes),
        opportunity_count=len(opportunities),
        repeat_supported_count=repeat_supported_count,
        capacity_gated_count=capacity_gated_count,
    )
    status = _status(
        repeat_supported_count=repeat_supported_count,
        capacity_gated_count=capacity_gated_count,
        candidate_count=len(rows),
    )
    top_opportunities = ", ".join(
        row.get("opportunity", "")
        for row in sorted(rows, key=lambda item: _float(item.get("priority_score")), reverse=True)[:5]
        if row.get("opportunity", "")
    )
    evidence = (
        f"sources={', '.join(source_lanes)}; "
        f"top_opportunities={top_opportunities}; "
        f"best_net_bps={best_net:.4f}; "
        f"mean_net_bps={mean_net:.4f}; "
        f"max_usage={max_visible_depth_usage:.4f}"
    )
    return CostAdjustedAlphaCluster(
        cluster_id=f"{asset.lower()}_{decision}",
        asset=asset,
        decision=decision,
        candidate_count=len(rows),
        source_lane_count=len(source_lanes),
        source_lanes=", ".join(source_lanes),
        opportunity_count=len(opportunities),
        top_opportunities=top_opportunities,
        best_net_after_cost_bps=best_net,
        mean_net_after_cost_bps=mean_net,
        best_priority_score=best_priority,
        cluster_score=cluster_score,
        repeat_supported_count=repeat_supported_count,
        capacity_gated_count=capacity_gated_count,
        max_visible_depth_usage=max_visible_depth_usage,
        status=status,
        evidence=evidence,
        missing_work=_missing_work(status),
        next_step=_next_step(status, asset=asset, decision=decision),
    )


def _cluster_score(
    *,
    best_net: float,
    mean_net: float,
    source_lane_count: int,
    opportunity_count: int,
    repeat_supported_count: int,
    capacity_gated_count: int,
) -> float:
    return (
        best_net
        + mean_net * 0.35
        + min(source_lane_count, 4) * 25.0
        + min(opportunity_count, 8) * 10.0
        + repeat_supported_count * 35.0
        - capacity_gated_count * 90.0
    )


def _status(*, repeat_supported_count: int, capacity_gated_count: int, candidate_count: int) -> str:
    if repeat_supported_count > 0 and candidate_count > capacity_gated_count:
        return "multi_lane_repeat_supported_alpha_cluster"
    if capacity_gated_count == candidate_count:
        return "capacity_gated_alpha_cluster"
    return "multi_candidate_cost_adjusted_alpha_cluster"


def _missing_work(status: str) -> str:
    if status == "capacity_gated_alpha_cluster":
        return "cluster has positive paper edge but current size or depth is not tradable enough"
    if status == "multi_lane_repeat_supported_alpha_cluster":
        return "cluster has repeat support but still needs independent fill, stop, and adverse-excursion evidence"
    return "cluster has cost-adjusted paper support but still needs repeat evidence"


def _next_step(status: str, *, asset: str, decision: str) -> str:
    if status == "capacity_gated_alpha_cluster":
        return f"open a smaller {asset} {decision} paper probe only if depth improves"
    return f"run one consolidated {asset} {decision} repeat probe with explicit fill and stop notes"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cost_adjusted_alpha_clusters.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cost_adjusted_alpha_clusters.md")
    args = parser.parse_args()

    rows = build_cost_adjusted_alpha_clusters()
    write_cost_adjusted_alpha_clusters_csv(rows, output_path=args.output_path)
    write_cost_adjusted_alpha_clusters_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.cluster_id, f"{row.cluster_score:.4f}")


if __name__ == "__main__":
    main()
