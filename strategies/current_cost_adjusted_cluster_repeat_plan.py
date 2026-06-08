from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP = 10


@dataclass(frozen=True)
class ClusterRepeatPlan:
    action_id: str
    cluster_id: str
    asset: str
    decision: str
    action: str
    representative_ticket_id: str
    representative_opportunity: str
    candidate_size_usd: str
    source_lanes: str
    candidate_count: str
    best_net_after_cost_bps: str
    cluster_score: str
    required_record: str
    next_step: str


def build_cluster_repeat_plan(
    *,
    clusters_path: Path = ROOT / "current_cost_adjusted_alpha_clusters.csv",
    candidates_path: Path = ROOT / "current_cost_adjusted_alpha_candidates.csv",
    top: int = DEFAULT_TOP,
) -> tuple[ClusterRepeatPlan, ...]:
    candidates_by_cluster = _candidates_by_cluster(candidates_path)
    rows: list[ClusterRepeatPlan] = []
    for rank, cluster in enumerate(_read_rows(clusters_path)[:top], start=1):
        representative = _representative_candidate(cluster, candidates_by_cluster)
        rows.append(_plan_row(rank=rank, cluster=cluster, representative=representative))
    return tuple(rows)


def write_cluster_repeat_plan_csv(rows: tuple[ClusterRepeatPlan, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "action_id",
                "cluster_id",
                "asset",
                "decision",
                "action",
                "representative_ticket_id",
                "representative_opportunity",
                "candidate_size_usd",
                "source_lanes",
                "candidate_count",
                "best_net_after_cost_bps",
                "cluster_score",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.action_id,
                    row.cluster_id,
                    row.asset,
                    row.decision,
                    row.action,
                    row.representative_ticket_id,
                    row.representative_opportunity,
                    row.candidate_size_usd,
                    row.source_lanes,
                    row.candidate_count,
                    row.best_net_after_cost_bps,
                    row.cluster_score,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_cluster_repeat_plan_md(rows: tuple[ClusterRepeatPlan, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cost Adjusted Cluster Repeat Plan\n\n")
        handle.write(
            "This converts cost-adjusted alpha clusters into one consolidated repeat action per asset and direction. "
            "It avoids multiplying paper tickets for duplicate opportunities inside the same symbol cluster.\n\n"
        )
        handle.write(
            "| action | cluster | decision | size USD | candidates | best net | score | representative | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.action} | "
                f"{row.cluster_id} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.candidate_count} | "
                f"{row.best_net_after_cost_bps} | "
                f"{row.cluster_score} | "
                f"{row.representative_opportunity} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _plan_row(
    *,
    rank: int,
    cluster: dict[str, str],
    representative: dict[str, str],
) -> ClusterRepeatPlan:
    action = _action(cluster)
    asset = cluster.get("asset", "")
    decision = cluster.get("decision", "")
    return ClusterRepeatPlan(
        action_id=f"cluster-repeat-{rank:02d}-{cluster.get('cluster_id', '')}",
        cluster_id=cluster.get("cluster_id", ""),
        asset=asset,
        decision=decision,
        action=action,
        representative_ticket_id=representative.get("ticket_id", ""),
        representative_opportunity=representative.get("opportunity", ""),
        candidate_size_usd=_candidate_size(cluster=cluster, representative=representative),
        source_lanes=cluster.get("source_lanes", ""),
        candidate_count=cluster.get("candidate_count", ""),
        best_net_after_cost_bps=cluster.get("best_net_after_cost_bps", ""),
        cluster_score=cluster.get("cluster_score", ""),
        required_record=_required_record(action),
        next_step=_next_step(action=action, asset=asset, decision=decision),
    )


def _candidates_by_cluster(path: Path) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_rows(path):
        groups.setdefault((row.get("asset", ""), row.get("decision", "")), []).append(row)
    return {
        key: tuple(sorted(rows, key=lambda row: _float(row.get("priority_score")), reverse=True))
        for key, rows in groups.items()
    }


def _representative_candidate(
    cluster: dict[str, str],
    candidates_by_cluster: dict[tuple[str, str], tuple[dict[str, str], ...]],
) -> dict[str, str]:
    rows = candidates_by_cluster.get((cluster.get("asset", ""), cluster.get("decision", "")), ())
    return rows[0] if rows else {}


def _action(cluster: dict[str, str]) -> str:
    if cluster.get("status") == "capacity_gated_alpha_cluster":
        return "resize_before_repeat_probe"
    return "open_consolidated_repeat_probe"


def _candidate_size(*, cluster: dict[str, str], representative: dict[str, str]) -> str:
    if cluster.get("status") == "capacity_gated_alpha_cluster":
        return "smaller_than_representative"
    return representative.get("candidate_size_usd", "")


def _required_record(action: str) -> str:
    if action == "resize_before_repeat_probe":
        return "smaller size, current depth, spread, funding, entry mark, stop, adverse excursion"
    return "entry mark, realized fill assumption, spread, funding, stop, adverse excursion, 15m/1h mark"


def _next_step(*, action: str, asset: str, decision: str) -> str:
    if action == "resize_before_repeat_probe":
        return f"only repeat {asset} {decision} after choosing a smaller size that fits current visible depth"
    return f"repeat {asset} {decision} once as a cluster-level paper probe instead of duplicating all source tickets"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cost_adjusted_cluster_repeat_plan.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cost_adjusted_cluster_repeat_plan.md")
    args = parser.parse_args()

    rows = build_cluster_repeat_plan(top=args.top)
    write_cluster_repeat_plan_csv(rows, output_path=args.output_path)
    write_cluster_repeat_plan_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.action, row.cluster_id, row.best_net_after_cost_bps)


if __name__ == "__main__":
    main()
