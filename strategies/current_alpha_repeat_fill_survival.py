from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaRepeatFillSurvivalRow:
    work_id: str
    cluster_id: str
    asset: str
    decision: str
    status: str
    survival_score: float
    best_net_after_cost_bps: float
    first_repeat_net_after_cost_bps: float
    second_repeat_net_after_cost_bps: float
    first_repeat_outcome: str
    second_repeat_outcome: str
    edge_decay_ratio: float
    latest_elapsed_minutes: str
    latest_checked_at: str
    required_evidence: str
    evidence: str
    next_step: str


def build_alpha_repeat_fill_survival(
    *,
    worklist_path: Path = ROOT / "current_alpha_promotion_worklist.csv",
    cost_survival_path: Path = ROOT / "current_cost_survival_cross_section.csv",
    first_repeat_outcomes_path: Path = ROOT / "current_promoted_ticket_repeat_outcomes.csv",
    second_repeat_outcomes_path: Path = ROOT / "current_second_promoted_ticket_repeat_outcomes.csv",
    first_fill_risk_path: Path = ROOT / "current_promoted_ticket_repeat_fill_risk_check.csv",
    second_fill_risk_path: Path = ROOT / "current_second_promoted_ticket_repeat_fill_risk_check.csv",
) -> tuple[AlphaRepeatFillSurvivalRow, ...]:
    clusters = {row.get("cluster_id", ""): row for row in _read_rows(cost_survival_path)}
    first_outcomes = _group_by_asset_decision(_read_rows(first_repeat_outcomes_path))
    second_outcomes = _group_by_asset_decision(_read_rows(second_repeat_outcomes_path))
    first_risk = _group_by_asset_decision(_read_rows(first_fill_risk_path))
    second_risk = _group_by_asset_decision(_read_rows(second_fill_risk_path))

    rows: list[AlphaRepeatFillSurvivalRow] = []
    for work in _read_rows(worklist_path):
        if work.get("work_kind") != "repeat_fill_risk_probe":
            continue
        cluster_id = _cluster_id_from_frontier(work.get("source_frontier_id", ""))
        cluster = clusters.get(cluster_id, {})
        asset = work.get("asset", "")
        decision = work.get("action", "")
        key = (asset, decision)
        first_best_risk = _best_numeric_row(first_risk.get(key, ()), key="estimated_net_after_cost_bps")
        second_best_risk = _best_numeric_row(second_risk.get(key, ()), key="estimated_net_after_cost_bps")
        first_latest = _latest_outcome(first_outcomes.get(key, ()))
        second_latest = _latest_outcome(second_outcomes.get(key, ()))
        rows.append(
            _survival_row(
                work=work,
                cluster=cluster,
                first_risk=first_best_risk,
                second_risk=second_best_risk,
                first_outcome=first_latest,
                second_outcome=second_latest,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_alpha_repeat_fill_survival_csv(
    rows: tuple[AlphaRepeatFillSurvivalRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "work_id",
                "cluster_id",
                "asset",
                "decision",
                "status",
                "survival_score",
                "best_net_after_cost_bps",
                "first_repeat_net_after_cost_bps",
                "second_repeat_net_after_cost_bps",
                "first_repeat_outcome",
                "second_repeat_outcome",
                "edge_decay_ratio",
                "latest_elapsed_minutes",
                "latest_checked_at",
                "required_evidence",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.work_id,
                    row.cluster_id,
                    row.asset,
                    row.decision,
                    row.status,
                    f"{row.survival_score:.8f}",
                    f"{row.best_net_after_cost_bps:.8f}",
                    f"{row.first_repeat_net_after_cost_bps:.8f}",
                    f"{row.second_repeat_net_after_cost_bps:.8f}",
                    row.first_repeat_outcome,
                    row.second_repeat_outcome,
                    f"{row.edge_decay_ratio:.8f}",
                    row.latest_elapsed_minutes,
                    row.latest_checked_at,
                    row.required_evidence,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_repeat_fill_survival_md(
    rows: tuple[AlphaRepeatFillSurvivalRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Repeat Fill Survival\n\n")
        handle.write(
            "This checks the top repeat/fill-risk worklist items against first and second repeat evidence. "
            "Rows still lack real fill, stop, and adverse-excursion records, so this is not a promotion report.\n\n"
        )
        handle.write(
            "| work | asset | status | score | best net | first net | second net | first outcome | second outcome | decay | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.work_id} | "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.survival_score:.4f} | "
                f"{row.best_net_after_cost_bps:.4f} | "
                f"{row.first_repeat_net_after_cost_bps:.4f} | "
                f"{row.second_repeat_net_after_cost_bps:.4f} | "
                f"{row.first_repeat_outcome} | "
                f"{row.second_repeat_outcome} | "
                f"{row.edge_decay_ratio:.4f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _survival_row(
    *,
    work: dict[str, str],
    cluster: dict[str, str],
    first_risk: dict[str, str],
    second_risk: dict[str, str],
    first_outcome: dict[str, str],
    second_outcome: dict[str, str],
) -> AlphaRepeatFillSurvivalRow:
    best_net = _float(cluster.get("best_net_after_cost_bps"))
    first_net = _float(first_risk.get("estimated_net_after_cost_bps"))
    second_net = _float(second_risk.get("estimated_net_after_cost_bps"))
    first_result = first_outcome.get("outcome", "")
    second_result = second_outcome.get("outcome", "")
    edge_decay_ratio = _edge_decay_ratio(first_net=first_net or best_net, second_net=second_net)
    status = _status(
        first_result=first_result,
        second_result=second_result,
        second_net=second_net,
        edge_decay_ratio=edge_decay_ratio,
    )
    score = _score(status=status, best_net=best_net, first_net=first_net, second_net=second_net)
    latest = second_outcome or first_outcome
    return AlphaRepeatFillSurvivalRow(
        work_id=work.get("work_id", ""),
        cluster_id=_cluster_id_from_frontier(work.get("source_frontier_id", "")),
        asset=work.get("asset", ""),
        decision=work.get("action", ""),
        status=status,
        survival_score=score,
        best_net_after_cost_bps=best_net,
        first_repeat_net_after_cost_bps=first_net,
        second_repeat_net_after_cost_bps=second_net,
        first_repeat_outcome=first_result or "missing",
        second_repeat_outcome=second_result or "missing",
        edge_decay_ratio=edge_decay_ratio,
        latest_elapsed_minutes=latest.get("elapsed_minutes", ""),
        latest_checked_at=latest.get("checked_at", ""),
        required_evidence=work.get("required_evidence", ""),
        evidence=(
            f"cluster_status={cluster.get('status', '')}; "
            f"best_net={best_net:.4f}; "
            f"first_ticket={first_risk.get('ticket_id', '')}; "
            f"first_net={first_net:.4f}; "
            f"first_outcome={first_result or 'missing'}; "
            f"second_ticket={second_risk.get('ticket_id', '') or second_outcome.get('ticket_id', '')}; "
            f"second_net={second_net:.4f}; "
            f"second_outcome={second_result or 'missing'}; "
            f"decay={edge_decay_ratio:.4f}"
        ),
        next_step=_next_step(status=status, asset=work.get("asset", ""), decision=work.get("action", "")),
    )


def _status(
    *,
    first_result: str,
    second_result: str,
    second_net: float,
    edge_decay_ratio: float,
) -> str:
    if second_result == "pending":
        return "second_repeat_pending"
    if second_result == "paper_mark_loss":
        return "second_repeat_failed"
    if second_result == "paper_mark_flat":
        return "second_repeat_flat"
    if second_result == "paper_mark_win" and second_net <= 5.0:
        return "repeat_edge_collapsed"
    if second_result == "paper_mark_win" and edge_decay_ratio >= 0.80:
        return "repeat_edge_mostly_decayed"
    if second_result == "paper_mark_win" and second_net > 25.0:
        return "second_repeat_cost_survived"
    if first_result == "paper_mark_win":
        return "first_repeat_survived_waiting_second"
    return "repeat_evidence_missing"


def _score(*, status: str, best_net: float, first_net: float, second_net: float) -> float:
    base = {
        "second_repeat_cost_survived": 800.0,
        "first_repeat_survived_waiting_second": 640.0,
        "second_repeat_pending": 600.0,
        "repeat_edge_mostly_decayed": 360.0,
        "repeat_edge_collapsed": 260.0,
        "second_repeat_flat": 220.0,
        "second_repeat_failed": 100.0,
    }.get(status, 160.0)
    edge = second_net if second_net else first_net if first_net else best_net
    return base + min(edge, 250.0)


def _next_step(*, status: str, asset: str, decision: str) -> str:
    if status == "second_repeat_cost_survived":
        return f"add real fill, stop, and adverse-excursion notes for {asset} {decision}"
    if status == "first_repeat_survived_waiting_second":
        return f"open or refresh a second repeat probe for {asset} {decision}"
    if status == "second_repeat_pending":
        return f"wait for the second repeat checkpoint, then rerun repeat fill survival for {asset}"
    if status in {"repeat_edge_collapsed", "repeat_edge_mostly_decayed"}:
        return f"do not promote {asset} {decision}; require a fresh independent repeat before keeping it alive"
    if status == "second_repeat_failed":
        return f"reject or isolate the failure regime for {asset} {decision}"
    if status == "second_repeat_flat":
        return f"keep observing {asset} {decision} only if the next mark moves with independent evidence"
    return f"collect repeat and fill evidence before ranking {asset} {decision}"


def _cluster_id_from_frontier(frontier_id: str) -> str:
    prefix = "cost_cluster:"
    if frontier_id.startswith(prefix):
        return frontier_id[len(prefix) :]
    return frontier_id


def _group_by_asset_decision(rows: tuple[dict[str, str], ...]) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("asset", ""), row.get("decision", ""))
        if all(key):
            grouped.setdefault(key, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _latest_outcome(rows: tuple[dict[str, str], ...]) -> dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: (_float(row.get("elapsed_minutes")), row.get("checked_at", "")))


def _best_numeric_row(rows: tuple[dict[str, str], ...], *, key: str) -> dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: _float(row.get(key)))


def _edge_decay_ratio(*, first_net: float, second_net: float) -> float:
    if first_net <= 0.0 or second_net <= 0.0:
        return 0.0
    return max((first_net - second_net) / first_net, 0.0)


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_repeat_fill_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_repeat_fill_survival.md")
    args = parser.parse_args()

    rows = build_alpha_repeat_fill_survival()
    write_alpha_repeat_fill_survival_csv(rows, output_path=args.output_path)
    write_alpha_repeat_fill_survival_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.status, row.asset, f"{row.survival_score:.4f}", row.next_step)


if __name__ == "__main__":
    main()
