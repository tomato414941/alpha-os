from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "current_paper_tickets.csv"
OUTCOMES_PATH = ROOT / "current_paper_ticket_outcomes.csv"
FILL_RISK_PATH = ROOT / "current_paper_ticket_fill_risk_check.csv"
EXPANSION_TARGETS_PATH = LOCAL_ROOT / "current_policy_expansion_targets.csv"


@dataclass(frozen=True)
class PolicyExpansionOutcomeFrontierRow:
    ticket_id: str
    target_id: str
    context: str
    seed_id: str
    asset: str
    action: str
    checkpoint_status: str
    outcome: str
    directional_return_bps: float
    estimated_net_after_cost_bps: str
    risk_action: str
    frontier_score: float
    decision: str
    evidence: str
    next_step: str


def build_policy_expansion_outcome_frontier(
    *,
    tickets_path: Path = TICKETS_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    fill_risk_path: Path = FILL_RISK_PATH,
    expansion_targets_path: Path = EXPANSION_TARGETS_PATH,
) -> tuple[PolicyExpansionOutcomeFrontierRow, ...]:
    outcomes_by_ticket = {row.get("ticket_id", ""): row for row in _read_rows(outcomes_path)}
    fill_by_ticket = {row.get("ticket_id", ""): row for row in _read_rows(fill_risk_path)}
    targets_by_id = {row.get("target_id", ""): row for row in _read_rows(expansion_targets_path)}
    rows = tuple(
        _frontier_row(
            ticket=ticket,
            outcome=outcomes_by_ticket.get(ticket.get("ticket_id", ""), {}),
            fill=fill_by_ticket.get(ticket.get("ticket_id", ""), {}),
            target=targets_by_id.get(ticket.get("opportunity", ""), {}),
        )
        for ticket in _read_rows(tickets_path)
        if ticket.get("probe_type", "") == "policy_expansion_probe"
    )
    return tuple(sorted(rows, key=lambda row: row.frontier_score, reverse=True))


def write_policy_expansion_outcome_frontier_csv(
    rows: tuple[PolicyExpansionOutcomeFrontierRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "target_id",
                "context",
                "seed_id",
                "asset",
                "action",
                "checkpoint_status",
                "outcome",
                "directional_return_bps",
                "estimated_net_after_cost_bps",
                "risk_action",
                "frontier_score",
                "decision",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.target_id,
                    row.context,
                    row.seed_id,
                    row.asset,
                    row.action,
                    row.checkpoint_status,
                    row.outcome,
                    f"{row.directional_return_bps:.8f}",
                    row.estimated_net_after_cost_bps,
                    row.risk_action,
                    f"{row.frontier_score:.8f}",
                    row.decision,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_policy_expansion_outcome_frontier_md(
    rows: tuple[PolicyExpansionOutcomeFrontierRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Policy Expansion Outcome Frontier\n\n")
        handle.write(
            "This isolates the policy-expansion paper tickets from the broader paper queue. "
            "It decides which expanded targets should repeat, wait, or rework. It is not a trained policy.\n\n"
        )
        handle.write(
            "| ticket | target | context | asset | outcome | dir bps | net bps | score | decision | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.ticket_id} | "
                f"{row.target_id} | "
                f"{row.context} | "
                f"{row.asset} | "
                f"{row.outcome} | "
                f"{row.directional_return_bps:.2f} | "
                f"{row.estimated_net_after_cost_bps} | "
                f"{row.frontier_score:.2f} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`repeat_policy_expansion_now` means the expanded target has a positive mark move and survives "
            "the rough fill/cost check. `check_fill_risk_before_repeat` means the mark moved but cost/depth "
            "evidence is not ready. `wait_for_checkpoint` and `keep_observing_flat` should not be promoted.\n"
        )
    return output_path


def _frontier_row(
    *,
    ticket: dict[str, str],
    outcome: dict[str, str],
    fill: dict[str, str],
    target: dict[str, str],
) -> PolicyExpansionOutcomeFrontierRow:
    dir_bps = _float(outcome.get("directional_return_bps"))
    net_bps = fill.get("estimated_net_after_cost_bps", "")
    risk_action = fill.get("risk_action", "")
    decision = _decision(
        checkpoint_status=outcome.get("checkpoint_status", ""),
        outcome=outcome.get("outcome", ""),
        dir_bps=dir_bps,
        net_bps=_float(net_bps),
        risk_action=risk_action,
    )
    score = _frontier_score(
        decision=decision,
        dir_bps=dir_bps,
        net_bps=_float(net_bps),
        context_score=_float(target.get("context_frontier_score")),
    )
    context = target.get("context", "")
    return PolicyExpansionOutcomeFrontierRow(
        ticket_id=ticket.get("ticket_id", ""),
        target_id=ticket.get("opportunity", ""),
        context=context,
        seed_id=target.get("seed_id", ""),
        asset=ticket.get("asset", ""),
        action=ticket.get("decision", ""),
        checkpoint_status=outcome.get("checkpoint_status", ""),
        outcome=outcome.get("outcome", ""),
        directional_return_bps=dir_bps,
        estimated_net_after_cost_bps=net_bps,
        risk_action=risk_action,
        frontier_score=score,
        decision=decision,
        evidence=(
            f"checkpoint={outcome.get('checkpoint_status', '')}; outcome={outcome.get('outcome', '')}; "
            f"dir_bps={dir_bps:.2f}; net_bps={net_bps or 'missing'}; risk={risk_action or 'missing'}"
        ),
        next_step=_next_step(
            decision=decision,
            ticket_id=ticket.get("ticket_id", ""),
            target_id=ticket.get("opportunity", ""),
            context=context,
        ),
    )


def _decision(*, checkpoint_status: str, outcome: str, dir_bps: float, net_bps: float, risk_action: str) -> str:
    if checkpoint_status == "pending" or outcome == "pending":
        return "wait_for_checkpoint"
    if outcome == "paper_mark_loss" or dir_bps < 0.0:
        return "rework_policy_expansion_target"
    if outcome == "paper_mark_flat":
        return "keep_observing_flat"
    if outcome == "paper_mark_win" and risk_action == "cost_adjusted_paper_probe" and net_bps > 0.0:
        return "repeat_policy_expansion_now"
    if outcome == "paper_mark_win":
        return "check_fill_risk_before_repeat"
    return "needs_current_mark"


def _frontier_score(*, decision: str, dir_bps: float, net_bps: float, context_score: float) -> float:
    decision_bonus = {
        "repeat_policy_expansion_now": 100.0,
        "check_fill_risk_before_repeat": 70.0,
        "wait_for_checkpoint": 30.0,
        "keep_observing_flat": 20.0,
        "needs_current_mark": 10.0,
        "rework_policy_expansion_target": 0.0,
    }.get(decision, 0.0)
    realized = net_bps if net_bps else dir_bps
    return decision_bonus + realized + min(context_score / 5.0, 50.0)


def _next_step(*, decision: str, ticket_id: str, target_id: str, context: str) -> str:
    if decision == "repeat_policy_expansion_now":
        return f"open a repeat policy-expansion ticket for {target_id} and compare it to its seed context {context}"
    if decision == "check_fill_risk_before_repeat":
        return f"run fill/cost/depth check for {ticket_id} before repeating {target_id}"
    if decision == "wait_for_checkpoint":
        return f"wait for the first checkpoint on {ticket_id}, then refresh marks"
    if decision == "keep_observing_flat":
        return f"keep {target_id} open until the quote moves or the source thesis changes"
    if decision == "rework_policy_expansion_target":
        return f"do not repeat {target_id}; isolate the failure regime against context {context}"
    return f"fill missing current mark evidence for {ticket_id}"


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
    parser.add_argument("--tickets-path", type=Path, default=TICKETS_PATH)
    parser.add_argument("--outcomes-path", type=Path, default=OUTCOMES_PATH)
    parser.add_argument("--fill-risk-path", type=Path, default=FILL_RISK_PATH)
    parser.add_argument("--expansion-targets-path", type=Path, default=EXPANSION_TARGETS_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_policy_expansion_outcome_frontier.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=LOCAL_ROOT / "current_policy_expansion_outcome_frontier.md",
    )
    args = parser.parse_args()

    rows = build_policy_expansion_outcome_frontier(
        tickets_path=args.tickets_path,
        outcomes_path=args.outcomes_path,
        fill_risk_path=args.fill_risk_path,
        expansion_targets_path=args.expansion_targets_path,
    )
    write_policy_expansion_outcome_frontier_csv(rows, output_path=args.output_path)
    write_policy_expansion_outcome_frontier_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.decision, f"{row.frontier_score:.4f}")


if __name__ == "__main__":
    main()
