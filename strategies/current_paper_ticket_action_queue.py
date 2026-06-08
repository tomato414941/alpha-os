from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperTicketAction:
    priority: float
    ticket_id: str
    action: str
    asset: str
    opportunity: str
    decision: str
    directional_return_bps: str
    outcome: str
    reason: str
    next_step: str


def build_paper_ticket_action_queue(
    *,
    outcomes_path: Path = ROOT / "current_paper_ticket_outcomes.csv",
) -> tuple[PaperTicketAction, ...]:
    rows = tuple(_action_from_outcome(row) for row in _read_rows(outcomes_path))
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_paper_ticket_action_queue_csv(
    rows: tuple[PaperTicketAction, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "priority",
                "ticket_id",
                "action",
                "asset",
                "opportunity",
                "decision",
                "directional_return_bps",
                "outcome",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    f"{row.priority:.8f}",
                    row.ticket_id,
                    row.action,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.directional_return_bps,
                    row.outcome,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_paper_ticket_action_queue_md(
    rows: tuple[PaperTicketAction, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Ticket Action Queue\n\n")
        handle.write(
            "This turns paper-ticket mark outcomes into the next observation work. "
            "It is not a trade instruction and does not promote a candidate without "
            "fill, funding, stop, and repeated-label evidence.\n\n"
        )
        handle.write(
            "| priority | ticket | action | asset | decision | dir bps | outcome | reason | next step |\n"
        )
        handle.write("| ---: | --- | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.priority:.4f} | "
                f"{row.ticket_id} | "
                f"{row.action} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.directional_return_bps} | "
                f"{row.outcome} | "
                f"{_escape(row.reason)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _action_from_outcome(row: dict[str, str]) -> PaperTicketAction:
    outcome = row.get("outcome", "")
    directional_bps = _float(row.get("directional_return_bps"))
    if row.get("checkpoint_status") == "pending":
        action = "wait_for_checkpoint"
        priority = 50.0
        reason = "ticket checkpoint has not matured"
        next_step = row.get("next_step", "")
    elif outcome == "paper_mark_win":
        action = "promote_to_fill_and_risk_check"
        priority = 100.0 + directional_bps
        reason = "paper mark moved in the ticket direction"
        next_step = "check fill assumption, funding, stop, adverse excursion, then repeat the label"
    elif outcome == "paper_mark_loss":
        action = "deprioritize_or_repeat_once"
        priority = 25.0 + directional_bps
        reason = "paper mark moved against the ticket direction"
        next_step = "repeat only if the original hypothesis has independent support; otherwise deprioritize"
    elif outcome == "paper_mark_flat":
        action = "keep_observing_quote"
        priority = 40.0
        reason = "paper mark has not moved"
        next_step = "keep observing until quote moves or external evidence changes"
    elif outcome.startswith("observe_"):
        action = "record_observation_only"
        priority = 35.0
        reason = "observation-only ticket is not a directional promotion candidate"
        next_step = "keep the context record; open a directional ticket only if the thesis becomes explicit"
    else:
        action = "fill_missing_observation"
        priority = 30.0
        reason = row.get("missing_evidence", "")
        next_step = row.get("next_step", "")
    return PaperTicketAction(
        priority=priority,
        ticket_id=row.get("ticket_id", ""),
        action=action,
        asset=row.get("asset", ""),
        opportunity=row.get("opportunity", ""),
        decision=row.get("decision", ""),
        directional_return_bps=row.get("directional_return_bps", ""),
        outcome=outcome,
        reason=reason,
        next_step=next_step,
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
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_paper_ticket_outcomes.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_ticket_action_queue.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_ticket_action_queue.md")
    args = parser.parse_args()

    rows = build_paper_ticket_action_queue(outcomes_path=args.outcomes_path)
    write_paper_ticket_action_queue_csv(rows, output_path=args.output_path)
    write_paper_ticket_action_queue_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.action, row.ticket_id, row.asset, row.directional_return_bps)


if __name__ == "__main__":
    main()
