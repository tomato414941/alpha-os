from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromotedRepeatTicket:
    ticket_id: str
    opened_at: str
    previous_ticket_id: str
    asset: str
    opportunity: str
    decision: str
    candidate_size_usd: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    estimated_net_after_cost_bps: str
    required_record: str
    next_step: str


def build_promoted_repeat_tickets(
    *,
    fill_risk_path: Path = ROOT / "current_paper_ticket_fill_risk_check.csv",
    outcomes_path: Path = ROOT / "current_paper_ticket_outcomes.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[PromotedRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing_opened_at = _existing_opened_at(existing_tickets_path)
    existing_tickets = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    outcomes = {row.get("ticket_id", ""): row for row in _read_rows(outcomes_path)}
    rows = []
    for row in _read_rows(fill_risk_path):
        if row.get("risk_action") != "cost_adjusted_paper_probe":
            continue
        previous_id = row.get("ticket_id", "")
        repeat_id = f"repeat-{previous_id}"
        if repeat_id in existing_tickets:
            rows.append(existing_tickets[repeat_id])
            continue
        outcome = outcomes.get(previous_id, {})
        rows.append(
            PromotedRepeatTicket(
                ticket_id=repeat_id,
                opened_at=existing_opened_at.get(repeat_id, opened_at),
                previous_ticket_id=previous_id,
                asset=row.get("asset", ""),
                opportunity=row.get("opportunity", ""),
                decision=row.get("decision", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                checkpoints="15m,1h",
                entry_mark=outcome.get("current_mark", ""),
                entry_source=outcome.get("current_source", ""),
                estimated_net_after_cost_bps=row.get("estimated_net_after_cost_bps", ""),
                required_record="repeat mark move, spread/fill assumption, funding, stop, adverse excursion",
                next_step="repeat this cost-adjusted paper probe and compare 15m/1h behavior against the first ticket",
            )
        )
    known_ticket_ids = {row.ticket_id for row in rows}
    rows.extend(
        row
        for row in existing_tickets.values()
        if row.ticket_id not in known_ticket_ids
    )
    return tuple(rows)


def write_promoted_repeat_tickets_csv(
    rows: tuple[PromotedRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "opened_at",
                "previous_ticket_id",
                "asset",
                "opportunity",
                "decision",
                "candidate_size_usd",
                "checkpoints",
                "entry_mark",
                "entry_source",
                "estimated_net_after_cost_bps",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.previous_ticket_id,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.candidate_size_usd,
                    row.checkpoints,
                    row.entry_mark,
                    row.entry_source,
                    row.estimated_net_after_cost_bps,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_promoted_repeat_tickets_md(
    rows: tuple[PromotedRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Promoted Ticket Repeat Tickets\n\n")
        handle.write(
            "These are repeat paper tickets opened only after a first paper ticket "
            "survives rough mark, spread, taker-fee, funding, and visible-depth checks.\n\n"
        )
        handle.write(
            "| ticket | previous | asset | decision | size USD | entry | checkpoints | net after cost | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.previous_ticket_id} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{row.estimated_net_after_cost_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _existing_opened_at(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        row.get("ticket_id", ""): row.get("opened_at", "")
        for row in _read_rows(path)
        if row.get("ticket_id") and row.get("opened_at")
    }


def _existing_tickets(path: Path | None) -> tuple[PromotedRepeatTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            PromotedRepeatTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                previous_ticket_id=row.get("previous_ticket_id", ""),
                asset=row.get("asset", ""),
                opportunity=row.get("opportunity", ""),
                decision=row.get("decision", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                checkpoints=row.get("checkpoints", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                estimated_net_after_cost_bps=row.get("estimated_net_after_cost_bps", ""),
                required_record=row.get("required_record", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill-risk-path", type=Path, default=ROOT / "current_paper_ticket_fill_risk_check.csv")
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_paper_ticket_outcomes.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_promoted_ticket_repeat_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_promoted_ticket_repeat_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_promoted_repeat_tickets(
        fill_risk_path=args.fill_risk_path,
        outcomes_path=args.outcomes_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_promoted_repeat_tickets_csv(rows, output_path=args.output_path)
    write_promoted_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.entry_mark, row.estimated_net_after_cost_bps)


if __name__ == "__main__":
    main()
