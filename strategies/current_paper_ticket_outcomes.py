from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperTicketOutcome:
    ticket_id: str
    opened_at: str
    checked_at: str
    elapsed_minutes: float
    checkpoint_status: str
    opportunity: str
    decision: str
    asset: str
    venue: str
    entry_mark: str
    current_mark: str
    current_source: str
    raw_return_bps: str
    directional_return_bps: str
    outcome: str
    missing_evidence: str
    next_step: str


def build_paper_ticket_outcomes(
    *,
    tickets_path: Path = ROOT / "current_paper_tickets.csv",
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
) -> tuple[PaperTicketOutcome, ...]:
    checked_at = datetime.now(UTC)
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    return tuple(
        _build_outcome(checked_at=checked_at, row=row, marks=marks)
        for row in _read_rows(tickets_path)
    )


def write_paper_ticket_outcomes_csv(
    rows: tuple[PaperTicketOutcome, ...],
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
                "checked_at",
                "elapsed_minutes",
                "checkpoint_status",
                "opportunity",
                "decision",
                "asset",
                "venue",
                "entry_mark",
                "current_mark",
                "current_source",
                "raw_return_bps",
                "directional_return_bps",
                "outcome",
                "missing_evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.checked_at,
                    f"{row.elapsed_minutes:.2f}",
                    row.checkpoint_status,
                    row.opportunity,
                    row.decision,
                    row.asset,
                    row.venue,
                    row.entry_mark,
                    row.current_mark,
                    row.current_source,
                    row.raw_return_bps,
                    row.directional_return_bps,
                    row.outcome,
                    row.missing_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_paper_ticket_outcomes_md(
    rows: tuple[PaperTicketOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Ticket Outcomes\n\n")
        handle.write(
            "This checks opened paper tickets against the latest available public marks. "
            "It is not a fill report and not a live trading PnL report.\n\n"
        )
        handle.write(
            "| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.checkpoint_status} | "
                f"{row.decision} | "
                f"{row.asset} | "
                f"{row.venue} | "
                f"{row.entry_mark} | "
                f"{row.current_mark} | "
                f"{row.directional_return_bps} | "
                f"{row.outcome} | "
                f"{_escape(row.missing_evidence)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _build_outcome(
    *,
    checked_at: datetime,
    row: dict[str, str],
    marks: dict[tuple[str, str], tuple[str, str]],
) -> PaperTicketOutcome:
    opened_at = _parse_time(row.get("opened_at", ""))
    elapsed_minutes = (checked_at - opened_at).total_seconds() / 60.0 if opened_at else 0.0
    checkpoint_status = _checkpoint_status(row.get("checkpoints", ""), elapsed_minutes)
    current_mark, current_source = _entry_mark(
        asset=row.get("asset", ""),
        venue=row.get("venue", ""),
        marks=marks,
    )
    raw_bps, dir_bps, outcome, missing = _mark_outcome(
        entry_mark=row.get("entry_mark", ""),
        current_mark=current_mark,
        decision=row.get("decision", ""),
        checkpoint_status=checkpoint_status,
    )
    return PaperTicketOutcome(
        ticket_id=row.get("ticket_id", ""),
        opened_at=row.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        elapsed_minutes=elapsed_minutes,
        checkpoint_status=checkpoint_status,
        opportunity=row.get("opportunity", ""),
        decision=row.get("decision", ""),
        asset=row.get("asset", ""),
        venue=row.get("venue", ""),
        entry_mark=row.get("entry_mark", ""),
        current_mark=current_mark,
        current_source=current_source,
        raw_return_bps=raw_bps,
        directional_return_bps=dir_bps,
        outcome=outcome,
        missing_evidence=missing,
        next_step=_next_step(checkpoint_status=checkpoint_status, outcome=outcome),
    )


def _entry_mark(*, asset: str, venue: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    keys = ((venue.upper(), asset.upper()), ("HL", asset.upper()), ("", asset.upper()))
    for key in keys:
        if key in marks:
            return marks[key]
    return "", ""


def _mark_outcome(
    *,
    entry_mark: str,
    current_mark: str,
    decision: str,
    checkpoint_status: str,
) -> tuple[str, str, str, str]:
    if checkpoint_status == "pending":
        return "", "", "pending", "checkpoint has not matured"
    if not entry_mark or not current_mark:
        return "", "", "missing_current_mark", "entry or current mark is missing"
    entry = _float(entry_mark)
    current = _float(current_mark)
    if entry <= 0.0 or current <= 0.0:
        return "", "", "missing_current_mark", "entry or current mark is invalid"
    raw_bps = (current / entry - 1.0) * 10_000.0
    if decision == "paper_short":
        directional_bps = -raw_bps
    else:
        directional_bps = raw_bps
    outcome = "paper_mark_win" if directional_bps > 0.0 else "paper_mark_loss"
    return f"{raw_bps:.8f}", f"{directional_bps:.8f}", outcome, "fill, funding, stop, and adverse excursion still missing"


def _checkpoint_status(checkpoints: str, elapsed_minutes: float) -> str:
    first_minutes = min((_checkpoint_minutes(value) for value in checkpoints.split(",")), default=0)
    if first_minutes <= 0:
        return "ready"
    return "ready" if elapsed_minutes >= first_minutes else "pending"


def _checkpoint_minutes(value: str) -> int:
    value = value.strip()
    if value == "15m":
        return 15
    if value == "1h":
        return 60
    if value == "4h":
        return 240
    if value == "12h":
        return 720
    if value == "24h":
        return 1440
    return 0


def _next_step(*, checkpoint_status: str, outcome: str) -> str:
    if checkpoint_status == "pending":
        return "wait for the first checkpoint and refresh marks"
    if outcome == "paper_mark_win":
        return "record fill, funding, stop, and adverse-excursion assumptions before promotion"
    if outcome == "paper_mark_loss":
        return "keep or reject based on repeated labels and failure regime"
    return "fill missing current mark before judging the ticket"


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets-path", type=Path, default=ROOT / "current_paper_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_ticket_outcomes.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_ticket_outcomes.md")
    args = parser.parse_args()

    rows = build_paper_ticket_outcomes(tickets_path=args.tickets_path)
    write_paper_ticket_outcomes_csv(rows, output_path=args.output_path)
    write_paper_ticket_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.checkpoint_status, row.outcome, row.directional_return_bps)


if __name__ == "__main__":
    main()
