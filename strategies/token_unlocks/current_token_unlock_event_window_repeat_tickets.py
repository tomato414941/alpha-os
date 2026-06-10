from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TokenUnlockEventWindowRepeatTicket:
    ticket_id: str
    opened_at: str
    previous_ticket_id: str
    asset: str
    opportunity: str
    decision: str
    venue: str
    candidate_size_usd: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    estimated_net_after_cost_bps: str
    prior_directional_bps: str
    required_record: str
    next_step: str


def build_token_unlock_event_window_repeat_tickets(
    *,
    risk_path: Path = ROOT / "current_token_unlock_event_window_risk_check.csv",
    outcomes_path: Path = ROOT / "current_token_unlock_event_window_outcomes.csv",
    tickets_path: Path = ROOT / "current_token_unlock_event_window_tickets.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[TokenUnlockEventWindowRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    outcomes = {row.get("ticket_id", ""): row for row in _read_rows(outcomes_path)}
    tickets = {row.get("ticket_id", ""): row for row in _read_rows(tickets_path)}
    rows: list[TokenUnlockEventWindowRepeatTicket] = []
    for row in _read_rows(risk_path):
        if row.get("risk_action") != "cost_adjusted_event_window_probe":
            continue
        previous_id = row.get("ticket_id", "")
        outcome = outcomes.get(previous_id, {})
        source_ticket = tickets.get(previous_id, {})
        asset = row.get("asset", "")
        repeat_id = f"token-unlock-repeat-{_slug(asset)}-{_slug(row.get('decision', ''))}"
        if repeat_id in existing:
            rows.append(existing[repeat_id])
            continue
        rows.append(
            TokenUnlockEventWindowRepeatTicket(
                ticket_id=repeat_id,
                opened_at=opened_at,
                previous_ticket_id=previous_id,
                asset=asset,
                opportunity=f"token_unlock_event_window_repeat:{asset}",
                decision=row.get("decision", ""),
                venue=outcome.get("venue") or source_ticket.get("venue") or "HL",
                candidate_size_usd="250",
                checkpoints="15m,1h,4h",
                entry_mark=outcome.get("current_mark", ""),
                entry_source=outcome.get("current_source", ""),
                estimated_net_after_cost_bps=row.get("net_directional_bps", ""),
                prior_directional_bps=row.get("directional_return_bps", ""),
                required_record="fresh repeat mark move, funding, spread, stop, and adverse excursion",
                next_step=f"repeat {asset} unlock event-window label and compare 15m/1h/4h behavior with the first ticket",
            )
        )
    return tuple(rows)


def write_token_unlock_event_window_repeat_tickets_csv(
    rows: tuple[TokenUnlockEventWindowRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(TokenUnlockEventWindowRepeatTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_token_unlock_event_window_repeat_tickets_md(
    rows: tuple[TokenUnlockEventWindowRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Event Window Repeat Tickets\n\n")
        handle.write(
            "These preserve fresh entry marks for token-unlock event-window probes "
            "after the first label survived rough cost checks. They are not live orders.\n\n"
        )
        handle.write(
            "| ticket | previous | asset | decision | entry | prior net | checkpoints | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.previous_ticket_id} | {row.asset} | "
                f"{row.decision} | {row.entry_mark} | {row.estimated_net_after_cost_bps} | "
                f"{row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _existing_tickets(path: Path | None) -> tuple[TokenUnlockEventWindowRepeatTicket, ...]:
    if path is None:
        return ()
    return tuple(
        TokenUnlockEventWindowRepeatTicket(
            ticket_id=row.get("ticket_id", ""),
            opened_at=row.get("opened_at", ""),
            previous_ticket_id=row.get("previous_ticket_id", ""),
            asset=row.get("asset", ""),
            opportunity=row.get("opportunity", ""),
            decision=row.get("decision", ""),
            venue=row.get("venue", ""),
            candidate_size_usd=row.get("candidate_size_usd", ""),
            checkpoints=row.get("checkpoints", ""),
            entry_mark=row.get("entry_mark", ""),
            entry_source=row.get("entry_source", ""),
            estimated_net_after_cost_bps=row.get("estimated_net_after_cost_bps", ""),
            prior_directional_bps=row.get("prior_directional_bps", ""),
            required_record=row.get("required_record", ""),
            next_step=row.get("next_step", ""),
        )
        for row in _read_rows(path)
        if row.get("ticket_id")
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-path", type=Path, default=ROOT / "current_token_unlock_event_window_risk_check.csv")
    parser.add_argument(
        "--outcomes-path",
        type=Path,
        default=ROOT / "current_token_unlock_event_window_outcomes.csv",
    )
    parser.add_argument("--tickets-path", type=Path, default=ROOT / "current_token_unlock_event_window_tickets.csv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_token_unlock_event_window_repeat_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_token_unlock_event_window_repeat_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_token_unlock_event_window_repeat_tickets(
        risk_path=args.risk_path,
        outcomes_path=args.outcomes_path,
        tickets_path=args.tickets_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_token_unlock_event_window_repeat_tickets_csv(rows, output_path=args.output_path)
    write_token_unlock_event_window_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark, row.estimated_net_after_cost_bps)


if __name__ == "__main__":
    main()
