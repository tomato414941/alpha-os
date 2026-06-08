from __future__ import annotations

import argparse
import csv
import re
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import PaperTicket, _load_marks, write_paper_tickets_csv, write_paper_tickets_md


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP = 8


def build_broad_alpha_repeat_tickets(
    *,
    outcomes_path: Path = ROOT / "current_broad_alpha_fill_audit_outcomes.csv",
    audit_tickets_path: Path = ROOT / "current_broad_alpha_fill_audit_tickets.csv",
    existing_tickets_path: Path | None = None,
    top: int = DEFAULT_TOP,
) -> tuple[PaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    audit_tickets = {row.get("ticket_id", ""): row for row in _read_rows(audit_tickets_path)}
    existing = {ticket.ticket_id: ticket for ticket in _existing_tickets(existing_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
        hl_context_path=ROOT / "candidate_validation" / "current_followup_execution_context.csv",
        okx_context_path=ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    )
    winners = _winning_outcomes(outcomes_path)[:top]
    tickets: list[PaperTicket] = []
    for rank, outcome in enumerate(winners, start=1):
        audit_ticket = audit_tickets.get(outcome.get("ticket_id", ""), {})
        ticket = _repeat_ticket(
            rank=rank,
            outcome=outcome,
            audit_ticket=audit_ticket,
            opened_at=opened_at,
            marks=marks,
        )
        tickets.append(existing.get(ticket.ticket_id, ticket))
    return tuple(tickets)


def _winning_outcomes(path: Path) -> tuple[dict[str, str], ...]:
    ready_rows = [row for row in _read_rows(path) if row.get("checkpoint_status") == "ready"]
    rows = [row for row in ready_rows if row.get("outcome") == "paper_fill_audit_win" and _longer_ready_path_did_not_fail(row, ready_rows)]
    rows.sort(
        key=lambda row: (
            _horizon_priority(row.get("horizon", "")),
            _float(row.get("close_return_bps")),
            -abs(_float(row.get("max_adverse_bps"))),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    winners = []
    for row in rows:
        source = row.get("ticket_id", "")
        if source in seen:
            continue
        seen.add(source)
        winners.append(row)
    return tuple(winners)


def _longer_ready_path_did_not_fail(row: dict[str, str], ready_rows: list[dict[str, str]]) -> bool:
    source = row.get("ticket_id", "")
    horizon_priority = _horizon_priority(row.get("horizon", ""))
    for peer in ready_rows:
        if peer.get("ticket_id", "") != source:
            continue
        if _horizon_priority(peer.get("horizon", "")) <= horizon_priority:
            continue
        if peer.get("outcome") != "paper_fill_audit_win":
            return False
    return True


def _repeat_ticket(
    *,
    rank: int,
    outcome: dict[str, str],
    audit_ticket: dict[str, str],
    opened_at: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> PaperTicket:
    asset = outcome.get("asset", "")
    venue = outcome.get("venue", "")
    decision = outcome.get("decision", "")
    entry_mark, entry_source = _entry_mark(asset=asset, venue=venue, marks=marks)
    source_ticket_id = audit_ticket.get("source_ticket_id", "")
    ticket_id = f"broad-repeat-{_slug(asset)}-{_side(decision)}-{_slug(source_ticket_id or outcome.get('ticket_id', ''))}"
    return PaperTicket(
        ticket_id=ticket_id,
        opened_at=opened_at,
        rank=rank,
        opportunity=f"broad_fill_audit_repeat:{outcome.get('ticket_id', '')}:{outcome.get('horizon', '')}",
        probe_type="broad_alpha_repeat",
        status="fresh_repeat_after_fill_audit_win",
        side=_side(decision),
        asset=asset,
        venue=venue,
        candidate_size_usd=audit_ticket.get("candidate_size_usd", "100"),
        observation_horizon="15m,1h",
        checkpoints="15m,1h",
        entry_mark=entry_mark,
        entry_source=entry_source,
        decision=decision,
        required_record=(
            "fresh repeat mark move, spread/depth, funding, stop status, adverse excursion, "
            "and comparison against the prior fill-audit path"
        ),
        next_step=(
            f"repeat {asset} {decision} after {outcome.get('horizon', '')} fill-audit win; "
            "keep only if the fresh path survives cost and stop checks again"
        ),
    )


def _entry_mark(
    *,
    asset: str,
    venue: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    for key in ((venue.upper(), asset.upper()), ("HL", asset.upper()), ("OKX", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _existing_tickets(path: Path | None) -> tuple[PaperTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            PaperTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                rank=int(float(row.get("rank") or 0)),
                opportunity=row.get("opportunity", ""),
                probe_type=row.get("probe_type", ""),
                status=row.get("status", ""),
                side=row.get("side", ""),
                asset=row.get("asset", ""),
                venue=row.get("venue", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                observation_horizon=row.get("observation_horizon", ""),
                checkpoints=row.get("checkpoints", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                decision=row.get("decision", ""),
                required_record=row.get("required_record", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _horizon_priority(value: str) -> int:
    if value == "1h":
        return 2
    if value == "15m":
        return 1
    return 0


def _side(decision: str) -> str:
    if decision == "paper_short":
        return "short"
    return "long"


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "na"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_broad_alpha_fill_audit_outcomes.csv")
    parser.add_argument("--audit-tickets-path", type=Path, default=ROOT / "current_broad_alpha_fill_audit_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_broad_alpha_repeat_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_broad_alpha_repeat_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = parser.parse_args()

    rows = build_broad_alpha_repeat_tickets(
        outcomes_path=args.outcomes_path,
        audit_tickets_path=args.audit_tickets_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top=args.top,
    )
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
