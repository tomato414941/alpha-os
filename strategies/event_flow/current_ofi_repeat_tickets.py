from __future__ import annotations

import argparse
import csv
import re
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import PaperTicket, _load_marks, write_paper_tickets_csv, write_paper_tickets_md


ROOT = Path(__file__).resolve().parents[1]
EVENT_FLOW_ROOT = ROOT / "event_flow"
DEFAULT_TOP = 8


def build_ofi_repeat_tickets(
    *,
    outcomes_path: Path = EVENT_FLOW_ROOT / "current_ofi_fill_audit_outcomes.csv",
    audit_tickets_path: Path = EVENT_FLOW_ROOT / "current_ofi_fill_audit_tickets.csv",
    existing_tickets_path: Path | None = None,
    ticket_prefix: str = "ofi-repeat",
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
    tickets = tuple(
        _repeat_ticket(
            rank=rank,
            outcome=outcome,
            audit_ticket=audit_tickets.get(outcome.get("ticket_id", ""), {}),
            opened_at=opened_at,
            marks=marks,
            ticket_prefix=ticket_prefix,
        )
        for rank, outcome in enumerate(winners, start=1)
    )
    return tuple(existing.get(ticket.ticket_id, ticket) for ticket in tickets)


def _winning_outcomes(path: Path) -> tuple[dict[str, str], ...]:
    rows = [
        row
        for row in _read_rows(path)
        if row.get("checkpoint_status") == "ready" and row.get("outcome") == "paper_fill_audit_win"
    ]
    rows.sort(key=lambda row: (_horizon_priority(row.get("horizon", "")), _float(row.get("close_return_bps"))), reverse=True)
    seen: set[str] = set()
    winners = []
    for row in rows:
        source = row.get("ticket_id", "")
        if source in seen:
            continue
        seen.add(source)
        winners.append(row)
    return tuple(winners)


def _repeat_ticket(
    *,
    rank: int,
    outcome: dict[str, str],
    audit_ticket: dict[str, str],
    opened_at: str,
    marks: dict[tuple[str, str], tuple[str, str]],
    ticket_prefix: str,
) -> PaperTicket:
    asset = outcome.get("asset", "")
    decision = outcome.get("decision", "")
    entry_mark, entry_source = _entry_mark(asset=asset, marks=marks)
    source_ticket_id = audit_ticket.get("source_ticket_id", "")
    ticket_id = f"{ticket_prefix}-{rank:02d}-{_slug(asset)}-{_side(decision)}"
    return PaperTicket(
        ticket_id=ticket_id,
        opened_at=opened_at,
        rank=rank,
        opportunity=f"ofi_execution_survival_repeat:{outcome.get('ticket_id', '')}:{outcome.get('horizon', '')}:{source_ticket_id}",
        probe_type="ofi_execution_survival_repeat",
        status="fresh_repeat_after_ofi_fill_audit_win",
        side=_side(decision),
        asset=asset,
        venue=outcome.get("venue", ""),
        candidate_size_usd=audit_ticket.get("candidate_size_usd", "100"),
        observation_horizon="5m/15m",
        checkpoints="5m,15m",
        entry_mark=entry_mark,
        entry_source=entry_source,
        decision=decision,
        required_record="fresh 5m/15m mark move, cost, stop status, queue/cancel note, adverse selection",
        next_step=f"repeat OFI {asset} {decision} after 15m fill-audit win; require another cost and stop-surviving label",
    )


def _entry_mark(*, asset: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _existing_tickets(path: Path | None) -> tuple[PaperTicket, ...]:
    if path is None or not path.exists():
        return ()
    rows = []
    for row in _read_rows(path):
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
    if value == "15m":
        return 2
    if value == "5m":
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
    parser.add_argument("--outcomes-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_fill_audit_outcomes.csv")
    parser.add_argument("--audit-tickets-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_fill_audit_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_repeat_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_repeat_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    parser.add_argument("--ticket-prefix", default="ofi-repeat")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = parser.parse_args()

    rows = build_ofi_repeat_tickets(
        outcomes_path=args.outcomes_path,
        audit_tickets_path=args.audit_tickets_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        ticket_prefix=args.ticket_prefix,
        top=args.top,
    )
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
