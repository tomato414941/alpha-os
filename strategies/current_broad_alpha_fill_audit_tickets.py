from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import requests

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent
OKX_BASE_URL = "https://www.okx.com"
DEFAULT_STOP_BPS = 50.0
DEFAULT_AUDIT_HORIZONS = "15m,1h"


@dataclass(frozen=True)
class BroadAlphaFillAuditTicket:
    ticket_id: str
    opened_at: str
    source_ticket_id: str
    opportunity: str
    asset: str
    venue: str
    decision: str
    side: str
    entry_mark: str
    entry_source: str
    candidate_size_usd: str
    stop_bps: float
    audit_horizons: str
    prior_net_after_cost_bps: str
    required_record: str
    next_step: str


def build_broad_alpha_fill_audit_tickets(
    *,
    fill_risk_path: Path = ROOT / "current_broad_alpha_paper_fill_risk_check.csv",
    broad_tickets_path: Path = ROOT / "current_broad_alpha_paper_tickets.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[BroadAlphaFillAuditTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    broad_tickets = {row.get("ticket_id", ""): row for row in _read_rows(broad_tickets_path)}
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
        hl_context_path=ROOT / "candidate_validation" / "current_followup_execution_context.csv",
        okx_context_path=ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    )
    marks.update(_okx_marks())
    rows = [
        row
        for row in _read_rows(fill_risk_path)
        if row.get("risk_action") == "cost_adjusted_paper_probe"
    ]
    tickets = tuple(
        _ticket_for_row(
            row=row,
            broad_ticket=broad_tickets.get(row.get("ticket_id", ""), {}),
            opened_at=opened_at,
            existing=existing,
            marks=marks,
        )
        for row in rows
    )
    active_ticket_ids = {ticket.ticket_id for ticket in tickets}
    carried_forward = tuple(ticket for ticket in existing.values() if ticket.ticket_id not in active_ticket_ids)
    return tickets + carried_forward


def write_broad_alpha_fill_audit_tickets_csv(
    rows: tuple[BroadAlphaFillAuditTicket, ...],
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
                "source_ticket_id",
                "opportunity",
                "asset",
                "venue",
                "decision",
                "side",
                "entry_mark",
                "entry_source",
                "candidate_size_usd",
                "stop_bps",
                "audit_horizons",
                "prior_net_after_cost_bps",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.source_ticket_id,
                    row.opportunity,
                    row.asset,
                    row.venue,
                    row.decision,
                    row.side,
                    row.entry_mark,
                    row.entry_source,
                    row.candidate_size_usd,
                    f"{row.stop_bps:.2f}",
                    row.audit_horizons,
                    row.prior_net_after_cost_bps,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_broad_alpha_fill_audit_tickets_md(
    rows: tuple[BroadAlphaFillAuditTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Broad Alpha Fill Audit Tickets\n\n")
        handle.write(
            "These are fresh paper fill-audit tickets for broad paper winners that survived rough cost/depth checks. "
            "They are not live trade instructions.\n\n"
        )
        handle.write("| ticket | source | asset | venue | side | entry | size USD | stop | horizons | prior net | next step |\n")
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.source_ticket_id} | "
                f"{row.asset} | "
                f"{row.venue} | "
                f"{row.side} | "
                f"{row.entry_mark} | "
                f"{row.candidate_size_usd} | "
                f"{row.stop_bps:.2f} | "
                f"{row.audit_horizons} | "
                f"{row.prior_net_after_cost_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _ticket_for_row(
    *,
    row: dict[str, str],
    broad_ticket: dict[str, str],
    opened_at: str,
    existing: dict[str, BroadAlphaFillAuditTicket],
    marks: dict[tuple[str, str], tuple[str, str]],
) -> BroadAlphaFillAuditTicket:
    asset = row.get("asset", "")
    venue = broad_ticket.get("venue", "")
    decision = row.get("decision", "")
    entry_mark, entry_source = _entry_mark(asset=asset, venue=venue, marks=marks)
    ticket = BroadAlphaFillAuditTicket(
        ticket_id=f"broad-fill-audit-{asset.lower()}-{_side(decision)}-50bps-stop",
        opened_at=opened_at,
        source_ticket_id=row.get("ticket_id", ""),
        opportunity=row.get("opportunity", ""),
        asset=asset,
        venue=venue,
        decision=decision,
        side=_side(decision),
        entry_mark=entry_mark,
        entry_source=entry_source,
        candidate_size_usd=row.get("candidate_size_usd", "") or broad_ticket.get("candidate_size_usd", ""),
        stop_bps=DEFAULT_STOP_BPS,
        audit_horizons=DEFAULT_AUDIT_HORIZONS,
        prior_net_after_cost_bps=row.get("estimated_net_after_cost_bps", ""),
        required_record="fresh entry, public 1m path, stop hit status, adverse excursion, funding and fill notes",
        next_step=f"audit fresh {asset} {decision} path with 50bps stop before any promotion",
    )
    prior = existing.get(ticket.ticket_id)
    if prior is None:
        return ticket
    return replace(
        ticket,
        opened_at=prior.opened_at,
        entry_mark=prior.entry_mark or ticket.entry_mark,
        entry_source=prior.entry_source or ticket.entry_source,
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


def _okx_marks() -> dict[tuple[str, str], tuple[str, str]]:
    try:
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/tickers",
            params={"instType": "SWAP"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return {}
    marks = {}
    for item in response.json().get("data", ()):
        inst_id = str(item.get("instId", ""))
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        asset = inst_id.removesuffix("-USDT-SWAP")
        mark = item.get("last", "")
        if asset and mark:
            marks[("OKX", asset)] = (str(mark), "okx_ticker")
    return marks


def _side(decision: str) -> str:
    if decision == "paper_short":
        return "short"
    if decision == "paper_long":
        return "long"
    return "observe"


def _existing_tickets(path: Path | None) -> tuple[BroadAlphaFillAuditTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        rows.append(
            BroadAlphaFillAuditTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                source_ticket_id=row.get("source_ticket_id", ""),
                opportunity=row.get("opportunity", ""),
                asset=row.get("asset", ""),
                venue=row.get("venue", ""),
                decision=row.get("decision", ""),
                side=row.get("side", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                stop_bps=float(row.get("stop_bps") or 0.0),
                audit_horizons=row.get("audit_horizons", ""),
                prior_net_after_cost_bps=row.get("prior_net_after_cost_bps", ""),
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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill-risk-path", type=Path, default=ROOT / "current_broad_alpha_paper_fill_risk_check.csv")
    parser.add_argument("--broad-tickets-path", type=Path, default=ROOT / "current_broad_alpha_paper_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_broad_alpha_fill_audit_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_broad_alpha_fill_audit_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_broad_alpha_fill_audit_tickets(
        fill_risk_path=args.fill_risk_path,
        broad_tickets_path=args.broad_tickets_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_broad_alpha_fill_audit_tickets_csv(rows, output_path=args.output_path)
    write_broad_alpha_fill_audit_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.venue, row.side, row.entry_mark)


if __name__ == "__main__":
    main()
