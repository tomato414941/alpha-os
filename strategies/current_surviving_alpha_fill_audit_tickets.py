from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP = 3
DEFAULT_STOP_BPS = 50.0
DEFAULT_AUDIT_HORIZONS = "15m,1h"


@dataclass(frozen=True)
class SurvivingAlphaFillAuditTicket:
    ticket_id: str
    opened_at: str
    work_id: str
    source_ticket_id: str
    asset: str
    decision: str
    side: str
    entry_mark: str
    entry_source: str
    candidate_size_usd: str
    stop_bps: float
    audit_horizons: str
    prior_second_net_after_cost_bps: str
    required_record: str
    next_step: str


def build_surviving_alpha_fill_audit_tickets(
    *,
    path_risk_path: Path = ROOT / "current_surviving_alpha_path_risk.csv",
    existing_tickets_path: Path | None = None,
    second_tickets_path: Path = ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    top: int = DEFAULT_TOP,
) -> tuple[SurvivingAlphaFillAuditTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    second_tickets = {row.get("ticket_id", ""): row for row in _read_rows(second_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = tuple(
        row for row in _read_rows(path_risk_path) if row.get("path_action") == "path_survived_paper_stop"
    )[:top]
    return tuple(
        _ticket_for_path_row(
            row=row,
            opened_at=opened_at,
            existing=existing,
            second_ticket=second_tickets.get(row.get("ticket_id", ""), {}),
            marks=marks,
        )
        for row in rows
    )


def write_surviving_alpha_fill_audit_tickets_csv(
    rows: tuple[SurvivingAlphaFillAuditTicket, ...],
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
                "work_id",
                "source_ticket_id",
                "asset",
                "decision",
                "side",
                "entry_mark",
                "entry_source",
                "candidate_size_usd",
                "stop_bps",
                "audit_horizons",
                "prior_second_net_after_cost_bps",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.work_id,
                    row.source_ticket_id,
                    row.asset,
                    row.decision,
                    row.side,
                    row.entry_mark,
                    row.entry_source,
                    row.candidate_size_usd,
                    f"{row.stop_bps:.2f}",
                    row.audit_horizons,
                    row.prior_second_net_after_cost_bps,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_surviving_alpha_fill_audit_tickets_md(
    rows: tuple[SurvivingAlphaFillAuditTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Fill Audit Tickets\n\n")
        handle.write(
            "These are fresh paper fill-audit tickets for second-repeat survivors whose public candle "
            "path already survived the paper stop review. They are not live trade instructions.\n\n"
        )
        handle.write(
            "| ticket | asset | side | entry | source | size USD | stop bps | horizons | prior net | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.entry_mark} | "
                f"{row.entry_source} | "
                f"{row.candidate_size_usd} | "
                f"{row.stop_bps:.2f} | "
                f"{row.audit_horizons} | "
                f"{row.prior_second_net_after_cost_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _ticket_for_path_row(
    *,
    row: dict[str, str],
    opened_at: str,
    existing: dict[str, SurvivingAlphaFillAuditTicket],
    second_ticket: dict[str, str],
    marks: dict[tuple[str, str], tuple[str, str]],
) -> SurvivingAlphaFillAuditTicket:
    asset = row.get("asset", "")
    decision = row.get("decision", "")
    ticket_id = _ticket_id(row)
    entry_mark, entry_source = _entry_mark(asset=asset, marks=marks)
    ticket = SurvivingAlphaFillAuditTicket(
        ticket_id=ticket_id,
        opened_at=opened_at,
        work_id=row.get("work_id", ""),
        source_ticket_id=row.get("ticket_id", ""),
        asset=asset,
        decision=decision,
        side=_side(decision),
        entry_mark=entry_mark,
        entry_source=entry_source,
        candidate_size_usd=second_ticket.get("candidate_size_usd", "") or "100.00",
        stop_bps=DEFAULT_STOP_BPS,
        audit_horizons=DEFAULT_AUDIT_HORIZONS,
        prior_second_net_after_cost_bps=row.get("second_net_after_cost_bps", ""),
        required_record="fresh paper entry, 15m/1h public path, stop hit status, adverse excursion, funding and fill notes",
        next_step=f"audit fresh {asset} {decision} path with 50bps stop before any promotion",
    )
    prior = existing.get(ticket_id)
    if prior is None:
        return ticket
    return replace(
        ticket,
        opened_at=prior.opened_at,
        entry_mark=prior.entry_mark or ticket.entry_mark,
        entry_source=prior.entry_source or ticket.entry_source,
    )


def _ticket_id(row: dict[str, str]) -> str:
    asset = row.get("asset", "").lower()
    side = _side(row.get("decision", ""))
    return f"fill-audit-{asset}-{side}-50bps-stop"


def _side(decision: str) -> str:
    if decision == "paper_short":
        return "short"
    if decision == "paper_long":
        return "long"
    return "observe"


def _entry_mark(*, asset: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", asset.upper()), ("", asset.upper()), ("OKX", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _existing_tickets(path: Path | None) -> tuple[SurvivingAlphaFillAuditTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        rows.append(
            SurvivingAlphaFillAuditTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                work_id=row.get("work_id", ""),
                source_ticket_id=row.get("source_ticket_id", ""),
                asset=row.get("asset", ""),
                decision=row.get("decision", ""),
                side=row.get("side", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                stop_bps=float(row.get("stop_bps") or 0.0),
                audit_horizons=row.get("audit_horizons", ""),
                prior_second_net_after_cost_bps=row.get("prior_second_net_after_cost_bps", ""),
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
    parser.add_argument("--path-risk-path", type=Path, default=ROOT / "current_surviving_alpha_path_risk.csv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_fill_audit_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_fill_audit_tickets.md",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_surviving_alpha_fill_audit_tickets(
        path_risk_path=args.path_risk_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top=args.top,
    )
    write_surviving_alpha_fill_audit_tickets_csv(rows, output_path=args.output_path)
    write_surviving_alpha_fill_audit_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.side, row.entry_mark, row.audit_horizons)


if __name__ == "__main__":
    main()
