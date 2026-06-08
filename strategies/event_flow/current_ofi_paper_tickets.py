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


def build_ofi_paper_tickets(
    *,
    survival_path: Path = EVENT_FLOW_ROOT / "current_ofi_execution_survival.csv",
    existing_tickets_path: Path | None = None,
    top: int = DEFAULT_TOP,
) -> tuple[PaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
        hl_context_path=ROOT / "candidate_validation" / "current_followup_execution_context.csv",
        okx_context_path=ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    )
    rows = _ticket_rows(survival_path)[:top]
    tickets = tuple(_ticket_for_row(row=row, rank=rank, opened_at=opened_at, marks=marks) for rank, row in enumerate(rows, start=1))
    return tuple(existing.get(ticket.ticket_id, ticket) for ticket in tickets)


def _ticket_rows(path: Path) -> tuple[dict[str, str], ...]:
    rows = [
        row
        for row in _read_rows(path)
        if row.get("status") in {"maker_ofi_survival_candidate", "short_horizon_maker_probe_only"}
        and row.get("action") in {"paper_long", "paper_short"}
    ]
    rows.sort(key=lambda row: _float(row.get("survival_score")), reverse=True)
    return tuple(rows)


def _ticket_for_row(
    *,
    row: dict[str, str],
    rank: int,
    opened_at: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> PaperTicket:
    asset = row.get("asset", "")
    decision = row.get("action", "")
    entry_mark, entry_source = _entry_mark(asset=asset, marks=marks)
    ticket_id = f"ofi-paper-{rank:02d}-{_slug(asset)}-{_side(decision)}"
    return PaperTicket(
        ticket_id=ticket_id,
        opened_at=opened_at,
        rank=rank,
        opportunity=f"ofi_execution_survival:{row.get('feature_route', '')}:{row.get('status', '')}",
        probe_type="ofi_execution_survival_probe",
        status=row.get("status", ""),
        side=_side(decision),
        asset=asset,
        venue="",
        candidate_size_usd="100",
        observation_horizon="5m/15m",
        checkpoints="5m,15m",
        entry_mark=entry_mark,
        entry_source=entry_source,
        decision=decision,
        required_record="5m/15m mark move, spread/depth, maker fill assumption, queue/cancel note, adverse selection",
        next_step=(
            f"paper-check {asset} OFI as a short-horizon execution-survival label; "
            "do not extend to 1h without a separate horizon rule"
        ),
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
    parser.add_argument("--survival-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_execution_survival.csv")
    parser.add_argument("--output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_paper_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_paper_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = parser.parse_args()

    rows = build_ofi_paper_tickets(
        survival_path=args.survival_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top=args.top,
    )
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
