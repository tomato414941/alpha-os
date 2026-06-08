from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeeRepeatTicket:
    ticket_id: str
    opened_at: str
    rank: int
    opportunity: str
    probe_type: str
    status: str
    side: str
    asset: str
    venue: str
    candidate_size_usd: str
    observation_horizon: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    decision: str
    required_record: str
    next_step: str


def build_protocol_fee_repeat_tickets(
    *,
    risk_path: Path = ROOT / "current_protocol_fee_repeat_risk_check.csv",
    execution_context_path: Path = ROOT / "current_protocol_fee_execution_context.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[ProtocolFeeRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    execution_by_key = {
        (row.get("token_symbol", ""), row.get("protocol", "")): row
        for row in _read_rows(execution_context_path)
    }
    tickets: list[ProtocolFeeRepeatTicket] = []
    for rank, row in enumerate(_read_rows(risk_path), start=1):
        if row.get("risk_action") != "cost_adjusted_repeat_probe":
            continue
        token = row.get("token_symbol", "")
        protocol = row.get("protocol", "")
        ticket_id = f"protocol-fee-repeat-{_slug(token)}-{_slug(protocol)}"
        if ticket_id in existing:
            tickets.append(existing[ticket_id])
            continue
        context = execution_by_key.get((token, protocol), {})
        tickets.append(
            ProtocolFeeRepeatTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=f"protocol_fee_repeat:{token}/{protocol}",
                probe_type="protocol_fee_repeat_probe",
                status=row.get("risk_action", ""),
                side="long_token",
                asset=token,
                venue="HL",
                candidate_size_usd=row.get("paper_notional_usd", ""),
                observation_horizon="15m,1h,4h",
                checkpoints="15m,1h,4h",
                entry_mark=context.get("hl_mark_price", ""),
                entry_source="protocol_fee_execution_context" if context.get("hl_mark_price") else "",
                decision="paper_long",
                required_record=(
                    "repeat forward label, spread/funding/depth refresh, stop behavior, "
                    "and adverse excursion"
                ),
                next_step=f"watch {token}/{protocol} repeat probe and record 15m, 1h, and 4h outcomes",
            )
        )
    return tuple(tickets)


def write_protocol_fee_repeat_tickets_csv(
    rows: tuple[ProtocolFeeRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(ProtocolFeeRepeatTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_protocol_fee_repeat_tickets_md(
    rows: tuple[ProtocolFeeRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Repeat Tickets\n\n")
        handle.write(
            "These preserve entry marks for protocol-fee repeat probes that survived rough cost checks. "
            "They are not live orders.\n\n"
        )
        handle.write("| ticket | asset | opportunity | decision | notional | entry | checkpoints | next step |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.asset} | {_escape(row.opportunity)} | {row.decision} | "
                f"{row.candidate_size_usd} | {row.entry_mark} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _existing_tickets(path: Path | None) -> tuple[ProtocolFeeRepeatTicket, ...]:
    if path is None:
        return ()
    return tuple(
        ProtocolFeeRepeatTicket(
            ticket_id=row.get("ticket_id", ""),
            opened_at=row.get("opened_at", ""),
            rank=_int(row.get("rank")),
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
        for row in _read_rows(path)
        if row.get("ticket_id")
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0.0))
    except ValueError:
        return 0


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-path", type=Path, default=ROOT / "current_protocol_fee_repeat_risk_check.csv")
    parser.add_argument(
        "--execution-context-path",
        type=Path,
        default=ROOT / "current_protocol_fee_execution_context.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_repeat_tickets.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_repeat_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_protocol_fee_repeat_tickets(
        risk_path=args.risk_path,
        execution_context_path=args.execution_context_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_protocol_fee_repeat_tickets_csv(rows, output_path=args.output_path)
    write_protocol_fee_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
