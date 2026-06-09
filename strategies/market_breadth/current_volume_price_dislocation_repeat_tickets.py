from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VolumePriceDislocationRepeatTicket:
    ticket_id: str
    opened_at: str
    rank: int
    opportunity: str
    probe_type: str
    status: str
    setup: str
    decision: str
    symbol: str
    name: str
    venue: str
    candidate_size_usd: str
    observation_horizon: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    conservative_net_4h_bps: str
    required_record: str
    next_step: str


def build_volume_price_dislocation_repeat_tickets(
    *,
    fill_risk_path: Path = ROOT / "current_volume_price_dislocation_fill_risk_check.csv",
    execution_gate_path: Path = ROOT / "current_volume_price_dislocation_execution_gate.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[VolumePriceDislocationRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    gate = {row.get("symbol", ""): row for row in _read_rows(execution_gate_path)}
    tickets: list[VolumePriceDislocationRepeatTicket] = []
    survivors = (
        row
        for row in _read_rows(fill_risk_path)
        if row.get("risk_action") == "cost_adjusted_probe_survived"
    )
    for rank, row in enumerate(survivors, start=1):
        symbol = row.get("symbol", "")
        context = gate.get(symbol, {})
        ticket_id = f"market-breadth-repeat-{_slug(symbol)}-{_slug(context.get('side', ''))}"
        if ticket_id in existing:
            tickets.append(existing[ticket_id])
            continue
        tickets.append(
            VolumePriceDislocationRepeatTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=f"volume_price_dislocation_repeat:{symbol}",
                probe_type="market_breadth_dislocation_second_repeat",
                status=row.get("risk_action", ""),
                setup=context.get("side", ""),
                decision=row.get("decision", ""),
                symbol=symbol,
                name=context.get("name", ""),
                venue=context.get("price_source", ""),
                candidate_size_usd=row.get("notional_usd", ""),
                observation_horizon="15m,1h,4h",
                checkpoints="15m,1h,4h",
                entry_mark=context.get("mark_price", ""),
                entry_source=context.get("price_source", ""),
                conservative_net_4h_bps=row.get("estimated_net_after_cost_bps", ""),
                required_record=(
                    "second repeat mark outcome, fill/slippage assumption, funding, depth, "
                    "stop behavior, and whether market-breadth trigger still exists"
                ),
                next_step=f"watch {symbol} second market-breadth repeat and record 15m, 1h, and 4h outcomes",
            )
        )
    return tuple(tickets)


def write_volume_price_dislocation_repeat_tickets_csv(
    rows: tuple[VolumePriceDislocationRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(VolumePriceDislocationRepeatTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_volume_price_dislocation_repeat_tickets_md(
    rows: tuple[VolumePriceDislocationRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volume Price Dislocation Repeat Tickets\n\n")
        handle.write(
            "These preserve entry marks for second-repeat market-breadth dislocation probes "
            "that survived rough fill-risk checks. They are not live orders.\n\n"
        )
        handle.write("| ticket | rank | symbol | setup | decision | notional | entry | net bps | checkpoints | next step |\n")
        handle.write("| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.rank} | {row.symbol} | {row.setup} | "
                f"{row.decision} | {row.candidate_size_usd} | {row.entry_mark} | "
                f"{row.conservative_net_4h_bps} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _existing_tickets(path: Path | None) -> tuple[VolumePriceDislocationRepeatTicket, ...]:
    if path is None:
        return ()
    return tuple(
        VolumePriceDislocationRepeatTicket(
            ticket_id=row.get("ticket_id", ""),
            opened_at=row.get("opened_at", ""),
            rank=_int(row.get("rank")),
            opportunity=row.get("opportunity", ""),
            probe_type=row.get("probe_type", ""),
            status=row.get("status", ""),
            setup=row.get("setup", ""),
            decision=row.get("decision", ""),
            symbol=row.get("symbol", ""),
            name=row.get("name", ""),
            venue=row.get("venue", ""),
            candidate_size_usd=row.get("candidate_size_usd", ""),
            observation_horizon=row.get("observation_horizon", ""),
            checkpoints=row.get("checkpoints", ""),
            entry_mark=row.get("entry_mark", ""),
            entry_source=row.get("entry_source", ""),
            conservative_net_4h_bps=row.get("conservative_net_4h_bps", ""),
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
    parser.add_argument(
        "--fill-risk-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_fill_risk_check.csv",
    )
    parser.add_argument(
        "--execution-gate-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_execution_gate.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_repeat_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_repeat_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_volume_price_dislocation_repeat_tickets(
        fill_risk_path=args.fill_risk_path,
        execution_gate_path=args.execution_gate_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_volume_price_dislocation_repeat_tickets_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.symbol, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
