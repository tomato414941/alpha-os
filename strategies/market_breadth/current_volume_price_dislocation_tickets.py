from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VolumePriceDislocationTicket:
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


def build_volume_price_dislocation_tickets(
    *,
    execution_gate_path: Path = ROOT / "current_volume_price_dislocation_execution_gate.csv",
    existing_tickets_path: Path | None = None,
    limit: int = 8,
) -> tuple[VolumePriceDislocationTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    tickets: list[VolumePriceDislocationTicket] = []
    candidates = (
        row
        for row in _read_rows(execution_gate_path)
        if row.get("action") == "paper_execution_probe"
        and _float(row.get("conservative_net_4h_bps")) > 0.0
    )
    for rank, row in enumerate(candidates, start=1):
        if rank > limit:
            break
        symbol = row.get("symbol", "")
        ticket_id = f"market-breadth-{_slug(symbol)}-{_slug(row.get('side', ''))}"
        if ticket_id in existing:
            tickets.append(existing[ticket_id])
            continue
        tickets.append(
            VolumePriceDislocationTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=f"volume_price_dislocation:{symbol}",
                probe_type="market_breadth_dislocation_repeat",
                status=row.get("label_status", ""),
                setup=row.get("side", ""),
                decision=_decision(row.get("side", "")),
                symbol=symbol,
                name=row.get("name", ""),
                venue=row.get("price_source", ""),
                candidate_size_usd="250",
                observation_horizon="15m,1h,4h",
                checkpoints="15m,1h,4h",
                entry_mark=row.get("mark_price", ""),
                entry_source=row.get("price_source", ""),
                conservative_net_4h_bps=row.get("conservative_net_4h_bps", ""),
                required_record=(
                    "repeat mark outcome, realized fill/slippage assumption, funding, depth, "
                    "stop behavior, and whether the cross-asset dislocation repeats"
                ),
                next_step=f"watch {symbol} market-breadth paper probe and record 15m, 1h, and 4h outcomes",
            )
        )
    return tuple(tickets)


def write_volume_price_dislocation_tickets_csv(
    rows: tuple[VolumePriceDislocationTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(VolumePriceDislocationTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_volume_price_dislocation_tickets_md(
    rows: tuple[VolumePriceDislocationTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volume Price Dislocation Tickets\n\n")
        handle.write(
            "These preserve entry marks for market-breadth dislocation paper probes. "
            "They are not live orders and not a reusable strategy abstraction.\n\n"
        )
        handle.write("| ticket | rank | symbol | setup | decision | notional | entry | net 4h bps | checkpoints | next step |\n")
        handle.write("| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.rank} | {row.symbol} | {row.setup} | "
                f"{row.decision} | {row.candidate_size_usd} | {row.entry_mark} | "
                f"{row.conservative_net_4h_bps} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _existing_tickets(path: Path | None) -> tuple[VolumePriceDislocationTicket, ...]:
    if path is None:
        return ()
    return tuple(
        VolumePriceDislocationTicket(
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


def _decision(setup: str) -> str:
    if setup.startswith("long_"):
        return "paper_long"
    return "paper_observe"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


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
        "--execution-gate-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_execution_gate.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_tickets.md",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_volume_price_dislocation_tickets(
        execution_gate_path=args.execution_gate_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        limit=args.limit,
    )
    write_volume_price_dislocation_tickets_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.symbol, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
