from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class VolumePriceDislocationOutcome:
    ticket_id: str
    opened_at: str
    checked_at: str
    elapsed_minutes: str
    checkpoint_status: str
    opportunity: str
    decision: str
    symbol: str
    venue: str
    entry_mark: str
    current_mark: str
    current_source: str
    raw_return_bps: str
    directional_return_bps: str
    outcome: str
    missing_evidence: str
    next_step: str


def build_volume_price_dislocation_outcomes(
    *,
    tickets_path: Path = ROOT / "current_volume_price_dislocation_tickets.csv",
    hyperliquid_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    minimum_elapsed_minutes: float = 15.0,
) -> tuple[VolumePriceDislocationOutcome, ...]:
    checked_at = datetime.now(UTC)
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    return tuple(
        _outcome_for_ticket(
            ticket=row,
            checked_at=checked_at,
            marks=marks,
            minimum_elapsed_minutes=minimum_elapsed_minutes,
        )
        for row in _read_rows(tickets_path)
    )


def write_volume_price_dislocation_outcomes_csv(
    rows: tuple[VolumePriceDislocationOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(VolumePriceDislocationOutcome.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_volume_price_dislocation_outcomes_md(
    rows: tuple[VolumePriceDislocationOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volume Price Dislocation Outcomes\n\n")
        handle.write(
            "These mark open market-breadth dislocation paper probes against current prices. "
            "They are paper observations, not realized fills.\n\n"
        )
        handle.write("| ticket | symbol | status | elapsed | entry | current | directional bps | outcome | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.symbol} | {row.checkpoint_status} | "
                f"{row.elapsed_minutes} | {row.entry_mark} | {row.current_mark} | "
                f"{row.directional_return_bps} | {row.outcome} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _outcome_for_ticket(
    *,
    ticket: dict[str, str],
    checked_at: datetime,
    marks: dict[tuple[str, str], tuple[str, str]],
    minimum_elapsed_minutes: float,
) -> VolumePriceDislocationOutcome:
    opened_at = _datetime(ticket.get("opened_at", ""))
    elapsed_minutes = (checked_at - opened_at).total_seconds() / 60.0 if opened_at else 0.0
    symbol = ticket.get("symbol", "")
    current_mark, current_source = _current_mark(symbol=symbol, venue=ticket.get("venue", ""), marks=marks)
    ready = elapsed_minutes >= minimum_elapsed_minutes
    raw_return_bps = _raw_return_bps(entry=ticket.get("entry_mark", ""), current=current_mark) if ready else None
    directional_return_bps = _directional_return_bps(
        raw_return_bps=raw_return_bps,
        decision=ticket.get("decision", ""),
    )
    return VolumePriceDislocationOutcome(
        ticket_id=ticket.get("ticket_id", ""),
        opened_at=ticket.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        elapsed_minutes=f"{elapsed_minutes:.2f}",
        checkpoint_status="ready" if ready else "pending",
        opportunity=ticket.get("opportunity", ""),
        decision=ticket.get("decision", ""),
        symbol=symbol,
        venue=ticket.get("venue", ""),
        entry_mark=ticket.get("entry_mark", ""),
        current_mark=current_mark,
        current_source=current_source,
        raw_return_bps="" if raw_return_bps is None else f"{raw_return_bps:.8f}",
        directional_return_bps="" if directional_return_bps is None else f"{directional_return_bps:.8f}",
        outcome=_outcome(directional_return_bps=directional_return_bps, ready=ready),
        missing_evidence="" if ready else "checkpoint has not matured",
        next_step=_next_step(
            symbol=symbol,
            outcome=_outcome(directional_return_bps=directional_return_bps, ready=ready),
            ready=ready,
        ),
    )


def _current_mark(
    *,
    symbol: str,
    venue: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    for key in ((venue.upper(), symbol.upper()), ("HL", symbol.upper()), ("", symbol.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _raw_return_bps(*, entry: str, current: str) -> float:
    entry_value = _float(entry)
    current_value = _float(current)
    if entry_value <= 0.0 or current_value <= 0.0:
        return 0.0
    return (current_value / entry_value - 1.0) * 10000.0


def _directional_return_bps(*, raw_return_bps: float | None, decision: str) -> float | None:
    if raw_return_bps is None:
        return None
    if decision == "paper_short":
        return -raw_return_bps
    if decision == "paper_long":
        return raw_return_bps
    return 0.0


def _outcome(*, directional_return_bps: float | None, ready: bool) -> str:
    if not ready:
        return "pending"
    if directional_return_bps is None:
        return "missing_mark"
    if directional_return_bps > 0.0:
        return "paper_mark_win"
    if directional_return_bps < 0.0:
        return "paper_mark_loss"
    return "paper_mark_flat"


def _next_step(*, symbol: str, outcome: str, ready: bool) -> str:
    if not ready:
        return f"wait for {symbol} first checkpoint and refresh marks"
    if outcome == "paper_mark_win":
        return f"refresh {symbol} fill/funding/depth and test whether the setup repeats"
    return f"do not promote {symbol} without another fresh positive dislocation label"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickets-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_outcomes.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_outcomes.md",
    )
    parser.add_argument("--minimum-elapsed-minutes", type=float, default=15.0)
    args = parser.parse_args()

    rows = build_volume_price_dislocation_outcomes(
        tickets_path=args.tickets_path,
        minimum_elapsed_minutes=args.minimum_elapsed_minutes,
    )
    write_volume_price_dislocation_outcomes_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.checkpoint_status, row.outcome, row.directional_return_bps)


if __name__ == "__main__":
    main()
