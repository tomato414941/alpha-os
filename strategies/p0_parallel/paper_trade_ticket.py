from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FundingTicketCandidate:
    timestamp: str
    asset: str
    long_venue: str
    short_venue: str
    annualized_spread: float
    hl_day_notional_volume: float
    hl_impact_spread: float
    notes: str


def select_ticket_candidate(
    input_path: Path = ROOT / "cross_exchange_funding" / "current_funding_feasibility.csv",
) -> FundingTicketCandidate:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    viable_rows = [
        row
        for row in rows
        if row.get("hl_involved") == "True"
        and row.get("hl_day_notional_volume")
        and row.get("hl_impact_spread")
        and float(row["hl_day_notional_volume"]) >= 500_000.0
        and float(row["hl_impact_spread"]) <= 0.005
    ]
    if not viable_rows:
        raise RuntimeError("no current candidate passes minimal feasibility filters")
    row = max(viable_rows, key=lambda item: float(item["annualized_spread"]))
    return FundingTicketCandidate(
        timestamp=str(row["timestamp"]),
        asset=str(row["asset"]),
        long_venue=str(row["long_venue"]),
        short_venue=str(row["short_venue"]),
        annualized_spread=float(row["annualized_spread"]),
        hl_day_notional_volume=float(row["hl_day_notional_volume"]),
        hl_impact_spread=float(row["hl_impact_spread"]),
        notes=str(row["notes"]),
    )


def write_paper_trade_ticket(
    candidate: FundingTicketCandidate,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed_at = datetime.now(UTC).isoformat()
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Paper Trade Ticket\n\n")
        handle.write(f"Generated: {observed_at}\n\n")
        handle.write("This is not a trade instruction. It is an operational feasibility ticket.\n\n")
        handle.write("## Candidate\n\n")
        handle.write(f"- Asset: `{candidate.asset}`\n")
        handle.write(f"- Long venue: `{candidate.long_venue}`\n")
        handle.write(f"- Short venue: `{candidate.short_venue}`\n")
        handle.write(f"- Annualized spread snapshot: `{candidate.annualized_spread:.8f}`\n")
        handle.write(f"- Hyperliquid 24h notional volume: `{candidate.hl_day_notional_volume:.2f}`\n")
        handle.write(f"- Hyperliquid impact spread: `{candidate.hl_impact_spread:.8f}`\n")
        handle.write(f"- Source timestamp: `{candidate.timestamp}`\n")
        handle.write(f"- Notes: {candidate.notes}\n\n")
        handle.write("## Required Checks Before Any Real Order\n\n")
        handle.write("- Confirm both venues are accessible from the actual account and jurisdiction.\n")
        handle.write("- Confirm symbol availability, lot size, min notional, and leverage limits.\n")
        handle.write("- Confirm maker/taker fees and whether the spread survives taker execution.\n")
        handle.write("- Confirm borrow, margin, liquidation buffer, and funding interval timing.\n")
        handle.write("- Confirm depth for the intended notional on both legs.\n")
        handle.write("- Confirm that predicted funding still exists immediately before entry.\n")
        handle.write("- Define exit condition, max loss, and kill switch.\n\n")
        handle.write("## First Falsification\n\n")
        handle.write(
            "If this ticket cannot be converted into executable venue-specific order "
            "details with fees, size, and risk limits, this lane is not operational yet.\n"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_trade_ticket.md",
    )
    args = parser.parse_args()

    candidate = select_ticket_candidate()
    write_paper_trade_ticket(candidate, output_path=args.output_path)
    print(candidate.asset, candidate.long_venue, candidate.short_venue, candidate.annualized_spread)


if __name__ == "__main__":
    main()

