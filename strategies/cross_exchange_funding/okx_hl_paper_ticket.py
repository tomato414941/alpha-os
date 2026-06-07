from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OkxHlPaperTicketCandidate:
    asset: str
    observations: int
    long_venue: str
    short_venue: str
    mean_annualized_spread: float
    mean_net_8h_proxy: float
    min_net_8h_proxy: float
    max_net_8h_proxy: float
    positive_net_8h_rate: float
    mean_net_24h_proxy: float
    mean_breakeven_hold_hours: float
    mean_capacity_proxy_notional: float
    paper_notional: float


def select_okx_hl_ticket_candidate(
    *,
    summary_path: Path = ROOT / "okx_hl_funding_persistence_summary.csv",
    max_paper_notional: float = 1_000.0,
) -> OkxHlPaperTicketCandidate:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    viable_rows = tuple(
        row
        for row in rows
        if float(row["positive_net_8h_rate"]) >= 1.0
        and float(row["mean_net_8h_proxy"]) > 0.0
        and float(row["mean_capacity_proxy_notional"]) > 0.0
    )
    if not viable_rows:
        raise RuntimeError("no OKX-Hyperliquid candidate passes paper-ticket filters")
    row = max(
        viable_rows,
        key=lambda item: (
            float(item["mean_net_8h_proxy"]),
            float(item["mean_capacity_proxy_notional"]),
        ),
    )
    capacity = float(row["mean_capacity_proxy_notional"])
    paper_notional = min(max_paper_notional, capacity * 0.01)
    return OkxHlPaperTicketCandidate(
        asset=str(row["asset"]),
        observations=int(row["observations"]),
        long_venue=str(row["dominant_long_venue"]),
        short_venue=str(row["dominant_short_venue"]),
        mean_annualized_spread=float(row["mean_annualized_spread"]),
        mean_net_8h_proxy=float(row["mean_net_8h_proxy"]),
        min_net_8h_proxy=float(row["min_net_8h_proxy"]),
        max_net_8h_proxy=float(row["max_net_8h_proxy"]),
        positive_net_8h_rate=float(row["positive_net_8h_rate"]),
        mean_net_24h_proxy=float(row["mean_net_24h_proxy"]),
        mean_breakeven_hold_hours=float(row["mean_breakeven_hold_hours"]),
        mean_capacity_proxy_notional=capacity,
        paper_notional=paper_notional,
    )


def write_okx_hl_paper_ticket(
    candidate: OkxHlPaperTicketCandidate,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Paper Ticket\n\n")
        handle.write(f"Generated: `{generated_at}`\n\n")
        handle.write("This is not a trade instruction. It is a paper feasibility ticket.\n\n")
        handle.write("## Candidate\n\n")
        handle.write(f"- Asset: `{candidate.asset}`\n")
        handle.write(f"- Long venue: `{candidate.long_venue}`\n")
        handle.write(f"- Short venue: `{candidate.short_venue}`\n")
        handle.write(f"- Persistence observations: `{candidate.observations}`\n")
        handle.write(f"- Positive 8h net rate: `{candidate.positive_net_8h_rate:.4f}`\n")
        handle.write(f"- Mean annualized spread: `{candidate.mean_annualized_spread:.8f}`\n")
        handle.write(f"- Mean 8h net proxy: `{candidate.mean_net_8h_proxy:.8f}`\n")
        handle.write(f"- Min 8h net proxy: `{candidate.min_net_8h_proxy:.8f}`\n")
        handle.write(f"- Max 8h net proxy: `{candidate.max_net_8h_proxy:.8f}`\n")
        handle.write(f"- Mean 24h net proxy: `{candidate.mean_net_24h_proxy:.8f}`\n")
        handle.write(
            f"- Mean breakeven holding time: `{candidate.mean_breakeven_hold_hours:.4f}` hours\n"
        )
        handle.write(
            f"- Mean capacity proxy notional: `{candidate.mean_capacity_proxy_notional:.2f}`\n"
        )
        handle.write(f"- Paper notional cap: `{candidate.paper_notional:.2f}` USDT\n\n")
        handle.write("## Paper Order Shape\n\n")
        handle.write(
            f"- Leg 1: open a long `{candidate.asset}` perp exposure on `{candidate.long_venue}`.\n"
        )
        handle.write(
            f"- Leg 2: open a short `{candidate.asset}` perp exposure on `{candidate.short_venue}`.\n"
        )
        handle.write("- Use equal notional on both legs.\n")
        handle.write("- Use paper/notional-only tracking until venue order constraints are verified.\n")
        handle.write("- Target notional is capped by the smaller of 1,000 USDT and 1% of capacity proxy.\n\n")
        handle.write("## Falsification Checks\n\n")
        handle.write("- Confirm OKX and Hyperliquid account access from the real trading environment.\n")
        handle.write("- Confirm exact instrument IDs, lot size, min notional, and leverage limits.\n")
        handle.write("- Confirm maker/taker fees and whether taker entry still leaves positive 8h net.\n")
        handle.write("- Confirm funding timestamp alignment on both venues.\n")
        handle.write("- Confirm that mark/index basis does not dominate expected funding capture.\n")
        handle.write("- Confirm margin, liquidation buffer, and collateral transfer path.\n")
        handle.write("- Define exit if net proxy turns negative or either leg cannot be adjusted.\n\n")
        handle.write("## Why This Candidate\n\n")
        handle.write(
            "This candidate survived the short persistence probe with positive 8h net "
            "proxy in every snapshot. That does not prove a real edge; it only makes "
            "it the first candidate worth converting from screen output into a "
            "venue-specific paper workflow.\n"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "okx_hl_funding_persistence_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "okx_hl_paper_ticket.md",
    )
    parser.add_argument("--max-paper-notional", type=float, default=1_000.0)
    args = parser.parse_args()

    candidate = select_okx_hl_ticket_candidate(
        summary_path=args.summary_path,
        max_paper_notional=args.max_paper_notional,
    )
    write_okx_hl_paper_ticket(candidate, output_path=args.output_path)
    print(
        candidate.asset,
        candidate.long_venue,
        candidate.short_venue,
        f"net8h={candidate.mean_net_8h_proxy:.8f}",
        f"paper_notional={candidate.paper_notional:.2f}",
    )


if __name__ == "__main__":
    main()
