from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HyperliquidDislocationPaperTicket:
    generated_at: str
    label_timestamp: str
    asset: str
    side: str
    status: str
    paper_notional_usd: float
    horizon: str
    gross_15m_bps: float
    conservative_cost_bps: float
    conservative_net_15m_bps: float
    spread_bps: float
    side_depth_10bps_notional: float
    visible_depth_usage_10bps: float
    entry_observation: str
    required_observations: str
    reject_if: str


def build_hyperliquid_dislocation_paper_tickets(
    *,
    input_path: Path = ROOT / "current_hyperliquid_dislocation_execution_check.csv",
) -> tuple[HyperliquidDislocationPaperTicket, ...]:
    generated_at = datetime.now(UTC).isoformat()
    rows = (
        row
        for row in _read_rows(input_path)
        if row.get("gate_action") == "paper_execution_probe"
        and _float(row.get("conservative_net_15m_bps")) > 0.0
    )
    selected = _smallest_probe_per_asset(rows)
    return tuple(
        HyperliquidDislocationPaperTicket(
            generated_at=generated_at,
            label_timestamp=row.get("label_timestamp", ""),
            asset=row.get("asset", ""),
            side=row.get("side", ""),
            status=row.get("status", ""),
            paper_notional_usd=_float(row.get("candidate_size_usd")),
            horizon="15m_observation_then_1h_confirmation",
            gross_15m_bps=_float(row.get("gross_15m_bps")),
            conservative_cost_bps=_float(row.get("conservative_cost_bps")),
            conservative_net_15m_bps=_float(row.get("conservative_net_15m_bps")),
            spread_bps=_float(row.get("spread_bps")),
            side_depth_10bps_notional=_float(row.get("side_depth_10bps_notional")),
            visible_depth_usage_10bps=_float(row.get("visible_depth_usage_10bps")),
            entry_observation="paper observe immediate market entry at the current public book",
            required_observations=(
                "mark/index move, funding drift, spread, side depth, adverse move, "
                "fresh snapshot persistence, and the next 1h label"
            ),
            reject_if=(
                "1h label fails, fresh snapshot no longer supports the lane, "
                "conservative net falls below zero, spread/depth worsens materially, "
                "or a stronger conflicting lane dominates the same asset"
            ),
        )
        for row in sorted(selected.values(), key=_ticket_sort_key, reverse=True)
    )


def write_hyperliquid_dislocation_paper_tickets_csv(
    tickets: tuple[HyperliquidDislocationPaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "generated_at",
                "label_timestamp",
                "asset",
                "side",
                "status",
                "paper_notional_usd",
                "horizon",
                "gross_15m_bps",
                "conservative_cost_bps",
                "conservative_net_15m_bps",
                "spread_bps",
                "side_depth_10bps_notional",
                "visible_depth_usage_10bps",
                "entry_observation",
                "required_observations",
                "reject_if",
            )
        )
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.generated_at,
                    ticket.label_timestamp,
                    ticket.asset,
                    ticket.side,
                    ticket.status,
                    f"{ticket.paper_notional_usd:.2f}",
                    ticket.horizon,
                    f"{ticket.gross_15m_bps:.8f}",
                    f"{ticket.conservative_cost_bps:.8f}",
                    f"{ticket.conservative_net_15m_bps:.8f}",
                    f"{ticket.spread_bps:.8f}",
                    f"{ticket.side_depth_10bps_notional:.8f}",
                    f"{ticket.visible_depth_usage_10bps:.8f}",
                    ticket.entry_observation,
                    ticket.required_observations,
                    ticket.reject_if,
                )
            )
    return output_path


def write_hyperliquid_dislocation_paper_tickets_md(
    tickets: tuple[HyperliquidDislocationPaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Paper Tickets\n\n")
        handle.write(
            "This is not a trade instruction. It converts current 15m-supported "
            "dislocation probes into paper observation and falsification tickets.\n\n"
        )
        handle.write(f"- generated tickets: `{len(tickets)}`\n\n")
        handle.write(
            "| asset | side | notional | horizon | gross15 | cost | net15 | spread | depth10 | usage |\n"
        )
        handle.write("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for ticket in tickets:
            handle.write(
                f"| {ticket.asset} | "
                f"{ticket.side} | "
                f"{ticket.paper_notional_usd:.0f} | "
                f"{ticket.horizon} | "
                f"{ticket.gross_15m_bps:.2f} | "
                f"{ticket.conservative_cost_bps:.2f} | "
                f"{ticket.conservative_net_15m_bps:.2f} | "
                f"{ticket.spread_bps:.2f} | "
                f"{ticket.side_depth_10bps_notional:.0f} | "
                f"{ticket.visible_depth_usage_10bps:.4f} |\n"
            )
        handle.write("\n## Required Observations\n\n")
        handle.write("- Mark/index move over the paper horizon.\n")
        handle.write("- Funding drift and whether the edge survives fees.\n")
        handle.write("- Spread, side depth, and visible-depth usage at observation time.\n")
        handle.write("- Fresh snapshot persistence before treating the lane as repeatable.\n")
        handle.write("- The next 1h label after the source snapshot matures.\n\n")
        handle.write("## Reject If\n\n")
        handle.write("- The 1h label fails.\n")
        handle.write("- A fresh snapshot no longer supports the lane.\n")
        handle.write("- Conservative net falls below zero.\n")
        handle.write("- Spread or depth worsens materially.\n")
        handle.write("- A stronger conflicting lane dominates the same asset.\n")
    return output_path


def _smallest_probe_per_asset(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        asset = row.get("asset", "")
        if not asset:
            continue
        previous = selected.get(asset)
        if previous is None or _float(row.get("candidate_size_usd")) < _float(previous.get("candidate_size_usd")):
            selected[asset] = row
    return selected


def _ticket_sort_key(row: dict[str, str]) -> tuple[float, float]:
    return (
        _float(row.get("conservative_net_15m_bps")),
        -_float(row.get("candidate_size_usd")),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_execution_check.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_paper_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_paper_tickets.md",
    )
    args = parser.parse_args()

    tickets = build_hyperliquid_dislocation_paper_tickets(input_path=args.input_path)
    write_hyperliquid_dislocation_paper_tickets_csv(tickets, output_path=args.output_path)
    write_hyperliquid_dislocation_paper_tickets_md(tickets, output_path=args.md_output_path)
    for ticket in tickets:
        print(
            ticket.asset,
            ticket.side,
            f"notional={ticket.paper_notional_usd:.0f}",
            f"net15={ticket.conservative_net_15m_bps:.2f}bps",
        )


if __name__ == "__main__":
    main()
