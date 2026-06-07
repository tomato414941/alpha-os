from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationPaperTicket:
    asset: str
    action: str
    paper_notional_usd: float
    gross_continuation_bps: float
    conservative_cost_bps: float
    conservative_net_bps: float
    near_touch_depth_5bps: float
    visible_depth_usage: float
    direction: str


def select_liquidation_paper_ticket(
    *,
    paper_gate_path: Path = ROOT / "current_okx_liquidation_paper_gate.csv",
) -> LiquidationPaperTicket:
    rows = _small_paper_probe_rows(paper_gate_path)
    if not rows:
        raise RuntimeError("no liquidation paper-gate row passed")
    row = max(
        rows,
        key=lambda item: (
            float(item["conservative_net_bps"]),
            -float(item["visible_depth_usage"]),
            float(item["candidate_size_usd"]),
        ),
    )
    return LiquidationPaperTicket(
        asset=row["asset"],
        action=row["action"],
        paper_notional_usd=float(row["candidate_size_usd"]),
        gross_continuation_bps=float(row["gross_continuation_bps"]),
        conservative_cost_bps=float(row["conservative_cost_bps"]),
        conservative_net_bps=float(row["conservative_net_bps"]),
        near_touch_depth_5bps=float(row["near_touch_depth_5bps"]),
        visible_depth_usage=float(row["visible_depth_usage"]),
        direction=_direction_for_action(row["action"]),
    )


def write_liquidation_paper_ticket(
    ticket: LiquidationPaperTicket,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX Liquidation Paper Ticket\n\n")
        handle.write(f"Generated: `{generated_at}`\n\n")
        handle.write("This is not a trade instruction. It is a paper observation ticket.\n\n")
        handle.write("## Candidate\n\n")
        handle.write(f"- Asset: `{ticket.asset}`\n")
        handle.write("- Venue: `OKX USDT swap`\n")
        handle.write(f"- Action: `{ticket.action}`\n")
        handle.write(f"- Paper direction: `{ticket.direction}`\n")
        handle.write(f"- Paper notional: `{ticket.paper_notional_usd:.2f}` USDT\n")
        handle.write(f"- 15m gross continuation: `{ticket.gross_continuation_bps:.2f}` bps\n")
        handle.write(f"- Conservative cost proxy: `{ticket.conservative_cost_bps:.2f}` bps\n")
        handle.write(f"- Conservative net proxy: `{ticket.conservative_net_bps:.2f}` bps\n")
        handle.write(f"- Near-touch depth 5bps: `{ticket.near_touch_depth_5bps:.2f}` USDT\n")
        handle.write(f"- Visible depth usage: `{ticket.visible_depth_usage:.4f}`\n\n")
        handle.write("## Paper Observation Shape\n\n")
        handle.write("- Record the current mark/mid price at observation start.\n")
        handle.write("- Record the simulated entry side implied by the paper direction.\n")
        handle.write("- Record the 15m and 1h mark/mid price after the event timestamp.\n")
        handle.write("- Subtract the same fee, spread, and depth-impact proxy used by the gate.\n")
        handle.write("- Do not average into the paper position if the signal moves against the ticket.\n\n")
        handle.write("## Falsification Checks\n\n")
        handle.write("- Reject if the next fresh event does not reproduce the same action family.\n")
        handle.write("- Reject if visible near-touch depth drops below the paper notional / 0.25 rule.\n")
        handle.write("- Reject if live spread widens enough to consume the conservative net proxy.\n")
        handle.write("- Reject if funding or broader perp pressure points against the paper direction.\n")
        handle.write("- Reject if the 15m paper result is negative after the conservative cost proxy.\n\n")
        handle.write("## Why This Candidate\n\n")
        handle.write(
            "This candidate is the strongest current liquidation paper-gate row by "
            "conservative short-window net while staying under the visible-depth "
            "usage cap. It only proves that the signal is worth paper observation; "
            "it does not prove deployable alpha.\n"
        )
    return output_path


def _small_paper_probe_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") == "small_paper_probe"
        )


def _direction_for_action(action: str) -> str:
    if action.startswith("long_liquidation"):
        return "short"
    if action.startswith("short_liquidation"):
        return "long"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-gate-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_gate.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_ticket.md",
    )
    args = parser.parse_args()

    ticket = select_liquidation_paper_ticket(paper_gate_path=args.paper_gate_path)
    write_liquidation_paper_ticket(ticket, output_path=args.output_path)
    print(
        ticket.asset,
        ticket.action,
        ticket.direction,
        f"notional={ticket.paper_notional_usd:.2f}",
        f"net={ticket.conservative_net_bps:.2f}bps",
    )


if __name__ == "__main__":
    main()
