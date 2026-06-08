from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeePaperTicket:
    token_symbol: str
    protocol: str
    side: str
    thesis_score: float
    paper_notional_usd: float
    price_change_7d: float
    fee_growth_7d: float
    hl_annualized_funding: float
    hl_spread_bps: float
    hl_near_depth_10bps_notional: float
    hl_visible_depth_usage: float
    observation_horizons: str
    hypothesis: str
    falsification: str
    next_step: str


def build_protocol_fee_paper_tickets(
    *,
    execution_context_path: Path = ROOT / "current_protocol_fee_execution_context.csv",
    paper_notional_usd: float = 1_000.0,
) -> tuple[ProtocolFeePaperTicket, ...]:
    rows = tuple(
        row
        for row in _read_rows(execution_context_path)
        if row.get("action") == "paper_observation_ready"
    )
    tickets = tuple(_build_ticket(row=row, paper_notional_usd=paper_notional_usd) for row in rows)
    return tuple(sorted(tickets, key=lambda ticket: ticket.thesis_score, reverse=True))


def write_protocol_fee_paper_tickets_csv(
    tickets: tuple[ProtocolFeePaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "token_symbol",
                "protocol",
                "side",
                "thesis_score",
                "paper_notional_usd",
                "price_change_7d",
                "fee_growth_7d",
                "hl_annualized_funding",
                "hl_spread_bps",
                "hl_near_depth_10bps_notional",
                "hl_visible_depth_usage",
                "observation_horizons",
                "hypothesis",
                "falsification",
                "next_step",
            )
        )
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.token_symbol,
                    ticket.protocol,
                    ticket.side,
                    f"{ticket.thesis_score:.8f}",
                    f"{ticket.paper_notional_usd:.2f}",
                    f"{ticket.price_change_7d:.8f}",
                    f"{ticket.fee_growth_7d:.8f}",
                    f"{ticket.hl_annualized_funding:.8f}",
                    f"{ticket.hl_spread_bps:.8f}",
                    f"{ticket.hl_near_depth_10bps_notional:.8f}",
                    f"{ticket.hl_visible_depth_usage:.8f}",
                    ticket.observation_horizons,
                    ticket.hypothesis,
                    ticket.falsification,
                    ticket.next_step,
                )
            )
    return output_path


def write_protocol_fee_paper_tickets_md(
    tickets: tuple[ProtocolFeePaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Paper Tickets\n\n")
        handle.write(
            "This turns protocol fee-growth lag candidates that pass the current "
            "execution context gate into paper observation tickets. It is not a "
            "live trade instruction.\n\n"
        )
        handle.write(
            "| token | protocol | side | score | notional | price 7d | fee growth 7d | funding | spread bps | depth 10bps USD | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for ticket in tickets:
            handle.write(
                "| "
                f"{ticket.token_symbol} | "
                f"{ticket.protocol} | "
                f"{ticket.side} | "
                f"{ticket.thesis_score:.4f} | "
                f"{ticket.paper_notional_usd:.0f} | "
                f"{ticket.price_change_7d:.2f} | "
                f"{ticket.fee_growth_7d:.2f} | "
                f"{ticket.hl_annualized_funding:.4f} | "
                f"{ticket.hl_spread_bps:.4f} | "
                f"{ticket.hl_near_depth_10bps_notional:.0f} | "
                f"{ticket.next_step} |\n"
            )
        handle.write("\n## Falsification\n\n")
        for ticket in tickets:
            handle.write(f"- {ticket.token_symbol}: {ticket.falsification}\n")
    return output_path


def _build_ticket(*, row: dict[str, str], paper_notional_usd: float) -> ProtocolFeePaperTicket:
    token = row.get("token_symbol", "")
    horizons = "4h,12h,24h"
    return ProtocolFeePaperTicket(
        token_symbol=token,
        protocol=row.get("protocol", ""),
        side=row.get("side", ""),
        thesis_score=_float(row.get("thesis_score")),
        paper_notional_usd=paper_notional_usd,
        price_change_7d=_float(row.get("price_change_7d")),
        fee_growth_7d=_float(row.get("fee_growth_7d")),
        hl_annualized_funding=_float(row.get("hl_annualized_funding")),
        hl_spread_bps=_float(row.get("hl_spread_bps")),
        hl_near_depth_10bps_notional=_float(row.get("hl_near_depth_10bps_notional")),
        hl_visible_depth_usage=_float(row.get("hl_visible_depth_usage_1k")),
        observation_horizons=horizons,
        hypothesis=(
            f"{token} fee growth is strong while the token is still weak on the week; "
            "a small long paper observation should beat rough funding/spread context."
        ),
        falsification=(
            f"deprioritize {token} if 4h and 12h directional labels fail, "
            "or if fresh venue context becomes thin, wide, or unavailable."
        ),
        next_step=(
            f"start {token} paper observation now and label {horizons} returns with "
            "funding, spread, and depth context"
        ),
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
        "--execution-context-path",
        type=Path,
        default=ROOT / "current_protocol_fee_execution_context.csv",
    )
    parser.add_argument("--paper-notional-usd", type=float, default=1_000.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_paper_tickets.md",
    )
    args = parser.parse_args()

    tickets = build_protocol_fee_paper_tickets(
        execution_context_path=args.execution_context_path,
        paper_notional_usd=args.paper_notional_usd,
    )
    write_protocol_fee_paper_tickets_csv(tickets, output_path=args.output_path)
    write_protocol_fee_paper_tickets_md(tickets, output_path=args.markdown_output_path)
    for ticket in tickets:
        print(
            ticket.token_symbol,
            ticket.side,
            f"score={ticket.thesis_score:.4f}",
            f"notional={ticket.paper_notional_usd:.0f}",
            ticket.next_step,
        )


if __name__ == "__main__":
    main()
