from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TokenUnlockPaperTicket:
    symbol: str
    name: str
    side: str
    days_until: int
    unlock_value_usd: float
    percent_supply: float
    annualized_funding: float
    day_notional_volume: float
    open_interest_notional: float
    impact_spread: float
    max_leverage: float
    score: float
    status: str
    reason: str


def build_paper_tickets(
    *,
    join_path: Path,
    hyperliquid_path: Path,
) -> tuple[TokenUnlockPaperTicket, ...]:
    market_by_asset = {row["asset"]: row for row in _read_rows(hyperliquid_path)}
    tickets: list[TokenUnlockPaperTicket] = []
    for row in _read_rows(join_path):
        market = market_by_asset.get(row["symbol"])
        if market is None:
            continue
        funding = _float(row["annualized_funding"])
        percent_supply = _float(row["percent_supply"])
        volume = _float(row["day_notional_volume"])
        open_interest = _float(row["open_interest_notional"])
        impact_spread = _float(market.get("impact_spread", ""))
        side, status, reason = _side_status_reason(
            unlock_action=row["unlock_action"],
            funding=funding,
            volume=volume,
            impact_spread=impact_spread,
        )
        tickets.append(
            TokenUnlockPaperTicket(
                symbol=row["symbol"],
                name=row["name"],
                side=side,
                days_until=int(float(row["days_until"])),
                unlock_value_usd=_float(row["unlock_value_usd"]),
                percent_supply=percent_supply,
                annualized_funding=funding,
                day_notional_volume=volume,
                open_interest_notional=open_interest,
                impact_spread=impact_spread,
                max_leverage=_float(market.get("max_leverage", "")),
                score=_score(
                    percent_supply=percent_supply,
                    unlock_value_usd=_float(row["unlock_value_usd"]),
                    annualized_funding=funding,
                    volume=volume,
                    impact_spread=impact_spread,
                    days_until=int(float(row["days_until"])),
                    status=status,
                ),
                status=status,
                reason=reason,
            )
        )
    return tuple(sorted(tickets, key=lambda ticket: ticket.score, reverse=True))


def write_tickets_csv(
    tickets: tuple[TokenUnlockPaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "name",
                "side",
                "days_until",
                "unlock_value_usd",
                "percent_supply",
                "annualized_funding",
                "day_notional_volume",
                "open_interest_notional",
                "impact_spread",
                "max_leverage",
                "score",
                "status",
                "reason",
            )
        )
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.symbol,
                    ticket.name,
                    ticket.side,
                    ticket.days_until,
                    f"{ticket.unlock_value_usd:.2f}",
                    f"{ticket.percent_supply:.4f}",
                    f"{ticket.annualized_funding:.8f}",
                    f"{ticket.day_notional_volume:.8f}",
                    f"{ticket.open_interest_notional:.8f}",
                    f"{ticket.impact_spread:.12f}",
                    f"{ticket.max_leverage:.4f}",
                    f"{ticket.score:.8f}",
                    ticket.status,
                    ticket.reason,
                )
            )
    return output_path


def write_tickets_md(
    tickets: tuple[TokenUnlockPaperTicket, ...],
    *,
    output_path: Path,
    top: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Paper Tickets\n\n")
        handle.write(
            "This converts current token unlock/perp overlaps into paper tickets. "
            "It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for ticket in tickets[:top]:
            handle.write(
                f"| {ticket.symbol} | {ticket.side} | {ticket.days_until} | "
                f"{ticket.unlock_value_usd:.0f} | {ticket.percent_supply:.4f} | "
                f"{ticket.annualized_funding:.8f} | {ticket.day_notional_volume:.0f} | "
                f"{ticket.impact_spread:.8f} | {ticket.max_leverage:.1f} | "
                f"{ticket.score:.6f} | {ticket.status} | {ticket.reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. "
            "A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.\n"
        )
    return output_path


def _side_status_reason(
    *,
    unlock_action: str,
    funding: float,
    volume: float,
    impact_spread: float,
) -> tuple[str, str, str]:
    if unlock_action not in {"unlock_supply_shock_watch", "large_unlock_watch"}:
        return "none", "context_only", "unlock is not large enough for a direct supply-shock ticket"
    if funding < 0.0:
        return "watch_squeeze", "crowded_short_risk", "supply shock overlaps negative funding, so new shorts may be crowded"
    if volume < 100_000.0:
        return "short", "too_illiquid", "perp venue volume is too low for paper priority"
    if impact_spread > 0.003:
        return "short", "wide_impact_watch", "short carry aligns, but visible impact spread is wide"
    return "short", "paper_short_candidate", "supply shock and short carry align on a tradable perp venue"


def _score(
    *,
    percent_supply: float,
    unlock_value_usd: float,
    annualized_funding: float,
    volume: float,
    impact_spread: float,
    days_until: int,
    status: str,
) -> float:
    urgency = max(0.0, 30.0 - float(days_until)) / 30.0
    liquidity = min(volume / 10_000_000.0, 3.0)
    value_score = min(unlock_value_usd / 100_000_000.0, 5.0)
    status_bonus = 20.0 if status == "paper_short_candidate" else 0.0
    crowded_penalty = 3.0 if status == "crowded_short_risk" else 0.0
    context_penalty = 20.0 if status == "context_only" else 0.0
    return (
        percent_supply
        + value_score
        + abs(annualized_funding)
        + liquidity
        + urgency
        + status_bonus
        - (impact_spread * 100.0)
        - crowded_penalty
        - context_penalty
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--join-path",
        type=Path,
        default=ROOT / "current_token_unlock_market_join.csv",
    )
    parser.add_argument(
        "--hyperliquid-path",
        type=Path,
        default=ROOT.parents[0] / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_token_unlock_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_token_unlock_paper_tickets.md",
    )
    args = parser.parse_args()

    tickets = build_paper_tickets(join_path=args.join_path, hyperliquid_path=args.hyperliquid_path)
    write_tickets_csv(tickets, output_path=args.output_path)
    write_tickets_md(tickets, output_path=args.markdown_output_path)
    for ticket in tickets[:10]:
        print(
            ticket.symbol,
            ticket.status,
            ticket.side,
            f"supply={ticket.percent_supply:.2f}",
            f"funding={ticket.annualized_funding:.6f}",
            f"score={ticket.score:.4f}",
            ticket.reason,
        )


if __name__ == "__main__":
    main()
