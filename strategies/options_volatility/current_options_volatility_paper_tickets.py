from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OptionsVolatilityPaperTicket:
    currency: str
    expiry: str
    structure: str
    days_to_expiry: float
    atm_iv: float
    realized_vol_24h: float
    iv_premium_24h: float
    skew_iv: float
    term_iv_spread_to_next: float
    selected_spread_pct: float
    open_interest: float
    volume_usd: float
    score: float
    status: str
    reason: str


def build_paper_tickets(
    *,
    labels_path: Path,
    surface_path: Path,
) -> tuple[OptionsVolatilityPaperTicket, ...]:
    surface_by_key = {
        (row["currency"], row["expiry"]): row for row in _read_rows(surface_path)
    }
    tickets: list[OptionsVolatilityPaperTicket] = []
    for row in _read_rows(labels_path):
        surface = surface_by_key.get((row["currency"], row["expiry"]), {})
        dte = _float(row["days_to_expiry"])
        atm_iv = _float(row["atm_iv"])
        rv24 = _float(row["realized_vol_24h"])
        premium24 = _float(row["iv_premium_24h"])
        skew = _float(row["skew_iv"])
        term = _float(row["term_iv_spread_to_next"])
        volume = _float(surface.get("volume_usd", ""))
        open_interest = _float(surface.get("open_interest", ""))
        spread_pct = _float(surface.get("selected_spread_pct", ""))
        structure, status, reason = _structure_status_reason(
            action=row["action"],
            days_to_expiry=dte,
            iv_premium_24h=premium24,
            skew_iv=skew,
            term_iv_spread_to_next=term,
            volume_usd=volume,
        )
        tickets.append(
            OptionsVolatilityPaperTicket(
                currency=row["currency"],
                expiry=row["expiry"],
                structure=structure,
                days_to_expiry=dte,
                atm_iv=atm_iv,
                realized_vol_24h=rv24,
                iv_premium_24h=premium24,
                skew_iv=skew,
                term_iv_spread_to_next=term,
                selected_spread_pct=spread_pct,
                open_interest=open_interest,
                volume_usd=volume,
                score=_score(
                    days_to_expiry=dte,
                    iv_premium_24h=premium24,
                    skew_iv=skew,
                    term_iv_spread_to_next=term,
                    volume_usd=volume,
                    status=status,
                ),
                status=status,
                reason=reason,
            )
        )
    return tuple(sorted(tickets, key=lambda ticket: ticket.score, reverse=True))


def write_tickets_csv(
    tickets: tuple[OptionsVolatilityPaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "currency",
                "expiry",
                "structure",
                "days_to_expiry",
                "atm_iv",
                "realized_vol_24h",
                "iv_premium_24h",
                "skew_iv",
                "term_iv_spread_to_next",
                "selected_spread_pct",
                "open_interest",
                "volume_usd",
                "score",
                "status",
                "reason",
            )
        )
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.currency,
                    ticket.expiry,
                    ticket.structure,
                    f"{ticket.days_to_expiry:.4f}",
                    f"{ticket.atm_iv:.4f}",
                    f"{ticket.realized_vol_24h:.4f}",
                    f"{ticket.iv_premium_24h:.4f}",
                    f"{ticket.skew_iv:.4f}",
                    f"{ticket.term_iv_spread_to_next:.4f}",
                    f"{ticket.selected_spread_pct:.6f}",
                    f"{ticket.open_interest:.4f}",
                    f"{ticket.volume_usd:.2f}",
                    f"{ticket.score:.8f}",
                    ticket.status,
                    ticket.reason,
                )
            )
    return output_path


def write_tickets_md(
    tickets: tuple[OptionsVolatilityPaperTicket, ...],
    *,
    output_path: Path,
    top: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Options Volatility Paper Tickets\n\n")
        handle.write(
            "This converts current Deribit IV-vs-realized and skew contexts into paper tickets. "
            "It is not a live options trade instruction.\n\n"
        )
        handle.write(
            "| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for ticket in tickets[:top]:
            handle.write(
                f"| {ticket.currency} | {ticket.expiry} | {ticket.structure} | "
                f"{ticket.days_to_expiry:.2f} | {ticket.atm_iv:.2f} | "
                f"{ticket.realized_vol_24h:.2f} | {ticket.iv_premium_24h:.2f} | "
                f"{ticket.skew_iv:.2f} | {ticket.term_iv_spread_to_next:.2f} | "
                f"{ticket.volume_usd:.0f} | {ticket.score:.6f} | "
                f"{ticket.status} | {ticket.reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. "
            "Short premium candidates must be treated as capped-risk structures, not naked short options.\n"
        )
    return output_path


def _structure_status_reason(
    *,
    action: str,
    days_to_expiry: float,
    iv_premium_24h: float,
    skew_iv: float,
    term_iv_spread_to_next: float,
    volume_usd: float,
) -> tuple[str, str, str]:
    if days_to_expiry < 1.0:
        return "expiry_gamma_watch", "too_close_to_expiry", "expiry is too close for a clean paper ticket"
    if action == "rich_put_skew_vol_premium_watch" and iv_premium_24h >= 20.0 and skew_iv >= 12.0:
        if volume_usd < 250_000.0:
            return "short_put_spread", "too_thin", "put skew is rich but option volume is thin"
        return "short_put_spread", "paper_short_put_spread_candidate", "put skew and IV premium are rich versus recent realized vol"
    if action == "cheap_vol_watch" and iv_premium_24h <= -20.0:
        if volume_usd < 250_000.0:
            return "long_vol_spread", "too_thin", "IV is cheap versus recent realized vol but option volume is thin"
        return "long_vol_spread", "paper_long_vol_candidate", "IV is cheap versus recent realized vol; test capped-premium long-vol structure"
    if action == "term_structure_watch" and term_iv_spread_to_next >= 5.0 and iv_premium_24h >= 10.0:
        return "calendar_spread", "paper_calendar_spread_watch", "front IV premium and term spread are elevated"
    return "none", "context_only", "surface context exists but no paper structure is selected"


def _score(
    *,
    days_to_expiry: float,
    iv_premium_24h: float,
    skew_iv: float,
    term_iv_spread_to_next: float,
    volume_usd: float,
    status: str,
) -> float:
    liquidity = min(volume_usd / 1_000_000.0, 3.0)
    dte_penalty = 5.0 if days_to_expiry < 1.0 else 0.0
    status_bonus = {
        "paper_short_put_spread_candidate": 20.0,
        "paper_long_vol_candidate": 16.0,
        "paper_calendar_spread_watch": 10.0,
        "too_thin": 2.0,
    }.get(status, 0.0)
    premium_component = abs(iv_premium_24h) if status == "paper_long_vol_candidate" else iv_premium_24h
    return (
        premium_component
        + skew_iv
        + max(term_iv_spread_to_next, 0.0)
        + liquidity
        + status_bonus
        - dte_penalty
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
        "--labels-path",
        type=Path,
        default=ROOT / "current_deribit_options_realized_vol_labels.csv",
    )
    parser.add_argument(
        "--surface-path",
        type=Path,
        default=ROOT / "current_deribit_options_surface.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_options_volatility_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_options_volatility_paper_tickets.md",
    )
    args = parser.parse_args()

    tickets = build_paper_tickets(labels_path=args.labels_path, surface_path=args.surface_path)
    write_tickets_csv(tickets, output_path=args.output_path)
    write_tickets_md(tickets, output_path=args.markdown_output_path)
    for ticket in tickets[:10]:
        print(
            ticket.currency,
            ticket.expiry,
            ticket.status,
            ticket.structure,
            f"prem24={ticket.iv_premium_24h:.2f}",
            f"skew={ticket.skew_iv:.2f}",
            f"score={ticket.score:.4f}",
            ticket.reason,
        )


if __name__ == "__main__":
    main()
