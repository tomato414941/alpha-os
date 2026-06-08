from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from strategies.options_volatility.current_deribit_options_surface import (
    OptionQuote,
    fetch_deribit_option_summaries,
    _nearest,
    _parse_quotes,
)


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
    atm_call: str
    atm_put: str
    call_ask_pct: float
    put_ask_pct: float
    quote_spread_pct: float
    max_loss_pct: float
    max_loss_usd: float
    breakeven_move_pct: float
    realized_move_pct: float
    premium_to_realized_move: float
    quote_status: str
    quote_reason: str
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
    quotes_by_key = _build_quote_checks(tuple(sorted({row["currency"] for row in _read_rows(labels_path)})))
    tickets: list[OptionsVolatilityPaperTicket] = []
    for row in _read_rows(labels_path):
        surface = surface_by_key.get((row["currency"], row["expiry"]), {})
        quote_check = quotes_by_key.get((row["currency"], row["expiry"]))
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
        quote_status, quote_reason, quote_metrics = _quote_status_reason(
            structure=structure,
            status=status,
            days_to_expiry=dte,
            realized_vol_24h=rv24,
            quote_check=quote_check,
        )
        final_status = _final_status(status=status, quote_status=quote_status)
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
                atm_call="" if quote_check is None else quote_check.atm_call.instrument_name,
                atm_put="" if quote_check is None else quote_check.atm_put.instrument_name,
                call_ask_pct=quote_metrics.call_ask_pct,
                put_ask_pct=quote_metrics.put_ask_pct,
                quote_spread_pct=quote_metrics.quote_spread_pct,
                max_loss_pct=quote_metrics.max_loss_pct,
                max_loss_usd=quote_metrics.max_loss_usd,
                breakeven_move_pct=quote_metrics.breakeven_move_pct,
                realized_move_pct=quote_metrics.realized_move_pct,
                premium_to_realized_move=quote_metrics.premium_to_realized_move,
                quote_status=quote_status,
                quote_reason=quote_reason,
                score=_score(
                    days_to_expiry=dte,
                    iv_premium_24h=premium24,
                    skew_iv=skew,
                    term_iv_spread_to_next=term,
                    volume_usd=volume,
                    status=final_status,
                ),
                status=final_status,
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
                "atm_call",
                "atm_put",
                "call_ask_pct",
                "put_ask_pct",
                "quote_spread_pct",
                "max_loss_pct",
                "max_loss_usd",
                "breakeven_move_pct",
                "realized_move_pct",
                "premium_to_realized_move",
                "quote_status",
                "quote_reason",
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
                    ticket.atm_call,
                    ticket.atm_put,
                    f"{ticket.call_ask_pct:.6f}",
                    f"{ticket.put_ask_pct:.6f}",
                    f"{ticket.quote_spread_pct:.6f}",
                    f"{ticket.max_loss_pct:.6f}",
                    f"{ticket.max_loss_usd:.2f}",
                    f"{ticket.breakeven_move_pct:.6f}",
                    f"{ticket.realized_move_pct:.6f}",
                    f"{ticket.premium_to_realized_move:.6f}",
                    ticket.quote_status,
                    ticket.quote_reason,
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
            "| currency | expiry | structure | dte | atm iv | rv24 | prem24 | quote spread | max loss % | realized move % | prem/rv move | score | status | quote status | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for ticket in tickets[:top]:
            handle.write(
                f"| {ticket.currency} | {ticket.expiry} | {ticket.structure} | "
                f"{ticket.days_to_expiry:.2f} | {ticket.atm_iv:.2f} | "
                f"{ticket.realized_vol_24h:.2f} | {ticket.iv_premium_24h:.2f} | "
                f"{ticket.quote_spread_pct:.4f} | {ticket.max_loss_pct:.2f} | "
                f"{ticket.realized_move_pct:.2f} | {ticket.premium_to_realized_move:.2f} | "
                f"{ticket.score:.6f} | "
                f"{ticket.status} | {ticket.quote_status} | {ticket.reason}; {ticket.quote_reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "Long-vol tickets use a simple ATM straddle proxy from public Deribit summary quotes. They still lack order-book depth, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. "
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
            return "long_atm_straddle", "too_thin", "IV is cheap versus recent realized vol but option volume is thin"
        return "long_atm_straddle", "paper_long_vol_candidate", "IV is cheap versus recent realized vol; test capped-premium long-vol structure"
    if action == "term_structure_watch" and term_iv_spread_to_next >= 5.0 and iv_premium_24h >= 10.0:
        return "calendar_spread", "paper_calendar_spread_watch", "front IV premium and term spread are elevated"
    return "none", "context_only", "surface context exists but no paper structure is selected"


@dataclass(frozen=True)
class QuoteCheck:
    atm_call: OptionQuote
    atm_put: OptionQuote


@dataclass(frozen=True)
class QuoteMetrics:
    call_ask_pct: float = 0.0
    put_ask_pct: float = 0.0
    quote_spread_pct: float = 0.0
    max_loss_pct: float = 0.0
    max_loss_usd: float = 0.0
    breakeven_move_pct: float = 0.0
    realized_move_pct: float = 0.0
    premium_to_realized_move: float = 0.0


def _build_quote_checks(currencies: tuple[str, ...]) -> dict[tuple[str, str], QuoteCheck]:
    checks: dict[tuple[str, str], QuoteCheck] = {}
    for currency in currencies:
        quotes = _parse_quotes(fetch_deribit_option_summaries(currency=currency))
        grouped: dict[str, list[OptionQuote]] = {}
        for quote in quotes:
            grouped.setdefault(quote.expiry.date().isoformat(), []).append(quote)
        for expiry, group_rows in grouped.items():
            group = tuple(group_rows)
            atm_call = _nearest(group, option_type="C", target_moneyness=0.0)
            atm_put = _nearest(group, option_type="P", target_moneyness=0.0)
            if atm_call is None or atm_put is None:
                continue
            checks[(currency, expiry)] = QuoteCheck(atm_call=atm_call, atm_put=atm_put)
    return checks


def _quote_status_reason(
    *,
    structure: str,
    status: str,
    days_to_expiry: float,
    realized_vol_24h: float,
    quote_check: QuoteCheck | None,
) -> tuple[str, str, QuoteMetrics]:
    if not status.startswith("paper_"):
        return "quote_not_needed", "no paper structure selected", QuoteMetrics()
    if quote_check is None:
        return "quote_missing", "ATM option pair was not found in Deribit summary", QuoteMetrics()
    bid_total = (quote_check.atm_call.bid_price or 0.0) + (quote_check.atm_put.bid_price or 0.0)
    ask_total = (quote_check.atm_call.ask_price or 0.0) + (quote_check.atm_put.ask_price or 0.0)
    underlying = (quote_check.atm_call.underlying_price + quote_check.atm_put.underlying_price) / 2.0
    if bid_total <= 0.0 or ask_total <= 0.0 or underlying <= 0.0:
        return "quote_missing", "ATM option pair has missing bid or ask", QuoteMetrics()
    mid_total = (bid_total + ask_total) / 2.0
    spread_pct = (ask_total - bid_total) / mid_total
    max_loss_pct = ask_total * 100.0
    realized_move_pct = _realized_move_pct(
        annualized_realized_vol=realized_vol_24h,
        days_to_expiry=days_to_expiry,
    )
    metrics = QuoteMetrics(
        call_ask_pct=(quote_check.atm_call.ask_price or 0.0) * 100.0,
        put_ask_pct=(quote_check.atm_put.ask_price or 0.0) * 100.0,
        quote_spread_pct=spread_pct,
        max_loss_pct=max_loss_pct,
        max_loss_usd=ask_total * underlying,
        breakeven_move_pct=max_loss_pct,
        realized_move_pct=realized_move_pct,
        premium_to_realized_move=(
            max_loss_pct / realized_move_pct if realized_move_pct > 0.0 else 0.0
        ),
    )
    if spread_pct > 0.12:
        return "quote_too_wide", "ATM option pair spread is too wide for a clean paper ticket", metrics
    if structure == "long_atm_straddle" and max_loss_pct > 12.0:
        return "premium_too_large", "ATM straddle premium is too large relative to notional", metrics
    return "quote_executable_watch", "ATM option pair quote is present with acceptable spread", metrics


def _realized_move_pct(*, annualized_realized_vol: float, days_to_expiry: float) -> float:
    if annualized_realized_vol <= 0.0 or days_to_expiry <= 0.0:
        return 0.0
    return annualized_realized_vol * sqrt(days_to_expiry / 365.0)


def _final_status(*, status: str, quote_status: str) -> str:
    if status == "paper_long_vol_candidate" and quote_status == "quote_executable_watch":
        return "paper_long_vol_quote_candidate"
    if status == "paper_long_vol_candidate":
        return "paper_long_vol_quote_blocked"
    return status


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
        "paper_long_vol_quote_candidate": 20.0,
        "paper_long_vol_quote_blocked": 4.0,
        "paper_calendar_spread_watch": 10.0,
        "too_thin": 2.0,
    }.get(status, 0.0)
    premium_component = abs(iv_premium_24h) if status.startswith("paper_long_vol") else iv_premium_24h
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
