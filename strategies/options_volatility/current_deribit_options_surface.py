from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path

import requests


DERIBIT_BOOK_SUMMARY_URL = (
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
)
ROOT = Path(__file__).resolve().parent
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class OptionQuote:
    currency: str
    instrument_name: str
    expiry: datetime
    days_to_expiry: float
    strike: float
    option_type: str
    underlying_price: float
    moneyness: float
    mark_iv: float
    bid_price: float | None
    ask_price: float | None
    bid_ask_spread_pct: float | None
    open_interest: float
    volume_usd: float


@dataclass(frozen=True)
class OptionsSurfaceRow:
    timestamp: str
    currency: str
    expiry: str
    days_to_expiry: float
    underlying_price: float
    atm_iv: float
    put_wing_iv: float | None
    call_wing_iv: float | None
    skew_iv: float | None
    term_iv_spread_to_next: float | None
    selected_spread_pct: float | None
    open_interest: float
    volume_usd: float
    action: str
    score: float
    reason: str


def fetch_deribit_option_summaries(
    *,
    currency: str,
    url: str = DERIBIT_BOOK_SUMMARY_URL,
) -> tuple[dict[str, object], ...]:
    response = requests.get(
        url,
        params={"currency": currency, "kind": "option"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return tuple(payload.get("result") or ())


def build_options_surface_rows(
    *,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
    timestamp: str | None = None,
) -> tuple[OptionsSurfaceRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    quotes = tuple(
        quote
        for currency in currencies
        for quote in _parse_quotes(fetch_deribit_option_summaries(currency=currency))
    )
    base_rows = _summarize_expiries(quotes=quotes, timestamp=observed_at)
    rows = _attach_term_spreads(base_rows)
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_options_surface_csv(
    rows: tuple[OptionsSurfaceRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "currency",
                "expiry",
                "days_to_expiry",
                "underlying_price",
                "atm_iv",
                "put_wing_iv",
                "call_wing_iv",
                "skew_iv",
                "term_iv_spread_to_next",
                "selected_spread_pct",
                "open_interest",
                "volume_usd",
                "action",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.currency,
                    row.expiry,
                    f"{row.days_to_expiry:.4f}",
                    f"{row.underlying_price:.8f}",
                    f"{row.atm_iv:.4f}",
                    "" if row.put_wing_iv is None else f"{row.put_wing_iv:.4f}",
                    "" if row.call_wing_iv is None else f"{row.call_wing_iv:.4f}",
                    "" if row.skew_iv is None else f"{row.skew_iv:.4f}",
                    (
                        ""
                        if row.term_iv_spread_to_next is None
                        else f"{row.term_iv_spread_to_next:.4f}"
                    ),
                    (
                        ""
                        if row.selected_spread_pct is None
                        else f"{row.selected_spread_pct:.6f}"
                    ),
                    f"{row.open_interest:.4f}",
                    f"{row.volume_usd:.4f}",
                    row.action,
                    f"{row.score:.6f}",
                    row.reason,
                )
            )
    return output_path


def write_options_surface_md(
    rows: tuple[OptionsSurfaceRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Deribit Options Surface\n\n")
        handle.write(
            "This compresses public Deribit BTC/ETH option summaries into ATM IV, "
            "simple 5% OTM skew, and adjacent-expiry term structure. It is a "
            "volatility-surface exploration probe, not a trade instruction.\n\n"
        )
        handle.write(
            "| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n"
        )
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.currency} | "
                f"{row.expiry} | "
                f"{row.days_to_expiry:.2f} | "
                f"{row.atm_iv:.2f} | "
                f"{'' if row.skew_iv is None else f'{row.skew_iv:.2f}'} | "
                f"{'' if row.term_iv_spread_to_next is None else f'{row.term_iv_spread_to_next:.2f}'} | "
                f"{'' if row.selected_spread_pct is None else f'{row.selected_spread_pct:.4f}'} | "
                f"{row.open_interest:.0f} | "
                f"{row.volume_usd:.0f} | "
                f"{row.action} | "
                f"{row.score:.4f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Large positive term spread means the nearer expiry has richer ATM IV "
            "than the next expiry. Positive skew means the 5% OTM put proxy is "
            "richer than the 5% OTM call proxy. This still needs realized-vol "
            "baselines, option execution costs, margin, and hedging rules.\n"
        )
    return output_path


def _parse_quotes(raw_rows: tuple[dict[str, object], ...]) -> tuple[OptionQuote, ...]:
    now = datetime.now(UTC)
    quotes: list[OptionQuote] = []
    for row in raw_rows:
        instrument_name = str(row.get("instrument_name") or "")
        parsed = _parse_instrument_name(instrument_name)
        if parsed is None:
            continue
        currency, expiry, strike, option_type = parsed
        underlying_price = _float_or_none(row.get("underlying_price"))
        mark_iv = _float_or_none(row.get("mark_iv"))
        if underlying_price is None or underlying_price <= 0.0 or mark_iv is None:
            continue
        bid_price = _float_or_none(row.get("bid_price"))
        ask_price = _float_or_none(row.get("ask_price"))
        spread_pct = _bid_ask_spread_pct(bid_price=bid_price, ask_price=ask_price)
        quotes.append(
            OptionQuote(
                currency=currency,
                instrument_name=instrument_name,
                expiry=expiry,
                days_to_expiry=(expiry - now).total_seconds() / 86_400.0,
                strike=strike,
                option_type=option_type,
                underlying_price=underlying_price,
                moneyness=(strike / underlying_price) - 1.0,
                mark_iv=mark_iv,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_ask_spread_pct=spread_pct,
                open_interest=_float_or_none(row.get("open_interest")) or 0.0,
                volume_usd=_float_or_none(row.get("volume_usd")) or 0.0,
            )
        )
    return tuple(quote for quote in quotes if quote.days_to_expiry > 0.0)


def _summarize_expiries(
    *,
    quotes: tuple[OptionQuote, ...],
    timestamp: str,
) -> tuple[OptionsSurfaceRow, ...]:
    grouped: dict[tuple[str, datetime], list[OptionQuote]] = {}
    for quote in quotes:
        grouped.setdefault((quote.currency, quote.expiry), []).append(quote)
    rows: list[OptionsSurfaceRow] = []
    for (currency, expiry), group_rows in grouped.items():
        group = tuple(group_rows)
        if len(group) < 4:
            continue
        atm_call = _nearest(group, option_type="C", target_moneyness=0.0)
        atm_put = _nearest(group, option_type="P", target_moneyness=0.0)
        if atm_call is None or atm_put is None:
            continue
        put_wing = _nearest(group, option_type="P", target_moneyness=-0.05)
        call_wing = _nearest(group, option_type="C", target_moneyness=0.05)
        atm_iv = (atm_call.mark_iv + atm_put.mark_iv) / 2.0
        skew_iv = (
            None
            if put_wing is None or call_wing is None
            else put_wing.mark_iv - call_wing.mark_iv
        )
        selected_spreads = tuple(
            quote.bid_ask_spread_pct
            for quote in (atm_call, atm_put, put_wing, call_wing)
            if quote is not None and quote.bid_ask_spread_pct is not None
        )
        row = OptionsSurfaceRow(
            timestamp=timestamp,
            currency=currency,
            expiry=expiry.date().isoformat(),
            days_to_expiry=atm_call.days_to_expiry,
            underlying_price=(atm_call.underlying_price + atm_put.underlying_price) / 2.0,
            atm_iv=atm_iv,
            put_wing_iv=None if put_wing is None else put_wing.mark_iv,
            call_wing_iv=None if call_wing is None else call_wing.mark_iv,
            skew_iv=skew_iv,
            term_iv_spread_to_next=None,
            selected_spread_pct=_mean(selected_spreads),
            open_interest=sum(quote.open_interest for quote in group),
            volume_usd=sum(quote.volume_usd for quote in group),
            action="surface_context",
            score=0.0,
            reason="surface summarized but no strong dislocation selected yet",
        )
        rows.append(row)
    return tuple(rows)


def _attach_term_spreads(
    rows: tuple[OptionsSurfaceRow, ...],
) -> tuple[OptionsSurfaceRow, ...]:
    by_currency: dict[str, list[OptionsSurfaceRow]] = {}
    for row in rows:
        by_currency.setdefault(row.currency, []).append(row)
    final_rows: list[OptionsSurfaceRow] = []
    for currency_rows in by_currency.values():
        sorted_rows = sorted(currency_rows, key=lambda row: row.days_to_expiry)
        for index, row in enumerate(sorted_rows):
            next_row = sorted_rows[index + 1] if index + 1 < len(sorted_rows) else None
            term_spread = None if next_row is None else row.atm_iv - next_row.atm_iv
            action, reason = _classify_action(row=row, term_spread=term_spread)
            final_rows.append(
                OptionsSurfaceRow(
                    timestamp=row.timestamp,
                    currency=row.currency,
                    expiry=row.expiry,
                    days_to_expiry=row.days_to_expiry,
                    underlying_price=row.underlying_price,
                    atm_iv=row.atm_iv,
                    put_wing_iv=row.put_wing_iv,
                    call_wing_iv=row.call_wing_iv,
                    skew_iv=row.skew_iv,
                    term_iv_spread_to_next=term_spread,
                    selected_spread_pct=row.selected_spread_pct,
                    open_interest=row.open_interest,
                    volume_usd=row.volume_usd,
                    action=action,
                    score=_score(row=row, term_spread=term_spread),
                    reason=reason,
                )
            )
    return tuple(final_rows)


def _classify_action(
    *,
    row: OptionsSurfaceRow,
    term_spread: float | None,
) -> tuple[str, str]:
    skew = row.skew_iv or 0.0
    term = term_spread or 0.0
    if abs(term) >= abs(skew) and term >= 5.0:
        return "front_vol_premium_watch", "near expiry ATM IV is richer than the next expiry"
    if abs(term) >= abs(skew) and term <= -5.0:
        return "back_vol_premium_watch", "next expiry ATM IV is richer than near expiry"
    if skew >= 5.0:
        return "put_skew_watch", "5% OTM put proxy is richer than the call proxy"
    if skew <= -5.0:
        return "call_skew_watch", "5% OTM call proxy is richer than the put proxy"
    return "surface_context", "surface summarized but no strong dislocation selected yet"


def _score(*, row: OptionsSurfaceRow, term_spread: float | None) -> float:
    term = abs(term_spread or 0.0)
    skew = abs(row.skew_iv or 0.0)
    liquidity = log10(row.open_interest + 1.0) + log10(row.volume_usd + 1.0)
    spread_penalty = 0.0 if row.selected_spread_pct is None else row.selected_spread_pct * 2.0
    return max(term * 2.0 + skew + liquidity - spread_penalty, 0.0)


def _nearest(
    rows: tuple[OptionQuote, ...],
    *,
    option_type: str,
    target_moneyness: float,
) -> OptionQuote | None:
    candidates = tuple(row for row in rows if row.option_type == option_type)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(row.moneyness - target_moneyness),
            row.bid_ask_spread_pct if row.bid_ask_spread_pct is not None else 10.0,
        ),
    )


def _parse_instrument_name(
    instrument_name: str,
) -> tuple[str, datetime, float, str] | None:
    parts = instrument_name.split("-")
    if len(parts) != 4:
        return None
    currency, expiry_raw, strike_raw, option_type = parts
    if option_type not in {"C", "P"}:
        return None
    day = ""
    month = ""
    year = ""
    for char in expiry_raw:
        if char.isdigit() and not month:
            day += char
        elif char.isalpha():
            month += char
        else:
            year += char
    if not day or month not in MONTHS or not year:
        return None
    expiry = datetime(
        2000 + int(year),
        MONTHS[month],
        int(day),
        8,
        tzinfo=UTC,
    )
    return currency, expiry, float(strike_raw), option_type


def _bid_ask_spread_pct(*, bid_price: float | None, ask_price: float | None) -> float | None:
    if bid_price is None or ask_price is None or bid_price <= 0.0 or ask_price <= 0.0:
        return None
    mid = (bid_price + ask_price) / 2.0
    return (ask_price - bid_price) / mid if mid > 0.0 else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_deribit_options_surface.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_deribit_options_surface.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_options_surface_rows(currencies=tuple(args.currencies))
    write_options_surface_csv(rows, output_path=args.output_path)
    write_options_surface_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.currency,
            row.expiry,
            row.action,
            f"atm_iv={row.atm_iv:.2f}",
            f"skew={'' if row.skew_iv is None else f'{row.skew_iv:.2f}'}",
            f"term={'' if row.term_iv_spread_to_next is None else f'{row.term_iv_spread_to_next:.2f}'}",
            f"score={row.score:.2f}",
        )


if __name__ == "__main__":
    main()
