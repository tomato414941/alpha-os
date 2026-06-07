from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


DERIBIT_BOOK_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_TICKER_URL = "https://www.deribit.com/api/v2/public/ticker"
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
class FuturesBasisRow:
    timestamp: str
    currency: str
    instrument_name: str
    expiry: str
    days_to_expiry: float
    mark_price: float
    index_price: float
    best_bid_price: float
    best_ask_price: float
    bid_ask_spread_pct: float
    volume_usd: float
    open_interest: float
    basis: float
    annualized_basis: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def build_deribit_futures_basis_rows(
    *,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
    timestamp: str | None = None,
) -> tuple[FuturesBasisRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    output: list[FuturesBasisRow] = []
    for currency in currencies:
        for summary in _fetch_future_summaries(currency=currency):
            instrument = str(summary.get("instrument_name") or "")
            if not instrument or instrument.endswith("PERPETUAL"):
                continue
            ticker = _fetch_ticker(instrument_name=instrument)
            row = _build_row(observed_at=observed_at, currency=currency, instrument=instrument, ticker=ticker)
            if row is not None:
                output.append(row)
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_futures_basis_csv(rows: tuple[FuturesBasisRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "currency",
                "instrument_name",
                "expiry",
                "days_to_expiry",
                "mark_price",
                "index_price",
                "best_bid_price",
                "best_ask_price",
                "bid_ask_spread_pct",
                "volume_usd",
                "open_interest",
                "basis",
                "annualized_basis",
                "score",
                "status",
                "side",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.currency,
                    row.instrument_name,
                    row.expiry,
                    f"{row.days_to_expiry:.4f}",
                    f"{row.mark_price:.8f}",
                    f"{row.index_price:.8f}",
                    f"{row.best_bid_price:.8f}",
                    f"{row.best_ask_price:.8f}",
                    f"{row.bid_ask_spread_pct:.8f}",
                    f"{row.volume_usd:.8f}",
                    f"{row.open_interest:.8f}",
                    f"{row.basis:.8f}",
                    f"{row.annualized_basis:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_futures_basis_md(rows: tuple[FuturesBasisRow, ...], *, output_path: Path, top: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Deribit Futures Basis\n\n")
        handle.write(
            "This screen compares Deribit dated futures marks against index price and annualizes the basis. "
            "It is a basis term-structure screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| currency | instrument | dte | mark | index | basis | ann basis | spread pct | volume USD | status | score |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.currency} | {row.instrument_name} | {row.days_to_expiry:.2f} | "
                f"{row.mark_price:.2f} | {row.index_price:.2f} | {row.basis:.5f} | "
                f"{row.annualized_basis:.4f} | {row.bid_ask_spread_pct:.5f} | "
                f"{row.volume_usd:.0f} | {row.status} | {row.score:.4f} |\n"
            )
    return output_path


def _fetch_future_summaries(*, currency: str) -> tuple[dict[str, object], ...]:
    response = requests.get(
        DERIBIT_BOOK_SUMMARY_URL,
        params={"currency": currency, "kind": "future"},
        timeout=30,
    )
    response.raise_for_status()
    return tuple(response.json().get("result") or ())


def _fetch_ticker(*, instrument_name: str) -> dict[str, object]:
    response = requests.get(
        DERIBIT_TICKER_URL,
        params={"instrument_name": instrument_name},
        timeout=20,
    )
    response.raise_for_status()
    return dict(response.json().get("result") or {})


def _build_row(
    *,
    observed_at: str,
    currency: str,
    instrument: str,
    ticker: dict[str, object],
) -> FuturesBasisRow | None:
    expiry = _parse_future_expiry(instrument)
    if expiry is None:
        return None
    now = datetime.now(UTC)
    days_to_expiry = (expiry - now).total_seconds() / 86_400.0
    mark_price = _float(ticker.get("mark_price"))
    index_price = _float(ticker.get("index_price") or ticker.get("estimated_delivery_price"))
    if days_to_expiry <= 0.0 or mark_price <= 0.0 or index_price <= 0.0:
        return None
    best_bid = _float(ticker.get("best_bid_price"))
    best_ask = _float(ticker.get("best_ask_price"))
    spread_pct = _spread_pct(best_bid=best_bid, best_ask=best_ask, mark_price=mark_price)
    basis = (mark_price / index_price) - 1.0
    annualized_basis = basis * 365.0 / days_to_expiry
    volume_usd = _float(ticker.get("stats", {}).get("volume_usd") if isinstance(ticker.get("stats"), dict) else None)
    open_interest = _float(ticker.get("open_interest"))
    status, side, reason = _status_side_reason(
        annualized_basis=annualized_basis,
        volume_usd=volume_usd,
        spread_pct=spread_pct,
    )
    return FuturesBasisRow(
        timestamp=observed_at,
        currency=currency,
        instrument_name=instrument,
        expiry=expiry.isoformat(),
        days_to_expiry=days_to_expiry,
        mark_price=mark_price,
        index_price=index_price,
        best_bid_price=best_bid,
        best_ask_price=best_ask,
        bid_ask_spread_pct=spread_pct,
        volume_usd=volume_usd,
        open_interest=open_interest,
        basis=basis,
        annualized_basis=annualized_basis,
        score=_score(annualized_basis=annualized_basis, volume_usd=volume_usd, spread_pct=spread_pct),
        status=status,
        side=side,
        reason=reason,
        next_step=f"check {instrument} hedge route, fees, margin, funding, and order-book depth",
    )


def _parse_future_expiry(instrument: str) -> datetime | None:
    parts = instrument.split("-")
    if len(parts) < 2:
        return None
    code = parts[1]
    if len(code) < 7:
        return None
    day = int(code[:2])
    month = MONTHS.get(code[2:5])
    year = 2000 + int(code[5:7])
    if month is None:
        return None
    return datetime(year, month, day, 8, 0, tzinfo=UTC)


def _status_side_reason(*, annualized_basis: float, volume_usd: float, spread_pct: float) -> tuple[str, str, str]:
    if volume_usd < 100_000.0 or spread_pct > 0.01:
        return "basis_liquidity_watch", "none", "basis exists but liquidity or spread is weak"
    if annualized_basis >= 0.03:
        return "paper_short_basis_watch", "short_future_long_spot_or_perp", "dated future is rich versus index"
    if annualized_basis <= -0.02:
        return "paper_long_basis_watch", "long_future_short_spot_or_perp", "dated future is cheap versus index"
    if abs(annualized_basis) >= 0.01:
        return "basis_term_structure_watch", "none", "basis is visible but not large enough yet"
    return "watch", "none", "basis is small"


def _score(*, annualized_basis: float, volume_usd: float, spread_pct: float) -> float:
    liquidity_score = min(volume_usd / 1_000_000.0, 20.0)
    spread_penalty = min(spread_pct * 1_000.0, 20.0)
    return abs(annualized_basis) * 1_000.0 + liquidity_score - spread_penalty


def _spread_pct(*, best_bid: float, best_ask: float, mark_price: float) -> float:
    if best_bid <= 0.0 or best_ask <= 0.0 or mark_price <= 0.0:
        return 1.0
    return (best_ask - best_bid) / mark_price


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_deribit_futures_basis.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_deribit_futures_basis.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_deribit_futures_basis_rows()
    write_futures_basis_csv(rows, output_path=args.output_path)
    write_futures_basis_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.instrument_name, f"ann_basis={row.annualized_basis:.4f}", f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
