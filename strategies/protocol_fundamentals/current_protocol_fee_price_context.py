from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests

from strategies.protocol_fundamentals.current_protocol_fee_valuation import (
    COINGECKO_IDS,
    COINGECKO_MARKETS_URL,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeePriceContextRow:
    token_symbol: str
    protocol: str
    fee_status: str
    fee_to_market_cap: float
    fee_to_fdv: float
    fee_growth_7d: float
    funding: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    volume_to_market_cap: float
    status: str
    side: str
    score: float
    evidence: str
    next_step: str


def build_protocol_fee_price_context_rows(
    *,
    valuation_path: Path = ROOT / "current_protocol_fee_valuation.csv",
) -> tuple[ProtocolFeePriceContextRow, ...]:
    valuation_rows = tuple(row for row in _read_rows(valuation_path) if row.get("token_symbol") in COINGECKO_IDS)
    markets = _fetch_markets(tuple(row.get("token_symbol", "") for row in valuation_rows))
    rows = tuple(
        _build_row(
            valuation=row,
            market=markets.get(row.get("token_symbol", ""), {}),
        )
        for row in valuation_rows
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_protocol_fee_price_context_csv(
    rows: tuple[ProtocolFeePriceContextRow, ...],
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
                "fee_status",
                "fee_to_market_cap",
                "fee_to_fdv",
                "fee_growth_7d",
                "funding",
                "price_change_1h",
                "price_change_24h",
                "price_change_7d",
                "price_change_30d",
                "volume_to_market_cap",
                "status",
                "side",
                "score",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.token_symbol,
                    row.protocol,
                    row.fee_status,
                    f"{row.fee_to_market_cap:.8f}",
                    f"{row.fee_to_fdv:.8f}",
                    f"{row.fee_growth_7d:.8f}",
                    f"{row.funding:.8f}",
                    f"{row.price_change_1h:.8f}",
                    f"{row.price_change_24h:.8f}",
                    f"{row.price_change_7d:.8f}",
                    f"{row.price_change_30d:.8f}",
                    f"{row.volume_to_market_cap:.8f}",
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_price_context_md(
    rows: tuple[ProtocolFeePriceContextRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Price Context\n\n")
        handle.write(
            "This joins protocol fee-growth valuation to current CoinGecko price movement. "
            "It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.\n\n"
        )
        handle.write(
            "| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.token_symbol} | {row.protocol} | {row.status} | {row.score:.4f} | "
                f"{row.fee_to_market_cap:.4f} | {row.fee_to_fdv:.4f} | {row.fee_growth_7d:.2f} | "
                f"{row.price_change_24h:.2f} | {row.price_change_7d:.2f} | {row.price_change_30d:.2f} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`fee_growth_price_lag_candidate` is the most interesting long setup class here: "
            "fees are strong, but the token has not obviously chased over the last week. "
            "`fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.\n"
        )
    return output_path


def _fetch_markets(tokens: tuple[str, ...]) -> dict[str, dict[str, object]]:
    coin_ids = tuple(COINGECKO_IDS[token].coin_id for token in tokens if token in COINGECKO_IDS)
    response = requests.get(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "ids": ",".join(sorted(set(coin_ids))),
            "per_page": 250,
            "price_change_percentage": "1h,24h,7d,30d",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    reverse = {mapping.coin_id: token for token, mapping in COINGECKO_IDS.items()}
    output: dict[str, dict[str, object]] = {}
    for row in response.json():
        token = reverse.get(str(row.get("id") or ""))
        if token:
            output[token] = row
    return output


def _build_row(
    *,
    valuation: dict[str, str],
    market: dict[str, object],
) -> ProtocolFeePriceContextRow:
    token = valuation.get("token_symbol", "")
    fee_to_market_cap = _float(valuation.get("fee_to_market_cap"))
    fee_to_fdv = _float(valuation.get("fee_to_fdv"))
    fee_growth = _float(valuation.get("change_7d_over_7d"))
    funding = _float(valuation.get("funding"))
    price_1h = _float(market.get("price_change_percentage_1h_in_currency"))
    price_24h = _float(market.get("price_change_percentage_24h_in_currency"))
    price_7d = _float(market.get("price_change_percentage_7d_in_currency"))
    price_30d = _float(market.get("price_change_percentage_30d_in_currency"))
    market_cap = _float(market.get("market_cap"))
    volume = _float(market.get("total_volume"))
    volume_to_market_cap = volume / market_cap if market_cap > 0.0 else 0.0
    status, side, score, next_step = _status_side_score_next_step(
        token=token,
        fee_to_market_cap=fee_to_market_cap,
        fee_to_fdv=fee_to_fdv,
        fee_growth=fee_growth,
        funding=funding,
        price_24h=price_24h,
        price_7d=price_7d,
        price_30d=price_30d,
    )
    evidence = (
        f"fee_to_mcap={fee_to_market_cap:.4f}; fee_to_fdv={fee_to_fdv:.4f}; "
        f"fee_growth_7d={fee_growth:.2f}; funding={funding:.4f}; "
        f"price_24h={price_24h:.2f}; price_7d={price_7d:.2f}; price_30d={price_30d:.2f}; "
        f"volume_to_mcap={volume_to_market_cap:.4f}"
    )
    return ProtocolFeePriceContextRow(
        token_symbol=token,
        protocol=valuation.get("protocol", ""),
        fee_status=valuation.get("status", ""),
        fee_to_market_cap=fee_to_market_cap,
        fee_to_fdv=fee_to_fdv,
        fee_growth_7d=fee_growth,
        funding=funding,
        price_change_1h=price_1h,
        price_change_24h=price_24h,
        price_change_7d=price_7d,
        price_change_30d=price_30d,
        volume_to_market_cap=volume_to_market_cap,
        status=status,
        side=side,
        score=score,
        evidence=evidence,
        next_step=next_step,
    )


def _status_side_score_next_step(
    *,
    token: str,
    fee_to_market_cap: float,
    fee_to_fdv: float,
    fee_growth: float,
    funding: float,
    price_24h: float,
    price_7d: float,
    price_30d: float,
) -> tuple[str, str, float, str]:
    base = min(fee_to_market_cap * 100.0, 30.0) + min(fee_to_fdv * 100.0, 30.0)
    growth_score = max(min(fee_growth, 250.0), -100.0) / 10.0
    funding_penalty = max(funding - 0.2, 0.0) * 10.0
    score = base + growth_score - funding_penalty
    if fee_growth >= 75.0 and fee_to_market_cap >= 0.10 and price_7d <= 0.0:
        return (
            "fee_growth_price_lag_candidate",
            "long_token",
            score + min(abs(price_7d), 15.0),
            f"paper-label {token} as fee-growth lag setup over 4h, 12h, 24h, and 7d",
        )
    if fee_growth >= 75.0 and fee_to_market_cap >= 0.10 and price_7d <= 15.0:
        return (
            "fee_growth_price_confirmation",
            "long_token",
            score + 5.0,
            f"paper-label {token} as confirmed fee-growth setup with strict entry timing",
        )
    if fee_growth >= 75.0 and price_7d > 15.0 and price_24h > 3.0:
        return (
            "fee_growth_price_chase_risk",
            "wait_or_pullback_long",
            score - 5.0 + min(price_30d / 10.0, 5.0),
            f"wait for {token} pullback or fresh fee-growth repeat before chasing price",
        )
    if fee_growth < 0.0 and price_7d < 0.0:
        return (
            "fee_decay_price_weakness_context",
            "watch_or_short",
            score + min(abs(price_7d), 10.0),
            f"test whether {token} fee decay and weak price persist before any short thesis",
        )
    return (
        "fee_price_context_watch",
        "none",
        score,
        f"collect another {token} fee and price snapshot before promotion",
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valuation-path", type=Path, default=ROOT / "current_protocol_fee_valuation.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_price_context.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_price_context.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_protocol_fee_price_context_rows(
        valuation_path=args.valuation_path,
    )
    write_protocol_fee_price_context_csv(rows, output_path=args.output_path)
    write_protocol_fee_price_context_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.token_symbol, f"score={row.score:.4f}", f"price7d={row.price_change_7d:.2f}")


if __name__ == "__main__":
    main()
