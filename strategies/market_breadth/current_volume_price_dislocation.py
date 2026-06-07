from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
SYMBOL_RE = re.compile(r"[A-Z0-9]{2,12}")
STABLE_SYMBOLS = {
    "BUSD",
    "DAI",
    "FDUSD",
    "PYUSD",
    "SUSD",
    "TUSD",
    "USDC",
    "USDD",
    "USDE",
    "USDG",
    "USDP",
    "USDS",
    "USDT",
    "USD1",
}


@dataclass(frozen=True)
class VolumePriceDislocationRow:
    symbol: str
    name: str
    coin_id: str
    market_cap_rank: int
    current_price: float
    market_cap: float
    total_volume: float
    volume_to_market_cap: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    status: str
    side: str
    score: float
    evidence: str
    next_step: str


def build_volume_price_dislocation_rows(
    *,
    pages: int = 2,
    per_page: int = 250,
    min_market_cap: float = 100_000_000.0,
) -> tuple[VolumePriceDislocationRow, ...]:
    raw_rows: list[dict[str, object]] = []
    for page in range(1, pages + 1):
        raw_rows.extend(_fetch_market_page(page=page, per_page=per_page))
    rows = tuple(
        _build_row(row)
        for row in raw_rows
        if _is_candidate_universe_row(row, min_market_cap=min_market_cap)
    )
    candidates = tuple(row for row in rows if row.status != "market_breadth_context_only")
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_volume_price_dislocation_csv(
    rows: tuple[VolumePriceDislocationRow, ...],
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
                "coin_id",
                "market_cap_rank",
                "current_price",
                "market_cap",
                "total_volume",
                "volume_to_market_cap",
                "price_change_1h",
                "price_change_24h",
                "price_change_7d",
                "price_change_30d",
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
                    row.symbol,
                    row.name,
                    row.coin_id,
                    row.market_cap_rank,
                    f"{row.current_price:.12f}",
                    f"{row.market_cap:.8f}",
                    f"{row.total_volume:.8f}",
                    f"{row.volume_to_market_cap:.8f}",
                    f"{row.price_change_1h:.8f}",
                    f"{row.price_change_24h:.8f}",
                    f"{row.price_change_7d:.8f}",
                    f"{row.price_change_30d:.8f}",
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_volume_price_dislocation_md(
    rows: tuple[VolumePriceDislocationRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volume Price Dislocation\n\n")
        handle.write(
            "This scans broad CoinGecko market data for volume-backed reversal, continuation, "
            "and chase-risk candidates. It is a candidate-generation screen, not a trade list.\n\n"
        )
        handle.write(
            "| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.name} | {row.status} | {row.side} | {row.score:.4f} | "
                f"{row.market_cap_rank} | {row.volume_to_market_cap:.4f} | "
                f"{row.price_change_24h:.2f} | {row.price_change_7d:.2f} | {row.price_change_30d:.2f} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. "
            "`capitulation_reversal_watch` is a falling setup that still needs a trigger. "
            "`breakout_continuation_watch` is already moving and needs stop/entry discipline. "
            "`chase_risk` should usually be avoided until pullback or fade labels exist.\n"
        )
    return output_path


def _fetch_market_page(*, page: int, per_page: int) -> tuple[dict[str, object], ...]:
    response = requests.get(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "price_change_percentage": "1h,24h,7d,30d",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    return tuple(response.json())


def _is_candidate_universe_row(row: dict[str, object], *, min_market_cap: float) -> bool:
    symbol = str(row.get("symbol") or "").upper()
    market_cap = _float(row.get("market_cap"))
    current_price = _float(row.get("current_price"))
    if not SYMBOL_RE.fullmatch(symbol) or symbol in STABLE_SYMBOLS:
        return False
    return market_cap >= min_market_cap and current_price > 0.0


def _build_row(row: dict[str, object]) -> VolumePriceDislocationRow:
    symbol = str(row.get("symbol") or "").upper()
    name = str(row.get("name") or "")
    rank = int(row.get("market_cap_rank") or 0)
    current_price = _float(row.get("current_price"))
    market_cap = _float(row.get("market_cap"))
    total_volume = _float(row.get("total_volume"))
    volume_to_market_cap = total_volume / market_cap if market_cap > 0.0 else 0.0
    price_1h = _float(row.get("price_change_percentage_1h_in_currency"))
    price_24h = _float(row.get("price_change_percentage_24h_in_currency") or row.get("price_change_percentage_24h"))
    price_7d = _float(row.get("price_change_percentage_7d_in_currency"))
    price_30d = _float(row.get("price_change_percentage_30d_in_currency"))
    status, side, score, next_step = _status_side_score_next_step(
        symbol=symbol,
        rank=rank,
        volume_to_market_cap=volume_to_market_cap,
        price_1h=price_1h,
        price_24h=price_24h,
        price_7d=price_7d,
        price_30d=price_30d,
    )
    evidence = (
        f"rank={rank}; price_1h={price_1h:.2f}; price_24h={price_24h:.2f}; "
        f"price_7d={price_7d:.2f}; price_30d={price_30d:.2f}; "
        f"volume_to_mcap={volume_to_market_cap:.4f}"
    )
    return VolumePriceDislocationRow(
        symbol=symbol,
        name=name,
        coin_id=str(row.get("id") or ""),
        market_cap_rank=rank,
        current_price=current_price,
        market_cap=market_cap,
        total_volume=total_volume,
        volume_to_market_cap=volume_to_market_cap,
        price_change_1h=price_1h,
        price_change_24h=price_24h,
        price_change_7d=price_7d,
        price_change_30d=price_30d,
        status=status,
        side=side,
        score=score,
        evidence=evidence,
        next_step=next_step,
    )


def _status_side_score_next_step(
    *,
    symbol: str,
    rank: int,
    volume_to_market_cap: float,
    price_1h: float,
    price_24h: float,
    price_7d: float,
    price_30d: float,
) -> tuple[str, str, float, str]:
    rank_score = max(300 - rank, 0) / 20.0
    volume_score = min(volume_to_market_cap * 60.0, 18.0)
    if price_7d <= -10.0 and price_24h >= 3.0 and volume_to_market_cap >= 0.04:
        score = rank_score + volume_score + min(abs(price_7d), 25.0) + min(price_24h, 15.0)
        return (
            "volume_reversal_candidate",
            "long_reversal",
            score,
            f"paper-label {symbol} volume-backed reversal over 1h, 4h, 12h, and 24h",
        )
    if price_7d <= -20.0 and price_24h < 0.0 and volume_to_market_cap >= 0.08:
        score = rank_score + volume_score + min(abs(price_7d), 35.0)
        return (
            "capitulation_reversal_watch",
            "watch_reversal_trigger",
            score,
            f"wait for {symbol} reversal trigger, then label capitulation rebound",
        )
    if 0.0 <= price_7d <= 25.0 and price_24h >= 8.0 and price_1h >= 0.0 and volume_to_market_cap >= 0.05:
        score = rank_score + volume_score + min(price_24h, 20.0) + min(price_7d, 15.0)
        return (
            "breakout_continuation_watch",
            "long_momentum_watch",
            score,
            f"paper-label {symbol} breakout continuation and stop behavior",
        )
    if (price_7d >= 35.0 or price_30d >= 75.0) and volume_to_market_cap >= 0.08:
        score = rank_score + volume_score + min(max(price_7d, price_30d) / 4.0, 30.0)
        return (
            "chase_risk",
            "wait_or_fade_watch",
            score,
            f"avoid chasing {symbol}; label pullback or fade setup first",
        )
    return "market_breadth_context_only", "none", rank_score + volume_score, f"keep {symbol} in market breadth context"


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=250)
    parser.add_argument("--min-market-cap", type=float, default=100_000_000.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_volume_price_dislocation.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_volume_price_dislocation.md")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_volume_price_dislocation_rows(
        pages=args.pages,
        per_page=args.per_page,
        min_market_cap=args.min_market_cap,
    )
    write_volume_price_dislocation_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.symbol, f"score={row.score:.4f}", f"vol_mcap={row.volume_to_market_cap:.4f}")


if __name__ == "__main__":
    main()
