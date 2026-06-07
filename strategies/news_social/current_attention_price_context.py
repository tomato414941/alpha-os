from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"


@dataclass(frozen=True)
class AttentionPriceContextRow:
    symbol: str
    name: str
    coin_id: str
    attention_rank: int
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


def build_attention_price_context_rows(
    *,
    attention_path: Path = ROOT / "current_attention_snapshot.csv",
) -> tuple[AttentionPriceContextRow, ...]:
    attention_rows = tuple(
        row for row in _read_rows(attention_path)
        if row.get("source") == "coingecko_trending" and row.get("asset_id")
    )
    markets = _fetch_markets(tuple(row.get("asset_id", "") for row in attention_rows))
    rows = tuple(_build_row(attention=row, market=markets.get(row.get("asset_id", ""), {})) for row in attention_rows)
    candidates = tuple(row for row in rows if row.status != "attention_only_watch")
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_attention_price_context_csv(
    rows: tuple[AttentionPriceContextRow, ...],
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
                "attention_rank",
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
                    row.attention_rank,
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


def write_attention_price_context_md(
    rows: tuple[AttentionPriceContextRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Attention Price Context\n\n")
        handle.write(
            "This joins CoinGecko trending attention to current price movement. "
            "It looks for attention-price lag, breakout continuation, and chase-risk candidates.\n\n"
        )
        handle.write(
            "| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.name} | {row.status} | {row.side} | {row.score:.4f} | "
                f"{row.attention_rank} | {row.price_change_24h:.2f} | {row.price_change_7d:.2f} | "
                f"{row.price_change_30d:.2f} | {row.volume_to_market_cap:.4f} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`attention_price_lag_candidate` means attention is high while 7d price remains weak. "
            "`attention_breakout_continuation_watch` means attention and price are already moving together. "
            "`attention_chase_risk` is a warning that the easy move may already be crowded.\n"
        )
    return output_path


def _fetch_markets(coin_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    if not coin_ids:
        return {}
    try:
        response = requests.get(
            COINGECKO_MARKETS_URL,
            params={
                "vs_currency": "usd",
                "ids": ",".join(sorted(set(coin_ids))),
                "per_page": 250,
                "price_change_percentage": "1h,24h,7d,30d",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return {}
    return {str(row.get("id") or ""): row for row in response.json()}


def _build_row(
    *,
    attention: dict[str, str],
    market: dict[str, object],
) -> AttentionPriceContextRow:
    symbol = attention.get("symbol", "").upper()
    rank = int(attention.get("rank") or "0")
    current_price = _float(market.get("current_price"))
    market_cap = _float(market.get("market_cap"))
    total_volume = _float(market.get("total_volume"))
    volume_to_market_cap = total_volume / market_cap if market_cap > 0.0 else 0.0
    price_1h = _float(market.get("price_change_percentage_1h_in_currency"))
    price_24h = _float(market.get("price_change_percentage_24h_in_currency") or attention.get("value"))
    price_7d = _float(market.get("price_change_percentage_7d_in_currency"))
    price_30d = _float(market.get("price_change_percentage_30d_in_currency"))
    status, side, score, next_step = _status_side_score_next_step(
        symbol=symbol,
        rank=rank,
        price_24h=price_24h,
        price_7d=price_7d,
        price_30d=price_30d,
        volume_to_market_cap=volume_to_market_cap,
        has_market=current_price > 0.0,
    )
    evidence = (
        f"rank={rank}; price_1h={price_1h:.2f}; price_24h={price_24h:.2f}; "
        f"price_7d={price_7d:.2f}; price_30d={price_30d:.2f}; "
        f"volume_to_mcap={volume_to_market_cap:.4f}"
    )
    return AttentionPriceContextRow(
        symbol=symbol,
        name=attention.get("name", ""),
        coin_id=attention.get("asset_id", ""),
        attention_rank=rank,
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
    price_24h: float,
    price_7d: float,
    price_30d: float,
    volume_to_market_cap: float,
    has_market: bool,
) -> tuple[str, str, float, str]:
    if not has_market:
        return "attention_only_watch", "none", 0.0, f"collect market data for {symbol} before promotion"
    attention_score = max(16 - rank, 0)
    liquidity_score = min(volume_to_market_cap * 50.0, 10.0)
    if rank <= 10 and price_7d <= 0.0 and price_24h >= 0.0:
        score = attention_score + min(abs(price_7d), 20.0) + min(price_24h, 10.0) + liquidity_score
        return (
            "attention_price_lag_candidate",
            "long_attention_lag",
            score,
            f"paper-label {symbol} attention-price lag over 1h, 4h, 12h, and 24h",
        )
    if rank <= 10 and price_7d <= -15.0 and price_24h < 0.0:
        score = attention_score + min(abs(price_7d), 20.0) + liquidity_score
        return (
            "attention_capitulation_reversal_watch",
            "watch_reversal_or_no_trade",
            score,
            f"wait for {symbol} reversal trigger, then label attention capitulation returns",
        )
    if rank <= 5 and 0.0 <= price_7d <= 20.0 and price_24h >= 5.0:
        score = attention_score + min(price_24h, 15.0) + min(price_7d, 10.0) + liquidity_score
        return (
            "attention_breakout_continuation_watch",
            "long_momentum_watch",
            score,
            f"paper-label {symbol} attention breakout continuation and stop behavior",
        )
    if rank <= 10 and (price_7d > 25.0 or price_30d > 60.0):
        score = attention_score + min(max(price_7d, price_30d) / 5.0, 20.0) + liquidity_score
        return (
            "attention_chase_risk",
            "wait_or_fade_watch",
            score,
            f"avoid chasing {symbol}; label pullback or fade setups instead",
        )
    return "attention_only_watch", "none", attention_score, f"collect another {symbol} attention snapshot"


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
    parser.add_argument("--attention-path", type=Path, default=ROOT / "current_attention_snapshot.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_attention_price_context.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_attention_price_context.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_attention_price_context_rows(attention_path=args.attention_path)
    write_attention_price_context_csv(rows, output_path=args.output_path)
    write_attention_price_context_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.symbol, f"score={row.score:.4f}", f"price7d={row.price_change_7d:.2f}")


if __name__ == "__main__":
    main()
