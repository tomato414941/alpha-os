from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import log10
from pathlib import Path

import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PolymarketMicrostructureRow:
    market_id: str
    question: str
    slug: str
    yes_token_id: str
    no_token_id: str
    action: str
    best_bid: float
    best_ask: float
    spread: float
    midpoint: float
    last_trade_price: float
    one_day_price_change: float
    volume_24h: float
    volume_1w: float
    liquidity: float
    competitive: float
    order_min_size: float
    min_tick: float
    end_date: str
    score: float
    reason: str


def fetch_polymarket_markets(
    *,
    limit: int = 200,
    order: str = "volume24hr",
    ascending: bool = False,
) -> tuple[dict[str, object], ...]:
    response = requests.get(
        f"{GAMMA_BASE_URL}/markets",
        params={
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": order,
            "ascending": str(ascending).lower(),
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple(response.json())


def build_polymarket_microstructure_rows(
    markets: tuple[dict[str, object], ...],
) -> tuple[PolymarketMicrostructureRow, ...]:
    rows = tuple(_build_row(market) for market in markets)
    candidates = tuple(row for row in rows if row.action != "ignore")
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_polymarket_microstructure_csv(
    rows: tuple[PolymarketMicrostructureRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "slug",
                "yes_token_id",
                "no_token_id",
                "action",
                "best_bid",
                "best_ask",
                "spread",
                "midpoint",
                "last_trade_price",
                "one_day_price_change",
                "volume_24h",
                "volume_1w",
                "liquidity",
                "competitive",
                "order_min_size",
                "min_tick",
                "end_date",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.slug,
                    row.yes_token_id,
                    row.no_token_id,
                    row.action,
                    f"{row.best_bid:.6f}",
                    f"{row.best_ask:.6f}",
                    f"{row.spread:.6f}",
                    f"{row.midpoint:.6f}",
                    f"{row.last_trade_price:.6f}",
                    f"{row.one_day_price_change:.6f}",
                    f"{row.volume_24h:.6f}",
                    f"{row.volume_1w:.6f}",
                    f"{row.liquidity:.6f}",
                    f"{row.competitive:.8f}",
                    f"{row.order_min_size:.6f}",
                    f"{row.min_tick:.6f}",
                    row.end_date,
                    f"{row.score:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_polymarket_microstructure_md(
    rows: tuple[PolymarketMicrostructureRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Polymarket Microstructure Screen\n\n")
        handle.write(
            "This screen looks for active event markets with enough public activity "
            "to justify prediction-model or market-making work. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.action} | "
                f"{_escape(row.question)} | "
                f"{row.best_bid:.4f} | "
                f"{row.best_ask:.4f} | "
                f"{row.spread:.4f} | "
                f"{row.midpoint:.4f} | "
                f"{row.one_day_price_change:.4f} | "
                f"{row.volume_24h:.2f} | "
                f"{row.liquidity:.2f} | "
                f"{row.score:.4f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`information_flow_watch` means a high-volume market moved materially in "
            "the last day with a tradable order book. `market_making_watch` means "
            "volume exists but the visible spread is still wide enough to deserve "
            "fill/adverse-selection research. This screen does not estimate true event probability.\n"
        )
    return output_path


def _build_row(market: dict[str, object]) -> PolymarketMicrostructureRow:
    yes_token_id, no_token_id = _token_ids(market.get("clobTokenIds"))
    best_bid = _float(market.get("bestBid"))
    best_ask = _float(market.get("bestAsk"))
    spread = _float(market.get("spread"))
    midpoint = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else 0.0
    one_day_price_change = _float(market.get("oneDayPriceChange"))
    volume_24h = _float(market.get("volume24hr"))
    volume_1w = _float(market.get("volume1wk"))
    liquidity = _float(market.get("liquidityNum") or market.get("liquidity"))
    competitive = _float(market.get("competitive"))
    action = _action(
        accepting_orders=bool(market.get("acceptingOrders")),
        enable_order_book=bool(market.get("enableOrderBook")),
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        one_day_price_change=one_day_price_change,
        volume_24h=volume_24h,
        liquidity=liquidity,
    )
    return PolymarketMicrostructureRow(
        market_id=str(market.get("id") or ""),
        question=str(market.get("question") or ""),
        slug=str(market.get("slug") or ""),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        action=action,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        midpoint=midpoint,
        last_trade_price=_float(market.get("lastTradePrice")),
        one_day_price_change=one_day_price_change,
        volume_24h=volume_24h,
        volume_1w=volume_1w,
        liquidity=liquidity,
        competitive=competitive,
        order_min_size=_float(market.get("orderMinSize")),
        min_tick=_float(market.get("orderPriceMinTickSize")),
        end_date=str(market.get("endDate") or market.get("endDateIso") or ""),
        score=_score(
            action=action,
            spread=spread,
            one_day_price_change=one_day_price_change,
            volume_24h=volume_24h,
            volume_1w=volume_1w,
            liquidity=liquidity,
            competitive=competitive,
        ),
        reason=_reason(action),
    )


def _action(
    *,
    accepting_orders: bool,
    enable_order_book: bool,
    best_bid: float,
    best_ask: float,
    spread: float,
    one_day_price_change: float,
    volume_24h: float,
    liquidity: float,
) -> str:
    if not accepting_orders or not enable_order_book or best_bid <= 0.0 or best_ask <= 0.0:
        return "ignore"
    if volume_24h >= 10_000.0 and abs(one_day_price_change) >= 0.02 and spread <= 0.05:
        return "information_flow_watch"
    if volume_24h >= 10_000.0 and liquidity >= 5_000.0 and spread >= 0.02:
        return "market_making_watch"
    return "ignore"


def _score(
    *,
    action: str,
    spread: float,
    one_day_price_change: float,
    volume_24h: float,
    volume_1w: float,
    liquidity: float,
    competitive: float,
) -> float:
    if action == "ignore":
        return float("-inf")
    activity = log10(max(volume_24h, 1.0)) + (0.25 * log10(max(volume_1w, 1.0)))
    liquidity_score = 0.5 * log10(max(liquidity, 1.0))
    if action == "information_flow_watch":
        return activity + liquidity_score + (abs(one_day_price_change) * 20.0) - spread
    if action == "market_making_watch":
        return activity + liquidity_score + (spread * 10.0) + competitive
    return float("-inf")


def _reason(action: str) -> str:
    if action == "information_flow_watch":
        return "high activity and material one-day price move"
    if action == "market_making_watch":
        return "high activity with non-trivial visible spread"
    return "not enough public activity or order-book signal"


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _token_ids(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = []
    yes_token_id = str(parsed[0]) if len(parsed) > 0 else ""
    no_token_id = str(parsed[1]) if len(parsed) > 1 else ""
    return yes_token_id, no_token_id


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    markets = fetch_polymarket_markets(limit=args.limit)
    rows = build_polymarket_microstructure_rows(markets)
    write_polymarket_microstructure_csv(rows, output_path=args.csv_output_path)
    write_polymarket_microstructure_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.action,
            f"spread={row.spread:.4f}",
            f"change={row.one_day_price_change:.4f}",
            f"vol24={row.volume_24h:.0f}",
            row.question,
        )


if __name__ == "__main__":
    main()
