from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_ASSETS = ("BTC", "ETH", "SOL", "HYPE")


@dataclass(frozen=True)
class OrderBookMetrics:
    timestamp: str
    asset: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    bid_depth_10_bps: float
    ask_depth_10_bps: float
    bid_depth_50_bps: float
    ask_depth_50_bps: float
    imbalance_10_bps: float
    top_bid_orders: int
    top_ask_orders: int


def fetch_l2_book(asset: str, url: str = HYPERLIQUID_INFO_URL) -> dict[str, object]:
    response = requests.post(url, json={"type": "l2Book", "coin": asset}, timeout=30)
    response.raise_for_status()
    return response.json()


def build_order_book_metrics(
    payload: dict[str, object],
    *,
    timestamp: str | None = None,
) -> OrderBookMetrics:
    bids, asks = payload["levels"]
    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    mid_price = (best_bid + best_ask) / 2.0
    bid_depth_10_bps = _depth_within_bps(bids, mid_price=mid_price, bps=10.0, side="bid")
    ask_depth_10_bps = _depth_within_bps(asks, mid_price=mid_price, bps=10.0, side="ask")
    bid_depth_50_bps = _depth_within_bps(bids, mid_price=mid_price, bps=50.0, side="bid")
    ask_depth_50_bps = _depth_within_bps(asks, mid_price=mid_price, bps=50.0, side="ask")
    return OrderBookMetrics(
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        asset=str(payload["coin"]),
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_bps=((best_ask - best_bid) / mid_price) * 10_000.0,
        bid_depth_10_bps=bid_depth_10_bps,
        ask_depth_10_bps=ask_depth_10_bps,
        bid_depth_50_bps=bid_depth_50_bps,
        ask_depth_50_bps=ask_depth_50_bps,
        imbalance_10_bps=_imbalance(bid_depth_10_bps, ask_depth_10_bps),
        top_bid_orders=int(bids[0]["n"]),
        top_ask_orders=int(asks[0]["n"]),
    )


def collect_order_book_metrics(
    assets: tuple[str, ...] = DEFAULT_ASSETS,
) -> tuple[OrderBookMetrics, ...]:
    observed_at = datetime.now(UTC).isoformat()
    return tuple(
        build_order_book_metrics(fetch_l2_book(asset), timestamp=observed_at)
        for asset in assets
    )


def write_order_book_metrics(
    rows: tuple[OrderBookMetrics, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "asset",
                "best_bid",
                "best_ask",
                "mid_price",
                "spread_bps",
                "bid_depth_10_bps",
                "ask_depth_10_bps",
                "bid_depth_50_bps",
                "ask_depth_50_bps",
                "imbalance_10_bps",
                "top_bid_orders",
                "top_ask_orders",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    f"{row.best_bid:.12f}",
                    f"{row.best_ask:.12f}",
                    f"{row.mid_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.bid_depth_10_bps:.8f}",
                    f"{row.ask_depth_10_bps:.8f}",
                    f"{row.bid_depth_50_bps:.8f}",
                    f"{row.ask_depth_50_bps:.8f}",
                    f"{row.imbalance_10_bps:.8f}",
                    row.top_bid_orders,
                    row.top_ask_orders,
                )
            )
    return output_path


def _depth_within_bps(
    levels: list[dict[str, object]],
    *,
    mid_price: float,
    bps: float,
    side: str,
) -> float:
    if side == "bid":
        threshold = mid_price * (1.0 - (bps / 10_000.0))
        return sum(float(level["sz"]) for level in levels if float(level["px"]) >= threshold)
    threshold = mid_price * (1.0 + (bps / 10_000.0))
    return sum(float(level["sz"]) for level in levels if float(level["px"]) <= threshold)


def _imbalance(bid_depth: float, ask_depth: float) -> float:
    denominator = bid_depth + ask_depth
    return ((bid_depth - ask_depth) / denominator) if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_l2_snapshot.csv",
    )
    args = parser.parse_args()

    rows = collect_order_book_metrics(tuple(args.assets))
    write_order_book_metrics(rows, output_path=args.output_path)
    for row in rows:
        print(
            row.asset,
            f"spread_bps={row.spread_bps:.4f}",
            f"depth10_bid={row.bid_depth_10_bps:.4f}",
            f"depth10_ask={row.ask_depth_10_bps:.4f}",
            f"imbalance10={row.imbalance_10_bps:.4f}",
        )


if __name__ == "__main__":
    main()

