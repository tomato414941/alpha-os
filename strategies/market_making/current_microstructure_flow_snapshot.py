from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from strategies.market_making.hyperliquid_l2_snapshot import (
    build_order_book_metrics,
    fetch_l2_book,
)


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent
DEFAULT_ASSETS = ("BTC", "ETH", "SOL", "HYPE", "NEAR", "BNB", "SUI", "ARB", "ADA", "MEGA")


@dataclass(frozen=True)
class MicrostructureFlowSnapshot:
    timestamp: str
    asset: str
    mid_price: float
    spread_bps: float
    depth_10bps_usd: float
    book_imbalance_10bps: float
    trade_window_seconds: float
    trade_count: int
    buy_notional: float
    sell_notional: float
    trade_imbalance: float
    pressure_score: float
    direction: int
    action: str
    reason: str


def build_microstructure_flow_snapshots(
    *,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[MicrostructureFlowSnapshot, ...]:
    observed_at = datetime.now(UTC)
    rows: list[MicrostructureFlowSnapshot] = []
    for asset in assets:
        try:
            rows.append(_build_snapshot(asset=asset, observed_at=observed_at, url=url))
        except (KeyError, IndexError, requests.RequestException, ValueError):
            continue
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_microstructure_flow_snapshots(
    rows: tuple[MicrostructureFlowSnapshot, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset",
                "mid_price",
                "spread_bps",
                "depth_10bps_usd",
                "book_imbalance_10bps",
                "trade_window_seconds",
                "trade_count",
                "buy_notional",
                "sell_notional",
                "trade_imbalance",
                "pressure_score",
                "direction",
                "action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    f"{row.mid_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.depth_10bps_usd:.8f}",
                    f"{row.book_imbalance_10bps:.8f}",
                    f"{row.trade_window_seconds:.2f}",
                    row.trade_count,
                    f"{row.buy_notional:.8f}",
                    f"{row.sell_notional:.8f}",
                    f"{row.trade_imbalance:.8f}",
                    f"{row.pressure_score:.8f}",
                    row.direction,
                    row.action,
                    row.reason,
                )
            )
    return output_path


def write_microstructure_flow_snapshots_md(
    rows: tuple[MicrostructureFlowSnapshot, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Microstructure Flow Snapshot\n\n")
        handle.write(
            "This joins Hyperliquid public book imbalance with recent trade-print "
            "imbalance. It is a short-horizon microstructure observation, not a "
            "deployable market-making model.\n\n"
        )
        handle.write(
            "| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.direction} | "
                f"{row.pressure_score:.4f} | "
                f"{row.book_imbalance_10bps:.4f} | "
                f"{row.trade_imbalance:.4f} | "
                f"{row.trade_count} | "
                f"{row.buy_notional:.0f} | "
                f"{row.sell_notional:.0f} | "
                f"{row.spread_bps:.4f} | "
                f"{row.depth_10bps_usd:.0f} | "
                f"{row.trade_window_seconds:.0f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`aligned_pressure_watch` means visible book pressure and recent "
            "taker flow point in the same direction. `book_trade_divergence_watch` "
            "means the book and taker flow disagree; that can be adverse selection "
            "or absorption, so it needs separate forward labels.\n"
        )
    return output_path


def _build_snapshot(*, asset: str, observed_at: datetime, url: str) -> MicrostructureFlowSnapshot:
    book = build_order_book_metrics(fetch_l2_book(asset, url=url), timestamp=observed_at.isoformat())
    trades = _fetch_recent_trades(asset, url=url)
    buy_notional = _trade_notional(trades, side="B")
    sell_notional = _trade_notional(trades, side="A")
    trade_imbalance = _imbalance(buy_notional, sell_notional)
    pressure_score = (book.imbalance_10_bps + trade_imbalance) / 2.0
    direction = _direction(pressure_score)
    action, reason = _action(
        spread_bps=book.spread_bps,
        book_imbalance=book.imbalance_10_bps,
        trade_imbalance=trade_imbalance,
        trade_count=len(trades),
    )
    return MicrostructureFlowSnapshot(
        timestamp=observed_at.isoformat(),
        asset=asset,
        mid_price=book.mid_price,
        spread_bps=book.spread_bps,
        depth_10bps_usd=min(book.bid_depth_10_bps, book.ask_depth_10_bps) * book.mid_price,
        book_imbalance_10bps=book.imbalance_10_bps,
        trade_window_seconds=_trade_window_seconds(trades),
        trade_count=len(trades),
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        trade_imbalance=trade_imbalance,
        pressure_score=pressure_score,
        direction=direction,
        action=action,
        reason=reason,
    )


def _fetch_recent_trades(asset: str, *, url: str) -> tuple[dict[str, object], ...]:
    response = requests.post(url, json={"type": "recentTrades", "coin": asset}, timeout=30)
    response.raise_for_status()
    return tuple(response.json())


def _trade_notional(trades: tuple[dict[str, object], ...], *, side: str) -> float:
    return sum(
        float(trade["px"]) * float(trade["sz"])
        for trade in trades
        if str(trade.get("side", "")) == side
    )


def _trade_window_seconds(trades: tuple[dict[str, object], ...]) -> float:
    times = tuple(float(trade["time"]) for trade in trades if "time" in trade)
    if len(times) < 2:
        return 0.0
    return (max(times) - min(times)) / 1000.0


def _imbalance(left: float, right: float) -> float:
    denominator = left + right
    return ((left - right) / denominator) if denominator > 0.0 else 0.0


def _direction(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _action(
    *,
    spread_bps: float,
    book_imbalance: float,
    trade_imbalance: float,
    trade_count: int,
) -> tuple[str, str]:
    if trade_count < 8:
        return "insufficient_trade_prints", "recent trade sample is too small"
    if spread_bps > 8.0:
        return "wide_spread_watch", "spread is too wide for a first microstructure probe"
    book_dir = _direction(book_imbalance)
    trade_dir = _direction(trade_imbalance)
    if abs(book_imbalance) < 0.15 and abs(trade_imbalance) < 0.15:
        return "no_clear_pressure", "book and taker-flow imbalance are both small"
    if book_dir != 0 and trade_dir != 0 and book_dir == trade_dir:
        return "aligned_pressure_watch", "book imbalance and taker flow point the same way"
    if book_dir != 0 and trade_dir != 0 and book_dir != trade_dir:
        return "book_trade_divergence_watch", "book imbalance and taker flow disagree"
    return "one_sided_pressure_watch", "only one side of book or taker flow is clearly imbalanced"


def _sort_key(row: MicrostructureFlowSnapshot) -> tuple[int, float, int, float]:
    action_rank = {
        "aligned_pressure_watch": 4,
        "book_trade_divergence_watch": 3,
        "one_sided_pressure_watch": 2,
        "no_clear_pressure": 1,
        "wide_spread_watch": 0,
        "insufficient_trade_prints": 0,
    }
    return (
        action_rank.get(row.action, 0),
        abs(row.pressure_score),
        row.trade_count,
        -row.spread_bps,
    )


def _assets_from_queue(path: Path, *, top: int) -> tuple[str, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    return tuple(row["asset"] for row in rows[:top] if row.get("asset"))


def _merge_assets(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for asset in left + right:
        normalized = asset.upper()
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument(
        "--asset-source-path",
        type=Path,
        default=STRATEGIES_ROOT / "candidate_validation" / "current_followup_queue.csv",
    )
    parser.add_argument("--asset-source-top", type=int, default=20)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_snapshot.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_snapshot.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    assets = _merge_assets(
        tuple(args.assets),
        _assets_from_queue(args.asset_source_path, top=args.asset_source_top),
    )
    rows = build_microstructure_flow_snapshots(assets=assets)
    write_microstructure_flow_snapshots(rows, output_path=args.output_path)
    write_microstructure_flow_snapshots_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"dir={row.direction}",
            f"pressure={row.pressure_score:.4f}",
            f"book={row.book_imbalance_10bps:.4f}",
            f"trade={row.trade_imbalance:.4f}",
            f"trades={row.trade_count}",
        )


if __name__ == "__main__":
    main()
