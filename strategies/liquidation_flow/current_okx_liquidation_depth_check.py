from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OkxDepthCheckRow:
    timestamp: str
    asset: str
    inst_id: str
    action: str
    spread_bps: float
    top_bid_notional: float
    top_ask_notional: float
    bid_depth_5bps: float
    ask_depth_5bps: float
    bid_depth_10bps: float
    ask_depth_10bps: float
    monitor_observations: int
    monitor_mean_score: float
    monitor_mean_liquidation_notional: float
    depth_score: float


def build_depth_check_rows(
    *,
    monitor_summary_path: Path = ROOT / "current_okx_liquidation_monitor_summary.csv",
    top: int = 10,
) -> tuple[OkxDepthCheckRow, ...]:
    timestamp = datetime.now(UTC).isoformat()
    monitor_rows = _read_monitor_rows(monitor_summary_path)[:top]
    instrument_by_asset = _fetch_usdt_swap_instruments()
    rows = tuple(
        _build_depth_check_row(
            timestamp=timestamp,
            monitor_row=row,
            contract_value=instrument_by_asset.get(row["asset"], 0.0),
        )
        for row in monitor_rows
        if instrument_by_asset.get(row["asset"], 0.0) > 0.0
    )
    return tuple(sorted(rows, key=lambda row: row.depth_score, reverse=True))


def write_depth_check_rows(
    rows: tuple[OkxDepthCheckRow, ...],
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
                "inst_id",
                "action",
                "spread_bps",
                "top_bid_notional",
                "top_ask_notional",
                "bid_depth_5bps",
                "ask_depth_5bps",
                "bid_depth_10bps",
                "ask_depth_10bps",
                "monitor_observations",
                "monitor_mean_score",
                "monitor_mean_liquidation_notional",
                "depth_score",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.inst_id,
                    row.action,
                    f"{row.spread_bps:.8f}",
                    f"{row.top_bid_notional:.8f}",
                    f"{row.top_ask_notional:.8f}",
                    f"{row.bid_depth_5bps:.8f}",
                    f"{row.ask_depth_5bps:.8f}",
                    f"{row.bid_depth_10bps:.8f}",
                    f"{row.ask_depth_10bps:.8f}",
                    row.monitor_observations,
                    f"{row.monitor_mean_score:.8f}",
                    f"{row.monitor_mean_liquidation_notional:.8f}",
                    f"{row.depth_score:.8f}",
                )
            )
    return output_path


def write_depth_check_md(
    rows: tuple[OkxDepthCheckRow, ...],
    *,
    output_path: Path,
    top: int = 10,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Depth Check\n\n")
        handle.write(
            "This checks visible OKX book depth for liquidation-monitor candidates. "
            "It is not a fill guarantee.\n\n"
        )
        handle.write(
            "| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.spread_bps:.4f} | "
                f"{row.bid_depth_5bps:.0f} | "
                f"{row.ask_depth_5bps:.0f} | "
                f"{row.bid_depth_10bps:.0f} | "
                f"{row.ask_depth_10bps:.0f} | "
                f"{row.monitor_mean_score:.6f} | "
                f"{row.depth_score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The useful follow-up candidates are liquidation signals that persist "
            "and have enough visible depth near touch. This still excludes account "
            "fees, hidden liquidity, maker fill probability, and slippage during a "
            "fast liquidation event.\n"
        )
    return output_path


def _read_monitor_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _fetch_usdt_swap_instruments() -> dict[str, float]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    return {
        str(item["instId"]).removesuffix("-USDT-SWAP"): float(item.get("ctVal") or 0.0)
        for item in response.json().get("data", ())
        if str(item.get("instId", "")).endswith("-USDT-SWAP")
    }


def _build_depth_check_row(
    *,
    timestamp: str,
    monitor_row: dict[str, str],
    contract_value: float,
) -> OkxDepthCheckRow:
    asset = monitor_row["asset"]
    inst_id = f"{asset}-USDT-SWAP"
    book = _fetch_okx_book(inst_id)
    bid_levels = tuple(_parse_book_level(row, contract_value=contract_value) for row in book["bids"])
    ask_levels = tuple(_parse_book_level(row, contract_value=contract_value) for row in book["asks"])
    best_bid = bid_levels[0][0]
    best_ask = ask_levels[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0.0 else 0.0
    bid_depth_5bps = _depth_within_bps(bid_levels, mid=mid, side="bid", bps=5.0)
    ask_depth_5bps = _depth_within_bps(ask_levels, mid=mid, side="ask", bps=5.0)
    bid_depth_10bps = _depth_within_bps(bid_levels, mid=mid, side="bid", bps=10.0)
    ask_depth_10bps = _depth_within_bps(ask_levels, mid=mid, side="ask", bps=10.0)
    monitor_mean_score = float(monitor_row.get("mean_cascade_score") or 0.0)
    monitor_mean_liquidation_notional = float(
        monitor_row.get("mean_total_liquidation_notional") or 0.0
    )
    near_touch_depth = min(bid_depth_5bps, ask_depth_5bps)
    depth_score = (
        monitor_mean_score
        * min(near_touch_depth / max(monitor_mean_liquidation_notional, 1.0), 2.0)
        / max(spread_bps, 0.1)
    )
    return OkxDepthCheckRow(
        timestamp=timestamp,
        asset=asset,
        inst_id=inst_id,
        action=monitor_row["action"],
        spread_bps=spread_bps,
        top_bid_notional=bid_levels[0][1],
        top_ask_notional=ask_levels[0][1],
        bid_depth_5bps=bid_depth_5bps,
        ask_depth_5bps=ask_depth_5bps,
        bid_depth_10bps=bid_depth_10bps,
        ask_depth_10bps=ask_depth_10bps,
        monitor_observations=int(monitor_row.get("observations") or "0"),
        monitor_mean_score=monitor_mean_score,
        monitor_mean_liquidation_notional=monitor_mean_liquidation_notional,
        depth_score=depth_score,
    )


def _fetch_okx_book(inst_id: str) -> dict[str, list[list[str]]]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": inst_id, "sz": "50"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", ())
    return data[0] if data else {"bids": [], "asks": []}


def _parse_book_level(row: list[str], *, contract_value: float) -> tuple[float, float]:
    price = float(row[0])
    size = float(row[1])
    return price, price * size * contract_value


def _depth_within_bps(
    levels: tuple[tuple[float, float], ...],
    *,
    mid: float,
    side: str,
    bps: float,
) -> float:
    if side == "bid":
        threshold = mid * (1.0 - bps / 10000.0)
        return sum(notional for price, notional in levels if price >= threshold)
    threshold = mid * (1.0 + bps / 10000.0)
    return sum(notional for price, notional in levels if price <= threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-summary-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_summary.csv",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_depth_check.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_depth_check.md",
    )
    args = parser.parse_args()

    rows = build_depth_check_rows(monitor_summary_path=args.monitor_summary_path, top=args.top)
    write_depth_check_rows(rows, output_path=args.output_path)
    write_depth_check_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"spread={row.spread_bps:.4f}",
            f"bid5={row.bid_depth_5bps:.0f}",
            f"ask5={row.ask_depth_5bps:.0f}",
            f"depth_score={row.depth_score:.4f}",
        )


if __name__ == "__main__":
    main()
