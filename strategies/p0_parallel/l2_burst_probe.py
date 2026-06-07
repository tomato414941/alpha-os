from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from strategies.market_making.hyperliquid_l2_snapshot import (
    DEFAULT_ASSETS,
    build_order_book_metrics,
    fetch_l2_book,
)


@dataclass(frozen=True)
class L2BurstRow:
    timestamp: str
    sequence: int
    asset: str
    mid_price: float
    spread_bps: float
    imbalance_10_bps: float
    bid_depth_10_bps: float
    ask_depth_10_bps: float


@dataclass(frozen=True)
class L2BurstSummary:
    asset: str
    samples: int
    mean_spread_bps: float
    mean_abs_imbalance_10_bps: float
    mean_next_mid_return_after_positive_imbalance: float
    mean_next_mid_return_after_negative_imbalance: float


def collect_l2_burst(
    *,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    samples: int = 8,
    delay_seconds: float = 1.0,
) -> tuple[L2BurstRow, ...]:
    rows: list[L2BurstRow] = []
    for sequence in range(samples):
        timestamp = datetime.now(UTC).isoformat()
        for asset in assets:
            metrics = build_order_book_metrics(fetch_l2_book(asset), timestamp=timestamp)
            rows.append(
                L2BurstRow(
                    timestamp=timestamp,
                    sequence=sequence,
                    asset=asset,
                    mid_price=metrics.mid_price,
                    spread_bps=metrics.spread_bps,
                    imbalance_10_bps=metrics.imbalance_10_bps,
                    bid_depth_10_bps=metrics.bid_depth_10_bps,
                    ask_depth_10_bps=metrics.ask_depth_10_bps,
                )
            )
        if sequence != samples - 1:
            time.sleep(delay_seconds)
    return tuple(rows)


def summarize_l2_burst(rows: tuple[L2BurstRow, ...]) -> tuple[L2BurstSummary, ...]:
    by_asset: dict[str, list[L2BurstRow]] = {}
    for row in rows:
        by_asset.setdefault(row.asset, []).append(row)
    summaries = []
    for asset, asset_rows in sorted(by_asset.items()):
        asset_rows = sorted(asset_rows, key=lambda row: row.sequence)
        positive_returns = []
        negative_returns = []
        for current, next_row in zip(asset_rows, asset_rows[1:]):
            if current.mid_price <= 0.0:
                continue
            next_return = (next_row.mid_price / current.mid_price) - 1.0
            if current.imbalance_10_bps >= 0.0:
                positive_returns.append(next_return)
            else:
                negative_returns.append(next_return)
        summaries.append(
            L2BurstSummary(
                asset=asset,
                samples=len(asset_rows),
                mean_spread_bps=mean(row.spread_bps for row in asset_rows),
                mean_abs_imbalance_10_bps=mean(abs(row.imbalance_10_bps) for row in asset_rows),
                mean_next_mid_return_after_positive_imbalance=(
                    mean(positive_returns) if positive_returns else 0.0
                ),
                mean_next_mid_return_after_negative_imbalance=(
                    mean(negative_returns) if negative_returns else 0.0
                ),
            )
        )
    return tuple(summaries)


def write_l2_burst_rows(rows: tuple[L2BurstRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "sequence",
                "asset",
                "mid_price",
                "spread_bps",
                "imbalance_10_bps",
                "bid_depth_10_bps",
                "ask_depth_10_bps",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.sequence,
                    row.asset,
                    f"{row.mid_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.imbalance_10_bps:.8f}",
                    f"{row.bid_depth_10_bps:.8f}",
                    f"{row.ask_depth_10_bps:.8f}",
                )
            )
    return output_path


def write_l2_burst_summaries(
    summaries: tuple[L2BurstSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "samples",
                "mean_spread_bps",
                "mean_abs_imbalance_10_bps",
                "mean_next_mid_return_after_positive_imbalance",
                "mean_next_mid_return_after_negative_imbalance",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.asset,
                    summary.samples,
                    f"{summary.mean_spread_bps:.8f}",
                    f"{summary.mean_abs_imbalance_10_bps:.8f}",
                    f"{summary.mean_next_mid_return_after_positive_imbalance:.12f}",
                    f"{summary.mean_next_mid_return_after_negative_imbalance:.12f}",
                )
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--rows-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "l2_burst.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "l2_burst_summary.csv",
    )
    args = parser.parse_args()

    rows = collect_l2_burst(
        assets=tuple(args.assets),
        samples=args.samples,
        delay_seconds=args.delay_seconds,
    )
    summaries = summarize_l2_burst(rows)
    write_l2_burst_rows(rows, output_path=args.rows_output_path)
    write_l2_burst_summaries(summaries, output_path=args.summary_output_path)
    for summary in summaries:
        print(
            summary.asset,
            f"samples={summary.samples}",
            f"spread={summary.mean_spread_bps:.4f}",
            f"abs_imb={summary.mean_abs_imbalance_10_bps:.4f}",
            f"pos_next={summary.mean_next_mid_return_after_positive_imbalance:.8f}",
            f"neg_next={summary.mean_next_mid_return_after_negative_imbalance:.8f}",
        )


if __name__ == "__main__":
    main()

