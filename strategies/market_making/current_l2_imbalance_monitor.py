from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from strategies.market_making.hyperliquid_l2_snapshot import (
    OrderBookMetrics,
    collect_order_book_metrics,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class L2ImbalanceMonitorSample:
    sample_index: int
    row: OrderBookMetrics


@dataclass(frozen=True)
class L2ImbalanceMonitorSummary:
    asset: str
    observations: int
    dominant_direction: int
    direction_persistence_rate: float
    mean_imbalance_10_bps: float
    mean_abs_imbalance_10_bps: float
    min_abs_imbalance_10_bps: float
    mean_spread_bps: float
    mean_near_depth_10bps_notional: float


def run_l2_imbalance_monitor(
    *,
    assets: tuple[str, ...],
    samples: int = 3,
    delay_seconds: float = 5.0,
) -> tuple[L2ImbalanceMonitorSample, ...]:
    rows: list[L2ImbalanceMonitorSample] = []
    for sample_index in range(samples):
        rows.extend(
            L2ImbalanceMonitorSample(sample_index=sample_index, row=row)
            for row in collect_order_book_metrics(assets)
        )
        if sample_index < samples - 1:
            sleep(delay_seconds)
    return tuple(rows)


def summarize_l2_imbalance_monitor(
    samples: tuple[L2ImbalanceMonitorSample, ...],
) -> tuple[L2ImbalanceMonitorSummary, ...]:
    grouped: dict[str, list[L2ImbalanceMonitorSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.row.asset, []).append(sample)
    summaries = tuple(
        _summarize_group(asset=asset, rows=tuple(rows))
        for asset, rows in grouped.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.observations,
                row.direction_persistence_rate,
                row.mean_abs_imbalance_10_bps,
                row.mean_near_depth_10bps_notional,
            ),
            reverse=True,
        )
    )


def write_l2_imbalance_monitor_samples(
    samples: tuple[L2ImbalanceMonitorSample, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "timestamp",
                "asset",
                "spread_bps",
                "bid_depth_10_bps",
                "ask_depth_10_bps",
                "mid_price",
                "imbalance_10_bps",
                "near_depth_10bps_notional",
            )
        )
        for sample in samples:
            row = sample.row
            writer.writerow(
                (
                    sample.sample_index,
                    row.timestamp,
                    row.asset,
                    f"{row.spread_bps:.8f}",
                    f"{row.bid_depth_10_bps:.8f}",
                    f"{row.ask_depth_10_bps:.8f}",
                    f"{row.mid_price:.12f}",
                    f"{row.imbalance_10_bps:.8f}",
                    f"{_near_depth_notional(row):.8f}",
                )
            )
    return output_path


def write_l2_imbalance_monitor_summary(
    summaries: tuple[L2ImbalanceMonitorSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "observations",
                "dominant_direction",
                "direction_persistence_rate",
                "mean_imbalance_10_bps",
                "mean_abs_imbalance_10_bps",
                "min_abs_imbalance_10_bps",
                "mean_spread_bps",
                "mean_near_depth_10bps_notional",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.asset,
                    row.observations,
                    row.dominant_direction,
                    f"{row.direction_persistence_rate:.8f}",
                    f"{row.mean_imbalance_10_bps:.8f}",
                    f"{row.mean_abs_imbalance_10_bps:.8f}",
                    f"{row.min_abs_imbalance_10_bps:.8f}",
                    f"{row.mean_spread_bps:.8f}",
                    f"{row.mean_near_depth_10bps_notional:.8f}",
                )
            )
    return output_path


def write_l2_imbalance_monitor_summary_md(
    summaries: tuple[L2ImbalanceMonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current L2 Imbalance Monitor\n\n")
        handle.write(
            "This repeats the broad Hyperliquid L2 imbalance snapshot over a short "
            "window. It is a persistence check, not a fill model or trade instruction.\n\n"
        )
        handle.write(
            "| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.observations} | "
                f"{row.dominant_direction} | "
                f"{row.direction_persistence_rate:.4f} | "
                f"{row.mean_imbalance_10_bps:.4f} | "
                f"{row.mean_abs_imbalance_10_bps:.4f} | "
                f"{row.min_abs_imbalance_10_bps:.4f} | "
                f"{row.mean_spread_bps:.4f} | "
                f"{row.mean_near_depth_10bps_notional:.0f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High persistence with high absolute imbalance is a better paper-label "
            "candidate than a one-off snapshot. It still needs 15m/1h forward labels "
            "and a real maker-fill/adverse-selection model.\n"
        )
    return output_path


def _summarize_group(
    *,
    asset: str,
    rows: tuple[L2ImbalanceMonitorSample, ...],
) -> L2ImbalanceMonitorSummary:
    signed_directions = tuple(_direction(row.row.imbalance_10_bps) for row in rows)
    dominant_direction = _dominant_direction(signed_directions)
    return L2ImbalanceMonitorSummary(
        asset=asset,
        observations=len(rows),
        dominant_direction=dominant_direction,
        direction_persistence_rate=(
            sum(direction == dominant_direction for direction in signed_directions) / len(rows)
            if dominant_direction != 0
            else 0.0
        ),
        mean_imbalance_10_bps=sum(row.row.imbalance_10_bps for row in rows) / len(rows),
        mean_abs_imbalance_10_bps=(
            sum(abs(row.row.imbalance_10_bps) for row in rows) / len(rows)
        ),
        min_abs_imbalance_10_bps=min(abs(row.row.imbalance_10_bps) for row in rows),
        mean_spread_bps=sum(row.row.spread_bps for row in rows) / len(rows),
        mean_near_depth_10bps_notional=sum(_near_depth_notional(row.row) for row in rows)
        / len(rows),
    )


def _assets_from_snapshot(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(row["asset"] for row in csv.DictReader(handle))


def _near_depth_notional(row: OrderBookMetrics) -> float:
    return min(row.bid_depth_10_bps, row.ask_depth_10_bps) * row.mid_price


def _direction(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _dominant_direction(values: tuple[int, ...]) -> int:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    if positives > negatives:
        return 1
    if negatives > positives:
        return -1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument(
        "--asset-source-path",
        type=Path,
        default=ROOT / "current_l2_snapshot.csv",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_monitor_summary.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    assets = tuple(args.assets) if args.assets else _assets_from_snapshot(args.asset_source_path)
    samples = run_l2_imbalance_monitor(
        assets=assets,
        samples=args.samples,
        delay_seconds=args.delay_seconds,
    )
    summaries = summarize_l2_imbalance_monitor(samples)
    write_l2_imbalance_monitor_samples(samples, output_path=args.samples_output_path)
    write_l2_imbalance_monitor_summary(summaries, output_path=args.summary_output_path)
    write_l2_imbalance_monitor_summary_md(
        summaries,
        output_path=args.md_output_path,
        top=args.top,
    )
    for row in summaries[: args.top]:
        print(
            row.asset,
            f"obs={row.observations}",
            f"dir={row.dominant_direction}",
            f"persist={row.direction_persistence_rate:.2f}",
            f"mean_abs={row.mean_abs_imbalance_10_bps:.4f}",
            f"depth={row.mean_near_depth_10bps_notional:.0f}",
        )


if __name__ == "__main__":
    main()
