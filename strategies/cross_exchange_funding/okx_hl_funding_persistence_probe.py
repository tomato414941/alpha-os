from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.cross_exchange_funding.current_okx_hl_funding_spread import (
    OkxHlFundingSpread,
    build_okx_hl_funding_spreads,
)


@dataclass(frozen=True)
class PersistenceSummary:
    asset: str
    observations: int
    dominant_long_venue: str
    dominant_short_venue: str
    mean_annualized_spread: float
    mean_net_8h_proxy: float
    min_net_8h_proxy: float
    max_net_8h_proxy: float
    positive_net_8h_rate: float
    mean_net_24h_proxy: float
    mean_breakeven_hold_hours: float
    mean_capacity_proxy_notional: float


def collect_okx_hl_persistence(
    *,
    samples: int,
    delay_seconds: float,
    max_workers: int,
    assets: tuple[str, ...] | None = None,
) -> tuple[OkxHlFundingSpread, ...]:
    rows: list[OkxHlFundingSpread] = []
    for sample_index in range(samples):
        rows.extend(
            build_okx_hl_funding_spreads(max_workers=max_workers, assets=assets)
        )
        if sample_index + 1 < samples:
            time.sleep(delay_seconds)
    return tuple(rows)


def summarize_persistence(
    rows: tuple[OkxHlFundingSpread, ...],
) -> tuple[PersistenceSummary, ...]:
    by_asset: dict[str, list[OkxHlFundingSpread]] = {}
    for row in rows:
        by_asset.setdefault(row.asset, []).append(row)
    summaries = tuple(
        _summarize_asset(asset=asset, rows=tuple(asset_rows))
        for asset, asset_rows in by_asset.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.positive_net_8h_rate,
                summary.mean_net_8h_proxy,
                summary.mean_capacity_proxy_notional,
            ),
            reverse=True,
        )
    )


def write_persistence_rows(
    rows: tuple[OkxHlFundingSpread, ...],
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
                "long_venue",
                "short_venue",
                "annualized_spread",
                "rough_round_trip_cost",
                "breakeven_hold_hours",
                "net_8h_proxy",
                "net_24h_proxy",
                "capacity_proxy_notional",
                "notes",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    f"{row.annualized_spread:.8f}",
                    f"{row.rough_round_trip_cost:.8f}",
                    f"{row.breakeven_hold_hours:.4f}",
                    f"{row.net_8h_proxy:.8f}",
                    f"{row.net_24h_proxy:.8f}",
                    f"{row.capacity_proxy_notional:.8f}",
                    row.notes,
                )
            )
    return output_path


def write_persistence_summary(
    summaries: tuple[PersistenceSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "observations",
                "dominant_long_venue",
                "dominant_short_venue",
                "mean_annualized_spread",
                "mean_net_8h_proxy",
                "min_net_8h_proxy",
                "max_net_8h_proxy",
                "positive_net_8h_rate",
                "mean_net_24h_proxy",
                "mean_breakeven_hold_hours",
                "mean_capacity_proxy_notional",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.asset,
                    summary.observations,
                    summary.dominant_long_venue,
                    summary.dominant_short_venue,
                    f"{summary.mean_annualized_spread:.8f}",
                    f"{summary.mean_net_8h_proxy:.8f}",
                    f"{summary.min_net_8h_proxy:.8f}",
                    f"{summary.max_net_8h_proxy:.8f}",
                    f"{summary.positive_net_8h_rate:.8f}",
                    f"{summary.mean_net_24h_proxy:.8f}",
                    f"{summary.mean_breakeven_hold_hours:.4f}",
                    f"{summary.mean_capacity_proxy_notional:.8f}",
                )
            )
    return output_path


def _summarize_asset(
    *,
    asset: str,
    rows: tuple[OkxHlFundingSpread, ...],
) -> PersistenceSummary:
    net_8h_values = tuple(row.net_8h_proxy for row in rows)
    return PersistenceSummary(
        asset=asset,
        observations=len(rows),
        dominant_long_venue=_mode(tuple(row.long_venue for row in rows)),
        dominant_short_venue=_mode(tuple(row.short_venue for row in rows)),
        mean_annualized_spread=mean(row.annualized_spread for row in rows),
        mean_net_8h_proxy=mean(net_8h_values),
        min_net_8h_proxy=min(net_8h_values),
        max_net_8h_proxy=max(net_8h_values),
        positive_net_8h_rate=mean(1.0 if value > 0.0 else 0.0 for value in net_8h_values),
        mean_net_24h_proxy=mean(row.net_24h_proxy for row in rows),
        mean_breakeven_hold_hours=mean(row.breakeven_hold_hours for row in rows),
        mean_capacity_proxy_notional=mean(row.capacity_proxy_notional for row in rows),
    )


def _mode(values: tuple[str, ...]) -> str:
    counts = {value: values.count(value) for value in set(values)}
    return max(counts, key=counts.get) if counts else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "okx_hl_funding_persistence.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "okx_hl_funding_persistence_summary.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--assets", nargs="+")
    args = parser.parse_args()

    assets = tuple(asset.upper() for asset in args.assets) if args.assets else None
    rows = collect_okx_hl_persistence(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        max_workers=args.max_workers,
        assets=assets,
    )
    summaries = summarize_persistence(rows)
    write_persistence_rows(rows, output_path=args.output_path)
    write_persistence_summary(summaries, output_path=args.summary_output_path)
    for summary in summaries[: args.top]:
        print(
            summary.asset,
            summary.dominant_long_venue,
            summary.dominant_short_venue,
            f"obs={summary.observations}",
            f"pos8h={summary.positive_net_8h_rate:.4f}",
            f"mean8h={summary.mean_net_8h_proxy:.6f}",
            f"mean24h={summary.mean_net_24h_proxy:.6f}",
            f"breakeven={summary.mean_breakeven_hold_hours:.2f}",
            f"capacity={summary.mean_capacity_proxy_notional:.0f}",
        )


if __name__ == "__main__":
    main()
