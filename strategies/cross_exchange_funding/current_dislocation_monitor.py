from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from strategies.cross_exchange_funding.current_dislocation_watchlist import (
    WatchRow,
    build_watchlist,
)
from strategies.cross_exchange_funding.current_funding_feasibility import (
    build_feasibility_rows,
    fetch_hyperliquid_market_contexts,
    write_feasibility_rows,
)
from strategies.cross_exchange_funding.current_funding_spread import (
    build_funding_spreads,
    fetch_predicted_fundings,
    write_funding_spreads,
)
from strategies.cross_exchange_funding.current_okx_hl_funding_spread import (
    build_okx_hl_funding_spreads,
    write_okx_hl_funding_spreads,
)
from strategies.perp_market_map.current_hyperliquid_snapshot import (
    build_perp_market_rows,
    fetch_hyperliquid_meta_and_contexts,
    write_perp_market_rows,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MonitorSample:
    sample_index: int
    timestamp: str
    row: WatchRow


@dataclass(frozen=True)
class MonitorSummary:
    source: str
    action: str
    asset: str
    long_venue: str
    short_venue: str
    observations: int
    mean_annualized_edge: float
    min_annualized_edge: float
    mean_net_8h_proxy: float | None
    mean_net_24h_proxy: float | None
    positive_net_24h_rate: float | None
    mean_liquidity_proxy: float
    mean_friction_proxy: float


def run_monitor(
    *,
    samples: int = 3,
    delay_seconds: float = 10.0,
    top: int = 25,
    max_workers: int = 16,
) -> tuple[MonitorSample, ...]:
    rows: list[MonitorSample] = []
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for sample_index in range(samples):
            timestamp = datetime.now(UTC).isoformat()
            watch_rows = _build_sample_watchlist(
                root=root,
                max_workers=max_workers,
            )
            rows.extend(
                MonitorSample(
                    sample_index=sample_index,
                    timestamp=timestamp,
                    row=row,
                )
                for row in watch_rows[:top]
            )
            if sample_index < samples - 1:
                sleep(delay_seconds)
    return tuple(rows)


def summarize_samples(samples: tuple[MonitorSample, ...]) -> tuple[MonitorSummary, ...]:
    grouped: dict[tuple[str, str, str, str, str], list[MonitorSample]] = {}
    for sample in samples:
        key = (
            sample.row.source,
            sample.row.action,
            sample.row.asset,
            sample.row.long_venue,
            sample.row.short_venue,
        )
        grouped.setdefault(key, []).append(sample)
    summaries = [
        _summarize_group(key=key, rows=tuple(rows))
        for key, rows in grouped.items()
    ]
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.observations,
                row.positive_net_24h_rate or 0.0,
                row.mean_annualized_edge,
                row.mean_liquidity_proxy,
            ),
            reverse=True,
        )
    )


def write_monitor_samples_csv(
    samples: tuple[MonitorSample, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "sample_index",
                "timestamp",
                "source",
                "action",
                "asset",
                "long_venue",
                "short_venue",
                "annualized_edge",
                "net_8h_proxy",
                "net_24h_proxy",
                "liquidity_proxy",
                "friction_proxy",
                "reason",
            )
        )
        for sample in samples:
            row = sample.row
            writer.writerow(
                (
                    sample.sample_index,
                    sample.timestamp,
                    row.source,
                    row.action,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    f"{row.annualized_edge:.8f}",
                    "" if row.net_8h_proxy is None else f"{row.net_8h_proxy:.8f}",
                    "" if row.net_24h_proxy is None else f"{row.net_24h_proxy:.8f}",
                    f"{row.liquidity_proxy:.8f}",
                    f"{row.friction_proxy:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_monitor_summary_csv(
    summaries: tuple[MonitorSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source",
                "action",
                "asset",
                "long_venue",
                "short_venue",
                "observations",
                "mean_annualized_edge",
                "min_annualized_edge",
                "mean_net_8h_proxy",
                "mean_net_24h_proxy",
                "positive_net_24h_rate",
                "mean_liquidity_proxy",
                "mean_friction_proxy",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.source,
                    row.action,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    row.observations,
                    f"{row.mean_annualized_edge:.8f}",
                    f"{row.min_annualized_edge:.8f}",
                    "" if row.mean_net_8h_proxy is None else f"{row.mean_net_8h_proxy:.8f}",
                    "" if row.mean_net_24h_proxy is None else f"{row.mean_net_24h_proxy:.8f}",
                    "" if row.positive_net_24h_rate is None else f"{row.positive_net_24h_rate:.8f}",
                    f"{row.mean_liquidity_proxy:.8f}",
                    f"{row.mean_friction_proxy:.8f}",
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[MonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Funding Dislocation Monitor\n\n")
        handle.write(
            "This repeats the current dislocation watchlist over a short window. "
            "It is a persistence check, not a trade instruction.\n\n"
        )
        handle.write(
            "| source | action | asset | long | short | obs | mean edge | min edge | mean net 8h | mean net 24h | positive net24 | liquidity | friction |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.source} | "
                f"{row.action} | "
                f"{row.asset} | "
                f"{row.long_venue} | "
                f"{row.short_venue} | "
                f"{row.observations} | "
                f"{row.mean_annualized_edge:.6f} | "
                f"{row.min_annualized_edge:.6f} | "
                f"{'' if row.mean_net_8h_proxy is None else f'{row.mean_net_8h_proxy:.6f}'} | "
                f"{'' if row.mean_net_24h_proxy is None else f'{row.mean_net_24h_proxy:.6f}'} | "
                f"{'' if row.positive_net_24h_rate is None else f'{row.positive_net_24h_rate:.6f}'} | "
                f"{row.mean_liquidity_proxy:.2f} | "
                f"{row.mean_friction_proxy:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows that appear in every sample with positive 24-hour proxy are the next "
            "monitor candidates. Rows without an executable hedge stay as funding "
            "alerts, not paper-trade candidates.\n"
        )
    return output_path


def _build_sample_watchlist(*, root: Path, max_workers: int) -> tuple[WatchRow, ...]:
    funding_spread_path = root / "current_funding_spread.csv"
    funding_feasibility_path = root / "current_funding_feasibility.csv"
    okx_hl_path = root / "current_okx_hl_funding_spread.csv"
    hl_snapshot_path = root / "current_hyperliquid_snapshot.csv"

    predicted_fundings = fetch_predicted_fundings()
    funding_spreads = build_funding_spreads(predicted_fundings)
    write_funding_spreads(funding_spreads, output_path=funding_spread_path)

    hl_contexts = fetch_hyperliquid_market_contexts()
    feasibility_rows = build_feasibility_rows(
        funding_spreads,
        hl_contexts=hl_contexts,
    )
    write_feasibility_rows(feasibility_rows, output_path=funding_feasibility_path)

    okx_hl_rows = build_okx_hl_funding_spreads(max_workers=max_workers)
    write_okx_hl_funding_spreads(okx_hl_rows, output_path=okx_hl_path)

    meta, contexts = fetch_hyperliquid_meta_and_contexts()
    hl_rows = build_perp_market_rows(meta=meta, contexts=contexts)
    write_perp_market_rows(hl_rows, output_path=hl_snapshot_path)

    return build_watchlist(
        funding_feasibility_path=funding_feasibility_path,
        okx_hl_path=okx_hl_path,
        hl_snapshot_path=hl_snapshot_path,
    )


def _summarize_group(
    *,
    key: tuple[str, str, str, str, str],
    rows: tuple[MonitorSample, ...],
) -> MonitorSummary:
    net_8h_values = tuple(
        sample.row.net_8h_proxy
        for sample in rows
        if sample.row.net_8h_proxy is not None
    )
    net_24h_values = tuple(
        sample.row.net_24h_proxy
        for sample in rows
        if sample.row.net_24h_proxy is not None
    )
    return MonitorSummary(
        source=key[0],
        action=key[1],
        asset=key[2],
        long_venue=key[3],
        short_venue=key[4],
        observations=len(rows),
        mean_annualized_edge=sum(sample.row.annualized_edge for sample in rows) / len(rows),
        min_annualized_edge=min(sample.row.annualized_edge for sample in rows),
        mean_net_8h_proxy=(
            sum(net_8h_values) / len(net_8h_values)
            if net_8h_values
            else None
        ),
        mean_net_24h_proxy=(
            sum(net_24h_values) / len(net_24h_values)
            if net_24h_values
            else None
        ),
        positive_net_24h_rate=(
            sum(value > 0.0 for value in net_24h_values) / len(net_24h_values)
            if net_24h_values
            else None
        ),
        mean_liquidity_proxy=sum(sample.row.liquidity_proxy for sample in rows) / len(rows),
        mean_friction_proxy=sum(sample.row.friction_proxy for sample in rows) / len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_dislocation_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_dislocation_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_dislocation_monitor_summary.md",
    )
    args = parser.parse_args()

    samples = run_monitor(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        top=args.top,
        max_workers=args.max_workers,
    )
    summaries = summarize_samples(samples)
    write_monitor_samples_csv(samples, output_path=args.samples_output_path)
    write_monitor_summary_csv(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.md_output_path, top=args.top)
    for row in summaries[: args.top]:
        print(
            row.source,
            row.action,
            row.asset,
            f"obs={row.observations}",
            f"edge={row.mean_annualized_edge:.4f}",
            f"net24={'' if row.mean_net_24h_proxy is None else f'{row.mean_net_24h_proxy:.6f}'}",
        )


if __name__ == "__main__":
    main()
