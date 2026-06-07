from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from strategies.liquidation_flow.current_okx_liquidation_flow import (
    LiquidationFlowRow,
    build_okx_liquidation_flow_rows,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationMonitorSample:
    sample_index: int
    row: LiquidationFlowRow


@dataclass(frozen=True)
class LiquidationMonitorSummary:
    asset: str
    action: str
    observations: int
    mean_cascade_score: float
    min_cascade_score: float
    mean_total_liquidation_notional: float
    mean_liquidation_to_volume: float
    mean_forced_buy_sell_imbalance: float
    latest_liquidation_at: str


def run_monitor(
    *,
    samples: int = 4,
    delay_seconds: float = 20.0,
    top: int = 25,
    lookback_minutes: int = 60,
    top_by_volume: int = 30,
) -> tuple[LiquidationMonitorSample, ...]:
    rows: list[LiquidationMonitorSample] = []
    for sample_index in range(samples):
        current_rows = build_okx_liquidation_flow_rows(
            lookback_minutes=lookback_minutes,
            top_by_volume=top_by_volume,
        )
        rows.extend(
            LiquidationMonitorSample(sample_index=sample_index, row=row)
            for row in current_rows[:top]
        )
        if sample_index < samples - 1:
            sleep(delay_seconds)
    return tuple(rows)


def summarize_samples(
    samples: tuple[LiquidationMonitorSample, ...],
) -> tuple[LiquidationMonitorSummary, ...]:
    grouped: dict[tuple[str, str], list[LiquidationMonitorSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.row.asset, sample.row.action), []).append(sample)
    summaries = tuple(
        _summarize_group(asset=key[0], action=key[1], rows=tuple(rows))
        for key, rows in grouped.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.observations,
                row.mean_cascade_score,
                row.mean_total_liquidation_notional,
                abs(row.mean_forced_buy_sell_imbalance),
            ),
            reverse=True,
        )
    )


def write_monitor_samples(
    samples: tuple[LiquidationMonitorSample, ...],
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
                "action",
                "observations",
                "latest_liquidation_at",
                "total_liquidation_notional",
                "liquidation_to_volume",
                "forced_buy_sell_imbalance",
                "cascade_score",
            )
        )
        for sample in samples:
            row = sample.row
            writer.writerow(
                (
                    sample.sample_index,
                    row.timestamp,
                    row.asset,
                    row.action,
                    row.observations,
                    row.latest_liquidation_at,
                    f"{row.total_liquidation_notional:.8f}",
                    f"{row.liquidation_to_volume:.8f}",
                    f"{row.forced_buy_sell_imbalance:.8f}",
                    f"{row.cascade_score:.8f}",
                )
            )
    return output_path


def write_monitor_summary(
    summaries: tuple[LiquidationMonitorSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "action",
                "observations",
                "mean_cascade_score",
                "min_cascade_score",
                "mean_total_liquidation_notional",
                "mean_liquidation_to_volume",
                "mean_forced_buy_sell_imbalance",
                "latest_liquidation_at",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.observations,
                    f"{row.mean_cascade_score:.8f}",
                    f"{row.min_cascade_score:.8f}",
                    f"{row.mean_total_liquidation_notional:.8f}",
                    f"{row.mean_liquidation_to_volume:.8f}",
                    f"{row.mean_forced_buy_sell_imbalance:.8f}",
                    row.latest_liquidation_at,
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[LiquidationMonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Monitor\n\n")
        handle.write(
            "This repeats the OKX liquidation-flow screen over a short window. "
            "It is a persistence check, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | obs | mean score | min score | mean liq USD | mean liq/vol | mean imbalance | latest liquidation |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.observations} | "
                f"{row.mean_cascade_score:.6f} | "
                f"{row.min_cascade_score:.6f} | "
                f"{row.mean_total_liquidation_notional:.0f} | "
                f"{row.mean_liquidation_to_volume:.6f} | "
                f"{row.mean_forced_buy_sell_imbalance:.6f} | "
                f"{row.latest_liquidation_at} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows that appear in every sample are persistence candidates. They "
            "still need forward labels, fee assumptions, and venue-depth checks.\n"
        )
    return output_path


def _summarize_group(
    *,
    asset: str,
    action: str,
    rows: tuple[LiquidationMonitorSample, ...],
) -> LiquidationMonitorSummary:
    return LiquidationMonitorSummary(
        asset=asset,
        action=action,
        observations=len(rows),
        mean_cascade_score=sum(row.row.cascade_score for row in rows) / len(rows),
        min_cascade_score=min(row.row.cascade_score for row in rows),
        mean_total_liquidation_notional=(
            sum(row.row.total_liquidation_notional for row in rows) / len(rows)
        ),
        mean_liquidation_to_volume=(
            sum(row.row.liquidation_to_volume for row in rows) / len(rows)
        ),
        mean_forced_buy_sell_imbalance=(
            sum(row.row.forced_buy_sell_imbalance for row in rows) / len(rows)
        ),
        latest_liquidation_at=max(row.row.latest_liquidation_at for row in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--top-by-volume", type=int, default=30)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_summary.md",
    )
    args = parser.parse_args()

    samples = run_monitor(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        top=args.top,
        lookback_minutes=args.lookback_minutes,
        top_by_volume=args.top_by_volume,
    )
    summaries = summarize_samples(samples)
    write_monitor_samples(samples, output_path=args.samples_output_path)
    write_monitor_summary(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.md_output_path, top=args.top)
    for row in summaries[: args.top]:
        print(
            row.asset,
            row.action,
            f"obs={row.observations}",
            f"mean_score={row.mean_cascade_score:.4f}",
            f"mean_liq={row.mean_total_liquidation_notional:.0f}",
            f"mean_imbalance={row.mean_forced_buy_sell_imbalance:.4f}",
        )


if __name__ == "__main__":
    main()
