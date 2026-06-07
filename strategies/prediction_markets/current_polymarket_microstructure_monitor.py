from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from time import sleep
from pathlib import Path

from strategies.prediction_markets.current_polymarket_microstructure import (
    PolymarketMicrostructureRow,
    build_polymarket_microstructure_rows,
    fetch_polymarket_markets,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PolymarketMonitorSample:
    sample_index: int
    row: PolymarketMicrostructureRow


@dataclass(frozen=True)
class PolymarketMonitorSummary:
    market_id: str
    question: str
    action: str
    observations: int
    mean_score: float
    min_score: float
    mean_spread: float
    mean_midpoint: float
    mean_one_day_price_change: float
    mean_volume_24h: float
    mean_liquidity: float


def run_monitor(
    *,
    samples: int = 5,
    delay_seconds: float = 10.0,
    limit: int = 200,
    top: int = 25,
) -> tuple[PolymarketMonitorSample, ...]:
    rows: list[PolymarketMonitorSample] = []
    for sample_index in range(samples):
        markets = fetch_polymarket_markets(limit=limit)
        screen_rows = build_polymarket_microstructure_rows(markets)
        rows.extend(
            PolymarketMonitorSample(
                sample_index=sample_index,
                row=row,
            )
            for row in screen_rows[:top]
        )
        if sample_index < samples - 1:
            sleep(delay_seconds)
    return tuple(rows)


def summarize_samples(
    samples: tuple[PolymarketMonitorSample, ...],
) -> tuple[PolymarketMonitorSummary, ...]:
    grouped: dict[tuple[str, str], list[PolymarketMonitorSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.row.market_id, sample.row.action), []).append(sample)
    summaries = tuple(
        _summarize_group(rows=tuple(rows))
        for rows in grouped.values()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.observations,
                row.mean_score,
                row.min_score,
                row.mean_volume_24h,
            ),
            reverse=True,
        )
    )


def write_monitor_samples_csv(
    samples: tuple[PolymarketMonitorSample, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "market_id",
                "question",
                "action",
                "spread",
                "midpoint",
                "one_day_price_change",
                "volume_24h",
                "liquidity",
                "score",
            )
        )
        for sample in samples:
            row = sample.row
            writer.writerow(
                (
                    sample.sample_index,
                    row.market_id,
                    row.question,
                    row.action,
                    f"{row.spread:.6f}",
                    f"{row.midpoint:.6f}",
                    f"{row.one_day_price_change:.6f}",
                    f"{row.volume_24h:.6f}",
                    f"{row.liquidity:.6f}",
                    f"{row.score:.8f}",
                )
            )
    return output_path


def write_monitor_summary_csv(
    summaries: tuple[PolymarketMonitorSummary, ...],
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
                "action",
                "observations",
                "mean_score",
                "min_score",
                "mean_spread",
                "mean_midpoint",
                "mean_one_day_price_change",
                "mean_volume_24h",
                "mean_liquidity",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.action,
                    row.observations,
                    f"{row.mean_score:.8f}",
                    f"{row.min_score:.8f}",
                    f"{row.mean_spread:.6f}",
                    f"{row.mean_midpoint:.6f}",
                    f"{row.mean_one_day_price_change:.6f}",
                    f"{row.mean_volume_24h:.6f}",
                    f"{row.mean_liquidity:.6f}",
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[PolymarketMonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Polymarket Microstructure Monitor\n\n")
        handle.write(
            "This repeats the Polymarket microstructure screen over a short window. "
            "It is a persistence check, not a trade instruction.\n\n"
        )
        handle.write(
            "| action | question | obs | mean score | min score | spread | midpoint | 1d change | vol24h | liquidity |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.action} | "
                f"{_escape(row.question)} | "
                f"{row.observations} | "
                f"{row.mean_score:.6f} | "
                f"{row.min_score:.6f} | "
                f"{row.mean_spread:.4f} | "
                f"{row.mean_midpoint:.4f} | "
                f"{row.mean_one_day_price_change:.4f} | "
                f"{row.mean_volume_24h:.2f} | "
                f"{row.mean_liquidity:.2f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows that appear in every sample are current event-market watch "
            "candidates. This still does not estimate true probability or adverse "
            "selection; it only checks that the public market-structure signal persists.\n"
        )
    return output_path


def _summarize_group(
    *,
    rows: tuple[PolymarketMonitorSample, ...],
) -> PolymarketMonitorSummary:
    first = rows[0].row
    return PolymarketMonitorSummary(
        market_id=first.market_id,
        question=first.question,
        action=first.action,
        observations=len(rows),
        mean_score=sum(sample.row.score for sample in rows) / len(rows),
        min_score=min(sample.row.score for sample in rows),
        mean_spread=sum(sample.row.spread for sample in rows) / len(rows),
        mean_midpoint=sum(sample.row.midpoint for sample in rows) / len(rows),
        mean_one_day_price_change=(
            sum(sample.row.one_day_price_change for sample in rows) / len(rows)
        ),
        mean_volume_24h=sum(sample.row.volume_24h for sample in rows) / len(rows),
        mean_liquidity=sum(sample.row.liquidity for sample in rows) / len(rows),
    )


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure_monitor_summary.md",
    )
    args = parser.parse_args()

    samples = run_monitor(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        limit=args.limit,
        top=args.top,
    )
    summaries = summarize_samples(samples)
    write_monitor_samples_csv(samples, output_path=args.samples_output_path)
    write_monitor_summary_csv(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.md_output_path)
    for row in summaries[: args.top]:
        print(
            row.action,
            f"obs={row.observations}",
            f"score={row.mean_score:.4f}",
            f"spread={row.mean_spread:.4f}",
            row.question,
        )


if __name__ == "__main__":
    main()
