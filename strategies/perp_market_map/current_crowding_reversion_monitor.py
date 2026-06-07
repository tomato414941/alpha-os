from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from strategies.perp_market_map.current_crowding_reversion_screen import (
    CrowdingReversionRow,
    build_crowding_reversion_rows,
)
from strategies.perp_market_map.current_hyperliquid_snapshot import (
    build_perp_market_rows,
    fetch_hyperliquid_meta_and_contexts,
    write_perp_market_rows,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CrowdingMonitorSample:
    sample_index: int
    timestamp: str
    row: CrowdingReversionRow


@dataclass(frozen=True)
class CrowdingMonitorSummary:
    asset: str
    action: str
    observations: int
    mean_score: float
    min_score: float
    mean_annualized_funding: float
    min_abs_annualized_funding: float
    mean_mark_oracle_diff: float
    mean_oi_volume_ratio: float
    mean_impact_spread: float


def run_monitor(
    *,
    samples: int = 6,
    delay_seconds: float = 10.0,
    top: int = 25,
) -> tuple[CrowdingMonitorSample, ...]:
    rows: list[CrowdingMonitorSample] = []
    with TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / "current_hyperliquid_snapshot.csv"
        for sample_index in range(samples):
            timestamp = datetime.now(UTC).isoformat()
            meta, contexts = fetch_hyperliquid_meta_and_contexts()
            market_rows = build_perp_market_rows(meta=meta, contexts=contexts, timestamp=timestamp)
            write_perp_market_rows(market_rows, output_path=snapshot_path)
            screen_rows = build_crowding_reversion_rows(snapshot_path=snapshot_path)
            rows.extend(
                CrowdingMonitorSample(
                    sample_index=sample_index,
                    timestamp=timestamp,
                    row=row,
                )
                for row in screen_rows[:top]
            )
            if sample_index < samples - 1:
                sleep(delay_seconds)
    return tuple(rows)


def summarize_samples(
    samples: tuple[CrowdingMonitorSample, ...],
) -> tuple[CrowdingMonitorSummary, ...]:
    grouped: dict[tuple[str, str], list[CrowdingMonitorSample]] = {}
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
                row.mean_score,
                row.min_score,
                abs(row.mean_annualized_funding),
            ),
            reverse=True,
        )
    )


def write_monitor_samples_csv(
    samples: tuple[CrowdingMonitorSample, ...],
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
                "annualized_funding",
                "mark_oracle_diff",
                "premium",
                "oi_volume_ratio",
                "impact_spread",
                "carry_reversion_score",
            )
        )
        for sample in samples:
            row = sample.row
            writer.writerow(
                (
                    sample.sample_index,
                    sample.timestamp,
                    row.asset,
                    row.action,
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.premium:.12f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.impact_spread:.12f}",
                    f"{row.carry_reversion_score:.8f}",
                )
            )
    return output_path


def write_monitor_summary_csv(
    summaries: tuple[CrowdingMonitorSummary, ...],
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
                "mean_score",
                "min_score",
                "mean_annualized_funding",
                "min_abs_annualized_funding",
                "mean_mark_oracle_diff",
                "mean_oi_volume_ratio",
                "mean_impact_spread",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.observations,
                    f"{row.mean_score:.8f}",
                    f"{row.min_score:.8f}",
                    f"{row.mean_annualized_funding:.8f}",
                    f"{row.min_abs_annualized_funding:.8f}",
                    f"{row.mean_mark_oracle_diff:.12f}",
                    f"{row.mean_oi_volume_ratio:.8f}",
                    f"{row.mean_impact_spread:.12f}",
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[CrowdingMonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Reversion Monitor\n\n")
        handle.write(
            "This repeats the current crowding/reversion screen over a short window. "
            "It is a persistence check, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | obs | mean score | min score | mean funding | min abs funding | mean mark/oracle | mean OI/volume | mean impact |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.observations} | "
                f"{row.mean_score:.6f} | "
                f"{row.min_score:.6f} | "
                f"{row.mean_annualized_funding:.6f} | "
                f"{row.min_abs_annualized_funding:.6f} | "
                f"{row.mean_mark_oracle_diff:.6f} | "
                f"{row.mean_oi_volume_ratio:.6f} | "
                f"{row.mean_impact_spread:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows that appear in every sample are persistence candidates. They still "
            "need future-return labels, funding-decay labels, and execution-cost checks "
            "before becoming strategy inputs.\n"
        )
    return output_path


def _summarize_group(
    *,
    asset: str,
    action: str,
    rows: tuple[CrowdingMonitorSample, ...],
) -> CrowdingMonitorSummary:
    return CrowdingMonitorSummary(
        asset=asset,
        action=action,
        observations=len(rows),
        mean_score=sum(row.row.carry_reversion_score for row in rows) / len(rows),
        min_score=min(row.row.carry_reversion_score for row in rows),
        mean_annualized_funding=sum(row.row.annualized_funding for row in rows) / len(rows),
        min_abs_annualized_funding=min(abs(row.row.annualized_funding) for row in rows),
        mean_mark_oracle_diff=sum(row.row.mark_oracle_diff for row in rows) / len(rows),
        mean_oi_volume_ratio=sum(row.row.oi_volume_ratio for row in rows) / len(rows),
        mean_impact_spread=sum(row.row.impact_spread for row in rows) / len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_monitor_summary.md",
    )
    args = parser.parse_args()

    samples = run_monitor(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        top=args.top,
    )
    summaries = summarize_samples(samples)
    write_monitor_samples_csv(samples, output_path=args.samples_output_path)
    write_monitor_summary_csv(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.md_output_path)
    for row in summaries[: args.top]:
        print(
            row.asset,
            row.action,
            f"obs={row.observations}",
            f"score={row.mean_score:.4f}",
            f"funding={row.mean_annualized_funding:.4f}",
        )


if __name__ == "__main__":
    main()
