from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from strategies.perp_market_map.current_hyperliquid_dislocation_candidates import (
    HyperliquidDislocationCandidate,
    build_hyperliquid_dislocation_candidates,
)
from strategies.perp_market_map.current_hyperliquid_snapshot import (
    build_perp_market_rows,
    fetch_hyperliquid_meta_and_contexts,
    write_perp_market_rows,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HyperliquidDislocationMonitorSample:
    monitor_timestamp: str
    sample_index: int
    candidate: HyperliquidDislocationCandidate


@dataclass(frozen=True)
class HyperliquidDislocationMonitorSummary:
    asset: str
    status: str
    side: str
    monitor_action: str
    observations: int
    first_seen_at: str
    last_seen_at: str
    mean_score: float
    max_score: float
    min_score: float
    mean_return_24h: float
    mean_annualized_funding: float
    mean_mark_oracle_diff: float
    mean_premium: float
    mean_oi_volume_ratio: float
    mean_impact_spread: float


def run_monitor(
    *,
    samples: int = 4,
    delay_seconds: float = 10.0,
    top: int = 40,
) -> tuple[HyperliquidDislocationMonitorSample, ...]:
    rows: list[HyperliquidDislocationMonitorSample] = []
    with TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / "current_hyperliquid_snapshot.csv"
        for sample_index in range(samples):
            timestamp = datetime.now(UTC).isoformat()
            meta, contexts = fetch_hyperliquid_meta_and_contexts()
            market_rows = build_perp_market_rows(meta=meta, contexts=contexts, timestamp=timestamp)
            write_perp_market_rows(market_rows, output_path=snapshot_path)
            candidates = build_hyperliquid_dislocation_candidates(snapshot_path=snapshot_path)
            rows.extend(
                HyperliquidDislocationMonitorSample(
                    monitor_timestamp=timestamp,
                    sample_index=sample_index,
                    candidate=candidate,
                )
                for candidate in candidates[:top]
            )
            if sample_index < samples - 1:
                sleep(delay_seconds)
    return tuple(rows)


def read_monitor_samples_csv(
    *,
    input_path: Path,
) -> tuple[HyperliquidDislocationMonitorSample, ...]:
    if not input_path.exists():
        return ()
    with input_path.open(newline="", encoding="utf-8") as handle:
        return tuple(_sample_from_row(row) for row in csv.DictReader(handle))


def merge_monitor_samples(
    existing: tuple[HyperliquidDislocationMonitorSample, ...],
    new: tuple[HyperliquidDislocationMonitorSample, ...],
) -> tuple[HyperliquidDislocationMonitorSample, ...]:
    seen: set[tuple[str, int, str, str, str]] = set()
    output: list[HyperliquidDislocationMonitorSample] = []
    for sample in (*existing, *new):
        key = (
            sample.monitor_timestamp,
            sample.sample_index,
            sample.candidate.asset,
            sample.candidate.status,
            sample.candidate.side,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(sample)
    return tuple(sorted(output, key=lambda row: (row.monitor_timestamp, row.sample_index, row.candidate.score)))


def summarize_monitor_samples(
    samples: tuple[HyperliquidDislocationMonitorSample, ...],
) -> tuple[HyperliquidDislocationMonitorSummary, ...]:
    grouped: dict[tuple[str, str, str], list[HyperliquidDislocationMonitorSample]] = {}
    for sample in samples:
        key = (sample.candidate.asset, sample.candidate.status, sample.candidate.side)
        grouped.setdefault(key, []).append(sample)
    summaries = tuple(
        _summarize_group(asset=key[0], status=key[1], side=key[2], samples=tuple(group_samples))
        for key, group_samples in grouped.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.monitor_action == "repeat_label_priority",
                row.observations,
                row.mean_score,
                abs(row.mean_annualized_funding),
            ),
            reverse=True,
        )
    )


def write_monitor_samples_csv(
    samples: tuple[HyperliquidDislocationMonitorSample, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "monitor_timestamp",
                "sample_index",
                "candidate_timestamp",
                "asset",
                "status",
                "side",
                "score",
                "return_24h",
                "annualized_funding",
                "mark_oracle_diff",
                "premium",
                "open_interest_notional",
                "day_notional_volume",
                "oi_volume_ratio",
                "impact_spread",
                "reason",
                "next_step",
            )
        )
        for sample in samples:
            row = sample.candidate
            writer.writerow(
                (
                    sample.monitor_timestamp,
                    sample.sample_index,
                    row.timestamp,
                    row.asset,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.return_24h:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.premium:.12f}",
                    f"{row.open_interest_notional:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_monitor_summary_csv(
    summaries: tuple[HyperliquidDislocationMonitorSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "status",
                "side",
                "monitor_action",
                "observations",
                "first_seen_at",
                "last_seen_at",
                "mean_score",
                "max_score",
                "min_score",
                "mean_return_24h",
                "mean_annualized_funding",
                "mean_mark_oracle_diff",
                "mean_premium",
                "mean_oi_volume_ratio",
                "mean_impact_spread",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.asset,
                    row.status,
                    row.side,
                    row.monitor_action,
                    row.observations,
                    row.first_seen_at,
                    row.last_seen_at,
                    f"{row.mean_score:.8f}",
                    f"{row.max_score:.8f}",
                    f"{row.min_score:.8f}",
                    f"{row.mean_return_24h:.8f}",
                    f"{row.mean_annualized_funding:.8f}",
                    f"{row.mean_mark_oracle_diff:.12f}",
                    f"{row.mean_premium:.12f}",
                    f"{row.mean_oi_volume_ratio:.8f}",
                    f"{row.mean_impact_spread:.12f}",
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[HyperliquidDislocationMonitorSummary, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Monitor\n\n")
        handle.write(
            "This repeats the dislocation screen and keeps the sample history. "
            "It separates one-off paper hypotheses from candidates that remain "
            "visible across observations. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | side | action | obs | mean score | max score | ret24 | funding ann | OI/vol | impact |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.monitor_action} | "
                f"{row.observations} | "
                f"{row.mean_score:.4f} | "
                f"{row.max_score:.4f} | "
                f"{row.mean_return_24h:.4f} | "
                f"{row.mean_annualized_funding:.4f} | "
                f"{row.mean_oi_volume_ratio:.4f} | "
                f"{row.mean_impact_spread:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`repeat_label_priority` means the same asset/status/side appeared more "
            "than once in the accumulated samples. These are better next candidates "
            "for forward labels and execution checks than single-observation rows.\n"
        )
    return output_path


def _summarize_group(
    *,
    asset: str,
    status: str,
    side: str,
    samples: tuple[HyperliquidDislocationMonitorSample, ...],
) -> HyperliquidDislocationMonitorSummary:
    timestamps = tuple(sample.monitor_timestamp for sample in samples)
    observations = len(samples)
    scores = tuple(sample.candidate.score for sample in samples)
    monitor_action = "repeat_label_priority" if observations >= 2 else "first_seen_label_candidate"
    return HyperliquidDislocationMonitorSummary(
        asset=asset,
        status=status,
        side=side,
        monitor_action=monitor_action,
        observations=observations,
        first_seen_at=min(timestamps),
        last_seen_at=max(timestamps),
        mean_score=sum(scores) / observations,
        max_score=max(scores),
        min_score=min(scores),
        mean_return_24h=sum(sample.candidate.return_24h for sample in samples) / observations,
        mean_annualized_funding=sum(sample.candidate.annualized_funding for sample in samples) / observations,
        mean_mark_oracle_diff=sum(sample.candidate.mark_oracle_diff for sample in samples) / observations,
        mean_premium=sum(sample.candidate.premium for sample in samples) / observations,
        mean_oi_volume_ratio=sum(sample.candidate.oi_volume_ratio for sample in samples) / observations,
        mean_impact_spread=sum(sample.candidate.impact_spread for sample in samples) / observations,
    )


def _sample_from_row(row: dict[str, str]) -> HyperliquidDislocationMonitorSample:
    return HyperliquidDislocationMonitorSample(
        monitor_timestamp=row["monitor_timestamp"],
        sample_index=int(row["sample_index"]),
        candidate=HyperliquidDislocationCandidate(
            timestamp=row["candidate_timestamp"],
            asset=row["asset"],
            status=row["status"],
            side=row["side"],
            score=_float(row["score"]),
            return_24h=_float(row["return_24h"]),
            annualized_funding=_float(row["annualized_funding"]),
            mark_oracle_diff=_float(row["mark_oracle_diff"]),
            premium=_float(row["premium"]),
            open_interest_notional=_float(row["open_interest_notional"]),
            day_notional_volume=_float(row["day_notional_volume"]),
            oi_volume_ratio=_float(row["oi_volume_ratio"]),
            impact_spread=_float(row["impact_spread"]),
            reason=row["reason"],
            next_step=row["next_step"],
        ),
    )


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--samples-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_monitor_samples.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_monitor_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_monitor_summary.md",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace existing samples instead of appending this run",
    )
    args = parser.parse_args()

    new_samples = run_monitor(samples=args.samples, delay_seconds=args.delay_seconds, top=args.top)
    existing_samples = () if args.replace else read_monitor_samples_csv(input_path=args.samples_output_path)
    samples = merge_monitor_samples(existing_samples, new_samples)
    summaries = summarize_monitor_samples(samples)
    write_monitor_samples_csv(samples, output_path=args.samples_output_path)
    write_monitor_summary_csv(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.md_output_path, top=args.top)
    for row in summaries[: args.top]:
        print(
            row.asset,
            row.status,
            row.side,
            row.monitor_action,
            f"obs={row.observations}",
            f"score={row.mean_score:.4f}",
        )


if __name__ == "__main__":
    main()
