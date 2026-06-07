from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean

from strategies.cross_exchange_funding.okx_hl_event_window_score import (
    EventWindowScore,
    build_event_window_scores,
)
from strategies.cross_exchange_funding.okx_hl_event_window_triage import (
    EventWindowTriage,
    build_event_window_triage_from_rows,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventWindowMonitorObservation:
    collected_at: str
    asset: str
    event_action: str
    previous_action: str
    long_venue: str
    short_venue: str
    capacity: Decimal
    very_low_fee_net_8h: Decimal | None
    very_low_fee_net_24h: Decimal | None
    low_fee_net_24h: Decimal | None
    one_bps_each_net_24h: Decimal | None
    max_entry_slippage_bps: Decimal
    reason: str


@dataclass(frozen=True)
class EventWindowMonitorSummary:
    asset: str
    observations: int
    dominant_event_action: str
    paper_8h_rate: float
    active_24h_rate: float
    watch_rate: float
    drop_rate: float
    mean_very_low_fee_net_8h: Decimal
    mean_very_low_fee_net_24h: Decimal
    mean_low_fee_net_24h: Decimal
    mean_one_bps_each_net_24h: Decimal
    mean_capacity: Decimal
    max_entry_slippage_bps: Decimal


def collect_event_window_monitor(
    *,
    samples: int,
    delay_seconds: float,
    triage_path: Path = ROOT / "okx_hl_candidate_triage.csv",
    execution_score_path: Path = ROOT / "okx_hl_execution_cost_score.csv",
) -> tuple[EventWindowMonitorObservation, ...]:
    observations: list[EventWindowMonitorObservation] = []
    for sample_index in range(samples):
        collected_at = datetime.now(UTC).isoformat()
        scores = build_event_window_scores(
            triage_path=triage_path,
            execution_score_path=execution_score_path,
        )
        triage = build_event_window_triage_from_rows(rows=_score_rows(scores))
        observations.extend(
            _observation(collected_at=collected_at, item=item) for item in triage
        )
        if sample_index + 1 < samples:
            time.sleep(delay_seconds)
    return tuple(observations)


def summarize_event_window_monitor(
    observations: tuple[EventWindowMonitorObservation, ...],
) -> tuple[EventWindowMonitorSummary, ...]:
    by_asset: dict[str, list[EventWindowMonitorObservation]] = {}
    for observation in observations:
        by_asset.setdefault(observation.asset, []).append(observation)
    return tuple(
        sorted(
            (
                _summarize_asset(asset=asset, observations=tuple(asset_observations))
                for asset, asset_observations in by_asset.items()
            ),
            key=lambda item: (
                item.paper_8h_rate,
                item.active_24h_rate,
                item.watch_rate,
                item.mean_low_fee_net_24h,
                item.mean_capacity,
            ),
            reverse=True,
        )
    )


def write_monitor_observations_csv(
    observations: tuple[EventWindowMonitorObservation, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "collected_at",
                "asset",
                "event_action",
                "previous_action",
                "long_venue",
                "short_venue",
                "capacity",
                "very_low_fee_net_8h",
                "very_low_fee_net_24h",
                "low_fee_net_24h",
                "one_bps_each_net_24h",
                "max_entry_slippage_bps",
                "reason",
            )
        )
        for observation in observations:
            writer.writerow(
                (
                    observation.collected_at,
                    observation.asset,
                    observation.event_action,
                    observation.previous_action,
                    observation.long_venue,
                    observation.short_venue,
                    _fmt(observation.capacity),
                    _fmt_optional(observation.very_low_fee_net_8h),
                    _fmt_optional(observation.very_low_fee_net_24h),
                    _fmt_optional(observation.low_fee_net_24h),
                    _fmt_optional(observation.one_bps_each_net_24h),
                    _fmt(observation.max_entry_slippage_bps),
                    observation.reason,
                )
            )
    return output_path


def write_monitor_summary_csv(
    summaries: tuple[EventWindowMonitorSummary, ...],
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
                "dominant_event_action",
                "paper_8h_rate",
                "active_24h_rate",
                "watch_rate",
                "drop_rate",
                "mean_very_low_fee_net_8h",
                "mean_very_low_fee_net_24h",
                "mean_low_fee_net_24h",
                "mean_one_bps_each_net_24h",
                "mean_capacity",
                "max_entry_slippage_bps",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.asset,
                    summary.observations,
                    summary.dominant_event_action,
                    f"{summary.paper_8h_rate:.8f}",
                    f"{summary.active_24h_rate:.8f}",
                    f"{summary.watch_rate:.8f}",
                    f"{summary.drop_rate:.8f}",
                    _fmt(summary.mean_very_low_fee_net_8h),
                    _fmt(summary.mean_very_low_fee_net_24h),
                    _fmt(summary.mean_low_fee_net_24h),
                    _fmt(summary.mean_one_bps_each_net_24h),
                    _fmt(summary.mean_capacity),
                    _fmt(summary.max_entry_slippage_bps),
                )
            )
    return output_path


def write_monitor_summary_md(
    summaries: tuple[EventWindowMonitorSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Event Window Monitor\n\n")
        handle.write(
            "This repeats event-window triage to check whether the current candidate "
            "classification is stable. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | obs | dominant action | paper 8h rate | active 24h rate | watch rate | drop rate | mean very-low 8h | mean low-fee 24h | mean one-bps 24h | capacity |\n"
        )
        handle.write(
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for summary in summaries:
            handle.write(
                "| "
                f"{summary.asset} | "
                f"{summary.observations} | "
                f"{summary.dominant_event_action} | "
                f"{summary.paper_8h_rate:.8f} | "
                f"{summary.active_24h_rate:.8f} | "
                f"{summary.watch_rate:.8f} | "
                f"{summary.drop_rate:.8f} | "
                f"{_fmt(summary.mean_very_low_fee_net_8h)} | "
                f"{_fmt(summary.mean_low_fee_net_24h)} | "
                f"{_fmt(summary.mean_one_bps_each_net_24h)} | "
                f"{_fmt(summary.mean_capacity)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A candidate should not move toward paper execution unless the event-window "
            "action is stable and the surviving scenario is realistic for the account's "
            "actual fee and maker-fill conditions.\n"
        )
    return output_path


def _summarize_asset(
    *,
    asset: str,
    observations: tuple[EventWindowMonitorObservation, ...],
) -> EventWindowMonitorSummary:
    actions = tuple(observation.event_action for observation in observations)
    return EventWindowMonitorSummary(
        asset=asset,
        observations=len(observations),
        dominant_event_action=Counter(actions).most_common(1)[0][0],
        paper_8h_rate=_rate(actions, {"paper_8h_candidate"}),
        active_24h_rate=_rate(actions, {"active_24h_monitor"}),
        watch_rate=_rate(
            actions,
            {
                "fee_dependent_24h_monitor",
                "very_low_fee_24h_watch",
                "thin_or_unstable_watch",
            },
        ),
        drop_rate=_rate(actions, {"drop_for_now"}),
        mean_very_low_fee_net_8h=_mean_optional(
            observation.very_low_fee_net_8h for observation in observations
        ),
        mean_very_low_fee_net_24h=_mean_optional(
            observation.very_low_fee_net_24h for observation in observations
        ),
        mean_low_fee_net_24h=_mean_optional(
            observation.low_fee_net_24h for observation in observations
        ),
        mean_one_bps_each_net_24h=_mean_optional(
            observation.one_bps_each_net_24h for observation in observations
        ),
        mean_capacity=Decimal(
            str(mean(observation.capacity for observation in observations))
        ),
        max_entry_slippage_bps=max(
            observation.max_entry_slippage_bps for observation in observations
        ),
    )


def _score_rows(scores: tuple[EventWindowScore, ...]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "asset": score.asset,
            "action": score.action,
            "scenario": score.scenario,
            "long_venue": score.long_venue,
            "short_venue": score.short_venue,
            "net_event_8h_after_all_in_cost": str(
                score.net_event_8h_after_all_in_cost
            ),
            "net_event_24h_after_all_in_cost": str(
                score.net_event_24h_after_all_in_cost
            ),
            "capacity": str(score.capacity),
            "max_entry_slippage_bps": str(score.max_entry_slippage_bps),
        }
        for score in scores
    )


def _observation(
    *,
    collected_at: str,
    item: EventWindowTriage,
) -> EventWindowMonitorObservation:
    return EventWindowMonitorObservation(
        collected_at=collected_at,
        asset=item.asset,
        event_action=item.event_action,
        previous_action=item.previous_action,
        long_venue=item.long_venue,
        short_venue=item.short_venue,
        capacity=item.capacity,
        very_low_fee_net_8h=item.very_low_fee_net_8h,
        very_low_fee_net_24h=item.very_low_fee_net_24h,
        low_fee_net_24h=item.low_fee_net_24h,
        one_bps_each_net_24h=item.one_bps_each_net_24h,
        max_entry_slippage_bps=item.max_entry_slippage_bps,
        reason=item.reason,
    )


def _rate(actions: tuple[str, ...], target_actions: set[str]) -> float:
    return mean(1.0 if action in target_actions else 0.0 for action in actions)


def _mean_optional(values: Iterable[Decimal | None]) -> Decimal:
    concrete_values = tuple(value for value in values if value is not None)
    if not concrete_values:
        return Decimal("0")
    return Decimal(str(mean(concrete_values)))


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def _fmt_optional(value: Decimal | None) -> str:
    return "" if value is None else _fmt(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument(
        "--triage-path",
        type=Path,
        default=ROOT / "okx_hl_candidate_triage.csv",
    )
    parser.add_argument(
        "--execution-score-path",
        type=Path,
        default=ROOT / "okx_hl_execution_cost_score.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_monitor.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_monitor_summary.csv",
    )
    parser.add_argument(
        "--summary-md-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_monitor_summary.md",
    )
    args = parser.parse_args()

    observations = collect_event_window_monitor(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        triage_path=args.triage_path,
        execution_score_path=args.execution_score_path,
    )
    summaries = summarize_event_window_monitor(observations)
    write_monitor_observations_csv(observations, output_path=args.output_path)
    write_monitor_summary_csv(summaries, output_path=args.summary_output_path)
    write_monitor_summary_md(summaries, output_path=args.summary_md_output_path)
    for summary in summaries:
        print(
            summary.asset,
            summary.dominant_event_action,
            f"obs={summary.observations}",
            f"paper8h={summary.paper_8h_rate:.4f}",
            f"watch={summary.watch_rate:.4f}",
            f"drop={summary.drop_rate:.4f}",
            f"low24h={_fmt(summary.mean_low_fee_net_24h)}",
        )


if __name__ == "__main__":
    main()
