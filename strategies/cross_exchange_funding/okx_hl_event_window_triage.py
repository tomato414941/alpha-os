from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventWindowTriage:
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


def build_event_window_triage(
    *,
    score_path: Path = ROOT / "okx_hl_event_window_score.csv",
    min_capacity_for_active_monitor: Decimal = Decimal("50000"),
    min_capacity_for_paper_8h: Decimal = Decimal("100000"),
) -> tuple[EventWindowTriage, ...]:
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    return build_event_window_triage_from_rows(
        rows=rows,
        min_capacity_for_active_monitor=min_capacity_for_active_monitor,
        min_capacity_for_paper_8h=min_capacity_for_paper_8h,
    )


def build_event_window_triage_from_rows(
    *,
    rows: tuple[dict[str, str], ...],
    min_capacity_for_active_monitor: Decimal = Decimal("50000"),
    min_capacity_for_paper_8h: Decimal = Decimal("100000"),
) -> tuple[EventWindowTriage, ...]:
    by_asset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_asset.setdefault(row["asset"], []).append(row)
    triage = tuple(
        _triage_asset(
            rows=tuple(asset_rows),
            min_capacity_for_active_monitor=min_capacity_for_active_monitor,
            min_capacity_for_paper_8h=min_capacity_for_paper_8h,
        )
        for asset_rows in by_asset.values()
    )
    return tuple(
        sorted(
            triage,
            key=lambda item: (
                _action_rank(item.event_action),
                item.one_bps_each_net_24h or Decimal("-999"),
                item.low_fee_net_24h or Decimal("-999"),
                item.very_low_fee_net_24h or Decimal("-999"),
                item.capacity,
            ),
            reverse=True,
        )
    )


def write_event_window_triage_csv(
    triage: tuple[EventWindowTriage, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
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
        for item in triage:
            writer.writerow(
                (
                    item.asset,
                    item.event_action,
                    item.previous_action,
                    item.long_venue,
                    item.short_venue,
                    _fmt(item.capacity),
                    _fmt_optional(item.very_low_fee_net_8h),
                    _fmt_optional(item.very_low_fee_net_24h),
                    _fmt_optional(item.low_fee_net_24h),
                    _fmt_optional(item.one_bps_each_net_24h),
                    _fmt(item.max_entry_slippage_bps),
                    item.reason,
                )
            )
    return output_path


def write_event_window_triage_md(
    triage: tuple[EventWindowTriage, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Event Window Triage\n\n")
        handle.write(
            "This turns event-window scores into research actions. It should override "
            "the smooth execution-cost triage when the two disagree.\n\n"
        )
        handle.write(
            "| asset | event action | previous action | long | short | capacity | very-low 8h | very-low 24h | low-fee 24h | one-bps 24h | max slippage bps | reason |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for item in triage:
            handle.write(
                "| "
                f"{item.asset} | "
                f"{item.event_action} | "
                f"{item.previous_action} | "
                f"{item.long_venue} | "
                f"{item.short_venue} | "
                f"{_fmt(item.capacity)} | "
                f"{_fmt_optional(item.very_low_fee_net_8h)} | "
                f"{_fmt_optional(item.very_low_fee_net_24h)} | "
                f"{_fmt_optional(item.low_fee_net_24h)} | "
                f"{_fmt_optional(item.one_bps_each_net_24h)} | "
                f"{_fmt(item.max_entry_slippage_bps)} | "
                f"{item.reason} |\n"
            )
    return output_path


def _triage_asset(
    *,
    rows: tuple[dict[str, str], ...],
    min_capacity_for_active_monitor: Decimal,
    min_capacity_for_paper_8h: Decimal,
) -> EventWindowTriage:
    by_scenario = {row["scenario"]: row for row in rows}
    first = rows[0]
    capacity = Decimal(first["capacity"])
    very_low_fee_net_8h = _scenario_value(
        by_scenario,
        "very_low_fee",
        "net_event_8h_after_all_in_cost",
    )
    very_low_fee_net_24h = _scenario_value(
        by_scenario,
        "very_low_fee",
        "net_event_24h_after_all_in_cost",
    )
    low_fee_net_24h = _scenario_value(
        by_scenario,
        "low_fee",
        "net_event_24h_after_all_in_cost",
    )
    one_bps_each_net_24h = _scenario_value(
        by_scenario,
        "one_bps_each",
        "net_event_24h_after_all_in_cost",
    )
    max_entry_slippage_bps = max(Decimal(row["max_entry_slippage_bps"]) for row in rows)
    if (
        very_low_fee_net_8h is not None
        and very_low_fee_net_8h > 0
        and capacity >= min_capacity_for_paper_8h
    ):
        event_action = "paper_8h_candidate"
        reason = "8h event-window survives only under very low fee assumptions"
    elif (
        one_bps_each_net_24h is not None
        and one_bps_each_net_24h > 0
        and capacity >= min_capacity_for_active_monitor
    ):
        event_action = "active_24h_monitor"
        reason = "24h event-window survives one-bps-each assumption with enough capacity"
    elif (
        low_fee_net_24h is not None
        and low_fee_net_24h > 0
        and capacity >= min_capacity_for_active_monitor
    ):
        event_action = "fee_dependent_24h_monitor"
        reason = "24h event-window survives low-fee assumption but not one-bps-each"
    elif (
        very_low_fee_net_24h is not None
        and very_low_fee_net_24h > 0
        and capacity >= min_capacity_for_active_monitor
    ):
        event_action = "very_low_fee_24h_watch"
        reason = "24h event-window only survives the very-low-fee assumption"
    elif any(Decimal(row["net_event_24h_after_all_in_cost"]) > 0 for row in rows):
        event_action = "thin_or_unstable_watch"
        reason = "24h event-window can be positive, but capacity or cost assumptions are weak"
    else:
        event_action = "drop_for_now"
        reason = "no current event-window scenario survives"
    return EventWindowTriage(
        asset=first["asset"],
        event_action=event_action,
        previous_action=first["action"],
        long_venue=first["long_venue"],
        short_venue=first["short_venue"],
        capacity=capacity,
        very_low_fee_net_8h=very_low_fee_net_8h,
        very_low_fee_net_24h=very_low_fee_net_24h,
        low_fee_net_24h=low_fee_net_24h,
        one_bps_each_net_24h=one_bps_each_net_24h,
        max_entry_slippage_bps=max_entry_slippage_bps,
        reason=reason,
    )


def _scenario_value(
    by_scenario: dict[str, dict[str, str]],
    scenario: str,
    field: str,
) -> Decimal | None:
    row = by_scenario.get(scenario)
    return Decimal(row[field]) if row is not None else None


def _action_rank(action: str) -> int:
    ranks = {
        "paper_8h_candidate": 5,
        "active_24h_monitor": 4,
        "fee_dependent_24h_monitor": 3,
        "very_low_fee_24h_watch": 2,
        "thin_or_unstable_watch": 1,
        "drop_for_now": 0,
    }
    return ranks[action]


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def _fmt_optional(value: Decimal | None) -> str:
    return "" if value is None else _fmt(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_score.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_triage.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_triage.md",
    )
    args = parser.parse_args()

    triage = build_event_window_triage(score_path=args.score_path)
    write_event_window_triage_csv(triage, output_path=args.csv_output_path)
    write_event_window_triage_md(triage, output_path=args.md_output_path)
    for item in triage:
        print(item.asset, item.event_action, item.reason)


if __name__ == "__main__":
    main()
