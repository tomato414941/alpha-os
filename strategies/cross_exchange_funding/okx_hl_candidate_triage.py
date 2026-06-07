from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CandidateTriage:
    asset: str
    action: str
    long_venue: str
    short_venue: str
    observations: int
    capacity: Decimal
    very_low_fee_net_8h: Decimal | None
    low_fee_net_24h: Decimal | None
    one_bps_each_net_24h: Decimal | None
    max_entry_slippage_bps: Decimal
    reason: str


def build_candidate_triage(
    *,
    score_path: Path = ROOT / "okx_hl_execution_cost_score.csv",
    min_capacity_for_active_monitor: Decimal = Decimal("50000"),
    min_capacity_for_paper_8h: Decimal = Decimal("100000"),
) -> tuple[CandidateTriage, ...]:
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    by_asset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_asset.setdefault(row["asset"], []).append(row)
    triage = tuple(
        _triage_asset(
            asset=asset,
            rows=tuple(asset_rows),
            min_capacity_for_active_monitor=min_capacity_for_active_monitor,
            min_capacity_for_paper_8h=min_capacity_for_paper_8h,
        )
        for asset, asset_rows in by_asset.items()
    )
    return tuple(
        sorted(
            triage,
            key=lambda item: (
                _action_rank(item.action),
                item.one_bps_each_net_24h or Decimal("-999"),
                item.low_fee_net_24h or Decimal("-999"),
                item.capacity,
            ),
            reverse=True,
        )
    )


def write_candidate_triage_csv(
    triage: tuple[CandidateTriage, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "action",
                "long_venue",
                "short_venue",
                "observations",
                "capacity",
                "very_low_fee_net_8h",
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
                    item.action,
                    item.long_venue,
                    item.short_venue,
                    item.observations,
                    _fmt(item.capacity),
                    _fmt_optional(item.very_low_fee_net_8h),
                    _fmt_optional(item.low_fee_net_24h),
                    _fmt_optional(item.one_bps_each_net_24h),
                    _fmt(item.max_entry_slippage_bps),
                    item.reason,
                )
            )
    return output_path


def write_candidate_triage_md(
    triage: tuple[CandidateTriage, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Candidate Triage\n\n")
        handle.write(
            "This turns execution-cost scores into research actions. It is not a "
            "trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | long | short | obs | capacity | very-low 8h | low-fee 24h | one-bps 24h | max slippage bps | reason |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for item in triage:
            handle.write(
                "| "
                f"{item.asset} | "
                f"{item.action} | "
                f"{item.long_venue} | "
                f"{item.short_venue} | "
                f"{item.observations} | "
                f"{_fmt(item.capacity)} | "
                f"{_fmt_optional(item.very_low_fee_net_8h)} | "
                f"{_fmt_optional(item.low_fee_net_24h)} | "
                f"{_fmt_optional(item.one_bps_each_net_24h)} | "
                f"{_fmt(item.max_entry_slippage_bps)} | "
                f"{item.reason} |\n"
            )
    return output_path


def _triage_asset(
    *,
    asset: str,
    rows: tuple[dict[str, str], ...],
    min_capacity_for_active_monitor: Decimal,
    min_capacity_for_paper_8h: Decimal,
) -> CandidateTriage:
    by_scenario = {row["scenario"]: row for row in rows}
    first = rows[0]
    capacity = Decimal(first["mean_capacity_proxy_notional"])
    very_low_fee_net_8h = _scenario_value(
        by_scenario,
        "very_low_fee",
        "net_8h_after_all_in_cost",
    )
    low_fee_net_24h = _scenario_value(
        by_scenario,
        "low_fee",
        "net_24h_after_all_in_cost",
    )
    one_bps_each_net_24h = _scenario_value(
        by_scenario,
        "one_bps_each",
        "net_24h_after_all_in_cost",
    )
    all_filled = all(
        row["okx_fully_filled"] == "True" and row["hl_fully_filled"] == "True"
        for row in rows
    )
    max_entry_slippage_bps = max(Decimal(row["entry_slippage_bps"]) for row in rows)
    if (
        all_filled
        and very_low_fee_net_8h is not None
        and very_low_fee_net_8h > 0
        and capacity >= min_capacity_for_paper_8h
    ):
        action = "paper_8h_candidate"
        reason = "8h survives only under very low fee assumptions; verify real fees and maker fills"
    elif (
        all_filled
        and one_bps_each_net_24h is not None
        and one_bps_each_net_24h > 0
        and capacity >= min_capacity_for_active_monitor
    ):
        action = "active_24h_monitor"
        reason = "24h survives one-bps-each assumption with enough visible capacity for monitoring"
    elif (
        all_filled
        and low_fee_net_24h is not None
        and low_fee_net_24h > 0
        and capacity >= min_capacity_for_active_monitor
    ):
        action = "fee_dependent_24h_monitor"
        reason = "24h survives low-fee assumption but not one-bps-each"
    elif any(Decimal(row["net_24h_after_all_in_cost"]) > 0 for row in rows):
        action = "thin_or_unstable_watch"
        reason = "24h can be positive, but capacity, slippage, or fee assumptions are weak"
    else:
        action = "drop_for_now"
        reason = "no current 24h all-in scenario survives"
    return CandidateTriage(
        asset=asset,
        action=action,
        long_venue=first["long_venue"],
        short_venue=first["short_venue"],
        observations=int(first["observations"]),
        capacity=capacity,
        very_low_fee_net_8h=very_low_fee_net_8h,
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
        "paper_8h_candidate": 4,
        "active_24h_monitor": 3,
        "fee_dependent_24h_monitor": 2,
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
        default=ROOT / "okx_hl_execution_cost_score.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_candidate_triage.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_candidate_triage.md",
    )
    args = parser.parse_args()

    triage = build_candidate_triage(score_path=args.score_path)
    write_candidate_triage_csv(triage, output_path=args.csv_output_path)
    write_candidate_triage_md(triage, output_path=args.md_output_path)
    for item in triage:
        print(item.asset, item.action, item.reason)


if __name__ == "__main__":
    main()
