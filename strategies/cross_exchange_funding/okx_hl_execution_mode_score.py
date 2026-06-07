from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from strategies.cross_exchange_funding.okx_hl_book_depth import build_book_depth_check
from strategies.cross_exchange_funding.okx_hl_candidate_score import (
    DEFAULT_SCENARIOS,
    FeeScenario,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    okx_cross: bool
    hl_cross: bool


@dataclass(frozen=True)
class ExecutionModeScore:
    asset: str
    scenario: str
    execution_mode: str
    long_venue: str
    short_venue: str
    gross_event_8h_rate: Decimal
    gross_event_24h_rate: Decimal
    okx_taker_slippage_bps: Decimal
    hl_taker_slippage_bps: Decimal
    entry_slippage_bps: Decimal
    round_trip_slippage_rate: Decimal
    fee_round_trip_rate: Decimal
    all_in_round_trip_cost_rate: Decimal
    net_event_8h_after_cost: Decimal
    net_event_24h_after_cost: Decimal
    both_touch_rate: Decimal | None
    okx_only_touch_rate: Decimal | None
    hl_only_touch_rate: Decimal | None
    capacity: Decimal


EXECUTION_MODES = (
    ExecutionMode("both_maker", okx_cross=False, hl_cross=False),
    ExecutionMode("okx_cross_hl_maker", okx_cross=True, hl_cross=False),
    ExecutionMode("okx_maker_hl_cross", okx_cross=False, hl_cross=True),
    ExecutionMode("both_cross", okx_cross=True, hl_cross=True),
)


def build_execution_mode_scores(
    *,
    event_score_path: Path = ROOT / "okx_hl_event_window_score.csv",
    pair_summary_path: Path = ROOT / "okx_hl_maker_touch_pair_summary.csv",
    assets: tuple[str, ...] | None = None,
    target_notional: Decimal = Decimal("1000"),
    scenarios: tuple[FeeScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[ExecutionModeScore, ...]:
    event_rows = _read_rows(event_score_path)
    if assets is not None:
        asset_set = set(assets)
        event_rows = tuple(row for row in event_rows if row["asset"] in asset_set)
    pair_rows = {row["asset"]: row for row in _read_rows(pair_summary_path)}
    by_asset_scenario = {
        (row["asset"], row["scenario"]): row
        for row in event_rows
        if row["scenario"] in {scenario.name for scenario in scenarios}
    }
    depth_by_asset = {
        asset: _build_asset_depth(row=row, target_notional=target_notional)
        for asset, row in _first_rows_by_asset(event_rows).items()
    }
    scores = [
        _score_mode(
            row=row,
            scenario=scenario,
            mode=mode,
            okx_taker_slippage_bps=depth_by_asset[row["asset"]].okx_check.slippage_bps,
            hl_taker_slippage_bps=depth_by_asset[row["asset"]].hl_check.slippage_bps,
            pair_row=pair_rows.get(row["asset"]),
        )
        for row in by_asset_scenario.values()
        for scenario in scenarios
        if row["scenario"] == scenario.name
        for mode in EXECUTION_MODES
    ]
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                score.net_event_24h_after_cost > 0,
                score.net_event_24h_after_cost,
                score.net_event_8h_after_cost > 0,
                score.net_event_8h_after_cost,
                score.capacity,
            ),
            reverse=True,
        )
    )


def write_execution_mode_scores_csv(
    scores: tuple[ExecutionModeScore, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "scenario",
                "execution_mode",
                "long_venue",
                "short_venue",
                "gross_event_8h_rate",
                "gross_event_24h_rate",
                "okx_taker_slippage_bps",
                "hl_taker_slippage_bps",
                "entry_slippage_bps",
                "round_trip_slippage_rate",
                "fee_round_trip_rate",
                "all_in_round_trip_cost_rate",
                "net_event_8h_after_cost",
                "net_event_24h_after_cost",
                "both_touch_rate",
                "okx_only_touch_rate",
                "hl_only_touch_rate",
                "capacity",
            )
        )
        for score in scores:
            writer.writerow(
                (
                    score.asset,
                    score.scenario,
                    score.execution_mode,
                    score.long_venue,
                    score.short_venue,
                    _fmt(score.gross_event_8h_rate),
                    _fmt(score.gross_event_24h_rate),
                    _fmt(score.okx_taker_slippage_bps),
                    _fmt(score.hl_taker_slippage_bps),
                    _fmt(score.entry_slippage_bps),
                    _fmt(score.round_trip_slippage_rate),
                    _fmt(score.fee_round_trip_rate),
                    _fmt(score.all_in_round_trip_cost_rate),
                    _fmt(score.net_event_8h_after_cost),
                    _fmt(score.net_event_24h_after_cost),
                    _fmt_optional(score.both_touch_rate),
                    _fmt_optional(score.okx_only_touch_rate),
                    _fmt_optional(score.hl_only_touch_rate),
                    _fmt(score.capacity),
                )
            )
    return output_path


def write_execution_mode_scores_md(
    scores: tuple[ExecutionModeScore, ...],
    *,
    output_path: Path,
    top: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Execution Mode Score\n\n")
        handle.write(
            "This compares maker/cross execution modes against event-window funding "
            "edge. Maker rebates and real queue position are not modeled.\n\n"
        )
        handle.write(
            "| asset | scenario | mode | gross 8h | gross 24h | entry slippage bps | cost | net 8h | net 24h | both touch | OKX only | HL only | capacity |\n"
        )
        handle.write(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for score in scores[:top]:
            handle.write(
                "| "
                f"{score.asset} | "
                f"{score.scenario} | "
                f"{score.execution_mode} | "
                f"{_fmt(score.gross_event_8h_rate)} | "
                f"{_fmt(score.gross_event_24h_rate)} | "
                f"{_fmt(score.entry_slippage_bps)} | "
                f"{_fmt(score.all_in_round_trip_cost_rate)} | "
                f"{_fmt(score.net_event_8h_after_cost)} | "
                f"{_fmt(score.net_event_24h_after_cost)} | "
                f"{_fmt_optional(score.both_touch_rate)} | "
                f"{_fmt_optional(score.okx_only_touch_rate)} | "
                f"{_fmt_optional(score.hl_only_touch_rate)} | "
                f"{_fmt(score.capacity)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "If only both_maker survives, the candidate depends on maker availability. "
            "If a one-leg-cross mode survives, execution may be easier, but real fees "
            "and adverse selection still need account-level validation. Missing touch "
            "rates mean the maker-touch probe has not been run for that asset yet.\n"
        )
    return output_path


def _score_mode(
    *,
    row: dict[str, str],
    scenario: FeeScenario,
    mode: ExecutionMode,
    okx_taker_slippage_bps: Decimal,
    hl_taker_slippage_bps: Decimal,
    pair_row: dict[str, str] | None,
) -> ExecutionModeScore:
    entry_slippage_bps = (
        (okx_taker_slippage_bps if mode.okx_cross else Decimal("0"))
        + (hl_taker_slippage_bps if mode.hl_cross else Decimal("0"))
    )
    round_trip_slippage_rate = (entry_slippage_bps / Decimal("10000")) * Decimal("2")
    fee_round_trip_rate = (
        Decimal("2")
        * (scenario.okx_fee_bps_per_fill + scenario.hl_fee_bps_per_fill)
        / Decimal("10000")
    )
    all_in_cost = fee_round_trip_rate + round_trip_slippage_rate
    gross_8h = Decimal(row["gross_event_8h_rate"])
    gross_24h = Decimal(row["gross_event_24h_rate"])
    return ExecutionModeScore(
        asset=row["asset"],
        scenario=scenario.name,
        execution_mode=mode.name,
        long_venue=row["long_venue"],
        short_venue=row["short_venue"],
        gross_event_8h_rate=gross_8h,
        gross_event_24h_rate=gross_24h,
        okx_taker_slippage_bps=okx_taker_slippage_bps,
        hl_taker_slippage_bps=hl_taker_slippage_bps,
        entry_slippage_bps=entry_slippage_bps,
        round_trip_slippage_rate=round_trip_slippage_rate,
        fee_round_trip_rate=fee_round_trip_rate,
        all_in_round_trip_cost_rate=all_in_cost,
        net_event_8h_after_cost=gross_8h - all_in_cost,
        net_event_24h_after_cost=gross_24h - all_in_cost,
        both_touch_rate=_pair_value(pair_row, "both_touch_rate"),
        okx_only_touch_rate=_pair_value(pair_row, "okx_only_touch_rate"),
        hl_only_touch_rate=_pair_value(pair_row, "hl_only_touch_rate"),
        capacity=Decimal(row["capacity"]),
    )


def _build_asset_depth(*, row: dict[str, str], target_notional: Decimal):
    long_venue = row["long_venue"]
    okx_side = "buy" if long_venue == "OkxSwap" else "sell"
    hl_side = "buy" if long_venue == "HlPerp" else "sell"
    return build_book_depth_check(
        asset=row["asset"],
        okx_target_notional=target_notional,
        hl_target_notional=target_notional,
        okx_side=okx_side,
        hl_side=hl_side,
    )


def _first_rows_by_asset(rows: tuple[dict[str, str], ...]) -> dict[str, dict[str, str]]:
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        first_rows.setdefault(row["asset"], row)
    return first_rows


def _pair_value(row: dict[str, str] | None, field: str) -> Decimal | None:
    return Decimal(row[field]) if row is not None else None


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def _fmt_optional(value: Decimal | None) -> str:
    return "" if value is None else _fmt(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-score-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_score.csv",
    )
    parser.add_argument(
        "--pair-summary-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_pair_summary.csv",
    )
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("1000"))
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_execution_mode_score.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_execution_mode_score.md",
    )
    parser.add_argument("--top", type=int, default=24)
    args = parser.parse_args()

    scores = build_execution_mode_scores(
        event_score_path=args.event_score_path,
        pair_summary_path=args.pair_summary_path,
        assets=tuple(asset.upper() for asset in args.assets) if args.assets else None,
        target_notional=args.target_notional,
    )
    write_execution_mode_scores_csv(scores, output_path=args.csv_output_path)
    write_execution_mode_scores_md(scores, output_path=args.md_output_path, top=args.top)
    for score in scores[: args.top]:
        print(
            score.asset,
            score.scenario,
            score.execution_mode,
            f"net8h={_fmt(score.net_event_8h_after_cost)}",
            f"net24h={_fmt(score.net_event_24h_after_cost)}",
            f"slip_bps={_fmt(score.entry_slippage_bps)}",
        )


if __name__ == "__main__":
    main()
