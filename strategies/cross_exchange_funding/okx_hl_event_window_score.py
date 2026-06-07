from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from strategies.cross_exchange_funding.okx_hl_funding_alignment import (
    FundingAlignment,
    build_funding_alignment,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventWindowScore:
    asset: str
    action: str
    scenario: str
    long_venue: str
    short_venue: str
    okx_events_8h: int
    hl_events_8h: int
    okx_events_24h: int
    hl_events_24h: int
    okx_funding_rate: Decimal
    hl_funding_rate: Decimal
    gross_event_8h_rate: Decimal
    gross_event_24h_rate: Decimal
    all_in_round_trip_cost_rate: Decimal
    net_event_8h_after_all_in_cost: Decimal
    net_event_24h_after_all_in_cost: Decimal
    capacity: Decimal
    max_entry_slippage_bps: Decimal


def build_event_window_scores(
    *,
    triage_path: Path = ROOT / "okx_hl_candidate_triage.csv",
    execution_score_path: Path = ROOT / "okx_hl_execution_cost_score.csv",
) -> tuple[EventWindowScore, ...]:
    triage_rows = _read_rows(triage_path)
    cost_rows = _read_rows(execution_score_path)
    costs_by_asset_scenario = {
        (row["asset"], row["scenario"]): row for row in cost_rows
    }
    alignments = {
        triage_row["asset"]: build_funding_alignment(asset=triage_row["asset"])
        for triage_row in triage_rows
    }
    scores = [
        _build_asset_scenario_score(
            triage_row=triage_row,
            cost_row=cost_row,
            alignment=alignments[triage_row["asset"]],
        )
        for triage_row in triage_rows
        for cost_row in _cost_rows_for_asset(
            triage_row=triage_row,
            costs_by_asset_scenario=costs_by_asset_scenario,
        )
    ]
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                score.net_event_24h_after_all_in_cost > 0,
                score.net_event_24h_after_all_in_cost,
                score.net_event_8h_after_all_in_cost > 0,
                score.net_event_8h_after_all_in_cost,
                score.capacity,
            ),
            reverse=True,
        )
    )


def write_event_window_scores_csv(
    scores: tuple[EventWindowScore, ...],
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
                "scenario",
                "long_venue",
                "short_venue",
                "okx_events_8h",
                "hl_events_8h",
                "okx_events_24h",
                "hl_events_24h",
                "okx_funding_rate",
                "hl_funding_rate",
                "gross_event_8h_rate",
                "gross_event_24h_rate",
                "all_in_round_trip_cost_rate",
                "net_event_8h_after_all_in_cost",
                "net_event_24h_after_all_in_cost",
                "capacity",
                "max_entry_slippage_bps",
            )
        )
        for score in scores:
            writer.writerow(
                (
                    score.asset,
                    score.action,
                    score.scenario,
                    score.long_venue,
                    score.short_venue,
                    score.okx_events_8h,
                    score.hl_events_8h,
                    score.okx_events_24h,
                    score.hl_events_24h,
                    _fmt(score.okx_funding_rate),
                    _fmt(score.hl_funding_rate),
                    _fmt(score.gross_event_8h_rate),
                    _fmt(score.gross_event_24h_rate),
                    _fmt(score.all_in_round_trip_cost_rate),
                    _fmt(score.net_event_8h_after_all_in_cost),
                    _fmt(score.net_event_24h_after_all_in_cost),
                    _fmt(score.capacity),
                    _fmt(score.max_entry_slippage_bps),
                )
            )
    return output_path


def write_event_window_scores_md(
    scores: tuple[EventWindowScore, ...],
    *,
    output_path: Path,
    top: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Event Window Score\n\n")
        handle.write(
            "This scores candidates by current funding event counts, not by a smooth "
            "hourly spread approximation. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | scenario | long | short | OKX 8h | HL 8h | OKX 24h | HL 24h | gross 8h | gross 24h | cost | net 8h | net 24h | capacity |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for score in scores[:top]:
            handle.write(
                "| "
                f"{score.asset} | "
                f"{score.action} | "
                f"{score.scenario} | "
                f"{score.long_venue} | "
                f"{score.short_venue} | "
                f"{score.okx_events_8h} | "
                f"{score.hl_events_8h} | "
                f"{score.okx_events_24h} | "
                f"{score.hl_events_24h} | "
                f"{_fmt(score.gross_event_8h_rate)} | "
                f"{_fmt(score.gross_event_24h_rate)} | "
                f"{_fmt(score.all_in_round_trip_cost_rate)} | "
                f"{_fmt(score.net_event_8h_after_all_in_cost)} | "
                f"{_fmt(score.net_event_24h_after_all_in_cost)} | "
                f"{_fmt(score.capacity)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A candidate that survives the smooth 24h proxy can still fail when the "
            "actual funding events inside the window are counted. This is especially "
            "important because Hyperliquid funds hourly while OKX generally funds "
            "every eight hours.\n"
        )
    return output_path


def _build_asset_scenario_score(
    *,
    triage_row: dict[str, str],
    cost_row: dict[str, str],
    alignment: FundingAlignment,
) -> EventWindowScore:
    gross_event_8h_rate = _event_income(
        alignment=alignment,
        long_venue=triage_row["long_venue"],
        okx_events=alignment.okx_events_8h,
        hl_events=alignment.hl_events_8h,
    )
    gross_event_24h_rate = _event_income(
        alignment=alignment,
        long_venue=triage_row["long_venue"],
        okx_events=alignment.okx_events_24h,
        hl_events=alignment.hl_events_24h,
    )
    all_in_cost = Decimal(cost_row["all_in_round_trip_cost_rate"])
    return EventWindowScore(
        asset=triage_row["asset"],
        action=triage_row["action"],
        scenario=cost_row["scenario"],
        long_venue=triage_row["long_venue"],
        short_venue=triage_row["short_venue"],
        okx_events_8h=alignment.okx_events_8h,
        hl_events_8h=alignment.hl_events_8h,
        okx_events_24h=alignment.okx_events_24h,
        hl_events_24h=alignment.hl_events_24h,
        okx_funding_rate=alignment.okx_funding_rate,
        hl_funding_rate=alignment.hl_funding_rate,
        gross_event_8h_rate=gross_event_8h_rate,
        gross_event_24h_rate=gross_event_24h_rate,
        all_in_round_trip_cost_rate=all_in_cost,
        net_event_8h_after_all_in_cost=gross_event_8h_rate - all_in_cost,
        net_event_24h_after_all_in_cost=gross_event_24h_rate - all_in_cost,
        capacity=Decimal(triage_row["capacity"]),
        max_entry_slippage_bps=Decimal(triage_row["max_entry_slippage_bps"]),
    )


def _event_income(
    *,
    alignment: FundingAlignment,
    long_venue: str,
    okx_events: int,
    hl_events: int,
) -> Decimal:
    if long_venue == "OkxSwap":
        okx_leg = -alignment.okx_funding_rate * Decimal(okx_events)
        hl_leg = alignment.hl_funding_rate * Decimal(hl_events)
    elif long_venue == "HlPerp":
        okx_leg = alignment.okx_funding_rate * Decimal(okx_events)
        hl_leg = -alignment.hl_funding_rate * Decimal(hl_events)
    else:
        raise RuntimeError(f"unknown long venue: {long_venue}")
    return okx_leg + hl_leg


def _cost_rows_for_asset(
    *,
    triage_row: dict[str, str],
    costs_by_asset_scenario: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, str], ...]:
    asset = triage_row["asset"]
    scenarios = ("very_low_fee", "low_fee", "one_bps_each")
    return tuple(
        costs_by_asset_scenario[(asset, scenario)]
        for scenario in scenarios
        if (asset, scenario) in costs_by_asset_scenario
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_score.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_score.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    scores = build_event_window_scores(
        triage_path=args.triage_path,
        execution_score_path=args.execution_score_path,
    )
    write_event_window_scores_csv(scores, output_path=args.csv_output_path)
    write_event_window_scores_md(scores, output_path=args.md_output_path, top=args.top)
    for score in scores[: args.top]:
        print(
            score.asset,
            score.action,
            score.scenario,
            f"net8h={_fmt(score.net_event_8h_after_all_in_cost)}",
            f"net24h={_fmt(score.net_event_24h_after_all_in_cost)}",
            f"okx_events_24h={score.okx_events_24h}",
            f"hl_events_24h={score.hl_events_24h}",
        )


if __name__ == "__main__":
    main()
