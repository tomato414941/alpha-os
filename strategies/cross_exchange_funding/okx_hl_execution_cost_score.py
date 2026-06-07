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
class ExecutionCostScore:
    asset: str
    scenario: str
    observations: int
    long_venue: str
    short_venue: str
    target_notional: Decimal
    gross_8h_rate: Decimal
    gross_24h_rate: Decimal
    entry_slippage_bps: Decimal
    round_trip_slippage_rate: Decimal
    fee_round_trip_rate: Decimal
    all_in_round_trip_cost_rate: Decimal
    net_8h_after_all_in_cost: Decimal
    net_24h_after_all_in_cost: Decimal
    mean_capacity_proxy_notional: Decimal
    okx_fully_filled: bool
    hl_fully_filled: bool


def build_execution_cost_scores(
    *,
    summary_path: Path = ROOT / "okx_hl_funding_persistence_focus_summary.csv",
    assets: tuple[str, ...] | None = None,
    target_notional: Decimal = Decimal("1000"),
    scenarios: tuple[FeeScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[ExecutionCostScore, ...]:
    rows = _read_summary_rows(summary_path=summary_path, assets=assets)
    scores = [
        score
        for row in rows
        for score in _score_asset(
            row,
            target_notional=target_notional,
            scenarios=scenarios,
        )
    ]
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                score.net_24h_after_all_in_cost > 0,
                score.net_24h_after_all_in_cost,
                score.net_8h_after_all_in_cost > 0,
                score.net_8h_after_all_in_cost,
                score.mean_capacity_proxy_notional,
            ),
            reverse=True,
        )
    )


def write_execution_cost_scores_csv(
    scores: tuple[ExecutionCostScore, ...],
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
                "observations",
                "long_venue",
                "short_venue",
                "target_notional",
                "gross_8h_rate",
                "gross_24h_rate",
                "entry_slippage_bps",
                "round_trip_slippage_rate",
                "fee_round_trip_rate",
                "all_in_round_trip_cost_rate",
                "net_8h_after_all_in_cost",
                "net_24h_after_all_in_cost",
                "mean_capacity_proxy_notional",
                "okx_fully_filled",
                "hl_fully_filled",
            )
        )
        for score in scores:
            writer.writerow(
                (
                    score.asset,
                    score.scenario,
                    score.observations,
                    score.long_venue,
                    score.short_venue,
                    _fmt(score.target_notional),
                    _fmt(score.gross_8h_rate),
                    _fmt(score.gross_24h_rate),
                    _fmt(score.entry_slippage_bps),
                    _fmt(score.round_trip_slippage_rate),
                    _fmt(score.fee_round_trip_rate),
                    _fmt(score.all_in_round_trip_cost_rate),
                    _fmt(score.net_8h_after_all_in_cost),
                    _fmt(score.net_24h_after_all_in_cost),
                    _fmt(score.mean_capacity_proxy_notional),
                    score.okx_fully_filled,
                    score.hl_fully_filled,
                )
            )
    return output_path


def write_execution_cost_scores_md(
    scores: tuple[ExecutionCostScore, ...],
    *,
    output_path: Path,
    top: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Execution Cost Score\n\n")
        handle.write(
            "This scores focused candidates by subtracting observed top-book taker "
            "slippage and simple fee assumptions from gross funding edge.\n\n"
        )
        handle.write(
            "| asset | scenario | long | short | gross 8h | gross 24h | entry slippage bps | all-in cost | net 8h | net 24h | capacity | filled |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for score in scores[:top]:
            handle.write(
                "| "
                f"{score.asset} | "
                f"{score.scenario} | "
                f"{score.long_venue} | "
                f"{score.short_venue} | "
                f"{_fmt(score.gross_8h_rate)} | "
                f"{_fmt(score.gross_24h_rate)} | "
                f"{_fmt(score.entry_slippage_bps)} | "
                f"{_fmt(score.all_in_round_trip_cost_rate)} | "
                f"{_fmt(score.net_8h_after_all_in_cost)} | "
                f"{_fmt(score.net_24h_after_all_in_cost)} | "
                f"{_fmt(score.mean_capacity_proxy_notional)} | "
                f"{score.okx_fully_filled and score.hl_fully_filled} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A positive 24h score here is still not enough to trade. It only means the "
            "current public top book and fee assumption do not immediately erase the "
            "funding edge. Real account fee tier, maker behavior, funding-event timing, "
            "and persistence still decide whether the candidate is executable.\n"
        )
    return output_path


def _score_asset(
    row: dict[str, str],
    *,
    target_notional: Decimal,
    scenarios: tuple[FeeScenario, ...],
) -> tuple[ExecutionCostScore, ...]:
    asset = row["asset"]
    long_venue = row["dominant_long_venue"]
    okx_side = "buy" if long_venue == "OkxSwap" else "sell"
    hl_side = "buy" if long_venue == "HlPerp" else "sell"
    depth = build_book_depth_check(
        asset=asset,
        okx_target_notional=target_notional,
        hl_target_notional=target_notional,
        okx_side=okx_side,
        hl_side=hl_side,
    )
    entry_slippage_bps = depth.combined_taker_slippage_bps
    round_trip_slippage_rate = (entry_slippage_bps / Decimal("10000")) * Decimal("2")
    annualized_spread = Decimal(row["mean_annualized_spread"])
    gross_8h_rate = annualized_spread * Decimal("8") / Decimal("24") / Decimal("365")
    gross_24h_rate = annualized_spread / Decimal("365")
    return tuple(
        _score_scenario(
            row=row,
            scenario=scenario,
            target_notional=target_notional,
            gross_8h_rate=gross_8h_rate,
            gross_24h_rate=gross_24h_rate,
            entry_slippage_bps=entry_slippage_bps,
            round_trip_slippage_rate=round_trip_slippage_rate,
            okx_fully_filled=depth.okx_check.fully_filled,
            hl_fully_filled=depth.hl_check.fully_filled,
        )
        for scenario in scenarios
    )


def _score_scenario(
    *,
    row: dict[str, str],
    scenario: FeeScenario,
    target_notional: Decimal,
    gross_8h_rate: Decimal,
    gross_24h_rate: Decimal,
    entry_slippage_bps: Decimal,
    round_trip_slippage_rate: Decimal,
    okx_fully_filled: bool,
    hl_fully_filled: bool,
) -> ExecutionCostScore:
    fee_round_trip_rate = (
        Decimal("2")
        * (scenario.okx_fee_bps_per_fill + scenario.hl_fee_bps_per_fill)
        / Decimal("10000")
    )
    all_in_cost = fee_round_trip_rate + round_trip_slippage_rate
    return ExecutionCostScore(
        asset=row["asset"],
        scenario=scenario.name,
        observations=int(row["observations"]),
        long_venue=row["dominant_long_venue"],
        short_venue=row["dominant_short_venue"],
        target_notional=target_notional,
        gross_8h_rate=gross_8h_rate,
        gross_24h_rate=gross_24h_rate,
        entry_slippage_bps=entry_slippage_bps,
        round_trip_slippage_rate=round_trip_slippage_rate,
        fee_round_trip_rate=fee_round_trip_rate,
        all_in_round_trip_cost_rate=all_in_cost,
        net_8h_after_all_in_cost=gross_8h_rate - all_in_cost,
        net_24h_after_all_in_cost=gross_24h_rate - all_in_cost,
        mean_capacity_proxy_notional=Decimal(row["mean_capacity_proxy_notional"]),
        okx_fully_filled=okx_fully_filled,
        hl_fully_filled=hl_fully_filled,
    )


def _read_summary_rows(
    *,
    summary_path: Path,
    assets: tuple[str, ...] | None,
) -> tuple[dict[str, str], ...]:
    asset_set = set(assets) if assets is not None else None
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if asset_set is None:
        return rows
    return tuple(row for row in rows if row["asset"] in asset_set)


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "okx_hl_funding_persistence_focus_summary.csv",
    )
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("1000"))
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_execution_cost_score.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_execution_cost_score.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    assets = tuple(asset.upper() for asset in args.assets) if args.assets else None
    scores = build_execution_cost_scores(
        summary_path=args.summary_path,
        assets=assets,
        target_notional=args.target_notional,
    )
    write_execution_cost_scores_csv(scores, output_path=args.csv_output_path)
    write_execution_cost_scores_md(
        scores,
        output_path=args.md_output_path,
        top=args.top,
    )
    for score in scores[: args.top]:
        print(
            score.asset,
            score.scenario,
            f"net8h={_fmt(score.net_8h_after_all_in_cost)}",
            f"net24h={_fmt(score.net_24h_after_all_in_cost)}",
            f"slippage_bps={_fmt(score.entry_slippage_bps)}",
            f"capacity={_fmt(score.mean_capacity_proxy_notional)}",
        )


if __name__ == "__main__":
    main()
