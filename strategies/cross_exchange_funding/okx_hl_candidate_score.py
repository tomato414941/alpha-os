from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeeScenario:
    name: str
    okx_fee_bps_per_fill: Decimal
    hl_fee_bps_per_fill: Decimal


@dataclass(frozen=True)
class CandidateScore:
    asset: str
    scenario: str
    observations: int
    long_venue: str
    short_venue: str
    mean_annualized_spread: Decimal
    positive_net_8h_rate: Decimal
    mean_net_8h_proxy: Decimal
    mean_net_24h_proxy: Decimal
    fee_round_trip_rate: Decimal
    net_8h_after_fee: Decimal
    net_24h_after_fee: Decimal
    mean_breakeven_hold_hours: Decimal
    mean_capacity_proxy_notional: Decimal
    survives_8h: bool
    survives_24h: bool


DEFAULT_SCENARIOS = (
    FeeScenario("very_low_fee", Decimal("0.2"), Decimal("0.2")),
    FeeScenario("low_fee", Decimal("0.5"), Decimal("0.5")),
    FeeScenario("one_bps_each", Decimal("1"), Decimal("1")),
)


def build_candidate_scores(
    *,
    summary_path: Path = ROOT / "okx_hl_funding_persistence_1m_summary.csv",
    min_observations: int = 3,
    scenarios: tuple[FeeScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[CandidateScore, ...]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    scores = [
        _score_row(row, scenario=scenario)
        for row in rows
        if int(row["observations"]) >= min_observations
        for scenario in scenarios
    ]
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                score.survives_24h,
                score.net_24h_after_fee,
                score.survives_8h,
                score.positive_net_8h_rate,
                score.mean_capacity_proxy_notional,
            ),
            reverse=True,
        )
    )


def write_candidate_scores(
    scores: tuple[CandidateScore, ...],
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
                "mean_annualized_spread",
                "positive_net_8h_rate",
                "mean_net_8h_proxy",
                "mean_net_24h_proxy",
                "fee_round_trip_rate",
                "net_8h_after_fee",
                "net_24h_after_fee",
                "mean_breakeven_hold_hours",
                "mean_capacity_proxy_notional",
                "survives_8h",
                "survives_24h",
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
                    _fmt(score.mean_annualized_spread),
                    _fmt(score.positive_net_8h_rate),
                    _fmt(score.mean_net_8h_proxy),
                    _fmt(score.mean_net_24h_proxy),
                    _fmt(score.fee_round_trip_rate),
                    _fmt(score.net_8h_after_fee),
                    _fmt(score.net_24h_after_fee),
                    _fmt(score.mean_breakeven_hold_hours),
                    _fmt(score.mean_capacity_proxy_notional),
                    score.survives_8h,
                    score.survives_24h,
                )
            )
    return output_path


def write_candidate_scores_md(
    scores: tuple[CandidateScore, ...],
    *,
    output_path: Path,
    top: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Candidate Score\n\n")
        handle.write(
            "This ranks all assets from the 1m persistence summary after simple fee assumptions.\n\n"
        )
        handle.write(
            "| asset | scenario | long | short | obs | net 8h after fee | net 24h after fee | capacity | survives 8h | survives 24h |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for score in scores[:top]:
            handle.write(
                "| "
                f"{score.asset} | "
                f"{score.scenario} | "
                f"{score.long_venue} | "
                f"{score.short_venue} | "
                f"{score.observations} | "
                f"{_fmt(score.net_8h_after_fee)} | "
                f"{_fmt(score.net_24h_after_fee)} | "
                f"{_fmt(score.mean_capacity_proxy_notional)} | "
                f"{score.survives_8h} | "
                f"{score.survives_24h} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The surviving set is dominated by the very-low-fee assumption. Under one "
            "bps per fill on both venues, the 1m persistence sample does not leave a "
            "robust top candidate. This makes fee tier and maker execution a hard "
            "requirement, not an optimization detail.\n"
        )
    return output_path


def _score_row(row: dict[str, str], *, scenario: FeeScenario) -> CandidateScore:
    fee_round_trip_rate = (
        Decimal("2")
        * (scenario.okx_fee_bps_per_fill + scenario.hl_fee_bps_per_fill)
        / Decimal("10000")
    )
    mean_net_8h_proxy = Decimal(row["mean_net_8h_proxy"])
    mean_net_24h_proxy = Decimal(row["mean_net_24h_proxy"])
    net_8h_after_fee = mean_net_8h_proxy - fee_round_trip_rate
    net_24h_after_fee = mean_net_24h_proxy - fee_round_trip_rate
    return CandidateScore(
        asset=row["asset"],
        scenario=scenario.name,
        observations=int(row["observations"]),
        long_venue=row["dominant_long_venue"],
        short_venue=row["dominant_short_venue"],
        mean_annualized_spread=Decimal(row["mean_annualized_spread"]),
        positive_net_8h_rate=Decimal(row["positive_net_8h_rate"]),
        mean_net_8h_proxy=mean_net_8h_proxy,
        mean_net_24h_proxy=mean_net_24h_proxy,
        fee_round_trip_rate=fee_round_trip_rate,
        net_8h_after_fee=net_8h_after_fee,
        net_24h_after_fee=net_24h_after_fee,
        mean_breakeven_hold_hours=Decimal(row["mean_breakeven_hold_hours"]),
        mean_capacity_proxy_notional=Decimal(row["mean_capacity_proxy_notional"]),
        survives_8h=net_8h_after_fee > 0,
        survives_24h=net_24h_after_fee > 0,
    )


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "okx_hl_funding_persistence_1m_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "okx_hl_candidate_score.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_candidate_score.md",
    )
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    scores = build_candidate_scores(
        summary_path=args.summary_path,
        min_observations=args.min_observations,
    )
    write_candidate_scores(scores, output_path=args.output_path)
    write_candidate_scores_md(scores, output_path=args.md_output_path, top=args.top)
    for score in scores[: args.top]:
        print(
            score.asset,
            score.scenario,
            score.long_venue,
            score.short_venue,
            f"8h={_fmt(score.net_8h_after_fee)}",
            f"24h={_fmt(score.net_24h_after_fee)}",
            f"capacity={_fmt(score.mean_capacity_proxy_notional)}",
            f"survives24h={score.survives_24h}",
        )


if __name__ == "__main__":
    main()
