from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.data import LOCAL_DATASET_DIR as SPOT_DATASET_DIR
from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR as PERP_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.crypto_market_structure.spot_perp_carry import (
    SpotDay,
    load_spot_days,
    run_spot_perp_carry_screen,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeeCeiling:
    candidate: str
    max_paired_leg_cost_rate: float
    max_paired_leg_cost_bps: float
    zero_cost_total_return: float
    zero_cost_sharpe: float
    default_cost_total_return: float
    default_cost_sharpe: float
    default_cost_rate: float
    max_drawdown_at_default_cost: float
    mean_daily_turnover: float


def build_fee_ceilings(
    *,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float = 0.0002,
    capital_per_notional: float = 2.0,
    default_cost_rate: float = 0.0004,
    max_search_cost_rate: float = 0.01,
    iterations: int = 24,
) -> tuple[FeeCeiling, ...]:
    zero_cost = _results_by_candidate(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=min_funding_rate_sum,
        capital_per_notional=capital_per_notional,
        paired_leg_cost_rate=0.0,
    )
    default_cost = _results_by_candidate(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=min_funding_rate_sum,
        capital_per_notional=capital_per_notional,
        paired_leg_cost_rate=default_cost_rate,
    )
    ceilings = tuple(
        FeeCeiling(
            candidate=candidate,
            max_paired_leg_cost_rate=_break_even_cost(
                candidate=candidate,
                spot_rows_by_symbol=spot_rows_by_symbol,
                perp_rows_by_symbol=perp_rows_by_symbol,
                min_funding_rate_sum=min_funding_rate_sum,
                capital_per_notional=capital_per_notional,
                max_search_cost_rate=max_search_cost_rate,
                iterations=iterations,
            ),
            max_paired_leg_cost_bps=0.0,
            zero_cost_total_return=result.total_return,
            zero_cost_sharpe=result.sharpe,
            default_cost_total_return=default_cost[candidate].total_return,
            default_cost_sharpe=default_cost[candidate].sharpe,
            default_cost_rate=default_cost_rate,
            max_drawdown_at_default_cost=default_cost[candidate].max_drawdown,
            mean_daily_turnover=default_cost[candidate].mean_daily_turnover,
        )
        for candidate, result in zero_cost.items()
    )
    return tuple(
        sorted(
            (
                FeeCeiling(
                    candidate=ceiling.candidate,
                    max_paired_leg_cost_rate=ceiling.max_paired_leg_cost_rate,
                    max_paired_leg_cost_bps=ceiling.max_paired_leg_cost_rate * 10000.0,
                    zero_cost_total_return=ceiling.zero_cost_total_return,
                    zero_cost_sharpe=ceiling.zero_cost_sharpe,
                    default_cost_total_return=ceiling.default_cost_total_return,
                    default_cost_sharpe=ceiling.default_cost_sharpe,
                    default_cost_rate=ceiling.default_cost_rate,
                    max_drawdown_at_default_cost=ceiling.max_drawdown_at_default_cost,
                    mean_daily_turnover=ceiling.mean_daily_turnover,
                )
                for ceiling in ceilings
            ),
            key=lambda row: (
                row.max_paired_leg_cost_rate,
                row.default_cost_sharpe,
                row.default_cost_total_return,
            ),
            reverse=True,
        )
    )


def write_fee_ceilings_csv(
    ceilings: tuple[FeeCeiling, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "max_paired_leg_cost_rate",
                "max_paired_leg_cost_bps",
                "zero_cost_total_return",
                "zero_cost_sharpe",
                "default_cost_rate",
                "default_cost_total_return",
                "default_cost_sharpe",
                "max_drawdown_at_default_cost",
                "mean_daily_turnover",
            )
        )
        for ceiling in ceilings:
            writer.writerow(
                (
                    ceiling.candidate,
                    f"{ceiling.max_paired_leg_cost_rate:.10f}",
                    f"{ceiling.max_paired_leg_cost_bps:.6f}",
                    f"{ceiling.zero_cost_total_return:.10f}",
                    f"{ceiling.zero_cost_sharpe:.10f}",
                    f"{ceiling.default_cost_rate:.10f}",
                    f"{ceiling.default_cost_total_return:.10f}",
                    f"{ceiling.default_cost_sharpe:.10f}",
                    f"{ceiling.max_drawdown_at_default_cost:.10f}",
                    f"{ceiling.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def write_fee_ceilings_md(
    ceilings: tuple[FeeCeiling, ...],
    *,
    output_path: Path,
    top: int = 8,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Spot/Perp Carry Fee Ceiling\n\n")
        handle.write(
            "This estimates the maximum paired-leg cost before each spot/perp carry "
            "candidate loses positive total return. It is based on the same historical "
            "spot/perp approximation as `spot_perp_carry.py`.\n\n"
        )
        handle.write(
            "| candidate | max paired-leg cost bps | zero-cost total | zero-cost sharpe | default total | default sharpe | drawdown | turnover |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for ceiling in ceilings[:top]:
            handle.write(
                "| "
                f"{ceiling.candidate} | "
                f"{ceiling.max_paired_leg_cost_bps:.6f} | "
                f"{ceiling.zero_cost_total_return:.6f} | "
                f"{ceiling.zero_cost_sharpe:.6f} | "
                f"{ceiling.default_cost_total_return:.6f} | "
                f"{ceiling.default_cost_sharpe:.6f} | "
                f"{ceiling.max_drawdown_at_default_cost:.6f} | "
                f"{ceiling.mean_daily_turnover:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Higher ceilings indicate more execution-cost room. A low-turnover 14-day "
            "candidate can survive much higher paired-leg costs than daily or 3-day "
            "variants. This still omits exchange-specific margin, borrow, liquidation, "
            "and order-book availability.\n"
        )
    return output_path


def _break_even_cost(
    *,
    candidate: str,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float,
    capital_per_notional: float,
    max_search_cost_rate: float,
    iterations: int,
) -> float:
    zero_result = _result_for_cost(
        candidate=candidate,
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=min_funding_rate_sum,
        capital_per_notional=capital_per_notional,
        paired_leg_cost_rate=0.0,
    )
    if zero_result.total_return <= 0.0:
        return 0.0
    low = 0.0
    high = max_search_cost_rate
    while (
        _result_for_cost(
            candidate=candidate,
            spot_rows_by_symbol=spot_rows_by_symbol,
            perp_rows_by_symbol=perp_rows_by_symbol,
            min_funding_rate_sum=min_funding_rate_sum,
            capital_per_notional=capital_per_notional,
            paired_leg_cost_rate=high,
        ).total_return
        > 0.0
    ):
        high *= 2.0
        if high > 0.2:
            return high
    for _ in range(iterations):
        mid = (low + high) / 2.0
        result = _result_for_cost(
            candidate=candidate,
            spot_rows_by_symbol=spot_rows_by_symbol,
            perp_rows_by_symbol=perp_rows_by_symbol,
            min_funding_rate_sum=min_funding_rate_sum,
            capital_per_notional=capital_per_notional,
            paired_leg_cost_rate=mid,
        )
        if result.total_return > 0.0:
            low = mid
        else:
            high = mid
    return low


def _result_for_cost(
    *,
    candidate: str,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float,
    capital_per_notional: float,
    paired_leg_cost_rate: float,
):
    return _results_by_candidate(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=min_funding_rate_sum,
        capital_per_notional=capital_per_notional,
        paired_leg_cost_rate=paired_leg_cost_rate,
    )[candidate]


def _results_by_candidate(
    *,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float,
    capital_per_notional: float,
    paired_leg_cost_rate: float,
):
    return {
        result.candidate: result
        for result in run_spot_perp_carry_screen(
            spot_rows_by_symbol=spot_rows_by_symbol,
            perp_rows_by_symbol=perp_rows_by_symbol,
            min_funding_rate_sum=min_funding_rate_sum,
            paired_leg_cost_rate=paired_leg_cost_rate,
            capital_per_notional=capital_per_notional,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dataset-dir", type=Path, default=SPOT_DATASET_DIR)
    parser.add_argument("--perp-dataset-dir", type=Path, default=PERP_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--capital-per-notional", type=float, default=2.0)
    parser.add_argument("--default-cost-rate", type=float, default=0.0004)
    parser.add_argument("--max-search-cost-rate", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_fee_ceiling.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_fee_ceiling.md",
    )
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    spot_rows_by_symbol = load_spot_days(
        dataset_dir=args.spot_dataset_dir,
        symbols=tuple(args.symbols),
    )
    perp_rows_by_symbol = load_market_structure_days(
        dataset_dir=args.perp_dataset_dir,
        symbols=tuple(args.symbols),
    )
    ceilings = build_fee_ceilings(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=args.min_funding_rate_sum,
        capital_per_notional=args.capital_per_notional,
        default_cost_rate=args.default_cost_rate,
        max_search_cost_rate=args.max_search_cost_rate,
        iterations=args.iterations,
    )
    write_fee_ceilings_csv(ceilings, output_path=args.csv_output_path)
    write_fee_ceilings_md(ceilings, output_path=args.md_output_path, top=args.top)
    for ceiling in ceilings[: args.top]:
        print(
            ceiling.candidate,
            f"max_cost_bps={ceiling.max_paired_leg_cost_bps:.6f}",
            f"default_total={ceiling.default_cost_total_return:.6f}",
            f"default_sharpe={ceiling.default_cost_sharpe:.6f}",
        )


if __name__ == "__main__":
    main()
