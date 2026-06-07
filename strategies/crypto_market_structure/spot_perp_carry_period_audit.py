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
DEFAULT_PERIODS = (
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026_to_date", "2026-01-01", "2026-06-07"),
)


@dataclass(frozen=True)
class PeriodResult:
    period: str
    candidate: str
    steps: int
    total_return: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


def build_period_audit(
    *,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    periods: tuple[tuple[str, str, str], ...] = DEFAULT_PERIODS,
    min_funding_rate_sum: float = 0.0002,
    paired_leg_cost_rate: float = 0.0004,
    capital_per_notional: float = 2.0,
) -> tuple[PeriodResult, ...]:
    rows: list[PeriodResult] = []
    for period, start_date, end_date in periods:
        period_spot_rows = _filter_spot_rows(
            spot_rows_by_symbol,
            start_date=start_date,
            end_date=end_date,
        )
        period_perp_rows = _filter_perp_rows(
            perp_rows_by_symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if not period_spot_rows or not period_perp_rows:
            continue
        for result in run_spot_perp_carry_screen(
            spot_rows_by_symbol=period_spot_rows,
            perp_rows_by_symbol=period_perp_rows,
            min_funding_rate_sum=min_funding_rate_sum,
            top_n_values=(1, 2, 3),
            rebalance_days_values=(14,),
            paired_leg_cost_rate=paired_leg_cost_rate,
            capital_per_notional=capital_per_notional,
        ):
            rows.append(
                PeriodResult(
                    period=period,
                    candidate=result.candidate,
                    steps=result.steps,
                    total_return=result.total_return,
                    sharpe=result.sharpe,
                    max_drawdown=result.max_drawdown,
                    mean_daily_turnover=result.mean_daily_turnover,
                )
            )
    return tuple(rows)


def write_period_audit_csv(
    rows: tuple[PeriodResult, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "period",
                "candidate",
                "steps",
                "total_return",
                "sharpe",
                "max_drawdown",
                "mean_daily_turnover",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.period,
                    row.candidate,
                    row.steps,
                    f"{row.total_return:.10f}",
                    f"{row.sharpe:.10f}",
                    f"{row.max_drawdown:.10f}",
                    f"{row.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def write_period_audit_md(
    rows: tuple[PeriodResult, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Spot/Perp Carry Period Audit\n\n")
        handle.write(
            "This checks whether the 14-day spot/perp carry candidate persists across "
            "calendar periods. It uses the broad spot/perp common universe and the "
            "default paired-leg cost.\n\n"
        )
        handle.write(
            "| period | candidate | steps | total return | sharpe | max drawdown | turnover |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in sorted(rows, key=lambda item: (item.period, -item.sharpe)):
            handle.write(
                "| "
                f"{row.period} | "
                f"{row.candidate} | "
                f"{row.steps} | "
                f"{row.total_return:.6f} | "
                f"{row.sharpe:.6f} | "
                f"{row.max_drawdown:.6f} | "
                f"{row.mean_daily_turnover:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A deployable carry candidate should not depend on one narrow calendar "
            "slice. If the edge is concentrated in one historical period, the next work should "
            "treat it as a current dislocation monitor rather than a stable historical "
            "strategy.\n"
        )
    return output_path


def _filter_spot_rows(
    rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    *,
    start_date: str,
    end_date: str,
) -> dict[str, tuple[SpotDay, ...]]:
    return {
        symbol: tuple(
            row
            for row in rows
            if start_date <= row.timestamp[:10] < end_date
        )
        for symbol, rows in rows_by_symbol.items()
    }


def _filter_perp_rows(
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    *,
    start_date: str,
    end_date: str,
) -> dict[str, tuple[MarketStructureDay, ...]]:
    return {
        symbol: tuple(
            row
            for row in rows
            if start_date <= row.timestamp[:10] < end_date
        )
        for symbol, rows in rows_by_symbol.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dataset-dir", type=Path, default=SPOT_DATASET_DIR)
    parser.add_argument("--perp-dataset-dir", type=Path, default=PERP_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--paired-leg-cost-rate", type=float, default=0.0004)
    parser.add_argument("--capital-per-notional", type=float, default=2.0)
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_period_audit.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_period_audit.md",
    )
    args = parser.parse_args()

    spot_rows_by_symbol = load_spot_days(
        dataset_dir=args.spot_dataset_dir,
        symbols=tuple(args.symbols),
    )
    perp_rows_by_symbol = load_market_structure_days(
        dataset_dir=args.perp_dataset_dir,
        symbols=tuple(args.symbols),
    )
    rows = build_period_audit(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=args.min_funding_rate_sum,
        paired_leg_cost_rate=args.paired_leg_cost_rate,
        capital_per_notional=args.capital_per_notional,
    )
    write_period_audit_csv(rows, output_path=args.csv_output_path)
    write_period_audit_md(rows, output_path=args.md_output_path)
    for row in sorted(rows, key=lambda item: (item.period, -item.sharpe)):
        print(
            row.period,
            row.candidate,
            f"total={row.total_return:.6f}",
            f"sharpe={row.sharpe:.6f}",
            f"drawdown={row.max_drawdown:.6f}",
        )


if __name__ == "__main__":
    main()
