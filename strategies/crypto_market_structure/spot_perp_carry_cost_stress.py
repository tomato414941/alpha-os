from __future__ import annotations

import argparse
import csv
from pathlib import Path

from strategies.crypto.data import LOCAL_DATASET_DIR as SPOT_DATASET_DIR
from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR as PERP_DATASET_DIR,
    load_market_structure_days,
)
from strategies.crypto_market_structure.spot_perp_carry import (
    load_spot_days,
    run_spot_perp_carry_screen,
)


DEFAULT_COST_RATES = (0.0004, 0.001, 0.002, 0.005)


def run_cost_stress(
    *,
    spot_dataset_dir: Path = SPOT_DATASET_DIR,
    perp_dataset_dir: Path = PERP_DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    cost_rates: tuple[float, ...] = DEFAULT_COST_RATES,
    min_funding_rate_sum: float = 0.0002,
    capital_per_notional: float = 2.0,
) -> tuple[dict[str, object], ...]:
    spot_rows_by_symbol = load_spot_days(dataset_dir=spot_dataset_dir, symbols=symbols)
    perp_rows_by_symbol = load_market_structure_days(
        dataset_dir=perp_dataset_dir,
        symbols=symbols,
    )
    rows: list[dict[str, object]] = []
    for cost_rate in cost_rates:
        for result in run_spot_perp_carry_screen(
            spot_rows_by_symbol=spot_rows_by_symbol,
            perp_rows_by_symbol=perp_rows_by_symbol,
            min_funding_rate_sum=min_funding_rate_sum,
            paired_leg_cost_rate=cost_rate,
            capital_per_notional=capital_per_notional,
        ):
            rows.append(
                {
                    "paired_leg_cost_rate": cost_rate,
                    "capital_per_notional": capital_per_notional,
                    "candidate": result.candidate,
                    "steps": result.steps,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "mean_daily_turnover": result.mean_daily_turnover,
                }
            )
    return tuple(rows)


def write_cost_stress(rows: tuple[dict[str, object], ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "paired_leg_cost_rate",
                "capital_per_notional",
                "candidate",
                "steps",
                "total_return",
                "sharpe",
                "max_drawdown",
                "mean_daily_turnover",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "paired_leg_cost_rate": row["paired_leg_cost_rate"],
                    "capital_per_notional": row["capital_per_notional"],
                    "candidate": row["candidate"],
                    "steps": row["steps"],
                    "total_return": f"{row['total_return']:.10f}",
                    "sharpe": f"{row['sharpe']:.10f}",
                    "max_drawdown": f"{row['max_drawdown']:.10f}",
                    "mean_daily_turnover": f"{row['mean_daily_turnover']:.10f}",
                }
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dataset-dir", type=Path, default=SPOT_DATASET_DIR)
    parser.add_argument("--perp-dataset-dir", type=Path, default=PERP_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--cost-rates", nargs="+", type=float, default=list(DEFAULT_COST_RATES))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--capital-per-notional", type=float, default=2.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "spot_perp_carry_cost_stress.csv",
    )
    args = parser.parse_args()

    rows = run_cost_stress(
        spot_dataset_dir=args.spot_dataset_dir,
        perp_dataset_dir=args.perp_dataset_dir,
        symbols=tuple(args.symbols),
        cost_rates=tuple(args.cost_rates),
        min_funding_rate_sum=args.min_funding_rate_sum,
        capital_per_notional=args.capital_per_notional,
    )
    write_cost_stress(rows, output_path=args.output_path)
    for row in rows[:12]:
        print(
            row["paired_leg_cost_rate"],
            row["candidate"],
            row["steps"],
            f"{row['total_return']:.6f}",
            f"{row['sharpe']:.6f}",
            f"{row['max_drawdown']:.6f}",
            f"{row['mean_daily_turnover']:.6f}",
        )


if __name__ == "__main__":
    main()
