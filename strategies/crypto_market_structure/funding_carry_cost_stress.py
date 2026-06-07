from __future__ import annotations

import argparse
import csv
from pathlib import Path

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    load_market_structure_days,
)
from strategies.crypto_market_structure.funding_carry import run_funding_carry_screen


DEFAULT_COST_RATES = (0.0004, 0.001, 0.002, 0.005)


def run_cost_stress(
    *,
    dataset_dir: Path = LOCAL_DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    cost_rates: tuple[float, ...] = DEFAULT_COST_RATES,
    min_funding_rate_sum: float = 0.0002,
) -> tuple[dict[str, object], ...]:
    rows_by_symbol = load_market_structure_days(dataset_dir=dataset_dir, symbols=symbols)
    rows: list[dict[str, object]] = []
    for cost_rate in cost_rates:
        for result in run_funding_carry_screen(
            rows_by_symbol,
            min_funding_rate_sum=min_funding_rate_sum,
            transaction_cost_rate=cost_rate,
        ):
            rows.append(
                {
                    "transaction_cost_rate": cost_rate,
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
                "transaction_cost_rate",
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
                    "transaction_cost_rate": row["transaction_cost_rate"],
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
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--cost-rates", nargs="+", type=float, default=list(DEFAULT_COST_RATES))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "funding_carry_cost_stress.csv",
    )
    args = parser.parse_args()

    rows = run_cost_stress(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
        cost_rates=tuple(args.cost_rates),
        min_funding_rate_sum=args.min_funding_rate_sum,
    )
    write_cost_stress(rows, output_path=args.output_path)
    for row in rows[:12]:
        print(
            row["transaction_cost_rate"],
            row["candidate"],
            row["steps"],
            f"{row['total_return']:.6f}",
            f"{row['sharpe']:.6f}",
            f"{row['max_drawdown']:.6f}",
            f"{row['mean_daily_turnover']:.6f}",
        )


if __name__ == "__main__":
    main()
