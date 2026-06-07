from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    load_market_structure_days,
)
from strategies.crypto_market_structure.predictive_screen import (
    build_prediction_rows,
    _walk_forward_predictions,
)


@dataclass(frozen=True)
class ExposureAuditRow:
    symbol: str
    mean_weight: float
    held_days: int
    gross_contribution: float
    mean_return_when_held: float


def audit_predictive_exposure(
    *,
    top_n: int,
    rebalance_days: int,
    min_train_days: int,
    ridge_penalty: float,
    dataset_dir: Path = LOCAL_DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> tuple[ExposureAuditRow, ...]:
    rows_by_symbol = load_market_structure_days(dataset_dir=dataset_dir, symbols=symbols)
    rows = build_prediction_rows(rows_by_symbol)
    timestamps = sorted({row.timestamp for row in rows})
    rows_by_timestamp = {
        timestamp: tuple(row for row in rows if row.timestamp == timestamp)
        for timestamp in timestamps
    }
    predictions_by_timestamp = _walk_forward_predictions(
        rows_by_timestamp,
        timestamps=timestamps,
        min_train_days=min_train_days,
        ridge_penalty=ridge_penalty,
    )
    exposure_by_symbol: dict[str, float] = defaultdict(float)
    held_days_by_symbol: dict[str, int] = defaultdict(int)
    contribution_by_symbol: dict[str, float] = defaultdict(float)
    held_returns_by_symbol: dict[str, list[float]] = defaultdict(list)
    target_weights: dict[str, float] = {}
    prediction_timestamps = sorted(predictions_by_timestamp)
    for index, timestamp in enumerate(prediction_timestamps):
        predictions = predictions_by_timestamp[timestamp]
        if index % rebalance_days == 0:
            positive_predictions = [
                (row, prediction)
                for row, prediction in predictions
                if prediction > 0.0
            ]
            selected = tuple(
                row.symbol
                for row, prediction in sorted(
                    positive_predictions,
                    key=lambda item: item[1],
                    reverse=True,
                )[:top_n]
            )
            target_weights = (
                {symbol: 1.0 / len(selected) for symbol in selected}
                if selected
                else {}
            )
        returns_by_symbol = {row.symbol: row.next_return for row, _ in predictions}
        for symbol, weight in target_weights.items():
            symbol_return = returns_by_symbol.get(symbol, 0.0)
            exposure_by_symbol[symbol] += weight
            held_days_by_symbol[symbol] += 1
            contribution_by_symbol[symbol] += weight * symbol_return
            held_returns_by_symbol[symbol].append(symbol_return)
    denominator = len(prediction_timestamps)
    return tuple(
        ExposureAuditRow(
            symbol=symbol,
            mean_weight=exposure_by_symbol[symbol] / denominator if denominator else 0.0,
            held_days=held_days_by_symbol[symbol],
            gross_contribution=contribution_by_symbol[symbol],
            mean_return_when_held=(
                mean(held_returns_by_symbol[symbol])
                if held_returns_by_symbol[symbol]
                else 0.0
            ),
        )
        for symbol in sorted(exposure_by_symbol, key=exposure_by_symbol.get, reverse=True)
    )


def write_exposure_audit(
    rows: tuple[ExposureAuditRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "symbol",
                "mean_weight",
                "held_days",
                "gross_contribution",
                "mean_return_when_held",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    f"{row.mean_weight:.10f}",
                    row.held_days,
                    f"{row.gross_contribution:.10f}",
                    f"{row.mean_return_when_held:.10f}",
                )
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--rebalance-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "predictive_exposure_audit.csv",
    )
    args = parser.parse_args()

    rows = audit_predictive_exposure(
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        min_train_days=args.min_train_days,
        ridge_penalty=args.ridge_penalty,
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    write_exposure_audit(rows, output_path=args.output_path)
    for row in rows:
        print(
            row.symbol,
            f"{row.mean_weight:.6f}",
            row.held_days,
            f"{row.gross_contribution:.6f}",
            f"{row.mean_return_when_held:.6f}",
        )


if __name__ == "__main__":
    main()
