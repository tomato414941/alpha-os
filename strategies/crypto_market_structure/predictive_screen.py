from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.daily_close.metrics import summarize_backtest


FEATURE_NAMES = (
    "return_1d",
    "return_3d",
    "return_7d",
    "funding_rate_sum",
    "premium_close",
    "taker_buy_imbalance",
    "volume_ratio_20d",
)


@dataclass(frozen=True)
class PredictionRow:
    timestamp: str
    symbol: str
    features: tuple[float, ...]
    next_return: float


@dataclass(frozen=True)
class PredictiveScreenSummary:
    candidate: str
    predictions: int
    mean_daily_rank_ic: float
    positive_prediction_hit_rate: float
    total_return: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


def build_prediction_rows(
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
) -> tuple[PredictionRow, ...]:
    rows: list[PredictionRow] = []
    for symbol, symbol_rows in rows_by_symbol.items():
        for index in range(20, len(symbol_rows) - 1):
            current = symbol_rows[index]
            next_row = symbol_rows[index + 1]
            if current.close <= 0.0 or current.volume <= 0.0:
                continue
            features = (
                _return(symbol_rows, index, 1),
                _return(symbol_rows, index, 3),
                _return(symbol_rows, index, 7),
                current.funding_rate_sum,
                current.premium_close,
                (current.taker_buy_volume / current.volume) - 0.5,
                _volume_ratio(symbol_rows, index, 20),
            )
            rows.append(
                PredictionRow(
                    timestamp=current.timestamp,
                    symbol=symbol,
                    features=features,
                    next_return=(next_row.close / current.close) - 1.0,
                )
            )
    return tuple(rows)


def run_predictive_screen(
    rows: tuple[PredictionRow, ...],
    *,
    min_train_days: int = 180,
    ridge_penalty: float = 10.0,
) -> tuple[PredictiveScreenSummary, ...]:
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
    summaries = []
    for top_n in (1, 2, 3):
        for rebalance_days in (1, 7):
            summaries.append(
                _summarize_candidate(
                    predictions_by_timestamp,
                    top_n=top_n,
                    rebalance_days=rebalance_days,
                )
            )
    return tuple(summaries)


def write_predictive_screen_summaries(
    summaries: tuple[PredictiveScreenSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "predictions",
                "mean_daily_rank_ic",
                "positive_prediction_hit_rate",
                "total_return",
                "sharpe",
                "max_drawdown",
                "mean_daily_turnover",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.candidate,
                    summary.predictions,
                    f"{summary.mean_daily_rank_ic:.10f}",
                    f"{summary.positive_prediction_hit_rate:.6f}",
                    f"{summary.total_return:.10f}",
                    f"{summary.sharpe:.10f}",
                    f"{summary.max_drawdown:.10f}",
                    f"{summary.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def _walk_forward_predictions(
    rows_by_timestamp: dict[str, tuple[PredictionRow, ...]],
    *,
    timestamps: list[str],
    min_train_days: int,
    ridge_penalty: float,
) -> dict[str, tuple[tuple[PredictionRow, float], ...]]:
    predictions_by_timestamp: dict[str, tuple[tuple[PredictionRow, float], ...]] = {}
    for index in range(min_train_days, len(timestamps)):
        train_rows = tuple(
            row
            for timestamp in timestamps[:index]
            for row in rows_by_timestamp[timestamp]
        )
        test_rows = rows_by_timestamp[timestamps[index]]
        if len(train_rows) <= len(FEATURE_NAMES) or not test_rows:
            continue
        coefficients, feature_mean, feature_std = _fit_ridge(
            train_rows,
            ridge_penalty=ridge_penalty,
        )
        predictions_by_timestamp[timestamps[index]] = tuple(
            (row, _predict(row, coefficients, feature_mean, feature_std))
            for row in test_rows
        )
    return predictions_by_timestamp


def _fit_ridge(
    rows: tuple[PredictionRow, ...],
    *,
    ridge_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array([row.features for row in rows], dtype=float)
    labels = np.array([row.next_return for row in rows], dtype=float)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    normalized = (features - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return coefficients, feature_mean, feature_std


def _predict(
    row: PredictionRow,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> float:
    normalized = (np.array(row.features, dtype=float) - feature_mean) / feature_std
    design_row = np.insert(normalized, 0, 1.0)
    return float(design_row @ coefficients)


def _summarize_candidate(
    predictions_by_timestamp: dict[str, tuple[tuple[PredictionRow, float], ...]],
    *,
    top_n: int,
    rebalance_days: int,
) -> PredictiveScreenSummary:
    rewards: list[float] = []
    equities: list[float] = []
    transaction_costs: list[float] = []
    rank_ics: list[float] = []
    positive_prediction_hits: list[float] = []
    equity = 1.0
    current_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    timestamps = sorted(predictions_by_timestamp)
    for index, timestamp in enumerate(timestamps):
        predictions = predictions_by_timestamp[timestamp]
        if len(predictions) >= 2:
            rank_ics.append(_correlation(
                tuple(prediction for _, prediction in predictions),
                tuple(row.next_return for row, _ in predictions),
            ))
        positive_predictions = [
            (row, prediction)
            for row, prediction in predictions
            if prediction > 0.0
        ]
        positive_prediction_hits.extend(
            1.0 if row.next_return > 0.0 else 0.0
            for row, _ in positive_predictions
        )
        if index % rebalance_days == 0:
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
        reward = sum(
            target_weights.get(row.symbol, 0.0) * row.next_return
            for row, _ in predictions
        )
        transaction_cost = _turnover(current_weights, target_weights) * 0.001
        net_reward = reward - transaction_cost
        equity *= 1.0 + net_reward
        rewards.append(net_reward)
        equities.append(equity)
        transaction_costs.append(transaction_cost)
        current_weights = dict(target_weights)
    summary = summarize_backtest(
        rewards=tuple(rewards),
        equities=tuple(equities),
        transaction_costs=tuple(transaction_costs),
        transaction_cost_rate=0.001,
    )
    return PredictiveScreenSummary(
        candidate=f"ridge_top_{top_n}_{rebalance_days}d",
        predictions=sum(len(items) for items in predictions_by_timestamp.values()),
        mean_daily_rank_ic=mean(rank_ics) if rank_ics else 0.0,
        positive_prediction_hit_rate=(
            mean(positive_prediction_hits) if positive_prediction_hits else 0.0
        ),
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        mean_daily_turnover=summary.mean_daily_turnover,
    )


def _return(rows: tuple[MarketStructureDay, ...], index: int, lookback: int) -> float:
    prior = rows[index - lookback]
    current = rows[index]
    return (current.close / prior.close) - 1.0 if prior.close > 0.0 else 0.0


def _volume_ratio(
    rows: tuple[MarketStructureDay, ...],
    index: int,
    lookback: int,
) -> float:
    window = rows[index - lookback + 1 : index + 1]
    average_volume = mean(row.volume for row in window)
    return rows[index].volume / average_volume if average_volume > 0.0 else 1.0


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_values = np.array(left, dtype=float)
    right_values = np.array(right, dtype=float)
    if left_values.std() == 0.0 or right_values.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in current_weights.keys() | target_weights.keys()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "predictive_screen.csv",
    )
    args = parser.parse_args()

    rows_by_symbol = load_market_structure_days(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    rows = build_prediction_rows(rows_by_symbol)
    summaries = run_predictive_screen(
        rows,
        min_train_days=args.min_train_days,
        ridge_penalty=args.ridge_penalty,
    )
    write_predictive_screen_summaries(summaries, output_path=args.output_path)
    for summary in summaries:
        print(
            summary.candidate,
            summary.predictions,
            f"{summary.mean_daily_rank_ic:.6f}",
            f"{summary.positive_prediction_hit_rate:.3f}",
            f"{summary.total_return:.6f}",
            f"{summary.sharpe:.6f}",
            f"{summary.max_drawdown:.6f}",
            f"{summary.mean_daily_turnover:.6f}",
        )


if __name__ == "__main__":
    main()
