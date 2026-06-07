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
    "return_14d",
    "return_30d",
    "volatility_20d",
    "funding_rate_sum",
    "funding_rate_mean",
    "abs_funding_rate_sum",
    "premium_close",
    "abs_premium_close",
    "taker_buy_imbalance",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "return_7d_x_taker_buy_imbalance",
    "funding_rate_sum_x_premium_close",
)

FEATURE_SETS = {
    "all": tuple(range(len(FEATURE_NAMES))),
    "momentum": (0, 1, 2, 3, 4, 5),
    "structure": (6, 7, 8, 9, 10, 11, 12, 13),
    "flow": (11, 12, 13, 14),
    "funding_premium": (6, 7, 8, 9, 10, 15),
}

MODEL_KINDS = (
    "return_ridge",
    "sign_ridge",
    "rank_ridge",
    "contrarian_return_ridge",
)


@dataclass(frozen=True)
class WidePredictionRow:
    timestamp: str
    symbol: str
    features: tuple[float, ...]
    next_return: float


@dataclass(frozen=True)
class ScreenResult:
    candidate: str
    predictions: int
    mean_daily_rank_ic: float
    positive_score_hit_rate: float
    total_return: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


def build_wide_prediction_rows(
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
) -> tuple[WidePredictionRow, ...]:
    rows: list[WidePredictionRow] = []
    for symbol, symbol_rows in rows_by_symbol.items():
        for index in range(30, len(symbol_rows) - 1):
            current = symbol_rows[index]
            next_row = symbol_rows[index + 1]
            if current.close <= 0.0 or current.volume <= 0.0:
                continue
            return_7d = _return(symbol_rows, index, 7)
            taker_buy_imbalance = (current.taker_buy_volume / current.volume) - 0.5
            rows.append(
                WidePredictionRow(
                    timestamp=current.timestamp,
                    symbol=symbol,
                    features=(
                        _return(symbol_rows, index, 1),
                        _return(symbol_rows, index, 3),
                        return_7d,
                        _return(symbol_rows, index, 14),
                        _return(symbol_rows, index, 30),
                        _volatility(symbol_rows, index, 20),
                        current.funding_rate_sum,
                        current.funding_rate_mean,
                        abs(current.funding_rate_sum),
                        current.premium_close,
                        abs(current.premium_close),
                        taker_buy_imbalance,
                        _volume_ratio(symbol_rows, index, 5),
                        _volume_ratio(symbol_rows, index, 20),
                        return_7d * taker_buy_imbalance,
                        current.funding_rate_sum * current.premium_close,
                    ),
                    next_return=(next_row.close / current.close) - 1.0,
                )
            )
    return tuple(rows)


def run_broad_model_screen(
    rows: tuple[WidePredictionRow, ...],
    *,
    min_train_days: int = 240,
    refit_days: int = 30,
    ridge_penalty: float = 10.0,
) -> tuple[ScreenResult, ...]:
    results: list[ScreenResult] = []
    for feature_set_name, feature_indices in FEATURE_SETS.items():
        projected_rows = _project_features(rows, feature_indices=feature_indices)
        for model_kind in MODEL_KINDS:
            predictions_by_timestamp = _walk_forward_predictions(
                projected_rows,
                model_kind=model_kind,
                min_train_days=min_train_days,
                refit_days=refit_days,
                ridge_penalty=ridge_penalty,
            )
            for top_n in (1, 2, 3):
                for rebalance_days in (1, 3, 7, 14):
                    results.append(
                        _summarize_policy(
                            predictions_by_timestamp,
                            candidate=(
                                f"{feature_set_name}_{model_kind}"
                                f"_top_{top_n}_{rebalance_days}d"
                            ),
                            top_n=top_n,
                            rebalance_days=rebalance_days,
                        )
                    )
    return tuple(sorted(results, key=lambda result: result.sharpe, reverse=True))


def write_screen_results(
    results: tuple[ScreenResult, ...],
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
                "positive_score_hit_rate",
                "total_return",
                "sharpe",
                "max_drawdown",
                "mean_daily_turnover",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.candidate,
                    result.predictions,
                    f"{result.mean_daily_rank_ic:.10f}",
                    f"{result.positive_score_hit_rate:.6f}",
                    f"{result.total_return:.10f}",
                    f"{result.sharpe:.10f}",
                    f"{result.max_drawdown:.10f}",
                    f"{result.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def _project_features(
    rows: tuple[WidePredictionRow, ...],
    *,
    feature_indices: tuple[int, ...],
) -> tuple[WidePredictionRow, ...]:
    return tuple(
        WidePredictionRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=tuple(row.features[index] for index in feature_indices),
            next_return=row.next_return,
        )
        for row in rows
    )


def _walk_forward_predictions(
    rows: tuple[WidePredictionRow, ...],
    *,
    model_kind: str,
    min_train_days: int,
    refit_days: int,
    ridge_penalty: float,
) -> dict[str, tuple[tuple[WidePredictionRow, float], ...]]:
    timestamps = sorted({row.timestamp for row in rows})
    rows_by_timestamp = _rows_by_timestamp(rows)
    predictions_by_timestamp: dict[str, tuple[tuple[WidePredictionRow, float], ...]] = {}
    coefficients: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    for index in range(min_train_days, len(timestamps)):
        if coefficients is None or (index - min_train_days) % refit_days == 0:
            train_rows = tuple(
                row
                for timestamp in timestamps[:index]
                for row in rows_by_timestamp[timestamp]
            )
            coefficients, feature_mean, feature_std = _fit_ridge(
                train_rows,
                model_kind=model_kind,
                ridge_penalty=ridge_penalty,
            )
        assert feature_mean is not None
        assert feature_std is not None
        test_rows = rows_by_timestamp[timestamps[index]]
        predictions_by_timestamp[timestamps[index]] = tuple(
            (
                row,
                _score(row, coefficients, feature_mean, feature_std, model_kind=model_kind),
            )
            for row in test_rows
        )
    return predictions_by_timestamp


def _rows_by_timestamp(
    rows: tuple[WidePredictionRow, ...],
) -> dict[str, tuple[WidePredictionRow, ...]]:
    grouped: dict[str, list[WidePredictionRow]] = {}
    for row in rows:
        grouped.setdefault(row.timestamp, []).append(row)
    return {timestamp: tuple(items) for timestamp, items in grouped.items()}


def _fit_ridge(
    rows: tuple[WidePredictionRow, ...],
    *,
    model_kind: str,
    ridge_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array([row.features for row in rows], dtype=float)
    labels = _labels(rows, model_kind=model_kind)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    normalized = (features - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return coefficients, feature_mean, feature_std


def _labels(rows: tuple[WidePredictionRow, ...], *, model_kind: str) -> np.ndarray:
    returns = np.array([row.next_return for row in rows], dtype=float)
    if model_kind == "sign_ridge":
        return np.where(returns > 0.0, 1.0, -1.0)
    if model_kind == "rank_ridge":
        return np.array(_cross_sectional_rank_labels(rows), dtype=float)
    return returns


def _cross_sectional_rank_labels(rows: tuple[WidePredictionRow, ...]) -> tuple[float, ...]:
    grouped = _rows_by_timestamp(rows)
    labels_by_key: dict[tuple[str, str], float] = {}
    for timestamp, timestamp_rows in grouped.items():
        sorted_rows = sorted(timestamp_rows, key=lambda row: row.next_return)
        if len(sorted_rows) == 1:
            labels_by_key[(timestamp, sorted_rows[0].symbol)] = 0.0
            continue
        for rank, row in enumerate(sorted_rows):
            labels_by_key[(timestamp, row.symbol)] = (rank / (len(sorted_rows) - 1)) - 0.5
    return tuple(labels_by_key[(row.timestamp, row.symbol)] for row in rows)


def _score(
    row: WidePredictionRow,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    model_kind: str,
) -> float:
    normalized = (np.array(row.features, dtype=float) - feature_mean) / feature_std
    score = float(np.insert(normalized, 0, 1.0) @ coefficients)
    return -score if model_kind == "contrarian_return_ridge" else score


def _summarize_policy(
    predictions_by_timestamp: dict[str, tuple[tuple[WidePredictionRow, float], ...]],
    *,
    candidate: str,
    top_n: int,
    rebalance_days: int,
) -> ScreenResult:
    rewards: list[float] = []
    equities: list[float] = []
    transaction_costs: list[float] = []
    rank_ics: list[float] = []
    positive_score_hits: list[float] = []
    equity = 1.0
    current_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    for index, timestamp in enumerate(sorted(predictions_by_timestamp)):
        predictions = predictions_by_timestamp[timestamp]
        if len(predictions) >= 2:
            rank_ics.append(
                _correlation(
                    tuple(score for _, score in predictions),
                    tuple(row.next_return for row, _ in predictions),
                )
            )
        positive_scores = [(row, score) for row, score in predictions if score > 0.0]
        positive_score_hits.extend(
            1.0 if row.next_return > 0.0 else 0.0 for row, _ in positive_scores
        )
        if index % rebalance_days == 0:
            selected = tuple(
                row.symbol
                for row, score in sorted(
                    positive_scores,
                    key=lambda item: item[1],
                    reverse=True,
                )[:top_n]
            )
            target_weights = (
                {symbol: 1.0 / len(selected) for symbol in selected}
                if selected
                else {}
            )
        gross_reward = sum(
            target_weights.get(row.symbol, 0.0) * row.next_return
            for row, _ in predictions
        )
        transaction_cost = _turnover(current_weights, target_weights) * 0.001
        reward = gross_reward - transaction_cost
        equity *= 1.0 + reward
        rewards.append(reward)
        equities.append(equity)
        transaction_costs.append(transaction_cost)
        current_weights = dict(target_weights)
    summary = summarize_backtest(
        rewards=tuple(rewards),
        equities=tuple(equities),
        transaction_costs=tuple(transaction_costs),
        transaction_cost_rate=0.001,
    )
    return ScreenResult(
        candidate=candidate,
        predictions=sum(len(items) for items in predictions_by_timestamp.values()),
        mean_daily_rank_ic=mean(rank_ics) if rank_ics else 0.0,
        positive_score_hit_rate=mean(positive_score_hits) if positive_score_hits else 0.0,
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        mean_daily_turnover=summary.mean_daily_turnover,
    )


def _return(rows: tuple[MarketStructureDay, ...], index: int, lookback: int) -> float:
    prior = rows[index - lookback]
    current = rows[index]
    return (current.close / prior.close) - 1.0 if prior.close > 0.0 else 0.0


def _volatility(rows: tuple[MarketStructureDay, ...], index: int, lookback: int) -> float:
    returns = tuple(_return(rows, step, 1) for step in range(index - lookback + 1, index + 1))
    return float(np.std(returns)) if returns else 0.0


def _volume_ratio(rows: tuple[MarketStructureDay, ...], index: int, lookback: int) -> float:
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


def _turnover(current_weights: dict[str, float], target_weights: dict[str, float]) -> float:
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in current_weights.keys() | target_weights.keys()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-train-days", type=int, default=240)
    parser.add_argument("--refit-days", type=int, default=30)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "broad_model_screen.csv",
    )
    args = parser.parse_args()

    rows_by_symbol = load_market_structure_days(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    rows = build_wide_prediction_rows(rows_by_symbol)
    results = run_broad_model_screen(
        rows,
        min_train_days=args.min_train_days,
        refit_days=args.refit_days,
        ridge_penalty=args.ridge_penalty,
    )
    write_screen_results(results, output_path=args.output_path)
    for result in results[:20]:
        print(
            result.candidate,
            result.predictions,
            f"{result.mean_daily_rank_ic:.6f}",
            f"{result.positive_score_hit_rate:.3f}",
            f"{result.total_return:.6f}",
            f"{result.sharpe:.6f}",
            f"{result.max_drawdown:.6f}",
            f"{result.mean_daily_turnover:.6f}",
        )


if __name__ == "__main__":
    main()
