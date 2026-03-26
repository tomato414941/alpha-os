from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..data.universe import required_raw_signals
from ..dsl import collect_feature_names, parse, temporal_expression_issues
from ..dsl.evaluator import evaluate_expression, normalize_signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSignalEvaluation:
    signal_yesterday: float
    signal_series: np.ndarray
    used_prediction_store: bool


def prepare_runtime_inputs(
    records,
    *,
    price_signal: str,
    store_signals: dict[str, float],
) -> tuple[list[str], list[tuple], int]:
    runtime_signals = {price_signal}
    parsed_records: list[tuple] = []
    n_failed = 0

    for record in records:
        if record.hypothesis_id in store_signals:
            parsed_records.append((record, None))
            continue
        if not record.expression:
            n_failed += 1
            continue
        try:
            expr = parse(record.expression)
        except SyntaxError as exc:
            logger.warning("Failed to parse %s: %s", record.hypothesis_id, exc)
            n_failed += 1
            continue
        issues = temporal_expression_issues(expr)
        if issues:
            logger.warning(
                "Skipping structurally invalid %s: %s",
                record.hypothesis_id,
                issues[0],
            )
            n_failed += 1
            continue
        runtime_signals.update(required_raw_signals(collect_feature_names(expr)))
        parsed_records.append((record, expr))

    return sorted(runtime_signals), parsed_records, n_failed


def filter_runtime_records_by_available_features(
    parsed_records: list[tuple],
    *,
    available_features: set[str],
    store_signals: dict[str, float],
) -> tuple[list[tuple], int]:
    filtered_records: list[tuple] = []
    n_feature_filtered = 0

    for record, expr in parsed_records:
        if record.hypothesis_id in store_signals:
            filtered_records.append((record, expr))
            continue
        required = set(required_raw_signals(collect_feature_names(expr)))
        if not required.issubset(available_features):
            n_feature_filtered += 1
            continue
        filtered_records.append((record, expr))

    return filtered_records, n_feature_filtered


def prediction_history_array(
    prediction_store,
    signal_id: str,
    asset: str,
    *,
    n_days: int,
    fallback_value: float,
) -> np.ndarray:
    rows = prediction_store.read_signal_history(signal_id, asset, n_days=n_days)
    values = [float(value) for _date, value in reversed(rows)]
    if not values:
        values = [fallback_value]
    if len(values) < n_days:
        values = [values[0]] * (n_days - len(values)) + values
    return np.asarray(values[-n_days:], dtype=np.float64)


def load_prediction_signal_arrays(
    prediction_store,
    parsed_records: list[tuple],
    *,
    asset: str,
    store_signals: dict[str, float],
    n_days: int,
) -> dict[str, np.ndarray]:
    signal_arrays: dict[str, np.ndarray] = {}
    for record, _expr in parsed_records:
        if record.hypothesis_id not in store_signals:
            continue
        value = store_signals[record.hypothesis_id]
        signal_arrays[record.hypothesis_id] = prediction_history_array(
            prediction_store,
            record.hypothesis_id,
            asset,
            n_days=n_days,
            fallback_value=value,
        )
    return signal_arrays


def evaluate_runtime_signal(
    record,
    expr,
    *,
    data: dict[str, np.ndarray],
    n_days: int,
    store_signals: dict[str, float],
    store_signal_arrays: dict[str, np.ndarray],
) -> RuntimeSignalEvaluation | None:
    if record.hypothesis_id in store_signals:
        signal_yesterday = float(store_signals[record.hypothesis_id])
        signal_series = store_signal_arrays.get(record.hypothesis_id)
        if signal_series is None:
            signal_series = np.full(n_days, signal_yesterday, dtype=np.float64)
        return RuntimeSignalEvaluation(
            signal_yesterday=signal_yesterday,
            signal_series=signal_series,
            used_prediction_store=True,
        )

    if expr is None:
        return None

    signal = evaluate_expression(expr, data, n_days)
    signal_series = normalize_signal(signal)
    signal_yesterday = _finite_signal_tail(signal_series)
    return RuntimeSignalEvaluation(
        signal_yesterday=signal_yesterday,
        signal_series=signal_series,
        used_prediction_store=False,
    )


def _finite_signal_tail(signal_series: np.ndarray) -> float:
    if len(signal_series) < 2:
        return 0.0
    signal_yesterday = float(signal_series[-2])
    if np.isfinite(signal_yesterday):
        return signal_yesterday
    for offset in range(3, min(10, len(signal_series))):
        fallback = float(signal_series[-offset])
        if np.isfinite(fallback):
            return fallback
    return signal_yesterday
