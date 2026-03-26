from __future__ import annotations

import logging

from ..data.universe import required_raw_signals
from ..dsl import collect_feature_names, parse, temporal_expression_issues

logger = logging.getLogger(__name__)


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
