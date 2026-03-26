from __future__ import annotations

import numpy as np

from ..config import Config, SIGNAL_CACHE_DB
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse
from ..dsl.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..legacy.registry_types import AlphaRecord, AlphaState


def build_registry_signal_map(
    asset: str,
    config: Config,
    records: list[AlphaRecord],
) -> dict[str, np.ndarray]:
    active_records = [
        record for record in records
        if AlphaState.canonical(record.state) == AlphaState.ACTIVE
    ]
    if not active_records:
        return {}

    lookback = max(int(config.deployment.signal_similarity_lookback), 0)
    if lookback <= 1 or config.deployment.signal_similarity_max >= 1.0:
        return {}

    store = DataStore(SIGNAL_CACHE_DB, None)
    try:
        features = build_feature_list(asset)
        matrix = store.get_matrix(features)
    finally:
        store.close()
    if matrix.empty:
        return {}

    if lookback > 0:
        matrix = matrix.tail(lookback)
    data = {column: matrix[column].to_numpy(dtype=np.float64) for column in matrix.columns}
    n_days = len(matrix)
    signal_by_id: dict[str, np.ndarray] = {}
    for record in active_records:
        try:
            expr = parse(record.expression)
            signal = normalize_signal(evaluate_expression(expr, data, n_days))
        except (EvaluationError, Exception):
            continue
        signal_by_id[record.alpha_id] = np.asarray(signal, dtype=np.float64)
    return signal_by_id
