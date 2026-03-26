from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.store import DataStore
from ..data.universe import price_signal
from ..forward.tracker import HypothesisObservationTracker
from .producer import collect_required_features, compute_hypothesis_prediction
from .store import HypothesisStore


@dataclass(frozen=True)
class ObservationBackfillSummary:
    n_hypotheses: int
    n_days: int
    n_records: int
    n_failures: int


def backfill_observation_returns(
    *,
    hypothesis_store: HypothesisStore,
    data_store: DataStore,
    forward_tracker: HypothesisObservationTracker,
    asset: str,
    lookback_days: int = 30,
) -> ObservationBackfillSummary:
    records = hypothesis_store.list_observation_active(asset=asset)
    if not records:
        return ObservationBackfillSummary(
            n_hypotheses=0,
            n_days=0,
            n_records=0,
            n_failures=0,
        )

    required_features = sorted(collect_required_features(records, [asset]))
    matrix = data_store.get_matrix(required_features)
    if len(matrix) < 2:
        return ObservationBackfillSummary(
            n_hypotheses=len(records),
            n_days=0,
            n_records=0,
            n_failures=0,
        )

    day_positions = range(max(1, len(matrix) - max(lookback_days, 1)), len(matrix))
    price_name = price_signal(asset)
    n_days = 0
    n_records = 0
    n_failures = 0

    for pos in day_positions:
        current_date = str(matrix.index[pos])
        prev_price = float(matrix.iloc[pos - 1][price_name])
        curr_price = float(matrix.iloc[pos][price_name])
        if not math.isfinite(prev_price) or not math.isfinite(curr_price) or abs(prev_price) <= 1e-12:
            continue

        price_return = (curr_price - prev_price) / prev_price
        history = matrix.iloc[:pos]
        data = {
            col: history[col].fillna(0.0).to_numpy(dtype=np.float64)
            for col in history.columns
        }
        wrote_day = False

        for record in records:
            try:
                signal_value = compute_hypothesis_prediction(record, data=data, asset=asset)
            except Exception:
                n_failures += 1
                continue
            if not math.isfinite(signal_value):
                n_failures += 1
                continue
            forward_tracker.record(
                record.hypothesis_id,
                current_date,
                float(signal_value),
                float(signal_value) * float(price_return),
            )
            n_records += 1
            wrote_day = True

        if wrote_day:
            n_days += 1

    return ObservationBackfillSummary(
        n_hypotheses=len(records),
        n_days=n_days,
        n_records=n_records,
        n_failures=n_failures,
    )
