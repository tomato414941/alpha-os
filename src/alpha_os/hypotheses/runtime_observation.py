from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimeObservationUpdate:
    daily_return: float
    all_returns: list[float]
    recent_returns: list[float]
    quality_estimate: object


def record_runtime_observation(
    record,
    *,
    signal_yesterday: float,
    today_date: str,
    price_return: float,
    forward_tracker,
    monitor,
    degradation_window: int,
    estimate_quality,
    supports_short: bool,
) -> RuntimeObservationUpdate:
    daily_return = (
        signal_yesterday * price_return if np.isfinite(signal_yesterday) else 0.0
    )
    forward_tracker.record(
        record.hypothesis_id,
        today_date,
        signal_yesterday,
        daily_return,
    )
    all_returns = list(
        forward_tracker.get_hypothesis_realizable_returns(
            record.hypothesis_id,
            supports_short=supports_short,
        )
    )
    recent_returns = all_returns[-degradation_window:]
    monitor.clear(record.hypothesis_id)
    monitor.record_batch(record.hypothesis_id, recent_returns)
    quality_estimate = estimate_quality(record, all_returns)
    return RuntimeObservationUpdate(
        daily_return=float(daily_return),
        all_returns=all_returns,
        recent_returns=recent_returns,
        quality_estimate=quality_estimate,
    )
