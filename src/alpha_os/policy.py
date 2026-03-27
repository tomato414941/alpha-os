from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CycleUpdate:
    signed_edge: float
    absolute_error: float
    quality_before: float
    quality_after: float
    quality_delta: float
    allocation_trust_before: float
    allocation_trust_after: float
    allocation_trust_delta: float


def update_quality(*, quality_before: float, prediction_value: float, observation_value: float) -> float:
    signed_edge = float(prediction_value) * float(observation_value)
    return (0.8 * float(quality_before)) + (0.2 * signed_edge)


def update_allocation_trust(*, trust_before: float, quality_after: float) -> float:
    return max(0.0, (0.7 * float(trust_before)) + (0.3 * max(float(quality_after), 0.0)))


def build_cycle_update(
    *,
    quality_before: float,
    allocation_trust_before: float,
    prediction_value: float,
    observation_value: float,
) -> CycleUpdate:
    signed_edge = float(prediction_value) * float(observation_value)
    absolute_error = abs(float(prediction_value) - float(observation_value))
    quality_after = update_quality(
        quality_before=quality_before,
        prediction_value=prediction_value,
        observation_value=observation_value,
    )
    allocation_trust_after = update_allocation_trust(
        trust_before=allocation_trust_before,
        quality_after=quality_after,
    )
    return CycleUpdate(
        signed_edge=signed_edge,
        absolute_error=absolute_error,
        quality_before=float(quality_before),
        quality_after=quality_after,
        quality_delta=quality_after - float(quality_before),
        allocation_trust_before=float(allocation_trust_before),
        allocation_trust_after=allocation_trust_after,
        allocation_trust_delta=allocation_trust_after - float(allocation_trust_before),
    )
