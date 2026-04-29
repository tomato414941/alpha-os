from __future__ import annotations

from .compression import compress_screening_result


def build_compressed_belief_from_screening_result(
    *,
    screening_result,
    prediction_values_by_signal_id: dict[str, float],
    created_at: str,
    strategy_adaptation_state=None,
    adaptation_blend: float = 0.2,
):
    return compress_screening_result(
        signal_discovery_id=screening_result.signal_discovery_id,
        screening_result_id=screening_result.screening_result_id,
        survivors=screening_result.survivors,
        prediction_values_by_signal_id=prediction_values_by_signal_id,
        created_at=created_at,
        strategy_adaptation_state=strategy_adaptation_state,
        adaptation_blend=adaptation_blend,
    )
