from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .evaluation_generation import (
    _daily_close_series,
    _load_price_frame_from_signal_noise,
    generate_evaluation_inputs_from_frame,
)
from .evaluation_runtime import apply_evaluation
from .hypothesis_registry import get_hypothesis_definition
from .meta_aggregation_service import refresh_target_meta_predictions
from .meta_metrics_service import refresh_target_meta_prediction_metrics
from .metrics_service import refresh_target_metrics
from .store import EvaluationStore
from .targets import get_target_definition
from .validation_spec import ValidationSpec


@dataclass(frozen=True)
class ValidationRunResult:
    run_id: str
    spec_json: str
    hypothesis_result_count: int
    meta_result_count: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ephemeral_hypothesis_id(base_hypothesis_id: str, target_id: str) -> str:
    return f"{base_hypothesis_id}@{target_id}"


def _build_target_variant_definition(*, hypothesis_id: str, target_id: str):
    base_definition = get_hypothesis_definition(hypothesis_id)
    target_definition = get_target_definition(target_id)
    return base_definition.__class__(
        hypothesis_id=_ephemeral_hypothesis_id(hypothesis_id, target_id),
        kind=base_definition.kind,
        signal_name=base_definition.signal_name,
        lookback=base_definition.lookback,
        target=target_definition,
        asset=base_definition.asset,
    )


def _clipped_date_range(
    *,
    frame,
    start_date: str,
    end_date: str,
    lookback: int,
    horizon_days: int,
) -> tuple[str, str] | None:
    daily_close = _daily_close_series(frame)
    dates = list(daily_close.index)
    if not dates:
        return None
    minimum_index = int(lookback)
    maximum_index = len(dates) - 1 - int(horizon_days)
    if maximum_index < minimum_index:
        return None
    minimum_date = dates[minimum_index]
    maximum_date = dates[maximum_index]
    clipped_start = max(start_date, minimum_date)
    clipped_end = min(end_date, maximum_date)
    if clipped_start > clipped_end:
        return None
    return clipped_start, clipped_end


def run_validation(
    store: EvaluationStore,
    *,
    spec: ValidationSpec,
    recorded_at: str | None = None,
) -> ValidationRunResult:
    store.ensure_schema()
    timestamp = recorded_at or _utc_now()
    run_id = timestamp
    spec_json = json.dumps(spec.to_document(), sort_keys=True)
    signal_frames: dict[str, object] = {}
    hypothesis_results: list[dict[str, object]] = []
    meta_results: list[dict[str, object]] = []

    for date_range in spec.date_ranges:
        with tempfile.TemporaryDirectory() as tmpdir:
            ephemeral_store = EvaluationStore(Path(tmpdir) / "validation_runtime.db")
            try:
                ephemeral_store.ensure_schema()
                touched_targets: set[str] = set()
                for base_hypothesis_id in spec.hypothesis_ids:
                    base_definition = get_hypothesis_definition(base_hypothesis_id)
                    frame = signal_frames.get(base_definition.signal_name)
                    if frame is None:
                        frame = _load_price_frame_from_signal_noise(
                            base_url=spec.base_url,
                            signal_name=base_definition.signal_name,
                        )
                        signal_frames[base_definition.signal_name] = frame
                    for target_id in spec.target_ids:
                        variant_definition = _build_target_variant_definition(
                            hypothesis_id=base_hypothesis_id,
                            target_id=target_id,
                        )
                        variant_hypothesis_id = variant_definition.hypothesis_id
                        ephemeral_store.register_hypothesis(
                            variant_hypothesis_id,
                            asset=spec.asset,
                            target_id=target_id,
                        )
                        clipped_range = _clipped_date_range(
                            frame=frame,
                            start_date=date_range.start_date,
                            end_date=date_range.end_date,
                            lookback=variant_definition.lookback,
                            horizon_days=variant_definition.horizon_days or 0,
                        )
                        if clipped_range is None:
                            continue
                        evaluation_inputs = generate_evaluation_inputs_from_frame(
                            frame=frame,
                            start_date=clipped_range[0],
                            end_date=clipped_range[1],
                            hypothesis_id=variant_hypothesis_id,
                            signal_name=variant_definition.signal_name,
                            definition=variant_definition,
                            asset=spec.asset,
                            target_id=target_id,
                        )
                        for evaluation_input in evaluation_inputs:
                            apply_evaluation(
                                ephemeral_store,
                                evaluation_id=(
                                    f"{evaluation_input.asset}:"
                                    f"{evaluation_input.target_id}:"
                                    f"{evaluation_input.date}"
                                ),
                                hypothesis_id=evaluation_input.hypothesis_id,
                                prediction_value=evaluation_input.prediction,
                                observation_value=evaluation_input.observation,
                                asset=evaluation_input.asset,
                                target_id=evaluation_input.target_id,
                                input_source="validation",
                                input_range_start=date_range.start_date,
                                input_range_end=date_range.end_date,
                                signal_name=variant_definition.signal_name,
                                refresh_metrics=False,
                            )
                        touched_targets.add(target_id)

                for target_id in sorted(touched_targets):
                    variant_hypothesis_ids = [
                        _ephemeral_hypothesis_id(base_hypothesis_id, target_id)
                        for base_hypothesis_id in spec.hypothesis_ids
                    ]
                    for window_size in spec.metric_windows:
                        refresh_target_metrics(
                            ephemeral_store,
                            asset=spec.asset,
                            target_id=target_id,
                            recorded_at=timestamp,
                            window_size=window_size,
                        )
                        refresh_target_meta_predictions(
                            ephemeral_store,
                            asset=spec.asset,
                            target_id=target_id,
                            recorded_at=timestamp,
                            window_size=window_size,
                            aggregation_kinds=spec.aggregation_kinds,
                        )
                        refresh_target_meta_prediction_metrics(
                            ephemeral_store,
                            asset=spec.asset,
                            target_id=target_id,
                            recorded_at=timestamp,
                            window_size=window_size,
                            aggregation_kinds=spec.aggregation_kinds,
                        )
                        for base_hypothesis_id, variant_hypothesis_id in zip(
                            spec.hypothesis_ids,
                            variant_hypothesis_ids,
                            strict=True,
                        ):
                            metric = ephemeral_store.get_hypothesis_metric(
                                variant_hypothesis_id
                            )
                            if metric is None:
                                continue
                            hypothesis_results.append(
                                {
                                    "run_id": run_id,
                                    "date_range_label": date_range.label,
                                    "start_date": date_range.start_date,
                                    "end_date": date_range.end_date,
                                    "target_id": target_id,
                                    "hypothesis_id": base_hypothesis_id,
                                    "window_size": window_size,
                                    "corr": metric.corr,
                                    "mmc": metric.mmc,
                                    "sample_count": metric.sample_count,
                                    "mmc_sample_count": metric.mmc_sample_count,
                                    "mmc_peer_count": metric.mmc_peer_count,
                                    "mmc_baseline_type": metric.mmc_baseline_type,
                                    "recorded_at": timestamp,
                                }
                            )
                        for meta_metric in ephemeral_store.list_meta_prediction_metrics(
                            asset=spec.asset,
                            target_id=target_id,
                        ):
                            if meta_metric.aggregation_kind not in spec.aggregation_kinds:
                                continue
                            if meta_metric.window_size != window_size:
                                continue
                            meta_results.append(
                                {
                                    "run_id": run_id,
                                    "date_range_label": date_range.label,
                                    "start_date": date_range.start_date,
                                    "end_date": date_range.end_date,
                                    "target_id": target_id,
                                    "aggregation_kind": meta_metric.aggregation_kind,
                                    "window_size": window_size,
                                    "corr": meta_metric.corr,
                                    "sample_count": meta_metric.sample_count,
                                    "recorded_at": timestamp,
                                }
                            )
            finally:
                ephemeral_store.close()

    store.create_validation_run(
        run_id=run_id,
        spec_json=spec_json,
        recorded_at=timestamp,
    )
    for item in hypothesis_results:
        store.upsert_validation_hypothesis_result(**item)
    for item in meta_results:
        store.upsert_validation_meta_result(**item)

    return ValidationRunResult(
        run_id=run_id,
        spec_json=spec_json,
        hypothesis_result_count=len(hypothesis_results),
        meta_result_count=len(meta_results),
    )
