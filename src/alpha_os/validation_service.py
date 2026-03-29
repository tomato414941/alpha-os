from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from .evaluation_generation import (
    _daily_close_series,
    _load_price_frame_from_signal_noise,
    generate_evaluation_inputs_from_frame,
)
from .hypothesis_registry import get_hypothesis_definition
from .store import EvaluationStore
from .targets import get_target_definition
from .validation_engine import (
    ValidationTargetBundle,
    compute_validation_hypothesis_metrics,
    compute_validation_meta_metrics,
    slice_validation_bundle,
)
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


def _global_date_bounds(spec: ValidationSpec) -> tuple[str, str]:
    start_date = min(item.start_date for item in spec.date_ranges)
    end_date = max(item.end_date for item in spec.date_ranges)
    return start_date, end_date


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


def _build_validation_bundle_for_target(
    *,
    spec: ValidationSpec,
    target_id: str,
    signal_frames: dict[str, object],
) -> ValidationTargetBundle | None:
    global_start_date, global_end_date = _global_date_bounds(spec)
    observations: pd.Series | None = None
    predictions_by_hypothesis: dict[str, pd.Series] = {}
    for base_hypothesis_id in spec.hypothesis_ids:
        variant_definition = _build_target_variant_definition(
            hypothesis_id=base_hypothesis_id,
            target_id=target_id,
        )
        frame = signal_frames.get(variant_definition.signal_name)
        if frame is None:
            frame = _load_price_frame_from_signal_noise(
                base_url=spec.base_url,
                signal_name=variant_definition.signal_name,
            )
            signal_frames[variant_definition.signal_name] = frame
        clipped_range = _clipped_date_range(
            frame=frame,
            start_date=global_start_date,
            end_date=global_end_date,
            lookback=variant_definition.lookback,
            horizon_days=variant_definition.horizon_days or 0,
        )
        if clipped_range is None:
            continue
        evaluation_inputs = generate_evaluation_inputs_from_frame(
            frame=frame,
            start_date=clipped_range[0],
            end_date=clipped_range[1],
            hypothesis_id=base_hypothesis_id,
            signal_name=variant_definition.signal_name,
            definition=variant_definition,
            asset=spec.asset,
            target_id=target_id,
        )
        if not evaluation_inputs:
            continue
        prediction_series = pd.Series(
            {item.date: item.prediction for item in evaluation_inputs},
            dtype=float,
        ).sort_index()
        observation_series = pd.Series(
            {item.date: item.observation for item in evaluation_inputs},
            dtype=float,
        ).sort_index()
        if observations is None:
            observations = observation_series
        else:
            overlap = observations.index.intersection(observation_series.index)
            if not overlap.empty and not observations.loc[overlap].equals(
                observation_series.loc[overlap]
            ):
                raise ValueError(
                    f"inconsistent observations for target_id={target_id} "
                    f"between hypotheses on overlapping dates"
                )
            observations = pd.concat([observations, observation_series]).groupby(level=0).first()
            observations = observations.sort_index()
        predictions_by_hypothesis[base_hypothesis_id] = prediction_series
    if observations is None or not predictions_by_hypothesis:
        return None
    return ValidationTargetBundle(
        target_id=target_id,
        observations=observations,
        predictions_by_hypothesis=predictions_by_hypothesis,
    )


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
    bundles_by_target: dict[str, ValidationTargetBundle] = {}

    for target_id in spec.target_ids:
        bundle = _build_validation_bundle_for_target(
            spec=spec,
            target_id=target_id,
            signal_frames=signal_frames,
        )
        if bundle is not None:
            bundles_by_target[target_id] = bundle

    for date_range in spec.date_ranges:
        for target_id, bundle in sorted(bundles_by_target.items()):
            sliced_bundle = slice_validation_bundle(
                bundle,
                start_date=date_range.start_date,
                end_date=date_range.end_date,
            )
            if sliced_bundle.observations.empty:
                continue
            for window_size in spec.metric_windows:
                for metric in compute_validation_hypothesis_metrics(
                    sliced_bundle,
                    window_size=window_size,
                ):
                    hypothesis_results.append(
                        {
                            "run_id": run_id,
                            "date_range_label": date_range.label,
                            "start_date": date_range.start_date,
                            "end_date": date_range.end_date,
                            "target_id": target_id,
                            "hypothesis_id": metric.hypothesis_id,
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
                for meta_metric in compute_validation_meta_metrics(
                    sliced_bundle,
                    aggregation_kinds=spec.aggregation_kinds,
                    window_size=window_size,
                ):
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
