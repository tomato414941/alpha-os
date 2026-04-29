from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_repositories import FeaturePlaneRepository
from .signal_registry import SignalDefinition
from .portfolio_decision import ObservationSpec, SubjectSet
from .feature_plane import PriceFeaturePlane


@dataclass(frozen=True)
class SubjectPlaneKey:
    asset: str
    observation_spec_id: str


def _cross_sectional_signal_frames(
    *,
    planes_by_key: dict[SubjectPlaneKey, PriceFeaturePlane],
    lookbacks: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    signal_frames: dict[int, pd.DataFrame] = {}
    for lookback in lookbacks:
        columns: dict[str, pd.Series] = {}
        for key, plane in planes_by_key.items():
            columns[key.asset] = plane.daily_returns.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        if columns:
            signal_frames[lookback] = pd.DataFrame(columns)
    return signal_frames


def _centered_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rank_frame = frame.rank(axis=1, method="first", ascending=True)
    counts = frame.notna().sum(axis=1)
    centered = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for column in frame.columns:
        rank_series = rank_frame[column]
        denominator = (counts - 1).replace(0, pd.NA)
        centered[column] = (((rank_series - 1.0) / denominator) * 2.0 - 1.0).where(
            denominator.notna(),
            0.0,
        )
    return centered.astype(float)


def build_subject_set_feature_planes(
    *,
    subject_set: SubjectSet,
    executable_definitions: list[SignalDefinition],
    base_url: str,
    feature_plane_repository: FeaturePlaneRepository | None = None,
) -> dict[SubjectPlaneKey, PriceFeaturePlane]:
    grouped_observation_specs: dict[SubjectPlaneKey, ObservationSpec] = {}
    for binding in subject_set.bindings:
        observation_spec = next(
            (
                spec
                for spec in subject_set.observation_specs
                if spec.observation_spec_id == binding.observation_spec_id
            ),
            None,
        )
        if observation_spec is None:
            raise ValueError(
                f"subject binding is missing observation spec: {binding.subject_id}"
            )
        grouped_observation_specs[
            SubjectPlaneKey(
                asset=binding.asset,
                observation_spec_id=observation_spec.observation_spec_id,
            )
        ] = observation_spec

    planes_by_key: dict[SubjectPlaneKey, PriceFeaturePlane] = {}
    repository = (
        feature_plane_repository
        if feature_plane_repository is not None
        else FeaturePlaneRepository()
    )
    for key, observation_spec in grouped_observation_specs.items():
        planes_by_key[key] = repository.load_signal_noise_feature_plane(
            observation_spec=observation_spec,
            asset=key.asset,
            base_url=base_url,
        )

    cross_sectional_lookbacks = tuple(
        sorted(
            {
                definition.lookback
                for definition in executable_definitions
                if definition.kind in {"relative_strength_rank", "peer_mean_reversion"}
            }
        )
    )
    if not cross_sectional_lookbacks:
        return planes_by_key

    signal_frames = _cross_sectional_signal_frames(
        planes_by_key=planes_by_key,
        lookbacks=cross_sectional_lookbacks,
    )
    for lookback, frame in signal_frames.items():
        centered_rank_frame = _centered_rank_frame(frame)
        for key, plane in planes_by_key.items():
            rank_series = centered_rank_frame.get(key.asset)
            if rank_series is None:
                continue
            plane.inject_signal_series(
                kind="relative_strength_rank",
                lookback=lookback,
                signal=rank_series,
            )
            plane.inject_signal_series(
                kind="peer_mean_reversion",
                lookback=lookback,
                signal=-rank_series,
            )
    return planes_by_key
