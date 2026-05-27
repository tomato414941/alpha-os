from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd

from .feature_plane_builder import prepare_feature_plane_from_frame
from .feature_plane import PriceFeaturePlane
from .observation_adapters import load_observation_frame as load_observation_frame_from_adapter
from .observation_adapters import observation_contract_key
from .portfolio_decision import ObservationSpec
from .store import EvaluationStore


def load_observation_frame(
    observation_spec: ObservationSpec,
    *,
    asset: str,
    base_url: str,
    client=None,
) -> pd.DataFrame:
    del client
    return load_observation_frame_from_adapter(
        observation_spec,
        asset=asset,
        base_url=base_url,
    )


@dataclass(frozen=True)
class ObservationFrameCacheKey:
    base_url: str
    asset: str
    observation_contract_key: str

    @property
    def value(self) -> str:
        return f"{self.base_url}:{self.observation_contract_key}"


@dataclass(frozen=True)
class FeaturePlaneCacheKey:
    base_url: str
    asset: str
    observation_contract_key: str


class ObservationFrameRepository:
    def __init__(self, *, store: EvaluationStore | None = None) -> None:
        self._store = store
        self._frames_by_key: dict[ObservationFrameCacheKey, pd.DataFrame] = {}

    def load_signal_noise_frame(
        self,
        *,
        observation_spec: ObservationSpec,
        asset: str,
        base_url: str,
    ) -> pd.DataFrame:
        cache_key = ObservationFrameCacheKey(
            base_url=base_url,
            asset=str(asset).strip().upper(),
            observation_contract_key=observation_contract_key(
                observation_spec,
                asset=asset,
            ),
        )
        cached = self._frames_by_key.get(cache_key)
        if cached is not None:
            return cached.copy(deep=True)
        if self._store is not None:
            cached_json = self._store.get_observation_frame_cache(cache_key.value)
            if cached_json is not None:
                restored = pd.read_json(StringIO(cached_json), orient="records")
                self._frames_by_key[cache_key] = restored
                return restored.copy(deep=True)
        frame = load_observation_frame(
            observation_spec=observation_spec,
            asset=asset,
            base_url=base_url,
        )
        persisted = frame.copy(deep=True)
        self._frames_by_key[cache_key] = persisted
        if self._store is not None:
            self._store.upsert_observation_frame_cache(
                cache_key.value,
                frame_json=frame.to_json(orient="records", date_format="iso"),
            )
        return persisted.copy(deep=True)


class FeaturePlaneRepository:
    def __init__(
        self,
        *,
        observation_repository: ObservationFrameRepository | None = None,
    ) -> None:
        self._observation_repository = (
            observation_repository
            if observation_repository is not None
            else ObservationFrameRepository()
        )
        self._planes_by_key: dict[FeaturePlaneCacheKey, PriceFeaturePlane] = {}

    def load_signal_noise_feature_plane(
        self,
        *,
        observation_spec: ObservationSpec,
        asset: str,
        base_url: str,
    ) -> PriceFeaturePlane:
        cache_key = FeaturePlaneCacheKey(
            base_url=base_url,
            asset=str(asset).strip().upper(),
            observation_contract_key=observation_contract_key(
                observation_spec,
                asset=asset,
            ),
        )
        cached = self._planes_by_key.get(cache_key)
        if cached is not None:
            return cached.clone()
        frame = self._observation_repository.load_signal_noise_frame(
            observation_spec=observation_spec,
            asset=asset,
            base_url=base_url,
        )
        plane = prepare_feature_plane_from_frame(frame=frame)
        self._planes_by_key[cache_key] = plane
        return plane.clone()
