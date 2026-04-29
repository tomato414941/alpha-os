from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd

from .evaluation_inputs import SubjectEvaluationInput
from .evaluation_generation import generate_evaluation_inputs_batch_from_feature_plane
from .evaluation_generation import prepare_feature_plane_from_frame
from .feature_plane import PriceFeaturePlane
from .signal_registry import SignalDefinition
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


@dataclass(frozen=True)
class EvaluationInputCacheKey:
    base_url: str
    asset: str
    observation_contract_key: str
    signal_id: str
    contract_multiplier: float | None = None
    contract_family: str | None = None
    quote_ccy: str | None = None
    collateral_ccy: str | None = None
    roll_rule: str | None = None


class EvaluationInputRepository:
    def __init__(self) -> None:
        self._inputs_by_key: dict[EvaluationInputCacheKey, tuple[SubjectEvaluationInput, ...]] = {}

    def load_inputs_for_range(
        self,
        *,
        plane: PriceFeaturePlane,
        definitions: list[SignalDefinition],
        start_date: str,
        end_date: str,
        observation_spec: ObservationSpec,
        asset: str,
        base_url: str,
        contract_multiplier: float | None = None,
        contract_family: str | None = None,
        quote_ccy: str | None = None,
        collateral_ccy: str | None = None,
        roll_rule: str | None = None,
    ) -> list[SubjectEvaluationInput]:
        if not definitions:
            return []
        selected_dates = {
            date for date in plane.dates if start_date <= date <= end_date
        }
        if not selected_dates:
            raise ValueError(f"no dates found in range: {start_date}..{end_date}")
        evaluation_inputs: list[SubjectEvaluationInput] = []
        for definition in definitions:
            cache_key = EvaluationInputCacheKey(
                base_url=base_url,
                asset=str(asset).strip().upper(),
                observation_contract_key=observation_contract_key(
                    observation_spec,
                    asset=asset,
                ),
                signal_id=definition.signal_id,
                contract_multiplier=(
                    None
                    if contract_multiplier is None
                    else float(contract_multiplier)
                ),
                contract_family=(
                    None if contract_family is None else str(contract_family)
                ),
                quote_ccy=None if quote_ccy is None else str(quote_ccy),
                collateral_ccy=(
                    None if collateral_ccy is None else str(collateral_ccy)
                ),
                roll_rule=None if roll_rule is None else str(roll_rule),
            )
            cached = self._inputs_by_key.get(cache_key)
            if cached is None:
                if definition.horizon_days is None:
                    raise ValueError(
                        f"signal definition must have fixed horizon_days: {definition.signal_id}"
                    )
                earliest_valid_index = int(definition.lookback)
                latest_valid_index = len(plane.dates) - int(definition.horizon_days) - 1
                if latest_valid_index < earliest_valid_index:
                    raise ValueError(
                        "feature plane does not contain enough valid dates for "
                        f"{definition.signal_id}"
                    )
                full_inputs = tuple(
                    generate_evaluation_inputs_batch_from_feature_plane(
                        plane=plane,
                        start_date=plane.dates[earliest_valid_index],
                        end_date=plane.dates[latest_valid_index],
                        definitions=[definition],
                        observation_spec=observation_spec,
                        contract_multiplier=contract_multiplier,
                        contract_family=contract_family,
                        quote_ccy=quote_ccy,
                        collateral_ccy=collateral_ccy,
                        roll_rule=roll_rule,
                    )
                )
                self._inputs_by_key[cache_key] = full_inputs
                cached = full_inputs
            evaluation_inputs.extend(
                item
                for item in cached
                if item.date in selected_dates
            )
        return evaluation_inputs
