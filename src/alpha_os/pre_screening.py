from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_plane import PriceFeaturePlane
from .signal_compiler import compile_signal_families
from .signal_registry import SignalDefinition


@dataclass(frozen=True)
class CheapPreScreenPolicy:
    min_abs_corr: float = 0.0
    top_k_per_kind: int | None = None


@dataclass(frozen=True)
class CheapPreScreenCandidate:
    signal_id: str
    kind: str
    lookback: int
    score: float
    corr: float
    sample_count: int


@dataclass(frozen=True)
class CheapPreScreenResult:
    selected_definitions: tuple[SignalDefinition, ...]
    candidates: tuple[CheapPreScreenCandidate, ...]


def cheap_pre_screen_on_feature_plane(
    *,
    plane: PriceFeaturePlane,
    start_date: str,
    end_date: str,
    definitions: list[SignalDefinition],
    policy: CheapPreScreenPolicy,
) -> CheapPreScreenResult:
    if not definitions:
        return CheapPreScreenResult(selected_definitions=(), candidates=())
    compiled_families = compile_signal_families(definitions)
    selected_dates = [
        date for date in plane.dates if start_date <= date <= end_date
    ]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    candidates: list[CheapPreScreenCandidate] = []
    definitions_by_signal_id = {
        definition.signal_id: definition for definition in definitions
    }
    for family in compiled_families:
        signal_frame = plane.signal_frame(
            kind=family.kind,
            lookbacks=family.lookbacks,
        ).loc[selected_dates]
        observation_slice = plane.observation_series(
            horizon_days=family.horizon_days
        ).loc[selected_dates]
        for lookback, definitions_at_lookback in family.definitions_by_lookback.items():
            signal_slice = signal_frame[lookback]
            valid_mask = (~signal_slice.isna()) & (~observation_slice.isna())
            valid_signal = signal_slice[valid_mask]
            valid_observation = observation_slice[valid_mask]
            sample_count = int(valid_mask.sum())
            corr = 0.0
            if sample_count >= 2:
                corr_value = valid_signal.corr(valid_observation)
                corr = 0.0 if pd.isna(corr_value) else float(corr_value)
            score = abs(corr)
            for definition in definitions_at_lookback:
                candidates.append(
                    CheapPreScreenCandidate(
                        signal_id=definition.signal_id,
                        kind=definition.kind,
                        lookback=definition.lookback,
                        score=score,
                        corr=corr,
                        sample_count=sample_count,
                    )
                )
    selected_signal_ids: list[str] = []
    if policy.top_k_per_kind is None:
        selected_signal_ids = [
            candidate.signal_id
            for candidate in candidates
            if candidate.score >= policy.min_abs_corr
        ]
    else:
        grouped_candidates: dict[str, list[CheapPreScreenCandidate]] = {}
        for candidate in candidates:
            if candidate.score < policy.min_abs_corr:
                continue
            grouped_candidates.setdefault(candidate.kind, []).append(candidate)
        for kind_candidates in grouped_candidates.values():
            ordered = sorted(
                kind_candidates,
                key=lambda item: (
                    -item.score,
                    -item.sample_count,
                    item.lookback,
                    item.signal_id,
                ),
            )
            selected_signal_ids.extend(
                item.signal_id for item in ordered[: policy.top_k_per_kind]
            )
    selected_id_set = set(selected_signal_ids)
    selected_definitions = tuple(
        definitions_by_signal_id[signal_id]
        for signal_id in selected_signal_ids
        if (
            signal_id in selected_id_set
            and signal_id in definitions_by_signal_id
        )
    )
    return CheapPreScreenResult(
        selected_definitions=selected_definitions,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.kind,
                    -item.score,
                    item.lookback,
                    item.signal_id,
                ),
            )
        ),
    )
