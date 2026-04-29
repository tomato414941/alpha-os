from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_plane import PriceFeaturePlane
from .signal_compiler import compile_signal_families
from .signal_registry import SignalDefinition


@dataclass(frozen=True)
class SurvivorScreenPolicy:
    min_sample_count: int = 0
    min_abs_corr: float = 0.0
    max_family_survivors_per_subject: int | None = None


@dataclass(frozen=True)
class SurvivorScreenCandidate:
    signal_id: str
    family_id: str | None
    kind: str
    lookback: int
    score: float
    corr: float
    sample_count: int
    keep: bool
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class SurvivorScreenResult:
    selected_definitions: tuple[SignalDefinition, ...]
    candidates: tuple[SurvivorScreenCandidate, ...]


def survivor_screen_on_feature_plane(
    *,
    plane: PriceFeaturePlane,
    start_date: str,
    end_date: str,
    definitions: list[SignalDefinition],
    policy: SurvivorScreenPolicy,
    family_ids_by_signal_id: dict[str, str] | None = None,
) -> SurvivorScreenResult:
    if not definitions:
        return SurvivorScreenResult(selected_definitions=(), candidates=())
    selected_dates = [
        date for date in plane.dates if start_date <= date <= end_date
    ]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    compiled_families = compile_signal_families(definitions)
    definitions_by_signal_id = {
        definition.signal_id: definition for definition in definitions
    }
    resolved_family_ids = (
        family_ids_by_signal_id
        if family_ids_by_signal_id is not None
        else {}
    )

    candidates: list[SurvivorScreenCandidate] = []
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
                reasons: list[str] = []
                if sample_count < policy.min_sample_count:
                    reasons.append("survivor_insufficient_samples")
                elif score < policy.min_abs_corr:
                    reasons.append("survivor_weak_signal")
                candidates.append(
                    SurvivorScreenCandidate(
                        signal_id=definition.signal_id,
                        family_id=resolved_family_ids.get(definition.signal_id),
                        kind=definition.kind,
                        lookback=definition.lookback,
                        score=score,
                        corr=corr,
                        sample_count=sample_count,
                        keep=not reasons,
                        reasons=tuple(reasons),
                    )
                )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            item.family_id or item.kind or "-",
            -item.score,
            -item.sample_count,
            item.lookback,
            item.signal_id,
        ),
    )

    if policy.max_family_survivors_per_subject is not None:
        kept_per_family: dict[str, int] = {}
        reduced_candidates: list[SurvivorScreenCandidate] = []
        for candidate in ordered_candidates:
            if not candidate.keep:
                reduced_candidates.append(candidate)
                continue
            family_id = candidate.family_id or candidate.kind or "-"
            kept_count = kept_per_family.get(family_id, 0)
            if kept_count >= policy.max_family_survivors_per_subject:
                reduced_candidates.append(
                    SurvivorScreenCandidate(
                        signal_id=candidate.signal_id,
                        family_id=candidate.family_id,
                        kind=candidate.kind,
                        lookback=candidate.lookback,
                        score=candidate.score,
                        corr=candidate.corr,
                        sample_count=candidate.sample_count,
                        keep=False,
                        reasons=("survivor_family_cap",),
                    )
                )
                continue
            kept_per_family[family_id] = kept_count + 1
            reduced_candidates.append(candidate)
        ordered_candidates = reduced_candidates

    selected_signal_ids = [
        candidate.signal_id for candidate in ordered_candidates if candidate.keep
    ]
    selected_definitions = tuple(
        definitions_by_signal_id[signal_id]
        for signal_id in selected_signal_ids
        if signal_id in definitions_by_signal_id
    )
    return SurvivorScreenResult(
        selected_definitions=selected_definitions,
        candidates=tuple(ordered_candidates),
    )
