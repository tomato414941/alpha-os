"""Pure planning helpers for legacy deployment and prune experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hypotheses.allocation_policy import (
    apply_ranked_feature_usage_cap,
    apply_ranked_replacement_policy,
    dedupe_ranked_ids_by_semantic_key,
    dedupe_ranked_ids_by_signal_similarity,
    resolve_ranked_current_ids,
    seed_ranked_selection,
)
from ..hypotheses.identity import expression_feature_names, expression_semantic_key
from ..hypotheses.quality import QualityEstimate
from ..legacy.registry_types import AlphaRecord, AlphaState


@dataclass(frozen=True)
class RankedDeployedAlpha:
    alpha_id: str
    prior_quality: float
    blended_quality: float
    confidence: float
    live_quality: float
    n_observations: int

    @property
    def score(self) -> float:
        return self.blended_quality

    @property
    def rank_key(self) -> tuple[float, float, float]:
        return (
            self.blended_quality,
            self.confidence,
            self.prior_quality,
        )

    def to_metadata(self) -> dict[str, float | int]:
        return {
            "prior_quality": self.prior_quality,
            "blended_quality": self.blended_quality,
            "confidence": self.confidence,
            "live_quality": self.live_quality,
            "n_observations": self.n_observations,
        }


@dataclass(frozen=True)
class DeployedAlphaPlan:
    active_count: int
    current_count: int
    deployed_count: int
    replacement_count: int
    kept_ids: list[str]
    added_ids: list[str]
    dropped_ids: list[str]
    skipped_semantic_duplicate_ids: list[str]
    skipped_signal_duplicate_ids: list[str]
    skipped_feature_cap_ids: list[str]
    selected: list[RankedDeployedAlpha]

    @property
    def selected_ids(self) -> list[str]:
        return [item.alpha_id for item in self.selected]

    @property
    def selected_scores(self) -> dict[str, float]:
        return {item.alpha_id: item.score for item in self.selected}

    @property
    def selected_metadata(self) -> dict[str, dict[str, float | int]]:
        return {item.alpha_id: item.to_metadata() for item in self.selected}

    @property
    def skipped_duplicate_ids(self) -> list[str]:
        return (
            self.skipped_semantic_duplicate_ids
            + self.skipped_signal_duplicate_ids
            + self.skipped_feature_cap_ids
        )


@dataclass(frozen=True)
class RegistryActivePrunePlan:
    active_count: int
    current_deployed_count: int
    kept_count: int
    demoted_count: int
    touched_deployed_count: int
    kept_ids: list[str]
    demoted_ids: list[str]
    skipped_semantic_duplicate_ids: list[str]
    skipped_signal_duplicate_ids: list[str]


@dataclass(frozen=True)
class RankedActiveAlphaInputs:
    active_records: list[AlphaRecord]
    ranked: list[RankedDeployedAlpha]
    semantic_key_by_id: dict[str, str]
    feature_names_by_id: dict[str, set[str]]


def plan_deployed_alphas(
    records: list[AlphaRecord],
    current_ids: list[str],
    estimate_for,
    *,
    max_alphas: int,
    max_replacements: int,
    promotion_margin: float,
    metric: str,
    signal_by_id: dict[str, np.ndarray] | None = None,
    signal_similarity_max: float = 1.0,
    max_feature_occurrences: int = 0,
) -> DeployedAlphaPlan:
    prepared = _prepare_ranked_active_alphas(
        records,
        estimate_for=estimate_for,
        metric=metric,
    )
    ranked = prepared.ranked

    if max_alphas <= 0 or not ranked:
        return _empty_deployed_alpha_plan(active_count=len(prepared.active_records))

    ranked_ids = [item.alpha_id for item in ranked]
    ranked_ids, skipped_semantic_duplicate_ids = dedupe_ranked_ids_by_semantic_key(
        ranked_ids,
        semantic_key_by_id=prepared.semantic_key_by_id,
    )
    ranked_ids, skipped_signal_duplicate_ids = dedupe_ranked_ids_by_signal_similarity(
        ranked_ids,
        signal_by_id=signal_by_id or {},
        similarity_max=signal_similarity_max,
    )
    ranked_ids, skipped_feature_cap_ids = apply_ranked_feature_usage_cap(
        ranked_ids,
        feature_names_by_id=prepared.feature_names_by_id,
        max_occurrences=max_feature_occurrences,
        min_keep=max_alphas,
    )
    ranked_by_id = {item.alpha_id: item for item in ranked}
    current = resolve_ranked_current_ids(current_ids, ranked_ids)
    current_set = set(current)
    selected_ids = seed_ranked_selection(
        current_ids=current,
        ranked_ids=ranked_ids,
        max_selected=max_alphas,
    )
    remaining_ranked = [
        item_id for item_id in ranked_ids if item_id not in set(selected_ids)
    ]
    selected_ids, replacements = apply_ranked_replacement_policy(
        selected_ids=selected_ids,
        current_ids=current,
        remaining_ids=remaining_ranked,
        rank_key_by_id={item_id: ranked_by_id[item_id].rank_key for item_id in ranked_ids},
        score_by_id={item_id: ranked_by_id[item_id].score for item_id in ranked_ids},
        max_replacements=max_replacements,
        promotion_margin=promotion_margin,
    )

    selected = sorted(
        [ranked_by_id[alpha_id] for alpha_id in selected_ids],
        key=lambda item: item.rank_key,
        reverse=True,
    )
    selected_set = {item.alpha_id for item in selected}
    kept_ids = [item.alpha_id for item in selected if item.alpha_id in current_set]
    added_ids = [item.alpha_id for item in selected if item.alpha_id not in current_set]
    dropped_ids = [alpha_id for alpha_id in current if alpha_id not in selected_set]

    return DeployedAlphaPlan(
        active_count=len(prepared.active_records),
        current_count=len(current),
        deployed_count=len(selected),
        replacement_count=replacements,
        kept_ids=kept_ids,
        added_ids=added_ids,
        dropped_ids=dropped_ids,
        skipped_semantic_duplicate_ids=skipped_semantic_duplicate_ids,
        skipped_signal_duplicate_ids=skipped_signal_duplicate_ids,
        skipped_feature_cap_ids=skipped_feature_cap_ids,
        selected=selected,
    )


def plan_registry_active_prune(
    records: list[AlphaRecord],
    current_deployed_ids: list[str],
    estimate_for,
    *,
    metric: str,
    signal_by_id: dict[str, np.ndarray] | None = None,
    signal_similarity_max: float = 1.0,
) -> RegistryActivePrunePlan:
    active_records = [
        record for record in records
        if AlphaState.canonical(record.state) == AlphaState.ACTIVE
    ]
    ranked = [
        _ranked_alpha(record, estimate_for(record), metric)
        for record in active_records
    ]
    ranked.sort(key=lambda item: item.rank_key, reverse=True)
    semantic_key_by_id = {
        record.alpha_id: expression_semantic_key(record.expression)
        for record in active_records
    }

    ranked_ids = [item.alpha_id for item in ranked]
    ranked_ids, skipped_semantic_duplicate_ids = dedupe_ranked_ids_by_semantic_key(
        ranked_ids,
        semantic_key_by_id=semantic_key_by_id,
    )
    kept_ids, skipped_signal_duplicate_ids = dedupe_ranked_ids_by_signal_similarity(
        ranked_ids,
        signal_by_id=signal_by_id or {},
        similarity_max=signal_similarity_max,
    )
    kept_set = set(kept_ids)
    demoted_ids = [
        record.alpha_id
        for record in active_records
        if record.alpha_id not in kept_set
    ]
    touched_deployed_count = sum(
        1 for alpha_id in current_deployed_ids if alpha_id in set(demoted_ids)
    )
    return RegistryActivePrunePlan(
        active_count=len(active_records),
        current_deployed_count=len(current_deployed_ids),
        kept_count=len(kept_ids),
        demoted_count=len(demoted_ids),
        touched_deployed_count=touched_deployed_count,
        kept_ids=kept_ids,
        demoted_ids=demoted_ids,
        skipped_semantic_duplicate_ids=skipped_semantic_duplicate_ids,
        skipped_signal_duplicate_ids=skipped_signal_duplicate_ids,
    )


def _prepare_ranked_active_alphas(
    records: list[AlphaRecord],
    *,
    estimate_for,
    metric: str,
) -> RankedActiveAlphaInputs:
    active_records = [
        record for record in records
        if AlphaState.canonical(record.state) == AlphaState.ACTIVE
    ]
    ranked = [
        _ranked_alpha(record, estimate_for(record), metric)
        for record in active_records
    ]
    ranked.sort(key=lambda item: item.rank_key, reverse=True)
    return RankedActiveAlphaInputs(
        active_records=active_records,
        ranked=ranked,
        semantic_key_by_id={
            record.alpha_id: expression_semantic_key(record.expression)
            for record in active_records
        },
        feature_names_by_id={
            record.alpha_id: expression_feature_names(record.expression)
            for record in active_records
        },
    )


def _empty_deployed_alpha_plan(*, active_count: int) -> DeployedAlphaPlan:
    return DeployedAlphaPlan(
        active_count=active_count,
        current_count=0,
        deployed_count=0,
        replacement_count=0,
        kept_ids=[],
        added_ids=[],
        dropped_ids=[],
        skipped_semantic_duplicate_ids=[],
        skipped_signal_duplicate_ids=[],
        skipped_feature_cap_ids=[],
        selected=[],
    )


def _ranked_alpha(
    record: AlphaRecord,
    estimate: QualityEstimate,
    metric: str,
) -> RankedDeployedAlpha:
    return RankedDeployedAlpha(
        alpha_id=record.alpha_id,
        prior_quality=record.oos_fitness(metric),
        blended_quality=estimate.blended_quality,
        confidence=estimate.confidence,
        live_quality=estimate.live_quality,
        n_observations=estimate.n_observations,
    )
