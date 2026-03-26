"""Helpers for enqueuing discovery-pool candidates into the legacy queue."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hypotheses.identity import expression_semantic_key
from .admission_queue import CandidateSeed
from .managed_alphas import ManagedAlphaStore


@dataclass(frozen=True)
class AdmissionQueueCandidate:
    expression: str
    fitness: float
    queue_score: float
    behavior: np.ndarray


def admission_queue_score(fitness: float) -> float:
    return float(fitness)


def existing_enqueue_semantic_keys(store: ManagedAlphaStore) -> set[str]:
    managed_keys = {
        expression_semantic_key(record.expression)
        for record in store.list_all()
        if record.state != "rejected"
    }
    queued_keys = {
        expression_semantic_key(expression)
        for expression in store.list_candidate_expressions(
            statuses=("pending", "validating", "adopted")
        )
    }
    return managed_keys | queued_keys


def dedupe_semantic_candidates(
    candidates: list[AdmissionQueueCandidate],
    *,
    existing_keys: set[str],
) -> list[AdmissionQueueCandidate]:
    unique_candidates: list[AdmissionQueueCandidate] = []
    for candidate in candidates:
        semantic_key = expression_semantic_key(candidate.expression)
        if semantic_key in existing_keys:
            continue
        existing_keys.add(semantic_key)
        unique_candidates.append(candidate)
    return unique_candidates


def candidate_seeds_for_enqueue(
    candidates: list[AdmissionQueueCandidate],
    *,
    asset: str,
) -> list[CandidateSeed]:
    return [
        CandidateSeed(
            expression=candidate.expression,
            source=f"alpha_generator_{asset.lower()}",
            fitness=candidate.fitness,
            behavior_json={
                "source": "alpha_generator",
                "asset": asset,
                "behavior": [float(x) for x in candidate.behavior.tolist()],
                "round": None,
                "enqueue": "manual_discovery_pool",
            },
        )
        for candidate in candidates
    ]
