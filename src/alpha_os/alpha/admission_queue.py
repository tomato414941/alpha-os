"""Compatibility wrapper for legacy admission queue helpers."""

import time

from alpha_os.config import asset_data_dir
from alpha_os.legacy import admission_queue as _legacy
from alpha_os.legacy.admission_queue import (
    CandidateSeed,
    PendingCandidatePruneStats,
    adopt_candidate,
    count_pending_candidates,
    ensure_candidate_source_metadata,
    fetch_pending_candidates,
    gc_old_candidate_results,
    mark_candidates_validating,
    queue_candidate_expressions,
    queue_candidates,
    reject_candidate,
    reset_candidates_to_pending,
)


def prune_stale_pending_candidates(
    asset: str,
    *,
    max_age_days: int,
    dry_run: bool = False,
) -> PendingCandidatePruneStats:
    _legacy.asset_data_dir = asset_data_dir
    _legacy.time = time
    return _legacy.prune_stale_pending_candidates(
        asset,
        max_age_days=max_age_days,
        dry_run=dry_run,
    )

__all__ = [
    "CandidateSeed",
    "PendingCandidatePruneStats",
    "adopt_candidate",
    "count_pending_candidates",
    "ensure_candidate_source_metadata",
    "fetch_pending_candidates",
    "gc_old_candidate_results",
    "mark_candidates_validating",
    "prune_stale_pending_candidates",
    "queue_candidate_expressions",
    "queue_candidates",
    "reject_candidate",
    "reset_candidates_to_pending",
]
