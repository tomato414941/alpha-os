"""Compatibility wrapper for legacy imports."""

from alpha_os.hypotheses.state_lifecycle import (
    ACTIVE_STATE,
    CANDIDATE_STATE,
    DORMANT_STATE,
    REJECTED_STATE,
    ST_ACTIVE,
    ST_CANDIDATE,
    ST_DORMANT,
    LifecycleConfig,
    batch_live_transitions,
    batch_transitions,
    canonical_state,
    compute_transition,
    next_live_state,
    passes_candidate_gate,
)

__all__ = [
    "ACTIVE_STATE",
    "CANDIDATE_STATE",
    "DORMANT_STATE",
    "REJECTED_STATE",
    "ST_ACTIVE",
    "ST_CANDIDATE",
    "ST_DORMANT",
    "LifecycleConfig",
    "batch_live_transitions",
    "batch_transitions",
    "canonical_state",
    "compute_transition",
    "next_live_state",
    "passes_candidate_gate",
]
