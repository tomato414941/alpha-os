"""Compatibility wrapper for hypothesis combination helpers."""

from alpha_os.hypotheses.combiner import (
    CombinerConfig,
    compute_diversity_scores,
    compute_stake_weights,
    compute_tc_scores,
    compute_tc_weights,
    cross_asset_neutralize,
    select_low_correlation,
    signal_consensus,
    weighted_combine,
    weighted_combine_scalar,
)

__all__ = [
    "CombinerConfig",
    "compute_diversity_scores",
    "compute_stake_weights",
    "compute_tc_scores",
    "compute_tc_weights",
    "cross_asset_neutralize",
    "select_low_correlation",
    "signal_consensus",
    "weighted_combine",
    "weighted_combine_scalar",
]
