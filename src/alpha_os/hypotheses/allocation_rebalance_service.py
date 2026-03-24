from __future__ import annotations

from ..config import Config, SIGNAL_CACHE_DB
from ..data.store import DataStore
from .breadth import (
    apply_capital_redundancy_cap,
    apply_live_proven_return_redundancy_cap,
    apply_weak_research_redundancy_cap,
    load_breadth_matrix,
)
from .lifecycle import AllocationRebalanceEntry, build_allocation_rebalance_plan
from .store import HypothesisStore


def build_capped_allocation_rebalance_plan(
    store: HypothesisStore,
    *,
    asset: str,
    config: Config,
    live_returns_for=None,
    signal_activity_for=None,
    data_store: DataStore | None = None,
) -> list[AllocationRebalanceEntry]:
    plan = build_allocation_rebalance_plan(
        store,
        metric=config.portfolio.objective,
        lookback=config.forward.degradation_window,
        min_observations=config.live_quality.min_observations,
        full_weight_observations=config.live_quality.full_weight_observations,
        early_stage_full_weight_observations=(
            config.live_quality.early_stage_full_weight_observations
        ),
        sharpe_clip_abs=config.live_quality.sharpe_clip_abs,
        log_growth_clip_abs=config.live_quality.log_growth_clip_abs,
        bootstrap_weight=config.lifecycle.bootstrap_weight,
        batch_research_weight=config.lifecycle.batch_research_weight,
        batch_research_normalized_quality_min=(
            config.lifecycle.batch_research_normalized_quality_min
        ),
        batch_research_capital_candidates_max=(
            config.lifecycle.batch_research_capital_candidates_max
        ),
        live_proven_quality_min=config.lifecycle.live_proven_quality_min,
        live_proven_marginal_contribution_min=(
            config.lifecycle.live_proven_marginal_contribution_min
        ),
        live_proven_signal_nonzero_ratio_min=(
            config.lifecycle.live_proven_signal_nonzero_ratio_min
        ),
        live_proven_signal_mean_abs_min=config.lifecycle.live_proven_signal_mean_abs_min,
        bootstrap_retention_quality_min=config.lifecycle.bootstrap_retention_quality_min,
        bootstrap_retention_marginal_contribution_min=(
            config.lifecycle.bootstrap_retention_marginal_contribution_min
        ),
        quality_weight=config.lifecycle.quality_weight,
        marginal_contribution_weight=config.lifecycle.marginal_contribution_weight,
        floor=config.lifecycle.target_stake_floor,
        live_returns_for=live_returns_for,
        signal_activity_for=signal_activity_for,
    )
    records = store.list_observation_active()
    close_data_store = data_store is None
    if data_store is None:
        data_store = DataStore(SIGNAL_CACHE_DB)
    try:
        breadth_data = load_breadth_matrix(
            data_store,
            records,
            asset=asset,
            lookback=config.forward.degradation_window,
        )
    finally:
        if close_data_store:
            data_store.close()
    plan = apply_weak_research_redundancy_cap(
        plan,
        records,
        data=breadth_data,
        asset=asset,
        corr_max=config.lifecycle.capital_redundancy_corr_max,
        floor=config.lifecycle.target_stake_floor,
    )
    plan = apply_capital_redundancy_cap(
        plan,
        records,
        data=breadth_data,
        asset=asset,
        corr_max=config.lifecycle.capital_redundancy_corr_max,
        floor=config.lifecycle.target_stake_floor,
    )
    if live_returns_for is not None:
        plan = apply_live_proven_return_redundancy_cap(
            plan,
            live_returns_for=live_returns_for,
            corr_max=config.lifecycle.capital_redundancy_corr_max,
            floor=config.lifecycle.target_stake_floor,
            min_observations=config.live_quality.min_observations,
        )
    return plan
