from __future__ import annotations

from dataclasses import replace

from ..config import Config, SIGNAL_CACHE_DB
from ..data.store import DataStore
from .breadth import (
    apply_capital_redundancy_cap,
    apply_live_proven_return_redundancy_cap,
    apply_weak_research_redundancy_cap,
    load_breadth_matrix,
)
from .lifecycle import AllocationRebalanceEntry, build_allocation_rebalance_plan
from .lifecycle import finalize_capital_backing
from .store import HypothesisStore


def prefilter_reference_rebalance_records(
    records,
    *,
    max_records: int,
):
    if max_records <= 0 or len(records) <= max_records:
        return list(records)

    def _rank(record) -> tuple:
        metadata = getattr(record, "metadata", {}) or {}
        source = str(getattr(record, "source", "") or "")
        return (
            1 if source.startswith("bootstrap_") else 0,
            1 if bool(metadata.get("lifecycle_capital_backed", False)) else 0,
            float(getattr(record, "stake", 0.0)),
            1 if bool(metadata.get("lifecycle_actionable_live", False)) else 0,
            1 if bool(metadata.get("lifecycle_live_proven", False)) else 0,
            1 if bool(metadata.get("lifecycle_research_retained", False)) else 0,
            float(metadata.get("lifecycle_blended_quality", 0.0) or 0.0),
            float(metadata.get("lifecycle_bootstrap_trust", 0.0) or 0.0),
            float(metadata.get("oos_sharpe", 0.0) or 0.0),
            float(getattr(record, "updated_at", 0.0) or 0.0),
            str(getattr(record, "hypothesis_id", "") or ""),
        )

    ranked = sorted(records, key=_rank, reverse=True)
    return ranked[:max_records]


def apply_capital_backed_count_cap(
    plan: list[AllocationRebalanceEntry],
    *,
    max_backed: int,
    floor: float,
    capital_reason: str = "reference_capital_capped",
) -> list[AllocationRebalanceEntry]:
    if max_backed <= 0:
        return plan

    ranked = sorted(
        (
            entry
            for entry in plan
            if entry.capital_eligible and float(entry.proposed_stake) > float(floor)
        ),
        key=lambda entry: (
            entry.proposed_stake,
            entry.target_stake,
            entry.bootstrap_trust_value,
            entry.blended_quality,
            entry.live_quality,
            entry.n_observations,
            entry.hypothesis_id,
        ),
        reverse=True,
    )
    if len(ranked) <= max_backed:
        return plan

    allowed_ids = {entry.hypothesis_id for entry in ranked[:max_backed]}
    capped: list[AllocationRebalanceEntry] = []
    for entry in plan:
        if (
            entry.capital_eligible
            and float(entry.proposed_stake) > float(floor)
            and entry.hypothesis_id not in allowed_ids
        ):
            capped.append(
                replace(
                    entry,
                    proposed_stake=float(floor),
                    capital_eligible=False,
                    capital_reason=capital_reason,
                )
            )
            continue
        capped.append(entry)
    return capped


def build_capped_allocation_rebalance_plan(
    store: HypothesisStore,
    *,
    asset: str,
    config: Config,
    live_returns_for=None,
    signal_activity_for=None,
    data_store: DataStore | None = None,
) -> list[AllocationRebalanceEntry]:
    records = store.list_observation_active(asset=asset)
    if (
        str(asset).upper() == str(config.cross_sectional.registry_asset).upper()
        and int(config.lifecycle.reference_rebalance_candidates_max) > 0
    ):
        records = prefilter_reference_rebalance_records(
            records,
            max_records=int(config.lifecycle.reference_rebalance_candidates_max),
        )
    plan = build_allocation_rebalance_plan(
        store,
        asset=asset,
        records=records,
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
    records = store.list_observation_active(asset=asset)
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
    if (
        str(asset).upper() == str(config.cross_sectional.registry_asset).upper()
        and int(config.lifecycle.reference_capital_backed_max) > 0
    ):
        plan = apply_capital_backed_count_cap(
            plan,
            max_backed=int(config.lifecycle.reference_capital_backed_max),
            floor=config.lifecycle.target_stake_floor,
        )
    return finalize_capital_backing(
        plan,
        floor=config.lifecycle.target_stake_floor,
    )
