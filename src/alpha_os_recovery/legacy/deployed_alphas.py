"""Deployed alphas deployment policy."""
# TODO: Legacy deployment policy for alpha_registry.db. Keep this outside the
# hypotheses-first runtime mainline until live-hypothesis selection fully
# replaces managed/deployed registry workflows.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config, HYPOTHESIS_OBSERVATIONS_DB_NAME
from ..forward.tracker import HypothesisObservationTracker
from ..legacy.deployment_planner import (
    DeployedAlphaPlan,
    RegistryActivePrunePlan,
    plan_deployed_alphas,
    plan_registry_active_prune,
)
from ..legacy.registry_signal_map import build_registry_signal_map
from .admission_replay import backup_registry_db
from .managed_alphas import ManagedAlphaStore
from .registry_types import AlphaState


@dataclass(frozen=True)
class DeployedAlphaRefreshStats:
    registry_db: Path
    backup_path: Path | None
    plan: DeployedAlphaPlan


@dataclass(frozen=True)
class RegistryActivePruneStats:
    registry_db: Path
    backup_path: Path | None
    plan: RegistryActivePrunePlan
    deployed_refresh: DeployedAlphaRefreshStats | None


def refresh_deployed_alphas(
    db_path: Path,
    config: Config,
    *,
    asset: str | None = None,
    forward_db_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> DeployedAlphaRefreshStats:
    registry = ManagedAlphaStore(db_path)
    tracker = HypothesisObservationTracker(
        db_path=forward_db_path or db_path.with_name(HYPOTHESIS_OBSERVATIONS_DB_NAME),
    )
    try:
        resolved_asset = (asset or db_path.parent.name).upper()
        records = registry.list_all()
        signal_by_id = build_registry_signal_map(
            resolved_asset,
            config,
            records,
        )
        objective = config.portfolio.objective
        plan = plan_deployed_alphas(
            records,
            registry.deployed_alpha_ids(),
            lambda record: config.estimate_alpha_quality(
                record.oos_fitness(objective),
                tracker.get_hypothesis_returns(record.alpha_id),
            ),
            max_alphas=config.deployment.max_deployed_alphas,
            max_replacements=config.deployment.max_replacements,
            promotion_margin=config.deployment.promotion_margin,
            metric=objective,
            signal_by_id=signal_by_id,
            signal_similarity_max=config.deployment.signal_similarity_max,
            max_feature_occurrences=config.deployment.max_feature_occurrences,
        )

        backup_path = None
        if not dry_run:
            if backup and db_path.exists():
                backup_path = backup_registry_db(db_path)
            registry.replace_deployed_alphas(
                plan.selected_ids,
                scores=plan.selected_scores,
                metadata=plan.selected_metadata,
            )
        return DeployedAlphaRefreshStats(
            registry_db=db_path,
            backup_path=backup_path,
            plan=plan,
        )
    finally:
        tracker.close()
        registry.close()


def prune_registry_active_duplicates(
    db_path: Path,
    config: Config,
    *,
    asset: str | None = None,
    forward_db_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
    refresh_deployed: bool = True,
) -> RegistryActivePruneStats:
    registry = ManagedAlphaStore(db_path)
    tracker = HypothesisObservationTracker(
        db_path=forward_db_path or db_path.with_name(HYPOTHESIS_OBSERVATIONS_DB_NAME),
    )
    try:
        resolved_asset = (asset or db_path.parent.name).upper()
        records = registry.list_all()
        current_deployed_ids = registry.deployed_alpha_ids()
        signal_by_id = build_registry_signal_map(
            resolved_asset,
            config,
            records,
        )
        objective = config.portfolio.objective
        plan = plan_registry_active_prune(
            records,
            current_deployed_ids,
            lambda record: config.estimate_alpha_quality(
                record.oos_fitness(objective),
                tracker.get_hypothesis_returns(record.alpha_id),
            ),
            metric=objective,
            signal_by_id=signal_by_id,
            signal_similarity_max=config.deployment.signal_similarity_max,
        )

        backup_path = None
        deployed_refresh = None
        if not dry_run and plan.demoted_ids:
            if backup and db_path.exists():
                backup_path = backup_registry_db(db_path)
            registry.bulk_update_states(plan.demoted_ids, AlphaState.DORMANT)
            if refresh_deployed:
                deployed_refresh = refresh_deployed_alphas(
                    db_path,
                    config,
                    asset=resolved_asset,
                    forward_db_path=forward_db_path,
                    dry_run=False,
                    backup=False,
                )
        return RegistryActivePruneStats(
            registry_db=db_path,
            backup_path=backup_path,
            plan=plan,
            deployed_refresh=deployed_refresh,
        )
    finally:
        tracker.close()
        registry.close()
