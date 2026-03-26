from __future__ import annotations

from dataclasses import dataclass

from ..forward.tracker import HypothesisObservationTracker
from .bootstrap import bootstrap_hypotheses
from .observation import ObservationBackfillSummary, backfill_observation_returns
from .serious_templates import serious_seed_specs


@dataclass(frozen=True)
class SeriousTemplateMaintenanceRun:
    asset: str
    template_total: int
    inserted: int
    refreshed: int
    backfill: ObservationBackfillSummary


def _refresh_serious_bootstrap_records(*, store, asset: str) -> tuple[int, int]:
    inserted = 0
    refreshed = 0
    serious_records = [
        record
        for record in bootstrap_hypotheses(asset)
        if str(getattr(record, "source", "") or "") == "bootstrap_serious"
    ]
    for record in serious_records:
        existing = store.get(record.hypothesis_id)
        if existing is None:
            store.register(record)
            inserted += 1
            continue
        merged_metadata = dict(existing.metadata)
        changed = existing.name != record.name or existing.definition != record.definition
        for key, value in record.metadata.items():
            if merged_metadata.get(key) != value:
                merged_metadata[key] = value
                changed = True
        if not changed:
            continue
        store.register(
            existing.__class__(
                hypothesis_id=existing.hypothesis_id,
                kind=existing.kind,
                name=record.name,
                definition=record.definition,
                status=existing.status,
                stake=existing.stake,
                target_kind=existing.target_kind,
                horizon=existing.horizon,
                source=existing.source,
                scope=existing.scope,
                metadata=merged_metadata,
                created_at=existing.created_at,
                updated_at=existing.updated_at,
            )
        )
        refreshed += 1
    return inserted, refreshed


def run_serious_template_maintenance(
    *,
    store,
    data_store,
    forward_tracker: HypothesisObservationTracker,
    asset: str,
    lookback_days: int = 30,
) -> SeriousTemplateMaintenanceRun:
    inserted, refreshed = _refresh_serious_bootstrap_records(store=store, asset=asset)
    serious_ids = {spec.hypothesis_id for spec in serious_seed_specs(asset)}
    records = [
        record
        for record in store.list_observation_active(asset=asset)
        if record.hypothesis_id in serious_ids
    ]
    backfill = backfill_observation_returns(
        hypothesis_store=store,
        data_store=data_store,
        forward_tracker=forward_tracker,
        asset=asset,
        lookback_days=lookback_days,
        records=records,
    )
    return SeriousTemplateMaintenanceRun(
        asset=str(asset).upper(),
        template_total=len(serious_ids),
        inserted=inserted,
        refreshed=refreshed,
        backfill=backfill,
    )
