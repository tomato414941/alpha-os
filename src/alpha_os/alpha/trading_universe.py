"""Trading universe deployment policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..forward.tracker import ForwardTracker
from .admission_replay import backup_registry_db
from .quality import QualityEstimate
from .registry import AlphaRecord, AlphaRegistry, AlphaState


@dataclass(frozen=True)
class RankedUniverseAlpha:
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
class TradingUniversePlan:
    active_count: int
    current_count: int
    deployed_count: int
    replacement_count: int
    kept_ids: list[str]
    added_ids: list[str]
    dropped_ids: list[str]
    selected: list[RankedUniverseAlpha]

    @property
    def selected_ids(self) -> list[str]:
        return [item.alpha_id for item in self.selected]

    @property
    def selected_scores(self) -> dict[str, float]:
        return {item.alpha_id: item.score for item in self.selected}

    @property
    def selected_metadata(self) -> dict[str, dict[str, float | int]]:
        return {item.alpha_id: item.to_metadata() for item in self.selected}


@dataclass(frozen=True)
class TradingUniverseRefreshStats:
    registry_db: Path
    backup_path: Path | None
    plan: TradingUniversePlan


def plan_trading_universe(
    records: list[AlphaRecord],
    current_ids: list[str],
    estimate_for,
    *,
    max_alphas: int,
    max_replacements: int,
    promotion_margin: float,
    metric: str,
) -> TradingUniversePlan:
    active_records = [
        record for record in records
        if AlphaState.canonical(record.state) == AlphaState.ACTIVE
    ]
    ranked = [
        _ranked_alpha(record, estimate_for(record), metric)
        for record in active_records
    ]
    ranked.sort(key=lambda item: item.rank_key, reverse=True)
    ranked_by_id = {item.alpha_id: item for item in ranked}

    if max_alphas <= 0 or not ranked:
        return TradingUniversePlan(
            active_count=len(active_records),
            current_count=0,
            deployed_count=0,
            replacement_count=0,
            kept_ids=[],
            added_ids=[],
            dropped_ids=[],
            selected=[],
        )

    current = []
    seen = set()
    for alpha_id in current_ids:
        if alpha_id in ranked_by_id and alpha_id not in seen:
            current.append(alpha_id)
            seen.add(alpha_id)

    selected_ids = current[:max_alphas]
    current_set = set(current)
    remaining_ranked = [item.alpha_id for item in ranked if item.alpha_id not in selected_ids]

    while len(selected_ids) < max_alphas and remaining_ranked:
        selected_ids.append(remaining_ranked.pop(0))

    replaced_out: set[str] = set()
    replacements = 0
    while replacements < max_replacements and remaining_ranked:
        incumbent_ids = [
            alpha_id for alpha_id in selected_ids
            if alpha_id in current_set and alpha_id not in replaced_out
        ]
        if not incumbent_ids:
            break
        challenger_id = remaining_ranked[0]
        weakest_id = min(
            incumbent_ids,
            key=lambda alpha_id: ranked_by_id[alpha_id].rank_key,
        )
        challenger = ranked_by_id[challenger_id]
        weakest = ranked_by_id[weakest_id]
        if challenger.score < weakest.score + promotion_margin:
            break
        selected_ids[selected_ids.index(weakest_id)] = challenger_id
        remaining_ranked.pop(0)
        replaced_out.add(weakest_id)
        replacements += 1

    selected = sorted(
        [ranked_by_id[alpha_id] for alpha_id in selected_ids],
        key=lambda item: item.rank_key,
        reverse=True,
    )
    selected_set = {item.alpha_id for item in selected}
    kept_ids = [item.alpha_id for item in selected if item.alpha_id in current_set]
    added_ids = [item.alpha_id for item in selected if item.alpha_id not in current_set]
    dropped_ids = [alpha_id for alpha_id in current if alpha_id not in selected_set]

    return TradingUniversePlan(
        active_count=len(active_records),
        current_count=len(current),
        deployed_count=len(selected),
        replacement_count=replacements,
        kept_ids=kept_ids,
        added_ids=added_ids,
        dropped_ids=dropped_ids,
        selected=selected,
    )


def refresh_trading_universe(
    db_path: Path,
    config: Config,
    *,
    forward_db_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> TradingUniverseRefreshStats:
    registry = AlphaRegistry(db_path)
    tracker = ForwardTracker(
        db_path=forward_db_path or db_path.with_name("forward_returns.db"),
    )
    try:
        plan = plan_trading_universe(
            registry.list_all(),
            registry.trading_universe_ids(),
            lambda record: config.estimate_alpha_quality(
                record.oos_fitness(config.fitness_metric),
                tracker.get_returns(record.alpha_id),
            ),
            max_alphas=config.universe.max_alphas,
            max_replacements=config.universe.max_replacements,
            promotion_margin=config.universe.promotion_margin,
            metric=config.fitness_metric,
        )

        backup_path = None
        if not dry_run:
            if backup and db_path.exists():
                backup_path = backup_registry_db(db_path)
            registry.replace_trading_universe(
                plan.selected_ids,
                scores=plan.selected_scores,
                metadata=plan.selected_metadata,
            )
        return TradingUniverseRefreshStats(
            registry_db=db_path,
            backup_path=backup_path,
            plan=plan,
        )
    finally:
        tracker.close()
        registry.close()


def _ranked_alpha(
    record: AlphaRecord,
    estimate: QualityEstimate,
    metric: str,
) -> RankedUniverseAlpha:
    return RankedUniverseAlpha(
        alpha_id=record.alpha_id,
        prior_quality=record.oos_fitness(metric),
        blended_quality=estimate.blended_quality,
        confidence=estimate.confidence,
        live_quality=estimate.live_quality,
        n_observations=estimate.n_observations,
    )
