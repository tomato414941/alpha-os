"""Deployed alphas deployment policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..config import Config
from ..config import DATA_DIR
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl.features import collect_feature_names
from ..dsl.canonical import canonical_string
from ..dsl import parse
from ..forward.tracker import ForwardTracker
from .admission_replay import backup_registry_db
from .quality import QualityEstimate
from .registry import AlphaRecord, AlphaRegistry, AlphaState


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
class DeployedAlphaRefreshStats:
    registry_db: Path
    backup_path: Path | None
    plan: DeployedAlphaPlan


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
    semantic_key_by_id = {
        record.alpha_id: _semantic_key(record.expression)
        for record in active_records
    }
    feature_names_by_id = {
        record.alpha_id: _feature_names(record.expression)
        for record in active_records
    }

    if max_alphas <= 0 or not ranked:
        return DeployedAlphaPlan(
            active_count=len(active_records),
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

    deduped_ranked: list[RankedDeployedAlpha] = []
    seen_keys: set[str] = set()
    skipped_semantic_duplicate_ids: list[str] = []
    for item in ranked:
        key = semantic_key_by_id[item.alpha_id]
        if key in seen_keys:
            skipped_semantic_duplicate_ids.append(item.alpha_id)
            continue
        seen_keys.add(key)
        deduped_ranked.append(item)
    ranked, skipped_signal_duplicate_ids = _dedupe_by_signal_similarity(
        deduped_ranked,
        signal_by_id=signal_by_id or {},
        similarity_max=signal_similarity_max,
    )
    ranked, skipped_feature_cap_ids = _apply_feature_usage_cap(
        ranked,
        feature_names_by_id=feature_names_by_id,
        max_occurrences=max_feature_occurrences,
        min_keep=max_alphas,
    )
    ranked_by_id = {item.alpha_id: item for item in ranked}

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

    return DeployedAlphaPlan(
        active_count=len(active_records),
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


def refresh_deployed_alphas(
    db_path: Path,
    config: Config,
    *,
    asset: str | None = None,
    forward_db_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> DeployedAlphaRefreshStats:
    registry = AlphaRegistry(db_path)
    tracker = ForwardTracker(
        db_path=forward_db_path or db_path.with_name("forward_returns.db"),
    )
    try:
        resolved_asset = (asset or db_path.parent.name).upper()
        records = registry.list_all()
        signal_by_id = _build_signal_map(
            resolved_asset,
            config,
            records,
        )
        plan = plan_deployed_alphas(
            records,
            registry.deployed_alpha_ids(),
            lambda record: config.estimate_alpha_quality(
                record.oos_fitness(config.fitness_metric),
                tracker.get_returns(record.alpha_id),
            ),
            max_alphas=config.deployment.max_alphas,
            max_replacements=config.deployment.max_replacements,
            promotion_margin=config.deployment.promotion_margin,
            metric=config.fitness_metric,
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


def _semantic_key(expression: str) -> str:
    try:
        return canonical_string(expression)
    except Exception:
        return expression


def _feature_names(expression: str) -> set[str]:
    try:
        return collect_feature_names(parse(expression))
    except Exception:
        return set()


def _abs_signal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    if n < 10:
        return 0.0
    lhs = np.asarray(left[:n], dtype=np.float64)
    rhs = np.asarray(right[:n], dtype=np.float64)
    if np.std(lhs) <= 1e-12 or np.std(rhs) <= 1e-12:
        return 0.0
    corr = np.corrcoef(lhs, rhs)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def _dedupe_by_signal_similarity(
    ranked: list[RankedDeployedAlpha],
    *,
    signal_by_id: dict[str, np.ndarray],
    similarity_max: float,
) -> tuple[list[RankedDeployedAlpha], list[str]]:
    if similarity_max >= 1.0 or not signal_by_id:
        return ranked, []

    kept: list[RankedDeployedAlpha] = []
    kept_signals: list[np.ndarray] = []
    skipped: list[str] = []
    for item in ranked:
        signal = signal_by_id.get(item.alpha_id)
        if signal is None:
            kept.append(item)
            continue
        if any(
            _abs_signal_correlation(signal, existing) >= similarity_max
            for existing in kept_signals
        ):
            skipped.append(item.alpha_id)
            continue
        kept.append(item)
        kept_signals.append(signal)
    return kept, skipped


def _apply_feature_usage_cap(
    ranked: list[RankedDeployedAlpha],
    *,
    feature_names_by_id: dict[str, set[str]],
    max_occurrences: int,
    min_keep: int,
) -> tuple[list[RankedDeployedAlpha], list[str]]:
    if max_occurrences <= 0:
        return ranked, []

    kept: list[RankedDeployedAlpha] = []
    overflow: list[RankedDeployedAlpha] = []
    feature_counts: dict[str, int] = {}
    skipped: list[str] = []

    for item in ranked:
        features = feature_names_by_id.get(item.alpha_id, set())
        if any(feature_counts.get(name, 0) >= max_occurrences for name in features):
            overflow.append(item)
            skipped.append(item.alpha_id)
            continue
        kept.append(item)
        for name in features:
            feature_counts[name] = feature_counts.get(name, 0) + 1

    if len(kept) >= min_keep:
        return kept, skipped

    deficit = min_keep - len(kept)
    if deficit > 0:
        kept.extend(overflow[:deficit])
        skipped = skipped[deficit:]
    return kept, skipped


def _build_signal_map(
    asset: str,
    config: Config,
    records: list[AlphaRecord],
) -> dict[str, np.ndarray]:
    active_records = [
        record for record in records
        if AlphaState.canonical(record.state) == AlphaState.ACTIVE
    ]
    if not active_records:
        return {}

    lookback = max(int(config.deployment.signal_similarity_lookback), 0)
    if lookback <= 1 or config.deployment.signal_similarity_max >= 1.0:
        return {}

    store = DataStore(DATA_DIR / "alpha_cache.db", None)
    try:
        features = build_feature_list(asset)
        matrix = store.get_matrix(features)
    finally:
        store.close()
    if matrix.empty:
        return {}

    if lookback > 0:
        matrix = matrix.tail(lookback)
    data = {column: matrix[column].to_numpy(dtype=np.float64) for column in matrix.columns}
    n_days = len(matrix)
    signal_by_id: dict[str, np.ndarray] = {}
    for record in active_records:
        try:
            expr = parse(record.expression)
            signal = normalize_signal(evaluate_expression(expr, data, n_days))
        except (EvaluationError, Exception):
            continue
        signal_by_id[record.alpha_id] = np.asarray(signal, dtype=np.float64)
    return signal_by_id
