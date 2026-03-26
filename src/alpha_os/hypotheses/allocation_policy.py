from __future__ import annotations

import numpy as np

DEFAULT_BOOTSTRAP_WEIGHT = 0.25
DEFAULT_BATCH_RESEARCH_WEIGHT = 0.10
DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN = 0.10
DEFAULT_QUALITY_WEIGHT = 1.0
DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT = 0.25
DEFAULT_LIVE_PROVEN_QUALITY_MIN = 0.05
DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN = 0.0
DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN = 0.20
DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN = 0.05
DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN = 0.0
DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN = 0.0


def dedupe_ranked_ids_by_semantic_key(
    ranked_ids: list[str],
    *,
    semantic_key_by_id: dict[str, str],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    seen_keys: set[str] = set()
    skipped: list[str] = []
    for item_id in ranked_ids:
        key = semantic_key_by_id[item_id]
        if key in seen_keys:
            skipped.append(item_id)
            continue
        seen_keys.add(key)
        kept.append(item_id)
    return kept, skipped


def resolve_ranked_current_ids(
    current_ids: list[str],
    ranked_ids: list[str],
) -> list[str]:
    ranked_set = set(ranked_ids)
    current: list[str] = []
    seen: set[str] = set()
    for item_id in current_ids:
        if item_id in ranked_set and item_id not in seen:
            current.append(item_id)
            seen.add(item_id)
    return current


def seed_ranked_selection(
    *,
    current_ids: list[str],
    ranked_ids: list[str],
    max_selected: int,
) -> list[str]:
    selected_ids = current_ids[:max_selected]
    for item_id in ranked_ids:
        if len(selected_ids) >= max_selected:
            break
        if item_id in selected_ids:
            continue
        selected_ids.append(item_id)
    return selected_ids


def apply_ranked_replacement_policy(
    *,
    selected_ids: list[str],
    current_ids: list[str],
    remaining_ids: list[str],
    rank_key_by_id: dict[str, tuple[float, ...]],
    score_by_id: dict[str, float],
    max_replacements: int,
    promotion_margin: float,
) -> tuple[list[str], int]:
    current_set = set(current_ids)
    replaced_out: set[str] = set()
    replacements = 0
    while replacements < max_replacements and remaining_ids:
        incumbent_ids = [
            item_id for item_id in selected_ids
            if item_id in current_set and item_id not in replaced_out
        ]
        if not incumbent_ids:
            break
        challenger_id = remaining_ids[0]
        weakest_id = min(
            incumbent_ids,
            key=lambda item_id: rank_key_by_id[item_id],
        )
        if score_by_id[challenger_id] < score_by_id[weakest_id] + promotion_margin:
            break
        selected_ids[selected_ids.index(weakest_id)] = challenger_id
        remaining_ids.pop(0)
        replaced_out.add(weakest_id)
        replacements += 1
    return selected_ids, replacements


def dedupe_ranked_ids_by_signal_similarity(
    ranked_ids: list[str],
    *,
    signal_by_id: dict[str, np.ndarray],
    similarity_max: float,
) -> tuple[list[str], list[str]]:
    if similarity_max >= 1.0 or not signal_by_id:
        return ranked_ids, []

    kept: list[str] = []
    kept_signals: list[np.ndarray] = []
    skipped: list[str] = []
    for item_id in ranked_ids:
        signal = signal_by_id.get(item_id)
        if signal is None:
            kept.append(item_id)
            continue
        if any(
            abs_signal_correlation(signal, existing) >= similarity_max
            for existing in kept_signals
        ):
            skipped.append(item_id)
            continue
        kept.append(item_id)
        kept_signals.append(signal)
    return kept, skipped


def apply_ranked_feature_usage_cap(
    ranked_ids: list[str],
    *,
    feature_names_by_id: dict[str, set[str]],
    max_occurrences: int,
    min_keep: int,
) -> tuple[list[str], list[str]]:
    if max_occurrences <= 0:
        return ranked_ids, []

    kept: list[str] = []
    overflow: list[str] = []
    feature_counts: dict[str, int] = {}
    skipped: list[str] = []

    for item_id in ranked_ids:
        features = feature_names_by_id.get(item_id, set())
        if any(feature_counts.get(name, 0) >= max_occurrences for name in features):
            overflow.append(item_id)
            skipped.append(item_id)
            continue
        kept.append(item_id)
        for name in features:
            feature_counts[name] = feature_counts.get(name, 0) + 1

    if len(kept) >= min_keep:
        return kept, skipped

    deficit = min_keep - len(kept)
    if deficit > 0:
        kept.extend(overflow[:deficit])
        skipped = skipped[deficit:]
    return kept, skipped


def abs_signal_correlation(left: np.ndarray, right: np.ndarray) -> float:
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


def trust_score(
    blended_quality: float,
    marginal_contribution: float,
    *,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
) -> float:
    score = (
        quality_weight * float(blended_quality)
        + marginal_contribution_weight * float(marginal_contribution)
    )
    return max(score, 0.0)


def normalized_research_quality(
    research_quality: float,
    *,
    metric: str = "sharpe",
) -> float:
    if metric == "sharpe":
        scale = 2.0
    elif metric == "log_growth":
        scale = 0.20
    else:
        raise ValueError(f"Unsupported fitness metric: {metric}")
    return min(max(float(research_quality) / scale, 0.0), 1.0)


def bootstrap_trust(
    research_quality: float,
    *,
    metric: str = "sharpe",
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    research_quality_source: str = "bootstrap_seed",
) -> float:
    if research_quality_source == "batch_research_score":
        weight = max(float(batch_research_weight), 0.0)
    elif research_quality_source in {"bootstrap_seed", ""}:
        weight = max(float(bootstrap_weight), 0.0)
    else:
        weight = 0.0
    return weight * normalized_research_quality(research_quality, metric=metric)


def is_research_backed(
    research_quality: float,
    *,
    metric: str,
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    batch_research_normalized_quality_min: float = (
        DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN
    ),
    research_quality_source: str = "bootstrap_seed",
    floor: float = 0.0,
) -> bool:
    trust = bootstrap_trust(
        research_quality,
        metric=metric,
        bootstrap_weight=bootstrap_weight,
        batch_research_weight=batch_research_weight,
        research_quality_source=research_quality_source,
    )
    if trust <= floor:
        return False
    if research_quality_source == "batch_research_score":
        return normalized_research_quality(
            research_quality,
            metric=metric,
        ) >= float(batch_research_normalized_quality_min)
    return True


def target_stake(
    blended_quality: float,
    quality_confidence: float,
    marginal_contribution: float,
    *,
    research_quality: float = 0.0,
    research_quality_source: str = "bootstrap_seed",
    metric: str = "sharpe",
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
    floor: float = 0.0,
) -> float:
    live_trust = trust_score(
        blended_quality,
        marginal_contribution,
        quality_weight=quality_weight,
        marginal_contribution_weight=marginal_contribution_weight,
    )
    confidence = min(max(float(quality_confidence), 0.0), 1.0)
    initial_trust = bootstrap_trust(
        research_quality,
        metric=metric,
        bootstrap_weight=bootstrap_weight,
        batch_research_weight=batch_research_weight,
        research_quality_source=research_quality_source,
    )
    trust = (1.0 - confidence) * initial_trust + confidence * live_trust
    return max(trust, floor)


def is_capital_eligible(
    *,
    research_quality: float,
    metric: str,
    bootstrap_weight: float,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    batch_research_normalized_quality_min: float = (
        DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN
    ),
    research_quality_source: str = "bootstrap_seed",
    has_min_observations: bool,
    live_quality: float = 0.0,
    marginal_contribution: float = 0.0,
    signal_nonzero_ratio: float = 0.0,
    signal_mean_abs: float = 0.0,
    live_proven_quality_min: float = DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    live_proven_marginal_contribution_min: float = DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    live_proven_signal_nonzero_ratio_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN,
    live_proven_signal_mean_abs_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
    bootstrap_retention_quality_min: float = DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN,
    bootstrap_retention_marginal_contribution_min: float = (
        DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN
    ),
    floor: float = 0.0,
) -> bool:
    research_backed = is_research_backed(
        research_quality,
        metric=metric,
        bootstrap_weight=bootstrap_weight,
        batch_research_weight=batch_research_weight,
        batch_research_normalized_quality_min=batch_research_normalized_quality_min,
        research_quality_source=research_quality_source,
        floor=floor,
    )
    live_proven = has_min_observations and (
        live_quality >= live_proven_quality_min
        and marginal_contribution >= live_proven_marginal_contribution_min
    )
    actionable_live = live_proven and (
        signal_nonzero_ratio >= live_proven_signal_nonzero_ratio_min
        and signal_mean_abs >= live_proven_signal_mean_abs_min
    )
    research_retained = research_backed and not has_min_observations
    return research_retained or actionable_live


def capital_eligibility_breakdown(
    *,
    research_quality: float,
    metric: str,
    bootstrap_weight: float,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    batch_research_normalized_quality_min: float = (
        DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN
    ),
    research_quality_source: str = "bootstrap_seed",
    has_min_observations: bool,
    live_quality: float,
    marginal_contribution: float,
    signal_nonzero_ratio: float = 0.0,
    signal_mean_abs: float = 0.0,
    live_proven_quality_min: float = DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    live_proven_marginal_contribution_min: float = DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    live_proven_signal_nonzero_ratio_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN,
    live_proven_signal_mean_abs_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
    bootstrap_retention_quality_min: float = DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN,
    bootstrap_retention_marginal_contribution_min: float = (
        DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN
    ),
    floor: float = 0.0,
) -> tuple[bool, bool, bool, bool, str]:
    research_backed = is_research_backed(
        research_quality,
        metric=metric,
        bootstrap_weight=bootstrap_weight,
        batch_research_weight=batch_research_weight,
        batch_research_normalized_quality_min=batch_research_normalized_quality_min,
        research_quality_source=research_quality_source,
        floor=floor,
    )
    live_proven = has_min_observations and (
        live_quality >= live_proven_quality_min
        and marginal_contribution >= live_proven_marginal_contribution_min
    )
    actionable_live = live_proven and (
        signal_nonzero_ratio >= live_proven_signal_nonzero_ratio_min
        and signal_mean_abs >= live_proven_signal_mean_abs_min
    )
    research_retained = research_backed and not has_min_observations
    if actionable_live:
        reason = "actionable_live"
    elif research_retained:
        reason = "research_backed"
    elif live_proven:
        reason = "live_not_actionable"
    elif research_backed:
        reason = "research_demoted"
    else:
        reason = "none"
    return research_backed, research_retained, live_proven, actionable_live, reason


def live_promotion_blocker(
    *,
    has_min_observations: bool,
    live_quality: float,
    marginal_contribution: float,
    signal_nonzero_ratio: float = 0.0,
    signal_mean_abs: float = 0.0,
    live_proven_quality_min: float = DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    live_proven_marginal_contribution_min: float = DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    live_proven_signal_nonzero_ratio_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN,
    live_proven_signal_mean_abs_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
) -> str:
    if not has_min_observations:
        return "insufficient_observations"
    weak_quality = live_quality < live_proven_quality_min
    weak_contribution = marginal_contribution < live_proven_marginal_contribution_min
    weak_signal_activity = (
        signal_nonzero_ratio < live_proven_signal_nonzero_ratio_min
        or signal_mean_abs < live_proven_signal_mean_abs_min
    )
    if weak_quality and weak_contribution:
        return "weak_live_quality_and_contribution"
    if weak_quality:
        return "weak_live_quality"
    if weak_contribution:
        return "weak_marginal_contribution"
    if weak_signal_activity:
        return "weak_signal_activity"
    return "eligible"
