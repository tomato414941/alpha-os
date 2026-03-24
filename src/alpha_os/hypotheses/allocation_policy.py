from __future__ import annotations

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
        and signal_nonzero_ratio >= live_proven_signal_nonzero_ratio_min
        and signal_mean_abs >= live_proven_signal_mean_abs_min
    )
    research_retained = research_backed and not has_min_observations
    return research_retained or live_proven


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
) -> tuple[bool, bool, bool, str]:
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
        and signal_nonzero_ratio >= live_proven_signal_nonzero_ratio_min
        and signal_mean_abs >= live_proven_signal_mean_abs_min
    )
    research_retained = research_backed and not has_min_observations
    if live_proven:
        reason = "live_proven"
    elif research_retained:
        reason = "research_backed"
    elif research_backed:
        reason = "research_demoted"
    else:
        reason = "none"
    return research_backed, research_retained, live_proven, reason


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
