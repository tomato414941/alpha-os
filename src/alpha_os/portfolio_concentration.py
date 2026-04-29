from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_EPSILON = 1e-12


@dataclass(frozen=True)
class PortfolioConcentrationSnapshot:
    active_position_count: int
    effective_n: float
    top1_gross_share: float
    top3_gross_share: float
    top5_gross_share: float
    max_subject_label: str | None
    max_subject_gross_share: float
    max_cluster_label: str | None
    max_cluster_gross_share: float


def portfolio_effective_n(
    weights: Iterable[float],
    *,
    min_abs_weight: float = 0.0,
) -> float:
    gross_weights = [
        abs(float(weight))
        for weight in weights
        if abs(float(weight)) > max(float(min_abs_weight), 0.0)
    ]
    total = sum(gross_weights)
    if total <= _EPSILON:
        return 0.0
    shares = [weight / total for weight in gross_weights]
    return float(1.0 / sum(share * share for share in shares))


def top_n_gross_share(weights: Iterable[float], *, top_n: int) -> float:
    if top_n < 1:
        return 0.0
    gross_weights = sorted((abs(float(weight)) for weight in weights), reverse=True)
    total = sum(gross_weights)
    if total <= _EPSILON:
        return 0.0
    return float(sum(gross_weights[:top_n]) / total)


def active_position_count(
    weights: Iterable[float],
    *,
    min_abs_weight: float = 0.001,
) -> int:
    threshold = max(float(min_abs_weight), 0.0)
    return sum(1 for weight in weights if abs(float(weight)) > threshold)


def group_gross_shares(
    weights_by_label: dict[str, float],
) -> dict[str, float]:
    total = sum(abs(float(weight)) for weight in weights_by_label.values())
    if total <= _EPSILON:
        return {}
    return {
        label: abs(float(weight)) / total
        for label, weight in weights_by_label.items()
    }


def concentration_snapshot(
    weights_by_subject: dict[str, float],
    *,
    cluster_by_subject: dict[str, str] | None = None,
    min_abs_weight: float = 0.001,
) -> PortfolioConcentrationSnapshot:
    cluster_by_subject = cluster_by_subject or {}
    weights = list(weights_by_subject.values())
    subject_shares = group_gross_shares(weights_by_subject)
    max_subject_label, max_subject_share = _max_share(subject_shares)
    cluster_weights: dict[str, float] = {}
    for subject_id, weight in weights_by_subject.items():
        label = cluster_by_subject.get(subject_id, "-")
        cluster_weights[label] = cluster_weights.get(label, 0.0) + abs(float(weight))
    cluster_shares = group_gross_shares(cluster_weights)
    max_cluster_label, max_cluster_share = _max_share(cluster_shares)
    return PortfolioConcentrationSnapshot(
        active_position_count=active_position_count(
            weights,
            min_abs_weight=min_abs_weight,
        ),
        effective_n=portfolio_effective_n(weights),
        top1_gross_share=top_n_gross_share(weights, top_n=1),
        top3_gross_share=top_n_gross_share(weights, top_n=3),
        top5_gross_share=top_n_gross_share(weights, top_n=5),
        max_subject_label=max_subject_label,
        max_subject_gross_share=max_subject_share,
        max_cluster_label=max_cluster_label,
        max_cluster_gross_share=max_cluster_share,
    )


def _max_share(shares: dict[str, float]) -> tuple[str | None, float]:
    if not shares:
        return None, 0.0
    label, value = max(shares.items(), key=lambda item: (item[1], item[0]))
    return label, float(value)
