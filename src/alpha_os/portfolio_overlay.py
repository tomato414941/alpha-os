from __future__ import annotations

import math

from .portfolio_decision import PortfolioTarget


def apply_active_weight_budget(
    targets: list[PortfolioTarget],
    *,
    active_weight_budget: float | None,
    direction_mode: str,
) -> list[PortfolioTarget]:
    if active_weight_budget is None or active_weight_budget <= 0.0:
        return targets
    return apply_rank_tilt_overlay(
        targets,
        direction_mode=direction_mode,
        active_weight_budget=float(active_weight_budget),
    )


def apply_rank_tilt_overlay(
    targets: list[PortfolioTarget],
    *,
    direction_mode: str,
    active_weight_budget: float,
) -> list[PortfolioTarget]:
    if not targets:
        return targets
    gross_weight = sum(abs(float(item.target_weight)) for item in targets)
    budget = gross_weight * max(float(active_weight_budget), 0.0)
    if gross_weight <= 0.0 or budget <= 0.0:
        return targets
    if direction_mode == "short_only":
        return _apply_one_sided_rank_tilt(
            targets,
            budget=budget,
            side="short",
        )
    if direction_mode == "long_only":
        return _apply_one_sided_rank_tilt(
            targets,
            budget=budget,
            side="long",
        )
    return _apply_signed_rank_tilt(targets, budget=budget)


def _apply_one_sided_rank_tilt(
    targets: list[PortfolioTarget],
    *,
    budget: float,
    side: str,
) -> list[PortfolioTarget]:
    if side == "long":
        active = [item for item in targets if float(item.target_weight) > 0.0]
        scores = {item.subject_id: float(item.target_weight) for item in active}
    else:
        active = [item for item in targets if float(item.target_weight) < 0.0]
        scores = {item.subject_id: abs(float(item.target_weight)) for item in active}
    centered = _centered_rank_scores(scores)
    if not centered:
        return targets
    raw_delta_abs = sum(abs(value) for value in centered.values())
    if raw_delta_abs <= 0.0:
        return targets
    scale = budget / raw_delta_abs
    adjusted: list[PortfolioTarget] = []
    for item in targets:
        score = centered.get(item.subject_id)
        if score is None:
            adjusted.append(item)
            continue
        delta = score * scale
        next_weight = float(item.target_weight) + (delta if side == "long" else -delta)
        if side == "long":
            next_weight = max(next_weight, 0.0)
        else:
            next_weight = min(next_weight, 0.0)
        adjusted.append(
            _target_with_weight(
                item,
                next_weight,
                entry_allowed=bool(item.entry_allowed) and abs(next_weight) > 0.0,
            )
        )
    return adjusted


def _apply_signed_rank_tilt(
    targets: list[PortfolioTarget],
    *,
    budget: float,
) -> list[PortfolioTarget]:
    weights = {
        item.subject_id: float(item.target_weight)
        for item in targets
        if abs(float(item.target_weight)) > 0.0
    }
    if len(weights) < 2:
        return targets
    centered = _centered_rank_scores(
        {
            subject_id: abs(weight)
            for subject_id, weight in weights.items()
        }
    )
    if not centered:
        return targets
    raw_delta = {
        subject_id: centered[subject_id] * math.copysign(1.0, weights[subject_id])
        for subject_id in centered
    }
    raw_delta_abs = sum(abs(value) for value in raw_delta.values())
    if raw_delta_abs <= 0.0:
        return targets
    scale = budget / raw_delta_abs
    return [
        (
            _target_with_weight(
                item,
                float(item.target_weight) + raw_delta[item.subject_id] * scale,
                entry_allowed=bool(item.entry_allowed),
            )
            if item.subject_id in raw_delta
            else item
        )
        for item in targets
    ]


def _centered_rank_scores(scores: dict[str, float]) -> dict[str, float]:
    if len(scores) < 2:
        return {}
    ordered = sorted(scores.items(), key=lambda item: item[1])
    if ordered[0][1] == ordered[-1][1]:
        return {}
    midpoint = (len(ordered) - 1) / 2.0
    return {
        subject_id: float(rank - midpoint)
        for rank, (subject_id, _score) in enumerate(ordered)
    }


def _target_with_weight(
    target: PortfolioTarget,
    target_weight: float,
    *,
    entry_allowed: bool,
) -> PortfolioTarget:
    return PortfolioTarget(
        subject_id=target.subject_id,
        target_weight=float(target_weight),
        position_delta=0.0,
        target_notional=None,
        target_quantity=target.target_quantity,
        entry_allowed=entry_allowed,
        risk_scale=float(target.risk_scale),
    )
