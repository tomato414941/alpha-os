from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .portfolio_concentration import portfolio_effective_n, top_n_gross_share
from .portfolio_decision import PortfolioTarget
from .portfolio_direction import normalize_portfolio_direction_mode
from .portfolio_overlay import apply_active_weight_budget


@dataclass(frozen=True)
class PortfolioConstructionRequest:
    targets: tuple[PortfolioTarget, ...]
    current_weights: dict[str, float]
    capital_base: float
    gross_exposure_cap: float | None
    gross_leverage_cap: float | None
    net_exposure_target: float | None
    target_vol: float | None
    risk_by_subject: dict[str, float]
    constraint_boundary: PortfolioConstraintBoundary
    long_only: bool
    direction_mode: str
    top_k: int | None
    active_weight_budget: float | None
    asset_class_by_subject: dict[str, str]
    cluster_by_subject: dict[str, str]
    asset_class_weight_caps: dict[str, float]
    cluster_weight_caps: dict[str, float]


@dataclass(frozen=True)
class PortfolioConstructionStageSnapshot:
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    effective_n: float
    top3_gross_share: float
    active_position_count: int


@dataclass(frozen=True)
class PortfolioConstructionStageTrace:
    stage_name: str
    before: PortfolioConstructionStageSnapshot
    after: PortfolioConstructionStageSnapshot
    gross_delta: float
    net_delta: float
    active_count_delta: int
    changed_subject_count: int


@dataclass(frozen=True)
class PortfolioConstructionResult:
    targets: dict[str, PortfolioTarget]
    trace: tuple[PortfolioConstructionStageTrace, ...]


class PortfolioConstructionStage(Protocol):
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        ...


@dataclass(frozen=True)
class PortfolioConstructionPipeline:
    stages: tuple[PortfolioConstructionStage, ...]

    def apply(self, request: PortfolioConstructionRequest) -> dict[str, PortfolioTarget]:
        return self.apply_with_trace(request).targets

    def apply_with_trace(
        self,
        request: PortfolioConstructionRequest,
    ) -> PortfolioConstructionResult:
        normalized = list(request.targets)
        trace: list[PortfolioConstructionStageTrace] = []
        for stage in self.stages:
            before = list(normalized)
            after = stage.apply(before, request)
            trace.append(_stage_trace(stage, before=before, after=after))
            normalized = after
        return PortfolioConstructionResult(
            targets=_finalize_targets(normalized, request),
            trace=tuple(trace),
        )


def construct_portfolio_targets(
    request: PortfolioConstructionRequest,
) -> PortfolioConstructionResult:
    return default_portfolio_construction_pipeline().apply_with_trace(request)


def default_portfolio_construction_pipeline() -> PortfolioConstructionPipeline:
    return PortfolioConstructionPipeline(
        stages=(
            DirectionModeStage(),
            ActiveWeightBudgetStage(),
            TopKStage(),
            GroupWeightCapStage(
                boundary_field="asset_class_weight_caps",
                group_field="asset_class",
            ),
            GroupWeightCapStage(
                boundary_field="cluster_weight_caps",
                group_field="cluster",
            ),
            TargetVolCapStage(),
            GrossExposureCapStage(),
            NetExposureTargetStage(),
            GrossExposureCapStage(),
        )
    )


def build_portfolio_construction_request(
    targets: tuple[PortfolioTarget, ...],
    *,
    current_weights: dict[str, float],
    capital_base: float,
    gross_exposure_cap: float | None,
    gross_leverage_cap: float | None,
    net_exposure_target: float | None,
    target_vol: float | None = None,
    risk_by_subject: dict[str, float] | None = None,
    constraint_boundary: PortfolioConstraintBoundary | None = None,
    long_only: bool,
    top_k: int | None,
    asset_class_by_subject: dict[str, str],
    cluster_by_subject: dict[str, str],
    asset_class_weight_caps: dict[str, float],
    cluster_weight_caps: dict[str, float],
    active_weight_budget: float | None = None,
    direction_mode: str | None = None,
) -> PortfolioConstructionRequest:
    normalized_direction_mode = normalize_portfolio_direction_mode(
        direction_mode,
        long_only=long_only,
    )
    return PortfolioConstructionRequest(
        targets=targets,
        current_weights=current_weights,
        capital_base=capital_base,
        gross_exposure_cap=gross_exposure_cap,
        gross_leverage_cap=gross_leverage_cap,
        net_exposure_target=net_exposure_target,
        target_vol=target_vol,
        risk_by_subject=risk_by_subject or {},
        constraint_boundary=constraint_boundary or default_portfolio_constraint_boundary(),
        long_only=normalized_direction_mode == "long_only",
        direction_mode=normalized_direction_mode,
        top_k=top_k,
        active_weight_budget=active_weight_budget,
        asset_class_by_subject=asset_class_by_subject,
        cluster_by_subject=cluster_by_subject,
        asset_class_weight_caps=asset_class_weight_caps,
        cluster_weight_caps=cluster_weight_caps,
    )


class DirectionModeStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        if request.direction_mode == "long_short":
            return targets
        if request.direction_mode == "long_only":
            return [
                _target_with_weight(
                    item,
                    max(float(item.target_weight), 0.0),
                    entry_allowed=bool(item.entry_allowed)
                    and float(item.target_weight) > 0.0,
                )
                for item in targets
            ]
        return [
            _target_with_weight(
                item,
                min(float(item.target_weight), 0.0),
                entry_allowed=bool(item.entry_allowed)
                and float(item.target_weight) < 0.0,
            )
            for item in targets
        ]


class ActiveWeightBudgetStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        return apply_active_weight_budget(
            targets,
            active_weight_budget=request.active_weight_budget,
            direction_mode=request.direction_mode,
        )


class TopKStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        if request.top_k is None:
            return targets
        ranked = sorted(
            targets,
            key=lambda item: _top_k_rank_value(item, request.direction_mode),
            reverse=True,
        )
        keep = {
            item.subject_id
            for item in ranked[: max(int(request.top_k), 0)]
            if _top_k_candidate_is_active(item, request.direction_mode)
        }
        return [
            item
            if item.subject_id in keep
            else _target_with_weight(item, 0.0, entry_allowed=False)
            for item in targets
        ]


def _top_k_rank_value(target: PortfolioTarget, direction_mode: str) -> float:
    if direction_mode == "long_only":
        return float(target.target_weight)
    return abs(float(target.target_weight))


def _top_k_candidate_is_active(target: PortfolioTarget, direction_mode: str) -> bool:
    target_weight = float(target.target_weight)
    if direction_mode == "long_only":
        return target_weight > 0.0
    if direction_mode == "short_only":
        return target_weight < 0.0
    return abs(target_weight) > 0.0


@dataclass(frozen=True)
class GroupWeightCapStage:
    boundary_field: str
    group_field: str

    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        if not request.constraint_boundary.is_post_sizing_normalization(self.boundary_field):
            return targets
        if self.group_field == "asset_class":
            return apply_group_weight_caps(
                targets,
                group_by_subject=request.asset_class_by_subject,
                weight_caps=request.asset_class_weight_caps,
            )
        return apply_group_weight_caps(
            targets,
            group_by_subject=request.cluster_by_subject,
            weight_caps=request.cluster_weight_caps,
        )


class TargetVolCapStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        if request.target_vol is None:
            return targets
        if not request.constraint_boundary.is_sizing_time("target_vol"):
            return targets
        weighted_risk_terms = [
            float(item.target_weight)
            * max(float(request.risk_by_subject.get(item.subject_id, 0.0)), 0.0)
            for item in targets
        ]
        estimated_vol = math.sqrt(sum(value * value for value in weighted_risk_terms))
        target_vol = max(float(request.target_vol), 0.0)
        if estimated_vol <= 0.0 or estimated_vol <= target_vol:
            return targets
        scale = target_vol / estimated_vol
        return [
            _target_with_weight(
                item,
                float(item.target_weight) * scale,
                entry_allowed=bool(item.entry_allowed),
            )
            for item in targets
        ]


class GrossExposureCapStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        effective_gross_cap = request.gross_leverage_cap
        if (
            effective_gross_cap is None
            and request.constraint_boundary.is_post_sizing_normalization("gross_exposure_cap")
        ):
            effective_gross_cap = request.gross_exposure_cap
        if effective_gross_cap is None:
            return targets
        gross_cap = max(float(effective_gross_cap), 0.0)
        gross_weight = _gross_weight(targets)
        if gross_weight <= 0.0 or gross_weight <= gross_cap:
            return targets
        scale = gross_cap / gross_weight
        return [
            _target_with_weight(
                item,
                float(item.target_weight) * scale,
                entry_allowed=bool(item.entry_allowed),
            )
            for item in targets
        ]


class NetExposureTargetStage:
    def apply(
        self,
        targets: list[PortfolioTarget],
        request: PortfolioConstructionRequest,
    ) -> list[PortfolioTarget]:
        if request.direction_mode == "long_only":
            return targets
        if request.net_exposure_target is None or not targets:
            return targets
        if not request.constraint_boundary.is_post_sizing_normalization("net_exposure_target"):
            return targets
        return apply_net_exposure_target_without_relevering(
            targets,
            net_exposure_target=float(request.net_exposure_target),
        )


def apply_group_weight_caps(
    targets: list[PortfolioTarget],
    *,
    group_by_subject: dict[str, str],
    weight_caps: dict[str, float],
) -> list[PortfolioTarget]:
    if not targets or not group_by_subject or not weight_caps:
        return targets
    adjusted = list(targets)
    subject_index = {item.subject_id: index for index, item in enumerate(adjusted)}
    for group_name, cap_value in weight_caps.items():
        group_subject_ids = [
            item.subject_id
            for item in adjusted
            if group_by_subject.get(item.subject_id) == group_name
        ]
        if not group_subject_ids:
            continue
        gross_weight = sum(
            abs(float(adjusted[subject_index[subject_id]].target_weight))
            for subject_id in group_subject_ids
        )
        cap = max(float(cap_value), 0.0)
        if gross_weight <= 0.0 or gross_weight <= cap:
            continue
        scale = cap / gross_weight
        for subject_id in group_subject_ids:
            index = subject_index[subject_id]
            item = adjusted[index]
            adjusted[index] = replace(
                item,
                target_weight=float(item.target_weight) * scale,
            )
    return adjusted


def apply_net_exposure_target_without_relevering(
    targets: list[PortfolioTarget],
    *,
    net_exposure_target: float,
) -> list[PortfolioTarget]:
    total_weight = sum(float(item.target_weight) for item in targets)
    delta = float(net_exposure_target) - total_weight
    if abs(delta) <= 1e-12:
        return targets
    adjusted = list(targets)
    if delta > 0.0:
        return _reduce_negative_exposure(adjusted, amount=delta)
    return _reduce_positive_exposure(adjusted, amount=abs(delta))


def _gross_weight(targets: list[PortfolioTarget]) -> float:
    return sum(abs(float(item.target_weight)) for item in targets)


def _stage_trace(
    stage: PortfolioConstructionStage,
    *,
    before: list[PortfolioTarget],
    after: list[PortfolioTarget],
) -> PortfolioConstructionStageTrace:
    before_snapshot = _stage_snapshot(before)
    after_snapshot = _stage_snapshot(after)
    return PortfolioConstructionStageTrace(
        stage_name=_stage_name(stage),
        before=before_snapshot,
        after=after_snapshot,
        gross_delta=after_snapshot.gross_exposure - before_snapshot.gross_exposure,
        net_delta=after_snapshot.net_exposure - before_snapshot.net_exposure,
        active_count_delta=(
            after_snapshot.active_position_count
            - before_snapshot.active_position_count
        ),
        changed_subject_count=_changed_subject_count(before, after),
    )


def _stage_snapshot(
    targets: list[PortfolioTarget],
) -> PortfolioConstructionStageSnapshot:
    weights = [float(item.target_weight) for item in targets]
    long_exposure = sum(weight for weight in weights if weight > 0.0)
    short_exposure = sum(abs(weight) for weight in weights if weight < 0.0)
    return PortfolioConstructionStageSnapshot(
        gross_exposure=sum(abs(weight) for weight in weights),
        net_exposure=sum(weights),
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        effective_n=portfolio_effective_n(weights),
        top3_gross_share=top_n_gross_share(weights, top_n=3),
        active_position_count=sum(1 for weight in weights if abs(weight) > 0.001),
    )


def _stage_name(stage: PortfolioConstructionStage) -> str:
    if isinstance(stage, GroupWeightCapStage):
        return f"{stage.group_field}_weight_cap"
    if isinstance(stage, DirectionModeStage):
        return "direction_mode"
    if isinstance(stage, ActiveWeightBudgetStage):
        return "active_weight_budget"
    if isinstance(stage, TopKStage):
        return "top_k"
    if isinstance(stage, TargetVolCapStage):
        return "target_vol_cap"
    if isinstance(stage, GrossExposureCapStage):
        return "gross_exposure_cap"
    if isinstance(stage, NetExposureTargetStage):
        return "net_exposure_target"
    return stage.__class__.__name__


def _changed_subject_count(
    before: list[PortfolioTarget],
    after: list[PortfolioTarget],
) -> int:
    before_by_subject = {item.subject_id: item for item in before}
    after_by_subject = {item.subject_id: item for item in after}
    subject_ids = set(before_by_subject) | set(after_by_subject)
    changed = 0
    for subject_id in subject_ids:
        before_item = before_by_subject.get(subject_id)
        after_item = after_by_subject.get(subject_id)
        before_weight = 0.0 if before_item is None else float(before_item.target_weight)
        after_weight = 0.0 if after_item is None else float(after_item.target_weight)
        before_entry = False if before_item is None else bool(before_item.entry_allowed)
        after_entry = False if after_item is None else bool(after_item.entry_allowed)
        if abs(after_weight - before_weight) > 1e-12 or after_entry != before_entry:
            changed += 1
    return changed


def _reduce_negative_exposure(
    targets: list[PortfolioTarget],
    *,
    amount: float,
) -> list[PortfolioTarget]:
    short_total = sum(abs(float(item.target_weight)) for item in targets if item.target_weight < 0.0)
    if short_total <= 0.0:
        return targets
    reduction = min(float(amount), short_total)
    scale = 1.0 - reduction / short_total
    return [
        _target_with_weight(
            item,
            float(item.target_weight) * scale if item.target_weight < 0.0 else item.target_weight,
            entry_allowed=bool(item.entry_allowed),
        )
        for item in targets
    ]


def _reduce_positive_exposure(
    targets: list[PortfolioTarget],
    *,
    amount: float,
) -> list[PortfolioTarget]:
    long_total = sum(float(item.target_weight) for item in targets if item.target_weight > 0.0)
    if long_total <= 0.0:
        return targets
    reduction = min(float(amount), long_total)
    scale = 1.0 - reduction / long_total
    return [
        _target_with_weight(
            item,
            float(item.target_weight) * scale if item.target_weight > 0.0 else item.target_weight,
            entry_allowed=bool(item.entry_allowed),
        )
        for item in targets
    ]


def _finalize_targets(
    targets: list[PortfolioTarget],
    request: PortfolioConstructionRequest,
) -> dict[str, PortfolioTarget]:
    return {
        item.subject_id: PortfolioTarget(
            subject_id=item.subject_id,
            target_weight=float(item.target_weight),
            position_delta=float(item.target_weight - request.current_weights.get(item.subject_id, 0.0)),
            target_notional=float(item.target_weight * request.capital_base),
            entry_allowed=bool(item.entry_allowed),
            risk_scale=float(item.risk_scale),
        )
        for item in targets
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
