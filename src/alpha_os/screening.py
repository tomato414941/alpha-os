from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ScreeningPolicy:
    min_sample_count: int = 10
    min_abs_corr: float = 0.05
    min_stability_score: float = 0.0
    adaptive_family_budget: bool = True
    adaptive_budget_stability_scale: float = 2.0
    max_family_survivors_per_subject: int = 2

    def to_document(self) -> dict[str, Any]:
        return {
            "min_sample_count": self.min_sample_count,
            "min_abs_corr": self.min_abs_corr,
            "min_stability_score": self.min_stability_score,
            "adaptive_family_budget": self.adaptive_family_budget,
            "adaptive_budget_stability_scale": self.adaptive_budget_stability_scale,
            "max_family_survivors_per_subject": self.max_family_survivors_per_subject,
        }


@dataclass(frozen=True)
class ScreeningCandidateResult:
    signal_id: str
    specification_signal_id: str | None
    family_id: str | None
    subject_id: str
    target_id: str
    kind: str | None
    lookback: int | None
    score: float
    corr: float | None
    stability_score: float
    sample_count: int
    keep: bool
    family_rank: int | None
    reasons: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "specification_signal_id": self.specification_signal_id,
            "family_id": self.family_id,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "kind": self.kind,
            "lookback": self.lookback,
            "score": self.score,
            "corr": self.corr,
            "stability_score": self.stability_score,
            "sample_count": self.sample_count,
            "keep": self.keep,
            "family_rank": self.family_rank,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ScreeningCandidateResult":
        reasons = document.get("reasons", [])
        return cls(
            signal_id=str(document["signal_id"]),
            specification_signal_id=(
                None
                if (
                    document.get("specification_signal_id") is None
                )
                else str(
                    document.get("specification_signal_id")
                )
            ),
            family_id=None if document.get("family_id") is None else str(document["family_id"]),
            subject_id=str(document["subject_id"]),
            target_id=str(document["target_id"]),
            kind=None if document.get("kind") is None else str(document["kind"]),
            lookback=None
            if document.get("lookback") is None
            else int(document["lookback"]),
            score=float(document["score"]),
            corr=None if document.get("corr") is None else float(document["corr"]),
            stability_score=float(document.get("stability_score", 0.0)),
            sample_count=int(document["sample_count"]),
            keep=bool(document["keep"]),
            family_rank=None
            if document.get("family_rank") is None
            else int(document["family_rank"]),
            reasons=tuple(str(item) for item in reasons),
        )


@dataclass(frozen=True)
class ScreeningResult:
    screening_result_id: str
    signal_discovery_id: str
    policy: ScreeningPolicy
    candidates: tuple[ScreeningCandidateResult, ...]
    created_at: str

    @property
    def survivors(self) -> tuple[ScreeningCandidateResult, ...]:
        return tuple(item for item in self.candidates if item.keep)

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_discovery_id": self.signal_discovery_id,
            "policy": self.policy.to_document(),
            "candidates": [item.to_document() for item in self.candidates],
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        screening_result_id: str,
        document: dict[str, Any],
    ) -> "ScreeningResult":
        policy_document = document.get("policy", {})
        return cls(
            screening_result_id=screening_result_id,
            signal_discovery_id=str(document["signal_discovery_id"]),
            policy=ScreeningPolicy(
                min_sample_count=int(policy_document.get("min_sample_count", 10)),
                min_abs_corr=float(policy_document.get("min_abs_corr", 0.05)),
                min_stability_score=float(policy_document.get("min_stability_score", 0.0)),
                adaptive_family_budget=bool(
                    policy_document.get("adaptive_family_budget", True)
                ),
                adaptive_budget_stability_scale=float(
                    policy_document.get("adaptive_budget_stability_scale", 2.0)
                ),
                max_family_survivors_per_subject=int(
                    policy_document.get("max_family_survivors_per_subject", 2)
                ),
            ),
            candidates=tuple(
                ScreeningCandidateResult.from_document(item)
                for item in document.get("candidates", [])
                if isinstance(item, dict)
            ),
            created_at=str(document["created_at"]),
        )


def screen_signals(
    *,
    signals,
    metrics_by_id: dict[str, Any],
    signal_discovery_id: str,
    policy: ScreeningPolicy,
    family_ids_by_signal_spec_id: dict[str, str] | None = None,
    family_budgets_by_family_id: dict[str, int] | None = None,
    created_at: str,
) -> ScreeningResult:
    ranked_candidates: list[ScreeningCandidateResult] = []
    family_groups: dict[tuple[str, str, str], list[ScreeningCandidateResult]] = {}
    resolved_family_ids = family_ids_by_signal_spec_id or {}
    resolved_family_budgets = family_budgets_by_family_id or {}

    for signal in signals:
        metric = metrics_by_id.get(signal.signal_id)
        reasons: list[str] = []
        corr = None if metric is None else float(metric.corr)
        sample_count = 0 if metric is None else int(metric.sample_count)
        score = 0.0 if corr is None else abs(corr)
        stability_score = score * math.sqrt(float(sample_count))
        if metric is None:
            reasons.append("cheap_missing_metric")
        if sample_count < policy.min_sample_count:
            reasons.append("cheap_insufficient_samples")
        if corr is None or abs(corr) < policy.min_abs_corr:
            reasons.append("cheap_weak_signal")
        if stability_score < policy.min_stability_score:
            reasons.append("stability_weak_signal")
        signal_spec_id = signal.signal_spec_id
        family_id = (
            None
            if signal_spec_id is None
            else resolved_family_ids.get(signal_spec_id)
        )
        candidate = ScreeningCandidateResult(
            signal_id=signal.signal_id,
            specification_signal_id=signal_spec_id,
            family_id=family_id,
            subject_id=signal.subject_id,
            target_id=signal.target_id,
            kind=signal.kind,
            lookback=signal.lookback,
            score=score,
            corr=corr,
            stability_score=stability_score,
            sample_count=sample_count,
            keep=False,
            family_rank=None,
            reasons=tuple(reasons),
        )
        family_key = (
            signal.subject_id,
            family_id or signal.kind or "-",
            signal.target_id,
        )
        family_groups.setdefault(family_key, []).append(candidate)

    for (_, group_family_id, _), candidates in family_groups.items():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.score,
                -item.sample_count,
                item.lookback if item.lookback is not None else 10**9,
                item.signal_id,
            ),
        )
        survivors_used = 0
        family_budget_ceiling = policy.max_family_survivors_per_subject
        if group_family_id != "-":
            family_budget_ceiling = resolved_family_budgets.get(
                group_family_id,
                family_budget_ceiling,
            )
        family_budget = _resolved_family_budget(
            ordered,
            policy=policy,
            family_budget_ceiling=family_budget_ceiling,
        )
        for rank, candidate in enumerate(ordered, start=1):
            reasons = list(candidate.reasons)
            keep = not reasons and survivors_used < family_budget
            if not keep and not reasons:
                if family_budget < family_budget_ceiling:
                    reasons.append("redundancy_adaptive_cap")
                else:
                    reasons.append("redundancy_family_cap")
            if keep:
                survivors_used += 1
            ranked_candidates.append(
                ScreeningCandidateResult(
                    signal_id=candidate.signal_id,
                    specification_signal_id=candidate.specification_signal_id,
                    family_id=candidate.family_id,
                    subject_id=candidate.subject_id,
                    target_id=candidate.target_id,
                    kind=candidate.kind,
                    lookback=candidate.lookback,
                    score=candidate.score,
                    corr=candidate.corr,
                    stability_score=candidate.stability_score,
                    sample_count=candidate.sample_count,
                    keep=keep,
                    family_rank=rank,
                    reasons=tuple(reasons),
                )
            )

    ordered_candidates = tuple(
        sorted(
            ranked_candidates,
            key=lambda item: (
                item.subject_id,
                item.kind or "",
                -int(item.keep),
                -(item.score),
                item.signal_id,
            ),
        )
    )
    return ScreeningResult(
        screening_result_id=f"{signal_discovery_id}:{created_at}",
        signal_discovery_id=signal_discovery_id,
        policy=policy,
        candidates=ordered_candidates,
        created_at=created_at,
    )


def _resolved_family_budget(
    candidates: list[ScreeningCandidateResult],
    *,
    policy: ScreeningPolicy,
    family_budget_ceiling: int,
) -> int:
    eligible = [item for item in candidates if not item.reasons]
    if not eligible:
        return 0
    effective_ceiling = min(family_budget_ceiling, len(eligible))
    if effective_ceiling <= 1 or not policy.adaptive_family_budget:
        return effective_ceiling
    mean_stability = sum(item.stability_score for item in eligible) / float(len(eligible))
    stability_threshold = max(policy.min_stability_score, 1e-9)
    target_stability = max(
        stability_threshold * policy.adaptive_budget_stability_scale,
        stability_threshold,
    )
    if target_stability <= 0.0:
        return effective_ceiling
    quality_ratio = min(1.0, mean_stability / target_stability)
    adaptive_budget = int(math.ceil(effective_ceiling * quality_ratio))
    return max(1, min(effective_ceiling, adaptive_budget))

