from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation_report import EvaluationTaskResult
from .screening import ScreeningResult


@dataclass(frozen=True)
class SignalReputation:
    signal_id: str
    family_id: str
    subject_id: str
    target_id: str
    orientation: int
    edge_score: float
    mmc: float | None
    contribution_score: float
    confidence: float
    stability_score: float
    sample_count: int
    update_count: int
    updated_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "family_id": self.family_id,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "orientation": self.orientation,
            "edge_score": self.edge_score,
            "mmc": self.mmc,
            "contribution_score": self.contribution_score,
            "confidence": self.confidence,
            "stability_score": self.stability_score,
            "sample_count": self.sample_count,
            "update_count": self.update_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SignalReputation":
        return cls(
            signal_id=str(document["signal_id"]),
            family_id=str(document["family_id"]),
            subject_id=str(document["subject_id"]),
            target_id=str(document["target_id"]),
            orientation=int(document.get("orientation", 1)),
            edge_score=float(document.get("edge_score", 0.0)),
            mmc=None if document.get("mmc") is None else float(document["mmc"]),
            contribution_score=float(document.get("contribution_score", 0.0)),
            confidence=float(document.get("confidence", 0.0)),
            stability_score=float(document.get("stability_score", 0.0)),
            sample_count=int(document.get("sample_count", 0)),
            update_count=int(document.get("update_count", 1)),
            updated_at=str(document["updated_at"]),
        )


@dataclass(frozen=True)
class FamilyReputation:
    family_id: str
    mean_edge_score: float
    mean_confidence: float
    mean_stability_score: float
    subject_coverage: int
    member_count: int
    update_count: int
    updated_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "mean_edge_score": self.mean_edge_score,
            "mean_confidence": self.mean_confidence,
            "mean_stability_score": self.mean_stability_score,
            "subject_coverage": self.subject_coverage,
            "member_count": self.member_count,
            "update_count": self.update_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "FamilyReputation":
        return cls(
            family_id=str(document["family_id"]),
            mean_edge_score=float(document.get("mean_edge_score", 0.0)),
            mean_confidence=float(document.get("mean_confidence", 0.0)),
            mean_stability_score=float(document.get("mean_stability_score", 0.0)),
            subject_coverage=int(document.get("subject_coverage", 0)),
            member_count=int(document.get("member_count", 0)),
            update_count=int(document.get("update_count", 1)),
            updated_at=str(document["updated_at"]),
        )


@dataclass(frozen=True)
class StrategyAdaptationState:
    strategy_id: str
    signal_train_id: str
    signal_discovery_id: str | None
    source_evaluation_report_id: str
    source_screening_result_id: str
    signal_reputations: tuple[SignalReputation, ...]
    family_reputations: tuple[FamilyReputation, ...]
    created_at: str

    def to_document(self) -> dict[str, Any]:
        document = {
            "strategy_id": self.strategy_id,
            "signal_train_id": self.signal_train_id,
            "source_evaluation_report_id": self.source_evaluation_report_id,
            "source_screening_result_id": self.source_screening_result_id,
            "signal_reputations": [
                item.to_document() for item in self.signal_reputations
            ],
            "family_reputations": [
                item.to_document() for item in self.family_reputations
            ],
            "created_at": self.created_at,
        }
        if self.signal_discovery_id is not None:
            document["signal_discovery_id"] = self.signal_discovery_id
        return document

    @classmethod
    def from_document(
        cls,
        *,
        strategy_id: str,
        document: dict[str, Any],
    ) -> "StrategyAdaptationState":
        signal_discovery_id_document = document.get("signal_discovery_id")
        signal_discovery_id = (
            None if signal_discovery_id_document is None else str(signal_discovery_id_document)
        )
        signal_train_id = str(
            document.get(
                "signal_train_id",
                f"signal-train:{strategy_id if signal_discovery_id is None else signal_discovery_id}",
            )
        )
        return cls(
            strategy_id=strategy_id,
            signal_train_id=signal_train_id,
            signal_discovery_id=signal_discovery_id,
            source_evaluation_report_id=str(document["source_evaluation_report_id"]),
            source_screening_result_id=str(document["source_screening_result_id"]),
            signal_reputations=tuple(
                SignalReputation.from_document(item)
                for item in document.get("signal_reputations", [])
                if isinstance(item, dict)
            ),
            family_reputations=tuple(
                FamilyReputation.from_document(item)
                for item in document.get("family_reputations", [])
                if isinstance(item, dict)
            ),
            created_at=str(document["created_at"]),
        )


def build_strategy_adaptation_state(
    *,
    evaluation_report_id: str,
    task_result: EvaluationTaskResult,
    screening_result: ScreeningResult,
    metrics_by_signal_id: dict[str, Any] | None,
    previous_state: StrategyAdaptationState | None,
    smoothing: float,
    created_at: str,
) -> StrategyAdaptationState:
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    previous_by_signal: dict[str, SignalReputation] = {}
    previous_family_by_id: dict[str, FamilyReputation] = {}
    if previous_state is not None:
        previous_by_signal = {
            item.signal_id: item
            for item in previous_state.signal_reputations
        }
        previous_family_by_id = {
            item.family_id: item for item in previous_state.family_reputations
        }

    signal_reputations = tuple(
        sorted(
            (
                _build_signal_reputation(
                    candidate=item,
                    metric=(
                        None
                        if metrics_by_signal_id is None
                        else metrics_by_signal_id.get(item.signal_id)
                    ),
                    previous=previous_by_signal.get(item.signal_id),
                    smoothing=smoothing,
                    created_at=created_at,
                )
                for item in screening_result.survivors
            ),
            key=lambda item: (item.subject_id, item.family_id, item.signal_id),
        )
    )
    family_reputations = tuple(
        _build_family_reputations(
            signal_reputations=signal_reputations,
            previous_family_by_id=previous_family_by_id,
            smoothing=smoothing,
            created_at=created_at,
        )
    )
    screening_result_ids = task_result.artifact_refs.get("screening_result_ids", ())
    return StrategyAdaptationState(
        strategy_id=task_result.strategy_id,
        signal_train_id=(
            str(task_result.artifact_refs["signal_train_ids"][0])
            if task_result.artifact_refs.get("signal_train_ids")
            else f"signal-train:{screening_result.signal_discovery_id}"
        ),
        signal_discovery_id=(
            screening_result.signal_discovery_id
            if task_result.signal_discovery_id is None
            else task_result.signal_discovery_id
        ),
        source_evaluation_report_id=evaluation_report_id,
        source_screening_result_id=(
            screening_result.screening_result_id
            if not screening_result_ids
            else str(screening_result_ids[0])
        ),
        signal_reputations=signal_reputations,
        family_reputations=family_reputations,
        created_at=created_at,
    )


def _build_signal_reputation(
    *,
    candidate,
    metric,
    previous: SignalReputation | None,
    smoothing: float,
    created_at: str,
) -> SignalReputation:
    corr = 0.0 if candidate.corr is None else float(candidate.corr)
    mmc = None if metric is None or metric.mmc is None else float(metric.mmc)
    orientation = 1 if corr >= 0.0 else -1
    edge_score = abs(corr)
    oriented_mmc = None if mmc is None else (mmc if orientation >= 0 else -mmc)
    contribution_score = max(0.0, 0.5 * edge_score) + max(
        0.0,
        0.5 * (0.0 if oriented_mmc is None else oriented_mmc),
    )
    confidence = min(1.0, edge_score * (max(int(candidate.sample_count), 1) ** 0.5))
    stability_score = float(candidate.stability_score)
    sample_count = int(candidate.sample_count)
    if previous is not None:
        edge_score = _smoothed(previous.edge_score, edge_score, smoothing=smoothing)
        if previous.mmc is not None or mmc is not None:
            mmc = _smoothed(
                0.0 if previous.mmc is None else previous.mmc,
                0.0 if mmc is None else mmc,
                smoothing=smoothing,
            )
        contribution_score = _smoothed(
            previous.contribution_score,
            contribution_score,
            smoothing=smoothing,
        )
        confidence = _smoothed(previous.confidence, confidence, smoothing=smoothing)
        stability_score = _smoothed(
            previous.stability_score,
            stability_score,
            smoothing=smoothing,
        )
        sample_count = max(previous.sample_count, sample_count)
        update_count = previous.update_count + 1
    else:
        update_count = 1
    return SignalReputation(
        signal_id=candidate.signal_id,
        family_id=candidate.family_id or candidate.kind or "-",
        subject_id=candidate.subject_id,
        target_id=candidate.target_id,
        orientation=orientation,
        edge_score=edge_score,
        mmc=mmc,
        contribution_score=contribution_score,
        confidence=confidence,
        stability_score=stability_score,
        sample_count=sample_count,
        update_count=update_count,
        updated_at=created_at,
    )


def _build_family_reputations(
    *,
    signal_reputations: tuple[SignalReputation, ...],
    previous_family_by_id: dict[str, FamilyReputation],
    smoothing: float,
    created_at: str,
) -> list[FamilyReputation]:
    family_groups: dict[str, list[SignalReputation]] = {}
    for item in signal_reputations:
        family_groups.setdefault(item.family_id, []).append(item)
    families: list[FamilyReputation] = []
    for family_id, items in sorted(family_groups.items()):
        mean_edge_score = _mean([item.edge_score for item in items])
        mean_confidence = _mean([item.confidence for item in items])
        mean_stability_score = _mean([item.stability_score for item in items])
        subject_coverage = len({item.subject_id for item in items})
        member_count = len(items)
        previous = previous_family_by_id.get(family_id)
        if previous is not None:
            mean_edge_score = _smoothed(
                previous.mean_edge_score,
                mean_edge_score,
                smoothing=smoothing,
            )
            mean_confidence = _smoothed(
                previous.mean_confidence,
                mean_confidence,
                smoothing=smoothing,
            )
            mean_stability_score = _smoothed(
                previous.mean_stability_score,
                mean_stability_score,
                smoothing=smoothing,
            )
            subject_coverage = max(previous.subject_coverage, subject_coverage)
            member_count = max(previous.member_count, member_count)
            update_count = previous.update_count + 1
        else:
            update_count = 1
        families.append(
            FamilyReputation(
                family_id=family_id,
                mean_edge_score=mean_edge_score,
                mean_confidence=mean_confidence,
                mean_stability_score=mean_stability_score,
                subject_coverage=subject_coverage,
                member_count=member_count,
                update_count=update_count,
                updated_at=created_at,
            )
        )
    return families


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _smoothed(previous: float, current: float, *, smoothing: float) -> float:
    return float((smoothing * previous) + ((1.0 - smoothing) * current))
