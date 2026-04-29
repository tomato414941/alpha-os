from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .belief_synthesis import (
    build_signal_contributions,
    synthesize_beliefs,
)


@dataclass(frozen=True, init=False)
class CompressedBeliefComponent:
    subject_id: str
    target_id: str
    belief_value: float
    confidence: float
    signal_contribution_count: int
    family_ids: tuple[str, ...]
    signal_ids: tuple[str, ...]
    representative_signal_ids: tuple[str, ...] = ()
    regime_tags: tuple[str, ...] = ()
    family_count: int = 0
    cluster_count: int = 0
    effective_belief_count: float = 1.0
    diversity_score: float = 1.0
    mean_marginal_signal_contribution: float = 0.0

    def __init__(
        self,
        *,
        subject_id: str,
        target_id: str,
        belief_value: float,
        confidence: float,
        signal_contribution_count: int,
        family_ids: tuple[str, ...],
        signal_ids: tuple[str, ...],
        representative_signal_ids: tuple[str, ...] = (),
        regime_tags: tuple[str, ...] = (),
        family_count: int = 0,
        cluster_count: int = 0,
        effective_belief_count: float = 1.0,
        diversity_score: float = 1.0,
        mean_marginal_signal_contribution: float = 0.0,
    ) -> None:
        resolved_signal_ids = signal_ids
        if resolved_signal_ids is None:
            raise ValueError("compressed belief component requires signal_ids")
        resolved_representatives = (
            representative_signal_ids
            if representative_signal_ids
            else (
                ()
                if representative_signal_ids is None
                else representative_signal_ids
            )
        )
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "belief_value", belief_value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "signal_contribution_count", signal_contribution_count
        )
        object.__setattr__(self, "family_ids", family_ids)
        object.__setattr__(self, "signal_ids", tuple(resolved_signal_ids))
        object.__setattr__(
            self,
            "representative_signal_ids",
            tuple(resolved_representatives),
        )
        object.__setattr__(self, "regime_tags", regime_tags)
        object.__setattr__(self, "family_count", family_count)
        object.__setattr__(self, "cluster_count", cluster_count)
        object.__setattr__(self, "effective_belief_count", effective_belief_count)
        object.__setattr__(self, "diversity_score", diversity_score)
        object.__setattr__(
            self,
            "mean_marginal_signal_contribution",
            mean_marginal_signal_contribution,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "belief_value": self.belief_value,
            "confidence": self.confidence,
            "signal_contribution_count": self.signal_contribution_count,
            "family_ids": list(self.family_ids),
            "signal_ids": list(self.signal_ids),
            "representative_signal_ids": list(self.representative_signal_ids),
            "regime_tags": list(self.regime_tags),
            "family_count": self.family_count,
            "cluster_count": self.cluster_count,
            "effective_belief_count": self.effective_belief_count,
            "diversity_score": self.diversity_score,
            "mean_marginal_signal_contribution": self.mean_marginal_signal_contribution,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CompressedBeliefComponent":
        family_ids = tuple(str(item) for item in document.get("family_ids", []))
        signal_ids = tuple(
            str(item)
            for item in document.get(
                "signal_ids",
                document.get("signal_ids", []),
            )
        )
        default_family_count = len(family_ids)
        if default_family_count == 0 and signal_ids:
            default_family_count = 1
        return cls(
            subject_id=str(document["subject_id"]),
            target_id=str(document["target_id"]),
            belief_value=float(document["belief_value"]),
            confidence=float(document["confidence"]),
            signal_contribution_count=int(document["signal_contribution_count"]),
            family_ids=family_ids,
            signal_ids=signal_ids,
            representative_signal_ids=tuple(
                str(item)
                for item in document.get(
                    "representative_signal_ids",
                    document.get("representative_signal_ids", []),
                )
            ),
            regime_tags=tuple(str(item) for item in document.get("regime_tags", [])),
            family_count=int(document.get("family_count", default_family_count)),
            cluster_count=int(document.get("cluster_count", default_family_count)),
            effective_belief_count=float(
                document.get("effective_belief_count", float(default_family_count or 1))
            ),
            diversity_score=float(document.get("diversity_score", 1.0)),
            mean_marginal_signal_contribution=float(
                document.get("mean_marginal_signal_contribution", 0.0)
            ),
        )


@dataclass(frozen=True)
class CompressedBelief:
    compressed_belief_id: str
    signal_discovery_id: str
    screening_result_id: str
    components: tuple[CompressedBeliefComponent, ...]
    created_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_discovery_id": self.signal_discovery_id,
            "screening_result_id": self.screening_result_id,
            "components": [item.to_document() for item in self.components],
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        compressed_belief_id: str,
        document: dict[str, Any],
    ) -> "CompressedBelief":
        return cls(
            compressed_belief_id=compressed_belief_id,
            signal_discovery_id=str(document["signal_discovery_id"]),
            screening_result_id=str(document["screening_result_id"]),
            components=tuple(
                CompressedBeliefComponent.from_document(item)
                for item in document.get("components", [])
                if isinstance(item, dict)
            ),
            created_at=str(document["created_at"]),
        )


def compress_screening_result(
    *,
    signal_discovery_id: str,
    screening_result_id: str,
    survivors,
    prediction_values_by_signal_id: dict[str, float] | None = None,
    created_at: str,
    strategy_adaptation_state=None,
    adaptation_blend: float = 0.2,
) -> CompressedBelief:
    signal_contributions = build_signal_contributions(
        survivors=survivors,
        prediction_values_by_signal_id=prediction_values_by_signal_id,
    )
    synthesis = synthesize_beliefs(
        signal_discovery_id=signal_discovery_id,
        screening_result_id=screening_result_id,
        signal_contributions=signal_contributions,
        created_at=created_at,
        strategy_adaptation_state=strategy_adaptation_state,
        adaptation_blend=adaptation_blend,
    )
    return CompressedBelief(
        compressed_belief_id=f"{screening_result_id}:compressed",
        signal_discovery_id=synthesis.signal_discovery_id,
        screening_result_id=synthesis.screening_result_id,
        components=tuple(
            CompressedBeliefComponent(
                subject_id=item.subject_id,
                target_id=item.target_id,
                belief_value=item.belief_value,
                confidence=item.confidence,
                signal_contribution_count=item.signal_contribution_count,
                family_ids=item.family_ids,
                signal_ids=item.signal_ids,
                representative_signal_ids=item.representative_signal_ids,
                regime_tags=item.regime_tags,
                family_count=item.family_count,
                cluster_count=item.cluster_count,
                effective_belief_count=item.effective_belief_count,
                diversity_score=item.diversity_score,
                mean_marginal_signal_contribution=(
                    item.mean_marginal_signal_contribution
                ),
            )
            for item in synthesis.components
        ),
        created_at=synthesis.created_at,
    )
