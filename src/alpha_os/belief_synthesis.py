from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .strategy_adaptation import StrategyAdaptationState
from .strategy_adaptation_weighting import build_strategy_adaptation_signal_weights


@dataclass(frozen=True)
class SignalContribution:
    subject_id: str
    target_id: str
    signal_id: str
    family_id: str
    prediction_value: float
    oriented_prediction: float
    confidence: float
    marginal_signal_contribution: float
    stability_score: float
    sample_count: int
    regime_tags: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "signal_id": self.signal_id,
            "family_id": self.family_id,
            "prediction_value": self.prediction_value,
            "oriented_prediction": self.oriented_prediction,
            "confidence": self.confidence,
            "marginal_signal_contribution": self.marginal_signal_contribution,
            "stability_score": self.stability_score,
            "sample_count": self.sample_count,
            "regime_tags": list(self.regime_tags),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SignalContribution":
        return cls(
            subject_id=str(document["subject_id"]),
            target_id=str(document["target_id"]),
            signal_id=str(document["signal_id"]),
            family_id=str(document["family_id"]),
            prediction_value=float(document["prediction_value"]),
            oriented_prediction=float(document["oriented_prediction"]),
            confidence=float(document["confidence"]),
            marginal_signal_contribution=float(
                document["marginal_signal_contribution"]
            ),
            stability_score=float(document["stability_score"]),
            sample_count=int(document["sample_count"]),
            regime_tags=tuple(str(item) for item in document.get("regime_tags", [])),
        )


@dataclass(frozen=True)
class BeliefSynthesisComponent:
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
    def from_document(cls, document: dict[str, Any]) -> "BeliefSynthesisComponent":
        family_ids = tuple(str(item) for item in document.get("family_ids", []))
        signal_ids = tuple(
            str(item)
            for item in document.get(
                "signal_ids",
                [],
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
                    [],
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
class BeliefSynthesisResult:
    belief_synthesis_id: str
    signal_discovery_id: str
    screening_result_id: str
    signal_contributions: tuple[SignalContribution, ...]
    components: tuple[BeliefSynthesisComponent, ...]
    created_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_discovery_id": self.signal_discovery_id,
            "screening_result_id": self.screening_result_id,
            "signal_contributions": [
                item.to_document() for item in self.signal_contributions
            ],
            "components": [item.to_document() for item in self.components],
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        belief_synthesis_id: str,
        document: dict[str, Any],
    ) -> "BeliefSynthesisResult":
        return cls(
            belief_synthesis_id=belief_synthesis_id,
            signal_discovery_id=str(document["signal_discovery_id"]),
            screening_result_id=str(document["screening_result_id"]),
            signal_contributions=tuple(
                SignalContribution.from_document(item)
                for item in document.get("signal_contributions", [])
                if isinstance(item, dict)
            ),
            components=tuple(
                BeliefSynthesisComponent.from_document(item)
                for item in document.get("components", [])
                if isinstance(item, dict)
            ),
            created_at=str(document["created_at"]),
        )


def build_signal_contributions(
    *,
    survivors,
    prediction_values_by_signal_id: dict[str, float] | None = None,
) -> tuple[SignalContribution, ...]:
    resolved_prediction_values = prediction_values_by_signal_id
    if resolved_prediction_values is None:
        raise ValueError("build_signal_contributions requires prediction values")
    signal_contributions: list[SignalContribution] = []
    for item in survivors:
        prediction_value = resolved_prediction_values.get(item.signal_id)
        if prediction_value is None:
            continue
        corr = 0.0 if item.corr is None else float(item.corr)
        oriented_prediction = float(prediction_value)
        if corr < 0.0:
            oriented_prediction = -oriented_prediction
        confidence = abs(corr) if item.corr is not None else float(item.score)
        signal_contributions.append(
            SignalContribution(
                subject_id=item.subject_id,
                target_id=item.target_id,
                signal_id=item.signal_id,
                family_id=item.family_id or item.signal_id,
                prediction_value=float(prediction_value),
                oriented_prediction=float(oriented_prediction),
                confidence=float(confidence),
                marginal_signal_contribution=float(
                    confidence
                    * (0.5 + 0.5 * min(float(item.sample_count) / 20.0, 1.0))
                ),
                stability_score=float(item.stability_score),
                sample_count=int(item.sample_count),
                regime_tags=_regime_tags_for_candidate(item),
            )
        )
    return tuple(signal_contributions)


def synthesize_beliefs(
    *,
    signal_discovery_id: str,
    screening_result_id: str,
    signal_contributions: tuple[SignalContribution, ...],
    created_at: str,
    strategy_adaptation_state: StrategyAdaptationState | None = None,
    adaptation_blend: float = 0.2,
) -> BeliefSynthesisResult:
    grouped: dict[tuple[str, str], list[SignalContribution]] = {}
    for item in signal_contributions:
        grouped.setdefault((item.subject_id, item.target_id), []).append(item)

    components: list[BeliefSynthesisComponent] = []
    for (subject_id, target_id), items in sorted(grouped.items()):
        family_rows = []
        weighted_contributions_by_family_id: dict[str, list[SignalContribution]] = {}
        for item in items:
            weighted_contributions_by_family_id.setdefault(
                item.family_id, []
            ).append(item)
        for family_id, family_contributions in sorted(
            weighted_contributions_by_family_id.items()
        ):
            ordered_family_contributions = sorted(
                family_contributions,
                key=lambda row: (-row.confidence, row.signal_id),
            )
            signal_weights = build_strategy_adaptation_signal_weights(
                signal_ids=tuple(
                    item.signal_id for item in ordered_family_contributions
                ),
                strategy_adaptation_state=strategy_adaptation_state,
                blend=adaptation_blend,
            )
            total_family_weight = sum(
                item.confidence
                * signal_weights[item.signal_id].blended_multiplier
                for item in ordered_family_contributions
            )
            if total_family_weight <= 0.0:
                total_family_weight = float(len(ordered_family_contributions))
                normalized_family_contributions = [
                    (item, 1.0 / total_family_weight)
                    for item in ordered_family_contributions
                ]
            else:
                normalized_family_contributions = [
                    (
                        item,
                        (
                            item.confidence
                            * signal_weights[item.signal_id].blended_multiplier
                        )
                        / total_family_weight,
                    )
                    for item in ordered_family_contributions
                ]
            family_belief = sum(
                item.oriented_prediction * normalized_weight
                for item, normalized_weight in normalized_family_contributions
            )
            family_confidence = sum(
                item.confidence for item in ordered_family_contributions
            ) / float(len(ordered_family_contributions))
            family_rows.append(
                {
                    "family_id": family_id,
                    "belief_value": float(family_belief),
                    "confidence": float(family_confidence),
                    "representative_signal_id": ordered_family_contributions[
                        0
                    ].signal_id,
                    "signal_ids": tuple(
                        item.signal_id for item in ordered_family_contributions
                    ),
                    "regime_tags": tuple(
                        sorted(
                            {
                                tag
                                for item in ordered_family_contributions
                                for tag in item.regime_tags
                            }
                        )
                    ),
                    "mean_marginal_signal_contribution": float(
                        sum(
                            item.marginal_signal_contribution
                            for item in ordered_family_contributions
                        )
                        / float(len(ordered_family_contributions))
                    ),
                }
            )
        cluster_rows = _cluster_family_rows(
            family_rows,
            strategy_adaptation_state=strategy_adaptation_state,
            adaptation_blend=adaptation_blend,
        )
        total_cluster_weight = sum(float(row["ensemble_weight"]) for row in cluster_rows)
        if total_cluster_weight <= 0.0:
            total_cluster_weight = float(len(cluster_rows))
            normalized_cluster_rows = [
                (
                    str(row["cluster_id"]),
                    float(row["belief_value"]),
                    1.0 / total_cluster_weight,
                    str(row["representative_signal_id"]),
                )
                for row in cluster_rows
            ]
        else:
            normalized_cluster_rows = [
                (
                    str(row["cluster_id"]),
                    float(row["belief_value"]),
                    float(row["ensemble_weight"]) / total_cluster_weight,
                    str(row["representative_signal_id"]),
                )
                for row in cluster_rows
            ]
        belief_value = sum(
            cluster_belief * normalized_weight
            for _, cluster_belief, normalized_weight, _ in normalized_cluster_rows
        )
        normalized_weights = [row[2] for row in normalized_cluster_rows]
        if len(normalized_weights) <= 1:
            effective_belief_count = 1.0
            diversity_score = 1.0
        else:
            effective_belief_count = 1.0 / sum(
                weight * weight for weight in normalized_weights
            )
            diversity_score = (
                effective_belief_count - 1.0
            ) / float(len(normalized_weights) - 1)
        mean_cluster_confidence = sum(
            float(row["confidence"]) for row in cluster_rows
        ) / float(len(cluster_rows))
        mean_marginal_signal_contribution = sum(
            float(row.get("mean_marginal_signal_contribution", 0.0))
            for row in cluster_rows
        ) / float(len(cluster_rows))
        confidence = min(
            mean_cluster_confidence * float(max(diversity_score, 0.0)),
            1.0,
        )
        components.append(
            BeliefSynthesisComponent(
                subject_id=subject_id,
                target_id=target_id,
                belief_value=float(belief_value),
                confidence=float(confidence),
                signal_contribution_count=len(items),
                family_ids=tuple(
                    str(row["family_id"])
                    for row in sorted(family_rows, key=lambda row: str(row["family_id"]))
                ),
                signal_ids=tuple(
                    item.signal_id
                    for item in sorted(
                        items,
                        key=lambda row: (-row.confidence, row.signal_id),
                    )
                ),
                representative_signal_ids=tuple(
                    str(row["representative_signal_id"])
                    for row in sorted(
                        cluster_rows,
                        key=lambda row: (
                            -float(row["confidence"]),
                            str(row["representative_signal_id"]),
                        ),
                    )
                ),
                regime_tags=tuple(
                    sorted(
                        {
                            tag
                            for row in cluster_rows
                            for tag in row.get("regime_tags", ())
                        }
                    )
                ),
                family_count=len(family_rows),
                cluster_count=len(cluster_rows),
                effective_belief_count=float(max(effective_belief_count, 1.0)),
                diversity_score=float(max(diversity_score, 0.0)),
                mean_marginal_signal_contribution=float(
                    mean_marginal_signal_contribution
                ),
            )
        )

    return BeliefSynthesisResult(
        belief_synthesis_id=f"{screening_result_id}:belief_synthesis",
        signal_discovery_id=signal_discovery_id,
        screening_result_id=screening_result_id,
        signal_contributions=signal_contributions,
        components=tuple(components),
        created_at=created_at,
    )


def _cluster_family_rows(
    family_rows: list[dict[str, Any]],
    *,
    strategy_adaptation_state: StrategyAdaptationState | None = None,
    adaptation_blend: float = 0.2,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    ordered_rows = sorted(
        family_rows,
        key=lambda row: (-float(row["confidence"]), str(row["family_id"])),
    )
    for row in ordered_rows:
        belief_value = float(row["belief_value"])
        confidence = float(row["confidence"])
        representative_signal_id = str(row["representative_signal_id"])
        row["ensemble_weight"] = confidence
        assigned_cluster: dict[str, Any] | None = None
        for cluster in clusters:
            if _same_cluster_direction(
                belief_value,
                float(cluster["belief_value"]),
            ) and _within_cluster_radius(
                belief_value,
                float(cluster["belief_value"]),
            ):
                assigned_cluster = cluster
                break
        if assigned_cluster is None:
            clusters.append(
                {
                    "cluster_id": f"cluster_{len(clusters) + 1}",
                    "belief_value": belief_value,
                    "confidence": confidence,
                    "representative_signal_id": representative_signal_id,
                    "family_ids": [str(row["family_id"])],
                    "regime_tags": list(row.get("regime_tags", ())),
                    "mean_marginal_signal_contribution": float(
                        row.get("mean_marginal_signal_contribution", 0.0)
                    ),
                    "ensemble_weight": float(row.get("ensemble_weight", confidence)),
                }
            )
            continue
        previous_confidence = float(assigned_cluster["confidence"])
        total_confidence = previous_confidence + confidence
        if total_confidence > 0.0:
            assigned_cluster["belief_value"] = (
                float(assigned_cluster["belief_value"]) * previous_confidence
                + belief_value * confidence
            ) / total_confidence
        assigned_cluster["confidence"] = total_confidence
        assigned_cluster["family_ids"].append(str(row["family_id"]))
        assigned_cluster["regime_tags"] = sorted(
            {
                *assigned_cluster.get("regime_tags", []),
                *row.get("regime_tags", ()),
            }
        )
        assigned_cluster["mean_marginal_signal_contribution"] = (
            float(assigned_cluster.get("mean_marginal_signal_contribution", 0.0))
            * previous_confidence
            + float(row.get("mean_marginal_signal_contribution", 0.0)) * confidence
        ) / max(total_confidence, 1e-9)
        assigned_cluster["ensemble_weight"] = float(
            assigned_cluster.get("ensemble_weight", previous_confidence)
        ) + float(row.get("ensemble_weight", confidence))
        if confidence > previous_confidence:
            assigned_cluster["representative_signal_id"] = (
                representative_signal_id
            )
    return clusters


def _same_cluster_direction(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return True
    return (left > 0.0) == (right > 0.0)


def _within_cluster_radius(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 0.02)
    radius = max(0.02, 0.35 * scale)
    return abs(left - right) <= radius


def _regime_tags_for_candidate(item: Any) -> tuple[str, ...]:
    kind = str(getattr(item, "kind", "") or "")
    tags = []
    if kind == "momentum":
        tags.append("trend")
    elif kind == "reversal":
        tags.append("mean_reversion")
    elif kind == "vol_compression_breakout":
        tags.extend(("volatility", "compression"))
    elif kind == "vol_expansion_reversal":
        tags.extend(("volatility", "expansion"))
    elif kind == "momentum_low_vol":
        tags.extend(("trend", "low_vol"))
    elif kind == "reversal_after_shock":
        tags.extend(("mean_reversion", "post_shock"))
    elif kind == "trend_volume_confirmation":
        tags.extend(("trend", "volume_confirmed"))
    elif kind == "relative_strength_rank":
        tags.extend(("cross_sectional", "relative_strength"))
    elif kind == "peer_mean_reversion":
        tags.extend(("cross_sectional", "peer_reversion"))
    return tuple(tags)
