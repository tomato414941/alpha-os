from __future__ import annotations

from dataclasses import dataclass, replace

from .portfolio_decision import (
    PortfolioDecisionInput,
    PredictiveSignalInput,
)
from .strategy_sleeves import (
    SleeveAttributionSummary,
    StrategySleeveCompositionSpec,
    StrategySleeveSpec,
    sleeve_normalized_budget,
)


@dataclass(frozen=True)
class SleeveSignalContribution:
    sleeve_id: str
    sleeve_kind: str
    subject_id: str
    raw_signal_value: float
    weighted_signal_value: float
    risk_budget: float
    normalized_risk_budget: float


@dataclass(frozen=True)
class SleeveCompositionResult:
    predictive_signals: tuple[PredictiveSignalInput, ...]
    contributions: tuple[SleeveSignalContribution, ...]
    sleeve_summaries: tuple[SleeveAttributionSummary, ...]


def compose_portfolio_decision_input(
    decision_input: PortfolioDecisionInput,
) -> tuple[PortfolioDecisionInput, SleeveCompositionResult | None]:
    composition = decision_input.sleeve_composition
    if composition is None:
        return decision_input, None
    result = compose_predictive_signals(
        decision_input.predictive_signals,
        composition=composition,
        subject_metadata_by_subject=decision_input.subject_metadata_by_subject,
    )
    return (
        replace(
            decision_input,
            observed_inputs=replace(
                decision_input.observed_inputs,
                predictive_signals=result.predictive_signals,
            ),
            sleeve_composition=None,
        ),
        result,
    )


def compose_predictive_signals(
    predictive_signals: tuple[PredictiveSignalInput, ...],
    *,
    composition: StrategySleeveCompositionSpec | None,
    subject_metadata_by_subject: dict[str, dict[str, str]] | None = None,
) -> SleeveCompositionResult:
    if composition is None:
        return SleeveCompositionResult(
            predictive_signals=predictive_signals,
            contributions=(),
            sleeve_summaries=(),
        )
    subject_metadata_by_subject = subject_metadata_by_subject or {}
    enabled_sleeves = composition.enabled_sleeves
    if not enabled_sleeves:
        raise ValueError("sleeve_composition requires at least one enabled sleeve")

    target_id_by_subject = _target_id_by_subject(predictive_signals)
    contributions: list[SleeveSignalContribution] = []
    composed_values: dict[str, float] = {}
    confidence_values: dict[str, list[float]] = {}
    for sleeve in enabled_sleeves:
        eligible_subject_ids = _eligible_subject_ids(
            sleeve,
            subject_ids=tuple(target_id_by_subject),
            subject_metadata_by_subject=subject_metadata_by_subject,
        )
        if not eligible_subject_ids:
            raise ValueError(f"enabled strategy sleeve has no eligible subjects: {sleeve.sleeve_id}")
        sleeve_signals = _aggregate_subject_signals(
            tuple(
                signal
                for signal in predictive_signals
                if signal.subject_id in eligible_subject_ids
                and _signal_matches_sleeve(signal, sleeve)
            )
        )
        if not sleeve_signals:
            raise ValueError(f"enabled strategy sleeve has no matching signals: {sleeve.sleeve_id}")
        normalized_budget = sleeve_normalized_budget(sleeve, composition)
        for subject_id, raw_value in sleeve_signals.items():
            weighted_value = float(raw_value * normalized_budget)
            composed_values[subject_id] = composed_values.get(subject_id, 0.0) + weighted_value
            contributions.append(
                SleeveSignalContribution(
                    sleeve_id=sleeve.sleeve_id,
                    sleeve_kind=sleeve.sleeve_kind,
                    subject_id=subject_id,
                    raw_signal_value=float(raw_value),
                    weighted_signal_value=weighted_value,
                    risk_budget=float(sleeve.risk_budget),
                    normalized_risk_budget=normalized_budget,
                )
            )
        for signal in predictive_signals:
            if signal.subject_id in sleeve_signals and signal.confidence is not None:
                confidence_values.setdefault(signal.subject_id, []).append(signal.confidence)

    composed_signals = tuple(
        PredictiveSignalInput(
            source_id="sleeve_composition",
            source_kind="sleeve_composite",
            subject_id=subject_id,
            target_id=target_id_by_subject[subject_id],
            value=value,
            confidence=_mean(confidence_values.get(subject_id, ())),
            sleeve_id=None,
        )
        for subject_id, value in sorted(composed_values.items())
    )
    return SleeveCompositionResult(
        predictive_signals=composed_signals,
        contributions=tuple(contributions),
        sleeve_summaries=_sleeve_signal_summaries(composition, tuple(contributions)),
    )


def _signal_matches_sleeve(
    signal: PredictiveSignalInput,
    sleeve: StrategySleeveSpec,
) -> bool:
    if sleeve.signal_source_kind is not None and signal.source_kind != sleeve.signal_source_kind:
        return False
    if sleeve.signal_discovery_id is not None and signal.source_id != sleeve.signal_discovery_id:
        return False
    return True


def _eligible_subject_ids(
    sleeve: StrategySleeveSpec,
    *,
    subject_ids: tuple[str, ...],
    subject_metadata_by_subject: dict[str, dict[str, str]],
) -> tuple[str, ...]:
    subject_filter = sleeve.subject_filter
    eligible = set(subject_ids)
    if subject_filter.subject_ids:
        eligible &= set(subject_filter.subject_ids)
    field_filters = (
        ("instrument_type", subject_filter.instrument_types),
        ("asset_class", subject_filter.asset_classes),
        ("region", subject_filter.regions),
        ("cluster", subject_filter.clusters),
    )
    for field_name, allowed_values in field_filters:
        if not allowed_values:
            continue
        allowed = set(allowed_values)
        eligible = {
            subject_id
            for subject_id in eligible
            if subject_metadata_by_subject.get(subject_id, {}).get(field_name) in allowed
        }
    return tuple(sorted(eligible))


def _target_id_by_subject(
    predictive_signals: tuple[PredictiveSignalInput, ...],
) -> dict[str, str]:
    target_ids: dict[str, str] = {}
    for signal in predictive_signals:
        target_ids.setdefault(signal.subject_id, signal.target_id)
    return target_ids


def _aggregate_subject_signals(
    predictive_signals: tuple[PredictiveSignalInput, ...],
) -> dict[str, float]:
    observed_subject_ids: set[str] = set()
    weighted_values: dict[str, float] = {}
    weights: dict[str, float] = {}
    for signal in predictive_signals:
        observed_subject_ids.add(signal.subject_id)
        confidence = signal.confidence if signal.confidence is not None else 1.0
        confidence = max(confidence, 0.0)
        weighted_values[signal.subject_id] = (
            weighted_values.get(signal.subject_id, 0.0) + signal.value * confidence
        )
        weights[signal.subject_id] = weights.get(signal.subject_id, 0.0) + confidence
    return {
        subject_id: (
            float(weighted_values[subject_id] / weights[subject_id])
            if weights.get(subject_id, 0.0) > 0.0
            else 0.0
        )
        for subject_id in observed_subject_ids
    }


def _sleeve_signal_summaries(
    composition: StrategySleeveCompositionSpec,
    contributions: tuple[SleeveSignalContribution, ...],
) -> tuple[SleeveAttributionSummary, ...]:
    summaries: list[SleeveAttributionSummary] = []
    for sleeve in composition.enabled_sleeves:
        sleeve_contributions = [
            item for item in contributions if item.sleeve_id == sleeve.sleeve_id
        ]
        summaries.append(
            SleeveAttributionSummary(
                sleeve_id=sleeve.sleeve_id,
                sleeve_kind=sleeve.sleeve_kind,
                risk_budget=sleeve.risk_budget,
                subject_count=len({item.subject_id for item in sleeve_contributions}),
                mean_signal=_mean(item.raw_signal_value for item in sleeve_contributions) or 0.0,
                mean_abs_signal=_mean(
                    abs(item.raw_signal_value) for item in sleeve_contributions
                )
                or 0.0,
            )
        )
    return tuple(summaries)


def _mean(values) -> float | None:
    values = tuple(float(value) for value in values)
    if not values:
        return None
    return float(sum(values) / len(values))
