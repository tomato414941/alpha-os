from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from .cross_instrument_contract import (
    CrossInstrumentReportContract,
    default_validation_result_set_cross_instrument_contract,
)
from .contract_boundaries import subject_set_contract_groups
from .evaluation_generation import (
    _daily_close_series,
    _resolve_observation_spec,
    generate_evaluation_inputs_from_frame,
)
from .signal_registry import get_signal_definition
from .observation_adapters import load_observation_frame, observation_contract_key
from .portfolio_decision import ObservationSpec, SubjectSet
from .store import EvaluationStore
from .trading_strategy import TradingStrategySpec
from .targets import get_target_definition
from .universe_contract import validate_subject_set_universe_contract
from .validation_engine import (
    ValidationTargetBundle,
    compute_validation_decision_metrics,
    compute_validation_signal_metrics,
    compute_validation_meta_metrics,
    slice_validation_bundle,
)
from .validation_spec import ValidationSpec
from .validation_result_set import ValidationResultSet, build_validation_result_set


@dataclass(frozen=True)
class ValidationRunResult:
    run_id: str
    spec_json: str
    signal_result_count: int
    meta_result_count: int
    decision_result_count: int
    cross_instrument_contract: CrossInstrumentReportContract
    validation_result_set: ValidationResultSet


@dataclass(frozen=True)
class StrategyValidationPlanEntry:
    strategy_id: str
    signal_discovery_id: str | None
    subject_set_id: str
    signal_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class StrategyValidationPlan:
    strategy_ids: tuple[str, ...]
    entries: tuple[StrategyValidationPlanEntry, ...]
    spec: ValidationSpec


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_price_frame_from_signal_noise(
    *,
    base_url: str,
    asset: str,
    observation_spec: ObservationSpec,
) -> pd.DataFrame:
    return load_observation_frame(
        observation_spec,
        asset=asset,
        base_url=base_url,
    )


def _ephemeral_signal_id(base_signal_id: str, target_id: str) -> str:
    return f"{base_signal_id}@{target_id}"


def _build_target_variant_definition(*, signal_id: str, target_id: str):
    base_definition = get_signal_definition(signal_id)
    target_definition = get_target_definition(target_id)
    return base_definition.__class__(
        signal_id=_ephemeral_signal_id(signal_id, target_id),
        kind=base_definition.kind,
        lookback=base_definition.lookback,
        target=target_definition,
        asset=base_definition.asset,
        observation_spec=base_definition.observation_spec,
    )


def _build_subject_target_variant_definition(
    *,
    signal_id: str,
    target_id: str,
    asset: str,
    observation_spec: ObservationSpec,
):
    base_definition = get_signal_definition(signal_id)
    target_definition = get_target_definition(target_id)
    return base_definition.__class__(
        signal_id=_ephemeral_signal_id(signal_id, target_id),
        kind=base_definition.kind,
        lookback=base_definition.lookback,
        target=target_definition,
        asset=asset,
        observation_spec=observation_spec,
    )


def _global_date_bounds(spec: ValidationSpec) -> tuple[str, str]:
    start_date = min(item.start_date for item in spec.date_ranges)
    end_date = max(item.end_date for item in spec.date_ranges)
    return start_date, end_date


def _clipped_date_range(
    *,
    frame,
    start_date: str,
    end_date: str,
    lookback: int,
    horizon_days: int,
) -> tuple[str, str] | None:
    daily_close = _daily_close_series(frame)
    dates = list(daily_close.index)
    if not dates:
        return None
    minimum_index = int(lookback)
    maximum_index = len(dates) - 1 - int(horizon_days)
    if maximum_index < minimum_index:
        return None
    minimum_date = dates[minimum_index]
    maximum_date = dates[maximum_index]
    clipped_start = max(start_date, minimum_date)
    clipped_end = min(end_date, maximum_date)
    if clipped_start > clipped_end:
        return None
    return clipped_start, clipped_end


def _build_validation_bundle_for_target(
    *,
    spec: ValidationSpec,
    target_id: str,
    signal_frames: dict[str, object],
    subject_id: str | None = None,
    asset: str | None = None,
    observation_spec: ObservationSpec | None = None,
) -> ValidationTargetBundle | None:
    global_start_date, global_end_date = _global_date_bounds(spec)
    observations: pd.Series | None = None
    predictions_by_signal: dict[str, pd.Series] = {}
    for base_signal_id in spec.signal_ids:
        if asset is None or observation_spec is None:
            variant_definition = _build_target_variant_definition(
                signal_id=base_signal_id,
                target_id=target_id,
            )
        else:
            variant_definition = _build_subject_target_variant_definition(
                signal_id=base_signal_id,
                target_id=target_id,
                asset=asset,
                observation_spec=observation_spec,
            )
        resolved_observation_spec = _resolve_observation_spec(
            definition=variant_definition,
            observation_spec=observation_spec,
        )
        frame_key = observation_contract_key(
            resolved_observation_spec,
            asset=variant_definition.asset,
        )
        frame = signal_frames.get(frame_key)
        if frame is None:
            frame = _load_price_frame_from_signal_noise(
                observation_spec=resolved_observation_spec,
                asset=variant_definition.asset,
                base_url=spec.base_url,
            )
            signal_frames[frame_key] = frame
        if frame is None:
            raise ValueError(
                f"validation variant is missing observation spec: {variant_definition.signal_id}"
            )
        clipped_range = _clipped_date_range(
            frame=frame,
            start_date=global_start_date,
            end_date=global_end_date,
            lookback=variant_definition.lookback,
            horizon_days=variant_definition.horizon_days or 0,
        )
        if clipped_range is None:
            continue
        evaluation_inputs = generate_evaluation_inputs_from_frame(
            frame=frame,
            start_date=clipped_range[0],
            end_date=clipped_range[1],
            signal_id=base_signal_id,
            definition=variant_definition,
            target_id=target_id,
            subject_id=subject_id,
        )
        if not evaluation_inputs:
            continue
        prediction_series = pd.Series(
            {item.date: item.prediction for item in evaluation_inputs},
            dtype=float,
        ).sort_index()
        observation_series = pd.Series(
            {item.date: item.observation for item in evaluation_inputs},
            dtype=float,
        ).sort_index()
        if observations is None:
            observations = observation_series
        else:
            overlap = observations.index.intersection(observation_series.index)
            if not overlap.empty and not observations.loc[overlap].equals(
                observation_series.loc[overlap]
            ):
                raise ValueError(
                    f"inconsistent observations for target_id={target_id} "
                    f"between hypotheses on overlapping dates"
                )
            observations = pd.concat([observations, observation_series]).groupby(level=0).first()
            observations = observations.sort_index()
        predictions_by_signal[base_signal_id] = prediction_series
    if observations is None or not predictions_by_signal:
        return None
    return ValidationTargetBundle(
        subject_id=subject_id,
        target_id=target_id,
        observations=observations,
        predictions_by_signal=predictions_by_signal,
    )


def _load_subject_sets(
    store: EvaluationStore,
    *,
    subject_set_ids: tuple[str, ...],
) -> dict[str, SubjectSet]:
    loaded: dict[str, SubjectSet] = {}
    for subject_set_id in subject_set_ids:
        state = store.get_subject_set(subject_set_id)
        if state is None:
            raise ValueError(f"unknown subject set: {subject_set_id}")
        validate_subject_set_universe_contract(state.definition)
        loaded[subject_set_id] = state.definition
    return loaded


def _load_strategy_specs(
    store: EvaluationStore,
    *,
    strategy_ids: tuple[str, ...],
) -> dict[str, TradingStrategySpec]:
    loaded: dict[str, TradingStrategySpec] = {}
    missing: list[str] = []
    for strategy_id in strategy_ids:
        state = store.get_trading_strategy(strategy_id)
        if state is None:
            missing.append(strategy_id)
            continue
        loaded[strategy_id] = state.trading_strategy
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"unknown strategy specs: {joined}")
    return loaded


def build_validation_plan_for_strategies(
    store: EvaluationStore,
    *,
    strategy_ids: tuple[str, ...],
    base_spec: ValidationSpec | None = None,
    base_url: str | None = None,
) -> StrategyValidationPlan:
    if not strategy_ids:
        raise ValueError("strategy validation plan is missing strategy_ids")
    strategies_by_id = _load_strategy_specs(
        store,
        strategy_ids=strategy_ids,
    )
    specification_states = store.list_signal_specs(limit=100000)
    specification_definitions = [item.definition for item in specification_states]
    entries: list[StrategyValidationPlanEntry] = []
    resolved_subject_set_ids: list[str] = []
    resolved_signal_ids: list[str] = []
    resolved_target_ids: list[str] = []
    for strategy_id in strategy_ids:
        strategy_spec = strategies_by_id[strategy_id]
        signal_discovery_id = strategy_spec.signal_discovery_id
        if signal_discovery_id is None:
            raise ValueError(
                f"validation strategy is missing signal discovery provenance: {strategy_id}"
            )
        signal_discovery_state = store.get_signal_discovery_spec(signal_discovery_id)
        if signal_discovery_state is None:
            raise ValueError(
                f"unknown signal discovery spec for strategy {strategy_id}: {signal_discovery_id}"
            )
        signal_discovery = signal_discovery_state.definition
        signal_ids = signal_discovery.resolve_signal_spec_ids(
            specification_definitions
        )
        if not signal_ids:
            raise ValueError(
                f"strategy resolves to no specifications: {strategy_id}"
            )
        target_ids: tuple[str, ...]
        if signal_discovery.target_id is not None:
            target_ids = (signal_discovery.target_id,)
        elif base_spec is not None:
            target_ids = base_spec.target_ids
        else:
            target_ids = ()
        entries.append(
            StrategyValidationPlanEntry(
                strategy_id=strategy_id,
                signal_discovery_id=signal_discovery_id,
                subject_set_id=signal_discovery.subject_set_id,
                signal_ids=signal_ids,
                target_ids=target_ids,
            )
        )
        if signal_discovery.subject_set_id not in resolved_subject_set_ids:
            resolved_subject_set_ids.append(signal_discovery.subject_set_id)
        for signal_id in signal_ids:
            if signal_id not in resolved_signal_ids:
                resolved_signal_ids.append(signal_id)
        for target_id in target_ids:
            if target_id not in resolved_target_ids:
                resolved_target_ids.append(target_id)
    if base_spec is None:
        from .validation_spec import default_validation_spec

        spec = default_validation_spec(subject_set_ids=tuple(resolved_subject_set_ids))
    else:
        spec = base_spec
    if not resolved_target_ids:
        resolved_target_ids = list(spec.target_ids)
    planned_spec = spec.__class__(
        signal_ids=tuple(resolved_signal_ids),
        target_ids=tuple(resolved_target_ids),
        date_ranges=spec.date_ranges,
        metric_windows=spec.metric_windows,
        aggregation_kinds=spec.aggregation_kinds,
        subject_set_ids=tuple(resolved_subject_set_ids),
        base_url=spec.base_url if base_url is None else str(base_url),
    )
    return StrategyValidationPlan(
        strategy_ids=strategy_ids,
        entries=tuple(entries),
        spec=planned_spec,
    )


def run_validation_for_strategies(
    store: EvaluationStore,
    *,
    strategy_ids: tuple[str, ...],
    spec: ValidationSpec | None = None,
    base_url: str | None = None,
    recorded_at: str | None = None,
) -> tuple[StrategyValidationPlan, ValidationRunResult]:
    plan = build_validation_plan_for_strategies(
        store,
        strategy_ids=strategy_ids,
        base_spec=spec,
        base_url=base_url,
    )
    result = run_validation(
        store,
        spec=plan.spec,
        recorded_at=recorded_at,
    )
    return plan, result


def run_validation(
    store: EvaluationStore,
    *,
    spec: ValidationSpec,
    recorded_at: str | None = None,
) -> ValidationRunResult:
    store.ensure_schema()
    timestamp = recorded_at or _utc_now()
    run_id = timestamp
    spec_json = json.dumps(spec.to_document(), sort_keys=True)
    cross_instrument_contract = default_validation_result_set_cross_instrument_contract()
    signal_frames: dict[str, object] = {}
    signal_results: list[dict[str, object]] = []
    meta_results: list[dict[str, object]] = []
    decision_results: list[dict[str, object]] = []
    bundles_by_target: dict[str, ValidationTargetBundle] = {}
    decision_bundles_by_target_and_subject_set: dict[tuple[str, str], dict[str, ValidationTargetBundle]] = {}
    subject_set_ids = spec.subject_set_ids
    subject_sets_by_id = _load_subject_sets(
        store,
        subject_set_ids=subject_set_ids,
    )

    for target_id in spec.target_ids:
        bundle = _build_validation_bundle_for_target(
            spec=spec,
            target_id=target_id,
            signal_frames=signal_frames,
        )
        if bundle is not None:
            bundles_by_target[target_id] = bundle
        for subject_set_id, subject_set in subject_sets_by_id.items():
            subject_bundles: dict[str, ValidationTargetBundle] = {}
            for binding in subject_set.bindings:
                observation_spec = subject_set.observation_spec_for_subject(
                    binding.subject_id
                )
                subject_bundle = _build_validation_bundle_for_target(
                    spec=spec,
                    target_id=target_id,
                    signal_frames=signal_frames,
                    subject_id=binding.subject_id,
                    asset=binding.asset,
                    observation_spec=observation_spec,
                )
                if subject_bundle is not None:
                    subject_bundles[binding.subject_id] = subject_bundle
            if subject_bundles:
                decision_bundles_by_target_and_subject_set[(target_id, subject_set_id)] = subject_bundles

    for date_range in spec.date_ranges:
        for target_id, bundle in sorted(bundles_by_target.items()):
            sliced_bundle = slice_validation_bundle(
                bundle,
                start_date=date_range.start_date,
                end_date=date_range.end_date,
            )
            if sliced_bundle.observations.empty:
                continue
            for window_size in spec.metric_windows:
                for metric in compute_validation_signal_metrics(
                    sliced_bundle,
                    window_size=window_size,
                ):
                    signal_results.append(
                        {
                            "run_id": run_id,
                            "date_range_label": date_range.label,
                            "start_date": date_range.start_date,
                            "end_date": date_range.end_date,
                            "target_id": target_id,
                            "signal_id": metric.signal_id,
                            "window_size": window_size,
                            "corr": metric.corr,
                            "mmc": metric.mmc,
                            "sample_count": metric.sample_count,
                            "mmc_sample_count": metric.mmc_sample_count,
                            "mmc_peer_count": metric.mmc_peer_count,
                            "mmc_baseline_type": metric.mmc_baseline_type,
                            "recorded_at": timestamp,
                        }
                    )
                for meta_metric in compute_validation_meta_metrics(
                    sliced_bundle,
                    aggregation_kinds=spec.aggregation_kinds,
                    window_size=window_size,
                ):
                    meta_results.append(
                        {
                            "run_id": run_id,
                            "date_range_label": date_range.label,
                            "start_date": date_range.start_date,
                            "end_date": date_range.end_date,
                            "target_id": target_id,
                            "aggregation_kind": meta_metric.aggregation_kind,
                            "window_size": window_size,
                            "corr": meta_metric.corr,
                            "sample_count": meta_metric.sample_count,
                            "recorded_at": timestamp,
                        }
                    )
                for subject_set_id in subject_set_ids:
                    decision_subject_bundles = decision_bundles_by_target_and_subject_set.get(
                        (target_id, subject_set_id),
                        {},
                    )
                    sliced_decision_bundles = {
                        subject_id: slice_validation_bundle(
                            subject_bundle,
                            start_date=date_range.start_date,
                            end_date=date_range.end_date,
                        )
                        for subject_id, subject_bundle in decision_subject_bundles.items()
                    }
                    for decision_metric in compute_validation_decision_metrics(
                        sliced_decision_bundles,
                        subject_set_id=subject_set_id,
                        aggregation_kinds=spec.aggregation_kinds,
                        window_size=window_size,
                    ):
                        decision_results.append(
                            {
                                "run_id": run_id,
                                "date_range_label": date_range.label,
                                "start_date": date_range.start_date,
                                "end_date": date_range.end_date,
                                "target_id": target_id,
                                "subject_set_id": decision_metric.subject_set_id,
                                "aggregation_kind": decision_metric.aggregation_kind,
                                "window_size": window_size,
                                "gross_return_total": decision_metric.gross_return_total,
                                "net_return_total": decision_metric.net_return_total,
                                "max_drawdown": decision_metric.max_drawdown,
                                "mean_turnover": decision_metric.mean_turnover,
                                "mean_gross_notional_exposure": decision_metric.mean_gross_notional_exposure,
                                "mean_net_notional_exposure": decision_metric.mean_net_notional_exposure,
                                "mean_long_notional_exposure": decision_metric.mean_long_notional_exposure,
                                "mean_short_notional_exposure": decision_metric.mean_short_notional_exposure,
                                "mean_traded_notional": decision_metric.mean_traded_notional,
                                "cost_notional_total": decision_metric.cost_notional_total,
                                "funding_cost_notional_total": decision_metric.funding_cost_notional_total,
                                "borrow_cost_notional_total": decision_metric.borrow_cost_notional_total,
                                "roll_cost_notional_total": decision_metric.roll_cost_notional_total,
                                "step_count": decision_metric.step_count,
                                "recorded_at": timestamp,
                            }
                        )

    validation_result_set = build_validation_result_set(
        signal_results=signal_results,
        meta_results=meta_results,
        decision_results=decision_results,
        subject_set_contract_groups_by_id={
            subject_set_id: subject_set_contract_groups(subject_set.definition.contract_boundary)
            for subject_set_id in spec.subject_set_ids
            if (subject_set := store.get_subject_set(subject_set_id)) is not None
        },
        universe_policy_by_subject_set_id={
            subject_set_id: subject_set.definition.universe_policy.to_document()
            for subject_set_id in spec.subject_set_ids
            if (subject_set := store.get_subject_set(subject_set_id)) is not None
        },
    )

    store.create_validation_run(
        run_id=run_id,
        spec_json=spec_json,
        cross_instrument_contract=cross_instrument_contract,
        validation_result_set=validation_result_set,
        recorded_at=timestamp,
    )
    for item in signal_results:
        store.upsert_validation_signal_result(**item)
    for item in meta_results:
        store.upsert_validation_meta_result(**item)
    for item in decision_results:
        store.upsert_validation_decision_result(**item)

    return ValidationRunResult(
        run_id=run_id,
        spec_json=spec_json,
        signal_result_count=len(signal_results),
        meta_result_count=len(meta_results),
        decision_result_count=len(decision_results),
        cross_instrument_contract=cross_instrument_contract,
        validation_result_set=validation_result_set,
    )
