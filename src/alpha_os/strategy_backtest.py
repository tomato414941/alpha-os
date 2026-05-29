from __future__ import annotations

import pandas as pd

from .evaluation_cost_config import TradingEnvironment
from .evaluation_spec import EvaluationDateRange
from .feature_plane import PriceFeaturePlane
from .feature_plane_builder import prepare_feature_plane_from_frame
from .observation_adapters import load_observation_frame
from .portfolio_construction_config import PortfolioConstructionSpec
from .portfolio_decision import ObservationSpec, SubjectObservationBinding, SubjectSet
from .strategy_backtest_evaluation import (
    build_direct_strategy_evaluation_metric_group_results,
)
from .universe_contract import validate_subject_set_universe_contract


def _observation_specs_by_id(subject_set: SubjectSet) -> dict[str, ObservationSpec]:
    return {spec.observation_spec_id: spec for spec in subject_set.observation_specs}


def _feature_plane_for_binding(
    *,
    binding: SubjectObservationBinding,
    observation_spec: ObservationSpec,
    base_url: str,
) -> PriceFeaturePlane:
    frame = load_observation_frame(
        observation_spec,
        asset=binding.asset,
        base_url=base_url,
    )
    return prepare_feature_plane_from_frame(frame=frame)


def subject_backtest_inputs_from_subject_set(
    *,
    subject_set: SubjectSet,
    base_url: str,
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, float],
]:
    subject_return_series_by_subject: dict[str, pd.Series] = {}
    funding_rate_series_by_subject: dict[str, pd.Series] = {}
    funding_cost_bps_series_by_subject: dict[str, pd.Series] = {}
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] = {}
    roll_cost_bps_series_by_subject: dict[str, pd.Series] = {}
    contract_multiplier_by_subject: dict[str, float] = {}
    observation_specs_by_id = _observation_specs_by_id(subject_set)
    for binding in subject_set.bindings:
        observation_spec = observation_specs_by_id.get(binding.observation_spec_id)
        if observation_spec is None:
            raise ValueError(
                f"subject binding is missing observation spec: {binding.subject_id}"
            )
        plane = _feature_plane_for_binding(
            binding=binding,
            observation_spec=observation_spec,
            base_url=base_url,
        )
        subject_return_series_by_subject[binding.subject_id] = (
            plane.daily_returns.astype(float).dropna()
        )
        funding_rate = plane.extra_observables.get("funding_rate")
        if funding_rate is not None:
            funding_rate_series_by_subject[binding.subject_id] = funding_rate.astype(
                float
            )
            funding_cost_bps_series_by_subject[binding.subject_id] = (
                funding_rate.astype(float) * 10000.0
            )
        borrow_fee = plane.extra_observables.get("borrow_fee")
        if borrow_fee is not None:
            borrow_fee_bps_series_by_subject[binding.subject_id] = (
                borrow_fee.astype(float) * 10000.0
            )
        roll_cost_bps = plane.extra_observables.get("roll_cost_bps")
        if roll_cost_bps is not None:
            roll_cost_bps_series_by_subject[binding.subject_id] = (
                roll_cost_bps.astype(float)
            )
        instrument = subject_set.instrument_for_subject(binding.subject_id)
        if instrument is not None and instrument.multiplier is not None:
            contract_multiplier_by_subject[binding.subject_id] = float(
                instrument.multiplier
            )
    return (
        subject_return_series_by_subject,
        funding_rate_series_by_subject,
        funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject,
    )


def run_strategy_backtest(
    *,
    subject_set: SubjectSet,
    target_id: str,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    base_url: str,
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    position_signal_series_by_subject: dict[str, pd.Series] | None = None,
):
    validate_subject_set_universe_contract(subject_set)
    (
        subject_return_series_by_subject,
        funding_rate_series_by_subject,
        funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject,
    ) = subject_backtest_inputs_from_subject_set(
        subject_set=subject_set,
        base_url=base_url,
    )
    return build_direct_strategy_evaluation_metric_group_results(
        subject_return_series_by_subject=subject_return_series_by_subject,
        evaluation_date_ranges=evaluation_date_ranges,
        target_id=target_id,
        subject_set=subject_set,
        signal_series_by_subject=position_signal_series_by_subject,
        funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject=contract_multiplier_by_subject,
        portfolio_construction=portfolio_construction,
        trading_environment=trading_environment,
    )
