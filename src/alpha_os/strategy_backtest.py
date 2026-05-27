from __future__ import annotations

import re
from typing import Protocol

import pandas as pd

from .position_rules import crypto_regime_momentum_eligibility_series_by_subject
from .data_repositories import FeaturePlaneRepository
from .evaluation_cost_config import TradingEnvironment
from .evaluation_spec import EvaluationDateRange
from .portfolio_construction_config import PortfolioConstructionSpec
from .portfolio_decision import SubjectSet
from .signal_discovery_strategy_evaluation import (
    build_direct_strategy_evaluation_metric_group_results,
)
from .subject_set_feature_plane import SubjectPlaneKey, build_subject_set_feature_planes
from .trading_strategy import TradingStrategySpec
from .universe_contract import validate_subject_set_universe_contract


# This module is intended to become the common backtest entrypoint for strategy
# candidates. Today it only supports trainless strategy candidates, so keep new
# behavior narrow until trained candidates are routed through the same
# boundary.


class DirectStrategyEvaluationReadPort(Protocol):
    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_subject_set(self, subject_set_id: str):
        ...


def subject_backtest_inputs_from_subject_set_planes(
    *,
    subject_set: SubjectSet,
    subject_planes: dict[SubjectPlaneKey, object],
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
    for binding in subject_set.bindings:
        plane = subject_planes.get(
            SubjectPlaneKey(
                asset=binding.asset,
                observation_spec_id=binding.observation_spec_id,
            )
        )
        if plane is None:
            raise ValueError(
                "strategy feature plane is missing: "
                f"{binding.asset}/{binding.observation_spec_id}"
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


def run_strategy_backtest_from_store(
    *,
    store: DirectStrategyEvaluationReadPort,
    strategy_id: str,
    subject_set_id: str,
    target_id: str,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    base_url: str,
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    feature_plane_repository: FeaturePlaneRepository | None,
):
    # Temporary DB adapter for legacy evaluation execution paths.
    # Prefer run_strategy_backtest() once strategy and subject set are resolved.
    strategy_state = store.get_trading_strategy(strategy_id)
    if strategy_state is None:
        raise ValueError(f"strategy does not exist: {strategy_id}")
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is None:
        raise ValueError(f"subject set does not exist: {subject_set_id}")
    return run_strategy_backtest(
        trading_strategy=strategy_state.trading_strategy,
        subject_set=subject_set_state.definition,
        target_id=target_id,
        evaluation_date_ranges=evaluation_date_ranges,
        base_url=base_url,
        portfolio_construction=portfolio_construction,
        trading_environment=trading_environment,
        feature_plane_repository=feature_plane_repository,
    )


def run_strategy_backtest(
    *,
    trading_strategy: TradingStrategySpec,
    subject_set: SubjectSet,
    target_id: str,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    base_url: str,
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    feature_plane_repository: FeaturePlaneRepository | None,
):
    selection_kind = trading_strategy.selection_kind
    selection_top_k = trading_strategy.top_k
    position_rule_id = trading_strategy.position_rule_id
    if position_rule_id not in {
        "constant_hold",
        "dual_momentum_hold",
        "crypto_regime_momentum_hold",
    }:
        raise ValueError(
            "current strategy backtest only supports "
            "position_rule=constant_hold, position_rule=dual_momentum_hold, or "
            "position_rule=crypto_regime_momentum_hold"
        )
    if selection_kind not in {"all_assets", "top_k"}:
        raise ValueError(
            "current strategy backtest only supports selection=all_assets or selection=top_k"
        )
    if selection_kind == "top_k" and selection_top_k is None:
        raise ValueError(
            "top_k strategy backtest requires strategy portfolio top_k"
        )
    validate_subject_set_universe_contract(subject_set)
    subject_planes = build_subject_set_feature_planes(
        subject_set=subject_set,
        executable_definitions=[],
        base_url=base_url,
        feature_plane_repository=feature_plane_repository,
    )
    (
        subject_return_series_by_subject,
        funding_rate_series_by_subject,
        funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject,
    ) = subject_backtest_inputs_from_subject_set_planes(
        subject_set=subject_set,
        subject_planes=subject_planes,
    )
    if position_rule_id == "constant_hold":
        position_signal_series_by_subject = None
    elif position_rule_id == "dual_momentum_hold":
        position_signal_series_by_subject = dual_momentum_signal_series_by_subject(
            subject_return_series_by_subject=subject_return_series_by_subject,
            family_mix=trading_strategy.family_mix,
        )
    else:
        position_signal_series_by_subject = crypto_regime_momentum_eligibility_series_by_subject(
            subject_return_series_by_subject=subject_return_series_by_subject,
            funding_rate_series_by_subject=funding_rate_series_by_subject,
        )
    return build_direct_strategy_evaluation_metric_group_results(
        subject_return_series_by_subject=subject_return_series_by_subject,
        evaluation_date_ranges=evaluation_date_ranges,
        target_id=target_id,
        subject_set_id=str(subject_set.subject_set_id),
        subject_set=subject_set,
        signal_series_by_subject=position_signal_series_by_subject,
        funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject=contract_multiplier_by_subject,
        portfolio_construction=portfolio_construction,
        trading_environment=trading_environment,
        top_k=selection_top_k,
    )


def _dual_momentum_lookback_from_family_mix(family_mix: str | None) -> int:
    if family_mix:
        match = re.search(r"lookback=(\d+)", family_mix)
        if match is not None:
            return max(int(match.group(1)), 2)
    return 252


def dual_momentum_signal_series_by_subject(
    *,
    subject_return_series_by_subject: dict[str, pd.Series],
    family_mix: str | None,
) -> dict[str, pd.Series]:
    lookback = _dual_momentum_lookback_from_family_mix(family_mix)
    trailing_returns_by_subject: dict[str, pd.Series] = {}
    for subject_id, full_returns in subject_return_series_by_subject.items():
        trailing_returns_by_subject[subject_id] = (
            (1.0 + full_returns.astype(float))
            .rolling(lookback, min_periods=lookback)
            .apply(lambda values: float(values.prod() - 1.0), raw=True)
            .shift(1)
        )
    if not trailing_returns_by_subject:
        return {}
    trailing_frame = pd.DataFrame(trailing_returns_by_subject, dtype=float)
    signal_series_by_subject: dict[str, pd.Series] = {}
    for subject_id in trailing_frame.columns:
        series = trailing_frame[subject_id].where(
            trailing_frame[subject_id] > 0.0,
            0.0,
        )
        signal_series_by_subject[str(subject_id)] = series.fillna(0.0).astype(float)
    return signal_series_by_subject
