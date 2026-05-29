from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .decision_backtest import DependenceBacktestSeries, SubjectBacktestSeries
from .evaluation_spec import EvaluationDateRange
from .prediction_diagnostics import PredictionDiagnostics, build_prediction_diagnostics


@dataclass(frozen=True)
class RangeBacktestDataset:
    label: str
    predictive_corr: float
    prediction_diagnostics: PredictionDiagnostics
    subject_series: tuple[SubjectBacktestSeries, ...]
    dependence_series: tuple[DependenceBacktestSeries, ...] = ()


def build_direct_range_backtest_dataset_for_range(
    *,
    date_range: EvaluationDateRange,
    target_id: str,
    subject_return_series_by_subject: dict[str, pd.Series],
    signal_series_by_subject: dict[str, pd.Series] | None,
    funding_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] | None,
    roll_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    contract_multiplier_by_subject: dict[str, float] | None,
    signal_value: float,
) -> RangeBacktestDataset | None:
    return build_direct_range_backtest_dataset(
        date_range=date_range,
        target_id=target_id,
        subject_return_series_by_subject=subject_return_series_by_subject,
        signal_series_by_subject=signal_series_by_subject,
        funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject=contract_multiplier_by_subject,
        signal_value=signal_value,
    )


def build_direct_range_backtest_dataset(
    *,
    date_range: EvaluationDateRange,
    target_id: str,
    subject_return_series_by_subject: dict[str, pd.Series],
    signal_series_by_subject: dict[str, pd.Series] | None,
    funding_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] | None,
    roll_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    contract_multiplier_by_subject: dict[str, float] | None,
    signal_value: float,
) -> RangeBacktestDataset | None:
    subject_series: list[SubjectBacktestSeries] = []
    for subject_id, full_returns in sorted(subject_return_series_by_subject.items()):
        range_returns = full_returns.loc[
            (full_returns.index >= date_range.start_date)
            & (full_returns.index <= date_range.end_date)
        ].dropna()
        if range_returns.empty:
            continue
        if signal_series_by_subject is None:
            signal_series = pd.Series(
                float(signal_value),
                index=range_returns.index,
                dtype=float,
            )
        else:
            subject_signal_series = signal_series_by_subject.get(subject_id)
            if subject_signal_series is None:
                signal_series = pd.Series(
                    0.0,
                    index=range_returns.index,
                    dtype=float,
                )
            else:
                signal_series = (
                    subject_signal_series.astype(float)
                    .reindex(range_returns.index)
                    .fillna(0.0)
                )
        subject_series.append(
            SubjectBacktestSeries(
                subject_id=subject_id,
                signal_series=signal_series,
                realized_return_series=range_returns.astype(float),
                target_id=target_id,
                historical_return_series=full_returns.astype(float),
                funding_cost_bps_series=(
                    None
                    if funding_cost_bps_series_by_subject is None
                    else funding_cost_bps_series_by_subject.get(subject_id)
                ),
                borrow_fee_bps_series=(
                    None
                    if borrow_fee_bps_series_by_subject is None
                    else borrow_fee_bps_series_by_subject.get(subject_id)
                ),
                roll_cost_bps_series=(
                    None
                    if roll_cost_bps_series_by_subject is None
                    else roll_cost_bps_series_by_subject.get(subject_id)
                ),
                contract_multiplier=(
                    None
                    if contract_multiplier_by_subject is None
                    else contract_multiplier_by_subject.get(subject_id)
                ),
            )
        )
    if not subject_series:
        return None
    prediction_diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={
            item.subject_id: item.signal_series for item in subject_series
        },
        forward_return_series_by_subject={
            item.subject_id: item.realized_return_series for item in subject_series
        },
        group_by_subject=None,
    )
    return RangeBacktestDataset(
        label=date_range.label,
        predictive_corr=prediction_diagnostics.mean_signal_forward_corr,
        prediction_diagnostics=prediction_diagnostics,
        subject_series=tuple(subject_series),
    )
