from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .decision_backtest import DecisionBacktestInput, DecisionBacktestResult, run_decision_backtest
from .meta_aggregation_service import (
    AGGREGATION_ACTIVE_EQUAL_MEAN,
    AGGREGATION_CORR_WEIGHTED_MEAN,
)
from .metrics_service import MMC_BASELINE_ACTIVE_PEER_MEAN
from .scoring import DEFAULT_METRIC_WINDOW, HypothesisMetrics, compute_hypothesis_metrics, numerai_corr


@dataclass(frozen=True)
class ValidationTargetBundle:
    target_id: str
    observations: pd.Series
    predictions_by_hypothesis: dict[str, pd.Series]


@dataclass(frozen=True)
class ValidationHypothesisMetric:
    hypothesis_id: str
    corr: float
    mmc: float | None
    sample_count: int
    mmc_sample_count: int
    mmc_peer_count: int
    mmc_baseline_type: str | None
    window_size: int


@dataclass(frozen=True)
class ValidationMetaMetric:
    aggregation_kind: str
    corr: float
    sample_count: int
    window_size: int


@dataclass(frozen=True)
class ValidationDecisionMetric:
    aggregation_kind: str
    gross_return_total: float
    net_return_total: float
    max_drawdown: float
    mean_turnover: float
    step_count: int
    window_size: int


def _sorted_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return series.astype(float).sort_index()


def slice_validation_bundle(
    bundle: ValidationTargetBundle,
    *,
    start_date: str,
    end_date: str,
) -> ValidationTargetBundle:
    observations = _sorted_series(
        bundle.observations.loc[
            (bundle.observations.index >= start_date) & (bundle.observations.index <= end_date)
        ]
    )
    predictions_by_hypothesis = {
        hypothesis_id: _sorted_series(
            prediction_series.loc[
                (prediction_series.index >= start_date) & (prediction_series.index <= end_date)
            ]
        )
        for hypothesis_id, prediction_series in bundle.predictions_by_hypothesis.items()
    }
    return ValidationTargetBundle(
        target_id=bundle.target_id,
        observations=observations,
        predictions_by_hypothesis=predictions_by_hypothesis,
    )


def _tail_aligned_pair(
    predictions: pd.Series,
    observations: pd.Series,
    *,
    window_size: int,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat([predictions, observations], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    aligned = aligned.tail(int(window_size))
    return aligned.iloc[:, 0].astype(float), aligned.iloc[:, 1].astype(float)


def _peer_meta_model(
    *,
    hypothesis_id: str,
    evaluation_ids: list[str],
    predictions_by_hypothesis: dict[str, pd.Series],
) -> tuple[pd.Series | None, int]:
    peer_series_list: list[pd.Series] = []
    peer_count = 0
    evaluation_index = pd.Index(evaluation_ids, dtype=object)
    for peer_hypothesis_id, series in predictions_by_hypothesis.items():
        if peer_hypothesis_id == hypothesis_id:
            continue
        aligned = series.reindex(evaluation_index).dropna()
        if aligned.empty:
            continue
        peer_count += 1
        peer_series_list.append(series.reindex(evaluation_index))
    if not peer_series_list:
        return None, 0
    peer_frame = pd.concat(peer_series_list, axis=1)
    return peer_frame.mean(axis=1, skipna=True).dropna().astype(float), peer_count


def compute_validation_hypothesis_metrics(
    bundle: ValidationTargetBundle,
    *,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> list[ValidationHypothesisMetric]:
    results: list[ValidationHypothesisMetric] = []
    for hypothesis_id, prediction_series in sorted(bundle.predictions_by_hypothesis.items()):
        predictions, observations = _tail_aligned_pair(
            _sorted_series(prediction_series),
            _sorted_series(bundle.observations),
            window_size=window_size,
        )
        evaluation_ids = [str(item) for item in predictions.index]
        if not evaluation_ids:
            continue
        meta_model, peer_count = _peer_meta_model(
            hypothesis_id=hypothesis_id,
            evaluation_ids=evaluation_ids,
            predictions_by_hypothesis=bundle.predictions_by_hypothesis,
        )
        metrics = compute_hypothesis_metrics(
            predictions=predictions,
            target=observations,
            meta_model=meta_model,
            window_size=window_size,
        )
        baseline_type = MMC_BASELINE_ACTIVE_PEER_MEAN if peer_count > 0 else None
        results.append(
            ValidationHypothesisMetric(
                hypothesis_id=hypothesis_id,
                corr=metrics.corr,
                mmc=metrics.mmc,
                sample_count=metrics.sample_count,
                mmc_sample_count=metrics.mmc_sample_count,
                mmc_peer_count=peer_count,
                mmc_baseline_type=baseline_type,
                window_size=metrics.window_size,
            )
        )
    return results


def _lagged_corr_weight(
    *,
    prediction_series: pd.Series,
    observations: pd.Series,
    current_date: str,
    window_size: int,
) -> float:
    prior_predictions = prediction_series[prediction_series.index < current_date]
    prior_observations = observations[observations.index < current_date]
    predictions, target = _tail_aligned_pair(
        _sorted_series(prior_predictions),
        _sorted_series(prior_observations),
        window_size=window_size,
    )
    return max(numerai_corr(predictions, target), 0.0)


def _compute_meta_prediction_series(
    bundle: ValidationTargetBundle,
    *,
    aggregation_kind: str,
    window_size: int,
) -> pd.Series:
    prediction_frame = pd.concat(
        {
            hypothesis_id: _sorted_series(series)
            for hypothesis_id, series in bundle.predictions_by_hypothesis.items()
        },
        axis=1,
    ).sort_index()
    if prediction_frame.empty:
        return pd.Series(dtype=float)

    if aggregation_kind == AGGREGATION_ACTIVE_EQUAL_MEAN:
        return prediction_frame.mean(axis=1, skipna=True).dropna().astype(float)

    if aggregation_kind != AGGREGATION_CORR_WEIGHTED_MEAN:
        raise ValueError(f"unknown aggregation kind: {aggregation_kind}")

    observations = _sorted_series(bundle.observations)
    values: dict[str, float] = {}
    for evaluation_date, row in prediction_frame.iterrows():
        contributors = row.dropna()
        if contributors.empty:
            continue
        weights = {
            hypothesis_id: _lagged_corr_weight(
                prediction_series=_sorted_series(bundle.predictions_by_hypothesis[hypothesis_id]),
                observations=observations,
                current_date=str(evaluation_date),
                window_size=window_size,
            )
            for hypothesis_id in contributors.index
        }
        total_weight = float(sum(weights.values()))
        if total_weight <= 0.0:
            normalized_weights = {
                hypothesis_id: 1.0 / float(len(contributors))
                for hypothesis_id in contributors.index
            }
        else:
            normalized_weights = {
                hypothesis_id: weight / total_weight
                for hypothesis_id, weight in weights.items()
            }
        values[str(evaluation_date)] = float(
            sum(float(contributors[hypothesis_id]) * normalized_weights[hypothesis_id] for hypothesis_id in contributors.index)
        )
    return pd.Series(values, dtype=float).sort_index()


def compute_validation_meta_prediction_series(
    bundle: ValidationTargetBundle,
    *,
    aggregation_kind: str,
    window_size: int,
) -> pd.Series:
    return _compute_meta_prediction_series(
        bundle,
        aggregation_kind=aggregation_kind,
        window_size=window_size,
    )


def compute_validation_meta_metrics(
    bundle: ValidationTargetBundle,
    *,
    aggregation_kinds: tuple[str, ...],
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> list[ValidationMetaMetric]:
    observations = _sorted_series(bundle.observations)
    results: list[ValidationMetaMetric] = []
    for aggregation_kind in aggregation_kinds:
        prediction_series = _compute_meta_prediction_series(
            bundle,
            aggregation_kind=aggregation_kind,
            window_size=window_size,
        )
        predictions, target = _tail_aligned_pair(
            prediction_series,
            observations,
            window_size=window_size,
        )
        if predictions.empty:
            continue
        results.append(
            ValidationMetaMetric(
                aggregation_kind=aggregation_kind,
                corr=numerai_corr(predictions, target),
                sample_count=len(predictions),
                window_size=int(window_size),
            )
        )
    return results


def compute_validation_decision_metrics(
    bundle: ValidationTargetBundle,
    *,
    aggregation_kinds: tuple[str, ...],
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> list[ValidationDecisionMetric]:
    observations = _sorted_series(bundle.observations)
    if observations.empty:
        return []
    risk_series = (
        observations.astype(float)
        .rolling(window=max(int(window_size), 1), min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
        .astype(float)
    )
    results: list[ValidationDecisionMetric] = []
    for aggregation_kind in aggregation_kinds:
        signal_series = compute_validation_meta_prediction_series(
            bundle,
            aggregation_kind=aggregation_kind,
            window_size=window_size,
        )
        if signal_series.empty:
            continue
        backtest_result = run_decision_backtest(
            DecisionBacktestInput(
                portfolio_id="validation",
                subject_id=bundle.target_id,
                target_id=bundle.target_id,
                signal_series=signal_series,
                realized_return_series=observations,
                risk_series=risk_series.reindex(signal_series.index).fillna(0.0),
                asset=None,
            )
        )
        results.append(_decision_metric_from_result(
            backtest_result,
            aggregation_kind=aggregation_kind,
            window_size=window_size,
        ))
    return results


def _decision_metric_from_result(
    result: DecisionBacktestResult,
    *,
    aggregation_kind: str,
    window_size: int,
) -> ValidationDecisionMetric:
    return ValidationDecisionMetric(
        aggregation_kind=aggregation_kind,
        gross_return_total=result.gross_return_total,
        net_return_total=result.net_return_total,
        max_drawdown=result.max_drawdown,
        mean_turnover=result.mean_turnover,
        step_count=len(result.steps),
        window_size=int(window_size),
    )
