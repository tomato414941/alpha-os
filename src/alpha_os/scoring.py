from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_METRIC_WINDOW = 20
_NORMAL = NormalDist()


@dataclass(frozen=True)
class HypothesisMetrics:
    corr: float
    mmc: float | None
    sample_count: int
    mmc_sample_count: int
    window_size: int


def _gaussianize(series: pd.Series) -> pd.Series:
    ranked = (series.rank(method="average") - 0.5) / series.count()
    clipped = ranked.clip(lower=1e-6, upper=1.0 - 1e-6)
    values = np.array([_NORMAL.inv_cdf(float(value)) for value in clipped], dtype=float)
    return pd.Series(values, index=series.index, dtype=float)


def _aligned_series(*series_list: pd.Series) -> list[pd.Series]:
    if not series_list:
        return []
    frame = pd.concat(series_list, axis=1, join="inner").dropna()
    return [frame.iloc[:, idx].astype(float) for idx in range(frame.shape[1])]


def numerai_corr(predictions: pd.Series, target: pd.Series) -> float:
    aligned_predictions, aligned_target = _aligned_series(predictions, target)
    if len(aligned_predictions) < 2:
        return 0.0

    gauss_ranked_predictions = _gaussianize(aligned_predictions)
    centered_target = aligned_target - aligned_target.mean()
    predictions_p15 = np.sign(gauss_ranked_predictions) * np.abs(gauss_ranked_predictions) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    corr = np.corrcoef(predictions_p15, target_p15)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def meta_model_contribution(
    predictions: pd.Series,
    meta_model: pd.Series,
    target: pd.Series,
) -> float | None:
    aligned_predictions, aligned_meta_model, aligned_target = _aligned_series(
        predictions,
        meta_model,
        target,
    )
    if len(aligned_predictions) < 2:
        return None

    p = _gaussianize(aligned_predictions).to_numpy(dtype=float)
    m = _gaussianize(aligned_meta_model).to_numpy(dtype=float)
    centered_target = (aligned_target - aligned_target.mean()).to_numpy(dtype=float)

    denominator = float(np.dot(m, m))
    if denominator <= 0.0:
        return None
    neutral_predictions = p - ((np.dot(p, m) / denominator) * m)
    mmc = float(np.dot(centered_target, neutral_predictions) / len(centered_target))
    if np.isnan(mmc):
        return None
    return mmc


def compute_hypothesis_metrics(
    *,
    predictions: pd.Series,
    target: pd.Series,
    meta_model: pd.Series | None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> HypothesisMetrics:
    aligned_predictions, aligned_target = _aligned_series(predictions, target)
    sample_count = len(aligned_predictions)
    corr = numerai_corr(aligned_predictions, aligned_target)
    mmc_sample_count = 0
    mmc = None
    if meta_model is not None:
        mmc_aligned_predictions, mmc_aligned_meta_model, mmc_aligned_target = _aligned_series(
            aligned_predictions,
            meta_model,
            aligned_target,
        )
        mmc_sample_count = len(mmc_aligned_predictions)
        mmc = meta_model_contribution(
            mmc_aligned_predictions,
            mmc_aligned_meta_model,
            mmc_aligned_target,
        )
    return HypothesisMetrics(
        corr=corr,
        mmc=mmc,
        sample_count=sample_count,
        mmc_sample_count=mmc_sample_count,
        window_size=int(window_size),
    )
