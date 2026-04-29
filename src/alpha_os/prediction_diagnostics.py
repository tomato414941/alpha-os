from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import pandas as pd

from .scoring import numerai_corr


@dataclass(frozen=True)
class PredictionDiagnostics:
    mean_signal_forward_corr: float
    mean_signal_hit_rate: float
    mean_long_short_forward_spread: float
    long_bucket_return: float
    short_bucket_return: float
    coverage: float
    positive_group_fraction: float = 0.0
    group_diagnostics: dict[str, "PredictionDiagnostics"] = field(
        default_factory=dict
    )


def build_prediction_diagnostics(
    *,
    signal_series_by_subject: dict[str, pd.Series],
    forward_return_series_by_subject: dict[str, pd.Series],
    group_by_subject: dict[str, str] | None = None,
    bucket_fraction: float = 0.30,
) -> PredictionDiagnostics:
    aligned = _aligned_prediction_frame(
        signal_series_by_subject=signal_series_by_subject,
        forward_return_series_by_subject=forward_return_series_by_subject,
    )
    if aligned.empty:
        return PredictionDiagnostics(
            mean_signal_forward_corr=0.0,
            mean_signal_hit_rate=0.0,
            mean_long_short_forward_spread=0.0,
            long_bucket_return=0.0,
            short_bucket_return=0.0,
            coverage=0.0,
        )
    row_metrics = [
        _cross_sectional_prediction_metrics(row, bucket_fraction=bucket_fraction)
        for _, row in aligned.groupby(level="date")
    ]
    group_diagnostics: dict[str, PredictionDiagnostics] = {}
    if group_by_subject:
        for group_name in sorted(set(group_by_subject.values())):
            group_subjects = {
                subject_id
                for subject_id, item_group in group_by_subject.items()
                if item_group == group_name
            }
            group_signals = {
                subject_id: series
                for subject_id, series in signal_series_by_subject.items()
                if subject_id in group_subjects
            }
            group_returns = {
                subject_id: series
                for subject_id, series in forward_return_series_by_subject.items()
                if subject_id in group_subjects
            }
            group_diagnostics[group_name] = build_prediction_diagnostics(
                signal_series_by_subject=group_signals,
                forward_return_series_by_subject=group_returns,
                group_by_subject=None,
                bucket_fraction=bucket_fraction,
            )
    possible_count = _possible_observation_count(
        signal_series_by_subject,
        forward_return_series_by_subject,
    )
    positive_group_fraction = (
        0.0
        if not group_diagnostics
        else sum(
            1
            for item in group_diagnostics.values()
            if item.mean_long_short_forward_spread > 0.0
        )
        / float(len(group_diagnostics))
    )
    return PredictionDiagnostics(
        mean_signal_forward_corr=_mean([item["corr"] for item in row_metrics]),
        mean_signal_hit_rate=_mean([item["hit_rate"] for item in row_metrics]),
        mean_long_short_forward_spread=_mean(
            [item["long_short_spread"] for item in row_metrics]
        ),
        long_bucket_return=_mean([item["long_bucket_return"] for item in row_metrics]),
        short_bucket_return=_mean([item["short_bucket_return"] for item in row_metrics]),
        coverage=(
            0.0 if possible_count <= 0 else float(len(aligned)) / float(possible_count)
        ),
        positive_group_fraction=positive_group_fraction,
        group_diagnostics=group_diagnostics,
    )


def _aligned_prediction_frame(
    *,
    signal_series_by_subject: dict[str, pd.Series],
    forward_return_series_by_subject: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, signal_series in signal_series_by_subject.items():
        return_series = forward_return_series_by_subject.get(subject_id)
        if return_series is None:
            continue
        aligned = pd.concat(
            [
                signal_series.astype(float).rename("signal"),
                return_series.astype(float).rename("forward_return"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        for date, row in aligned.iterrows():
            rows.append(
                {
                    "date": date,
                    "subject_id": subject_id,
                    "signal": float(row["signal"]),
                    "forward_return": float(row["forward_return"]),
                }
            )
    if not rows:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["date", "subject_id"],
        )
        return pd.DataFrame(
            columns=["signal", "forward_return"],
            index=empty_index,
        )
    frame = pd.DataFrame(rows).set_index(["date", "subject_id"]).sort_index()
    return frame[["signal", "forward_return"]]


def _cross_sectional_prediction_metrics(
    frame: pd.DataFrame,
    *,
    bucket_fraction: float,
) -> dict[str, float]:
    if len(frame) < 2:
        corr = 0.0
    else:
        try:
            corr = float(
                numerai_corr(
                    frame["signal"].astype(float),
                    frame["forward_return"].astype(float),
                )
            )
        except Exception:
            corr = 0.0
    signal_sign = frame["signal"].astype(float).apply(_sign)
    return_sign = frame["forward_return"].astype(float).apply(_sign)
    non_zero = (signal_sign != 0) & (return_sign != 0)
    hit_rate = (
        0.0
        if not bool(non_zero.any())
        else float((signal_sign[non_zero] == return_sign[non_zero]).mean())
    )
    bucket_size = max(1, int(ceil(len(frame) * max(float(bucket_fraction), 0.0))))
    ranked = frame.sort_values("signal", ascending=False)
    long_return = float(ranked.head(bucket_size)["forward_return"].mean())
    short_return = float(ranked.tail(bucket_size)["forward_return"].mean())
    return {
        "corr": corr,
        "hit_rate": hit_rate,
        "long_short_spread": long_return - short_return,
        "long_bucket_return": long_return,
        "short_bucket_return": short_return,
    }


def _possible_observation_count(
    signal_series_by_subject: dict[str, pd.Series],
    forward_return_series_by_subject: dict[str, pd.Series],
) -> int:
    count = 0
    for subject_id, signal_series in signal_series_by_subject.items():
        return_series = forward_return_series_by_subject.get(subject_id)
        if return_series is None:
            continue
        count += len(signal_series.index.union(return_series.index))
    return count


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
