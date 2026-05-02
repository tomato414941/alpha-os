from __future__ import annotations

import pandas as pd


def crypto_regime_momentum_eligibility_series_by_subject(
    *,
    subject_return_series_by_subject: dict[str, pd.Series],
    funding_rate_series_by_subject: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    signal_series_by_subject: dict[str, pd.Series] = {}
    for subject_id, full_returns in subject_return_series_by_subject.items():
        funding_rate = funding_rate_series_by_subject.get(subject_id)
        if funding_rate is None:
            raise ValueError(
                "crypto regime momentum requires funding_rate series: "
                f"{subject_id}"
            )
        returns = full_returns.astype(float)
        return_7d = (
            (1.0 + returns)
            .rolling(7, min_periods=7)
            .apply(lambda values: float(values.prod() - 1.0), raw=True)
        )
        return_30d = (
            (1.0 + returns)
            .rolling(30, min_periods=30)
            .apply(lambda values: float(values.prod() - 1.0), raw=True)
        )
        aligned_funding = funding_rate.astype(float).reindex(returns.index)
        funding_median_60d = aligned_funding.rolling(
            60,
            min_periods=60,
        ).median()
        funding_overheated = (aligned_funding > 0.0) & (
            aligned_funding > funding_median_60d
        )
        signal = ((return_7d > 0.0) & (return_30d > 0.0) & ~funding_overheated)
        signal_series_by_subject[subject_id] = signal.fillna(False).astype(float)
    return signal_series_by_subject
