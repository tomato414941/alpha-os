from __future__ import annotations

import pandas as pd
import pytest


def test_prediction_diagnostics_detects_positive_cross_sectional_signal():
    from alpha_os.prediction_diagnostics import build_prediction_diagnostics

    index = pd.Index(["2024-01-01", "2024-01-02"])
    diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={
            "A": pd.Series([1.0, 1.0], index=index),
            "B": pd.Series([-1.0, -1.0], index=index),
        },
        forward_return_series_by_subject={
            "A": pd.Series([0.02, 0.03], index=index),
            "B": pd.Series([-0.01, -0.02], index=index),
        },
    )

    assert diagnostics.mean_signal_forward_corr > 0.0
    assert diagnostics.mean_signal_hit_rate == pytest.approx(1.0)
    assert diagnostics.mean_long_short_forward_spread > 0.0
    assert diagnostics.long_bucket_return > diagnostics.short_bucket_return
    assert diagnostics.coverage == pytest.approx(1.0)


def test_prediction_diagnostics_detects_negative_cross_sectional_signal():
    from alpha_os.prediction_diagnostics import build_prediction_diagnostics

    index = pd.Index(["2024-01-01", "2024-01-02"])
    diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={
            "A": pd.Series([1.0, 1.0], index=index),
            "B": pd.Series([-1.0, -1.0], index=index),
        },
        forward_return_series_by_subject={
            "A": pd.Series([-0.02, -0.03], index=index),
            "B": pd.Series([0.01, 0.02], index=index),
        },
    )

    assert diagnostics.mean_signal_forward_corr < 0.0
    assert diagnostics.mean_signal_hit_rate == pytest.approx(0.0)
    assert diagnostics.mean_long_short_forward_spread < 0.0


def test_prediction_diagnostics_reports_group_fraction():
    from alpha_os.prediction_diagnostics import build_prediction_diagnostics

    index = pd.Index(["2024-01-01", "2024-01-02"])
    diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={
            "A": pd.Series([1.0, 1.0], index=index),
            "B": pd.Series([-1.0, -1.0], index=index),
            "C": pd.Series([1.0, 1.0], index=index),
            "D": pd.Series([-1.0, -1.0], index=index),
        },
        forward_return_series_by_subject={
            "A": pd.Series([0.02, 0.03], index=index),
            "B": pd.Series([-0.01, -0.02], index=index),
            "C": pd.Series([-0.02, -0.03], index=index),
            "D": pd.Series([0.01, 0.02], index=index),
        },
        group_by_subject={
            "A": "good",
            "B": "good",
            "C": "bad",
            "D": "bad",
        },
    )

    assert diagnostics.positive_group_fraction == pytest.approx(0.5)
    assert diagnostics.group_diagnostics["good"].mean_long_short_forward_spread > 0.0
    assert diagnostics.group_diagnostics["bad"].mean_long_short_forward_spread < 0.0


def test_prediction_diagnostics_handles_empty_inputs():
    from alpha_os.prediction_diagnostics import build_prediction_diagnostics

    diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={},
        forward_return_series_by_subject={},
    )

    assert diagnostics.mean_signal_forward_corr == pytest.approx(0.0)
    assert diagnostics.coverage == pytest.approx(0.0)
