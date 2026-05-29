from __future__ import annotations

import pytest


def test_build_direct_range_backtest_dataset_fills_missing_signals_with_zero():
    import pandas as pd

    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.range_backtest_dataset import (
        build_direct_range_backtest_dataset,
    )

    subject_series = build_direct_range_backtest_dataset(
        date_range=EvaluationDateRange(
            label="direct",
            start_date="2026-04-01",
            end_date="2026-04-02",
        ),
        target_id="residual_return_3d",
        subject_return_series_by_subject={
            "A": pd.Series(
                {"2026-04-01": 0.01, "2026-04-02": 0.02},
                dtype=float,
            ),
            "B": pd.Series(
                {"2026-04-01": -0.01, "2026-04-02": -0.02},
                dtype=float,
            ),
        },
        signal_series_by_subject={
            "A": pd.Series({"2026-04-01": 0.5}, dtype=float),
        },
        funding_cost_bps_series_by_subject=None,
        borrow_fee_bps_series_by_subject=None,
        roll_cost_bps_series_by_subject=None,
        contract_multiplier_by_subject=None,
        signal_value=1.0,
    )

    assert subject_series is not None
    series_by_subject = {item.subject_id: item.signal_series for item in subject_series}
    assert series_by_subject["A"].to_dict() == {
        "2026-04-01": pytest.approx(0.5),
        "2026-04-02": pytest.approx(0.0),
    }
    assert series_by_subject["B"].to_dict() == {
        "2026-04-01": pytest.approx(0.0),
        "2026-04-02": pytest.approx(0.0),
    }
