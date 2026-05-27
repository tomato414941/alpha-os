from __future__ import annotations

import pytest


def test_weighted_signal_series_orients_negative_corr_survivors():
    from types import SimpleNamespace

    import pandas as pd

    from alpha_os.strategy_backtest_evaluation import _weighted_signal_series

    signal_series = _weighted_signal_series(
        {
            "anti_trend": pd.Series(
                {"2026-04-01": 1.0, "2026-04-02": 0.5}
            ),
            "trend": pd.Series({"2026-04-01": 1.0, "2026-04-02": 0.5}),
        },
        survivor_metrics={
            "anti_trend": SimpleNamespace(corr=-0.75, score=0.75),
            "trend": SimpleNamespace(corr=0.25, score=0.25),
        },
    )

    assert signal_series.to_dict() == {
        "2026-04-01": pytest.approx(-0.5),
        "2026-04-02": pytest.approx(-0.25),
    }


def test_build_range_backtest_dataset_creates_subject_series():
    from types import SimpleNamespace

    import pandas as pd

    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.strategy_backtest_evaluation import (
        _bundles_by_subject,
        build_range_backtest_dataset,
    )
    from alpha_os.evaluation_snapshot import EvaluationSnapshot

    snapshots = [
        EvaluationSnapshot(
            evaluation_id="trend:2026-04-01",
            subject_id="BTC_spot",
            asset="BTC",
            target_id="return_1d",
            signal_id="trend",
            prediction_value=0.25,
            observation_value=0.05,
            signed_edge=0.01,
            absolute_error=0.0,
            input_source="test",
            input_range_start="2026-04-01",
            input_range_end="2026-04-01",
            observable_id="daily_close",
            adapter_kind="fixture",
            created_at="2026-04-02T00:00:00+00:00",
        ),
        EvaluationSnapshot(
            evaluation_id="trend:2026-04-02",
            subject_id="BTC_spot",
            asset="BTC",
            target_id="return_1d",
            signal_id="trend",
            prediction_value=0.50,
            observation_value=0.10,
            signed_edge=0.02,
            absolute_error=0.0,
            input_source="test",
            input_range_start="2026-04-02",
            input_range_end="2026-04-02",
            observable_id="daily_close",
            adapter_kind="fixture",
            created_at="2026-04-03T00:00:00+00:00",
        ),
    ]
    survivor_metrics = {"trend": SimpleNamespace(corr=0.5, score=0.5)}

    dataset = build_range_backtest_dataset(
        date_range=EvaluationDateRange(
            label="fold",
            start_date="2026-04-01",
            end_date="2026-04-02",
        ),
        snapshots=snapshots,
        all_bundles_by_subject=_bundles_by_subject(
            snapshots,
            survivor_metrics=survivor_metrics,
        ),
        survivor_metrics=survivor_metrics,
        component_by_subject_id={"BTC_spot": SimpleNamespace(confidence=0.75)},
        metric_window=2,
        funding_cost_bps_series_by_subject={
            "BTC_spot": pd.Series({"2026-04-01": 1.0}, dtype=float)
        },
        borrow_fee_bps_series_by_subject={},
        roll_cost_bps_series_by_subject={},
        contract_multiplier_by_subject={"BTC_spot": 5.0},
    )

    assert dataset is not None
    assert dataset.label == "fold"
    assert dataset.predictive_corr == pytest.approx(1.0)
    assert dataset.dependence_series == ()
    subject_series = dataset.subject_series[0]
    assert subject_series.subject_id == "BTC_spot"
    assert subject_series.signal_series.to_dict() == {
        "2026-04-01": pytest.approx(0.25),
        "2026-04-02": pytest.approx(0.5),
    }
    assert subject_series.confidence_series.to_dict() == {
        "2026-04-01": pytest.approx(0.75),
        "2026-04-02": pytest.approx(0.75),
    }
    assert subject_series.contract_multiplier == pytest.approx(5.0)


def test_build_direct_range_backtest_dataset_fills_missing_signals_with_zero():
    import pandas as pd

    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.strategy_backtest_evaluation import (
        build_direct_range_backtest_dataset,
    )

    dataset = build_direct_range_backtest_dataset(
        date_range=EvaluationDateRange(
            label="direct",
            start_date="2026-04-01",
            end_date="2026-04-02",
        ),
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

    assert dataset is not None
    series_by_subject = {item.subject_id: item.signal_series for item in dataset.subject_series}
    assert series_by_subject["A"].to_dict() == {
        "2026-04-01": pytest.approx(0.5),
        "2026-04-02": pytest.approx(0.0),
    }
    assert series_by_subject["B"].to_dict() == {
        "2026-04-01": pytest.approx(0.0),
        "2026-04-02": pytest.approx(0.0),
    }
