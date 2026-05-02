from __future__ import annotations

import pandas as pd
import pytest

from alpha_os.portfolio_allocation import PositionCandidate
from alpha_os.simple_candidate_backtest import run_equal_weight_long_only_backtest


def test_equal_weight_long_only_backtest_allocates_active_candidates() -> None:
    result = run_equal_weight_long_only_backtest(
        returns_by_subject={
            "BTC": pd.Series(
                [0.10, 0.20],
                index=("2026-01-01", "2026-01-02"),
            ),
            "ETH": pd.Series(
                [0.00, 0.10],
                index=("2026-01-01", "2026-01-02"),
            ),
        },
        candidates_by_date={
            "2026-01-01": (
                PositionCandidate(subject_id="BTC", direction="long"),
                PositionCandidate(subject_id="ETH", direction="flat"),
            ),
            "2026-01-02": (
                PositionCandidate(subject_id="BTC", direction="long"),
                PositionCandidate(subject_id="ETH", direction="long"),
            ),
        },
    )

    assert result.daily_returns.loc["2026-01-01", "gross_return"] == pytest.approx(0.10)
    assert result.daily_returns.loc["2026-01-02", "gross_return"] == pytest.approx(0.15)
    assert result.daily_returns.loc["2026-01-02", "active_assets"] == pytest.approx(2.0)


def test_equal_weight_long_only_backtest_charges_turnover_cost() -> None:
    result = run_equal_weight_long_only_backtest(
        returns_by_subject={
            "BTC": pd.Series([0.10], index=("2026-01-01",)),
            "ETH": pd.Series([0.00], index=("2026-01-01",)),
        },
        candidates_by_date={
            "2026-01-01": (
                PositionCandidate(subject_id="BTC", direction="long"),
                PositionCandidate(subject_id="ETH", direction="long"),
            ),
        },
        cost_bps_per_unit_turnover=10.0,
    )

    assert result.daily_returns.loc["2026-01-01", "gross_return"] == pytest.approx(0.05)
    assert result.daily_returns.loc["2026-01-01", "turnover"] == pytest.approx(1.0)
    assert result.daily_returns.loc["2026-01-01", "net_return"] == pytest.approx(0.049)


def test_equal_weight_long_only_backtest_flats_missing_candidate_dates() -> None:
    result = run_equal_weight_long_only_backtest(
        returns_by_subject={
            "BTC": pd.Series([0.10], index=("2026-01-01",)),
        },
        candidates_by_date={},
    )

    assert result.daily_returns.loc["2026-01-01", "gross_return"] == pytest.approx(0.0)
    assert result.daily_returns.loc["2026-01-01", "turnover"] == pytest.approx(0.0)
