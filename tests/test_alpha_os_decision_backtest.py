from __future__ import annotations

import pandas as pd
import pytest


def test_run_decision_backtest_replays_signal_into_equity_curve():
    from alpha_os.decision_backtest import DecisionBacktestInput, run_decision_backtest

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_id="paper_core",
            subject_id="BTC",
            target_id="residual_return_3d",
            signal_series=pd.Series(
                {"2026-03-24": 0.5, "2026-03-25": 0.25},
                dtype=float,
            ),
            realized_return_series=pd.Series(
                {"2026-03-24": 0.1, "2026-03-25": 0.2},
                dtype=float,
            ),
        )
    )

    assert len(result.steps) == 2
    assert result.steps[0].target_weight == pytest.approx(0.5)
    assert result.steps[0].gross_return == pytest.approx(0.05)
    assert result.steps[0].net_equity == pytest.approx(1.05)
    assert result.steps[1].target_weight == pytest.approx(0.3333333333)
    assert result.gross_return_total == pytest.approx(0.12)
    assert result.net_return_total == pytest.approx(0.12)


def test_run_decision_backtest_respects_no_trade_band_and_turnover_cost():
    from alpha_os.decision_backtest import DecisionBacktestInput, run_decision_backtest

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_id="paper_core",
            subject_id="BTC",
            target_id="residual_return_3d",
            signal_series=pd.Series(
                {"2026-03-24": 0.5, "2026-03-25": 0.45},
                dtype=float,
            ),
            realized_return_series=pd.Series(
                {"2026-03-24": 0.1, "2026-03-25": 0.1},
                dtype=float,
            ),
            no_trade_band=0.05,
            turnover_penalty=0.1,
        )
    )

    assert result.steps[0].turnover == pytest.approx(0.4166666667)
    assert result.steps[0].cost == pytest.approx(0.0416666667)
    assert result.steps[1].target_weight == pytest.approx(result.steps[0].target_weight)
    assert result.steps[1].position_delta == pytest.approx(0.0)
    assert result.steps[1].turnover == pytest.approx(0.0)
    assert result.steps[1].cost == pytest.approx(0.0)


def test_run_decision_backtest_tracks_drawdown_and_risk_scaling():
    from alpha_os.decision_backtest import DecisionBacktestInput, run_decision_backtest

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_id="paper_core",
            subject_id="BTC",
            target_id="residual_return_3d",
            signal_series=pd.Series(
                {"2026-03-24": 1.0, "2026-03-25": 1.0},
                dtype=float,
            ),
            realized_return_series=pd.Series(
                {"2026-03-24": 0.1, "2026-03-25": -0.2},
                dtype=float,
            ),
            risk_series=pd.Series(
                {"2026-03-24": 0.0, "2026-03-25": 1.0},
                dtype=float,
            ),
            gross_exposure_cap=1.0,
        )
    )

    assert result.steps[0].risk_scale == pytest.approx(1.0)
    assert result.steps[1].risk_scale == pytest.approx(0.5)
    assert result.steps[1].target_weight == pytest.approx(0.75)
    assert result.max_drawdown == pytest.approx(0.15)


def test_run_decision_backtest_feeds_state_into_next_decision():
    from alpha_os.decision_backtest import DecisionBacktestInput, run_decision_backtest

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_id="paper_core",
            subject_id="BTC",
            target_id="residual_return_3d",
            signal_series=pd.Series(
                {"2026-03-24": 1.0, "2026-03-25": 1.0},
                dtype=float,
            ),
            realized_return_series=pd.Series(
                {"2026-03-24": -0.2, "2026-03-25": 0.1},
                dtype=float,
            ),
        )
    )

    assert result.steps[0].target_weight == pytest.approx(1.0)
    assert result.steps[1].risk_scale < 1.0
    assert result.steps[1].target_weight < result.steps[0].target_weight
