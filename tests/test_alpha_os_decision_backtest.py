from __future__ import annotations

import pandas as pd
import pytest

from alpha_os.evaluation_cost_config import TradingEnvironment
from alpha_os.portfolio_construction_config import PortfolioConstructionSpec


def _subject_step(step, subject_id: str):
    return step.subject_step_by_subject[subject_id]


def test_run_decision_backtest_replays_signal_into_equity_curve():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series(
                        {"2026-03-24": 0.5, "2026-03-25": 0.25},
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        {"2026-03-24": 0.1, "2026-03-25": 0.2},
                        dtype=float,
                    ),
                ),
            ),
        )
    )

    first_step = _subject_step(result.steps[0], "BTC")
    second_step = _subject_step(result.steps[1], "BTC")

    assert len(result.steps) == 2
    assert result.subject_ids == ("BTC",)
    assert first_step.target_weight == pytest.approx(0.5)
    assert first_step.target_notional == pytest.approx(0.5)
    assert first_step.traded_notional == pytest.approx(0.5)
    assert result.steps[0].gross_return == pytest.approx(0.05)
    assert result.steps[0].gross_pnl_notional == pytest.approx(0.05)
    assert result.steps[0].gross_leverage_exposure == pytest.approx(0.5)
    assert result.steps[0].net_leverage_exposure == pytest.approx(0.5)
    assert result.steps[0].long_leverage_exposure == pytest.approx(0.5)
    assert result.steps[0].short_leverage_exposure == pytest.approx(0.0)
    assert result.steps[0].gross_notional_exposure == pytest.approx(0.5)
    assert result.steps[0].net_notional_exposure == pytest.approx(0.5)
    assert result.steps[0].long_notional_exposure == pytest.approx(0.5)
    assert result.steps[0].short_notional_exposure == pytest.approx(0.0)
    assert result.steps[0].net_equity == pytest.approx(1.05)
    assert second_step.target_weight == pytest.approx(0.25)
    assert second_step.target_notional == pytest.approx(0.2625)
    assert result.steps[1].gross_leverage_exposure == pytest.approx(0.25)
    assert result.steps[1].net_leverage_exposure == pytest.approx(0.25)
    assert result.steps[1].long_leverage_exposure == pytest.approx(0.25)
    assert result.steps[1].short_leverage_exposure == pytest.approx(0.0)
    assert result.gross_return_total == pytest.approx(0.1025)
    assert result.net_return_total == pytest.approx(0.1025)
    assert result.mean_traded_notional == pytest.approx(0.39375)
    assert result.mean_gross_leverage_exposure == pytest.approx(0.375)
    assert result.mean_net_leverage_exposure == pytest.approx(0.375)
    assert result.mean_long_leverage_exposure == pytest.approx(0.375)
    assert result.mean_short_leverage_exposure == pytest.approx(0.0)
    assert result.mean_gross_notional_exposure == pytest.approx(0.38125)
    assert result.mean_long_notional_exposure == pytest.approx(0.38125)
    assert result.mean_short_notional_exposure == pytest.approx(0.0)
    assert result.cost_notional_total == pytest.approx(0.0)
    assert result.funding_cost_notional_total == pytest.approx(0.0)
    assert result.borrow_cost_notional_total == pytest.approx(0.0)
    assert result.roll_cost_notional_total == pytest.approx(0.0)


def test_run_decision_backtest_reports_leverage_separately_from_notional():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_construction=PortfolioConstructionSpec(gross_leverage_cap=1.0),
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="ES",
                    signal_series=pd.Series(
                        {"2026-03-24": 1.0, "2026-03-25": 1.0},
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        {"2026-03-24": 1.0, "2026-03-25": 0.0},
                        dtype=float,
                    ),
                ),
            ),
        )
    )

    assert result.steps[0].gross_leverage_exposure == pytest.approx(1.0)
    assert result.steps[0].gross_notional_exposure == pytest.approx(1.0)
    assert result.steps[1].gross_leverage_exposure == pytest.approx(1.0)
    assert result.steps[1].gross_notional_exposure == pytest.approx(2.0)
    assert result.mean_gross_leverage_exposure == pytest.approx(1.0)
    assert result.mean_gross_notional_exposure == pytest.approx(1.5)


def test_run_decision_backtest_can_run_short_only_direction_mode():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_construction=PortfolioConstructionSpec(
                direction_mode="short_only"
            ),
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="LONG_SIGNAL",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="SHORT_SIGNAL",
                    signal_series=pd.Series({"2026-03-24": -0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
        )
    )

    long_step = _subject_step(result.steps[0], "LONG_SIGNAL")
    short_step = _subject_step(result.steps[0], "SHORT_SIGNAL")

    assert long_step.target_weight == 0.0
    assert short_step.target_weight == pytest.approx(-0.5)
    assert result.steps[0].short_leverage_exposure == pytest.approx(0.5)
    assert result.steps[0].gross_return == pytest.approx(-0.05)


def test_run_decision_backtest_drifts_current_weights_between_rebalances():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    index = ["2026-03-24", "2026-03-25", "2026-03-26"]
    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_construction=PortfolioConstructionSpec(
                gross_exposure_cap=1.0,
                rebalance_interval_steps=2,
                direction_mode="long_only",
            ),
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="A",
                    signal_series=pd.Series([1.0, 1.0, 1.0], index=index, dtype=float),
                    realized_return_series=pd.Series(
                        [1.0, 0.0, 0.0],
                        index=index,
                        dtype=float,
                    ),
                ),
                SubjectBacktestSeries(
                    subject_id="B",
                    signal_series=pd.Series([1.0, 1.0, 1.0], index=index, dtype=float),
                    realized_return_series=pd.Series(
                        [0.0, 0.0, 0.0],
                        index=index,
                        dtype=float,
                    ),
                ),
            ),
        )
    )

    second_a = _subject_step(result.steps[1], "A")
    second_b = _subject_step(result.steps[1], "B")

    assert result.steps[1].turnover == pytest.approx(0.0)
    assert second_a.target_weight == pytest.approx(2.0 / 3.0)
    assert second_a.target_notional == pytest.approx(1.0)
    assert second_b.target_weight == pytest.approx(1.0 / 3.0)
    assert second_b.target_notional == pytest.approx(0.5)
    assert result.steps[2].turnover == pytest.approx(1.0 / 3.0)


def test_advance_portfolio_state_drifts_weights_after_returns():
    from alpha_os.decision_backtest import (
        BacktestStepAccounting,
        DecisionBacktestSubjectStep,
        PortfolioBacktestState,
        advance_portfolio_state,
    )

    state = PortfolioBacktestState(
        current_weights={"A": 0.5, "B": 0.5},
        gross_equity=1.0,
        net_equity=1.0,
        net_peak_equity=1.0,
        current_drawdown=0.0,
        holding_period_days=0,
        recent_turnover=0.0,
        rebalance_step=0,
    )
    subject_steps = (
        DecisionBacktestSubjectStep(
            subject_id="A",
            signal_value=1.0,
            realized_return=1.0,
            target_weight=0.5,
            position_delta=0.5,
            target_notional=0.5,
            traded_notional=0.5,
            risk_scale=1.0,
            entry_allowed=True,
        ),
        DecisionBacktestSubjectStep(
            subject_id="B",
            signal_value=1.0,
            realized_return=0.0,
            target_weight=0.5,
            position_delta=0.5,
            target_notional=0.5,
            traded_notional=0.5,
            risk_scale=1.0,
            entry_allowed=True,
        ),
    )
    accounting = BacktestStepAccounting(
        gross_pnl_notional=0.5,
        gross_return=0.5,
        gross_leverage_exposure=1.0,
        net_leverage_exposure=1.0,
        long_leverage_exposure=1.0,
        short_leverage_exposure=0.0,
        gross_notional_exposure=1.0,
        net_notional_exposure=1.0,
        long_notional_exposure=1.0,
        short_notional_exposure=0.0,
        turnover=1.0,
        traded_notional=1.0,
        cost=0.0,
        cost_notional=0.0,
        net_pnl_notional=0.5,
        net_return=0.5,
        funding_cost_notional=0.0,
        borrow_cost_notional=0.0,
        roll_cost_notional=0.0,
    )

    next_state = advance_portfolio_state(
        state,
        subject_steps=subject_steps,
        accounting=accounting,
    )

    assert next_state.current_weights["A"] == pytest.approx(2.0 / 3.0)
    assert next_state.current_weights["B"] == pytest.approx(1.0 / 3.0)
    assert next_state.net_equity == pytest.approx(1.5)
    assert next_state.holding_period_days == 1
    assert next_state.recent_turnover == pytest.approx(1.0)
    assert next_state.rebalance_step == 1


def test_build_backtest_step_accounting_splits_cost_components():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        DecisionBacktestSubjectStep,
        build_backtest_step_accounting,
    )

    accounting = build_backtest_step_accounting(
        subject_steps=(
            DecisionBacktestSubjectStep(
                subject_id="A",
                signal_value=1.0,
                realized_return=0.1,
                target_weight=0.6,
                position_delta=0.2,
                target_notional=0.6,
                traded_notional=0.2,
                risk_scale=1.0,
                entry_allowed=True,
                funding_cost_bps=1.0,
                roll_cost_bps=2.0,
            ),
            DecisionBacktestSubjectStep(
                subject_id="B",
                signal_value=-1.0,
                realized_return=-0.2,
                target_weight=-0.4,
                position_delta=-0.1,
                target_notional=-0.4,
                traded_notional=0.1,
                risk_scale=1.0,
                entry_allowed=True,
                borrow_fee_bps=3.0,
            ),
        ),
        capital_base=1.0,
        backtest_input=DecisionBacktestInput(
            subject_series=(),
            trading_environment=TradingEnvironment(
                turnover_cost_rate=0.01,
                market_impact_bps=1.0,
                fee_bps=2.0,
                bid_ask_spread_bps=3.0,
                funding_bps_per_step=4.0,
                borrow_fee_bps_per_step=5.0,
            ),
        ),
    )

    assert accounting.gross_pnl_notional == pytest.approx(0.14)
    assert accounting.turnover == pytest.approx(0.3)
    assert accounting.traded_notional == pytest.approx(0.3)
    assert accounting.gross_leverage_exposure == pytest.approx(1.0)
    assert accounting.net_leverage_exposure == pytest.approx(0.2)
    assert accounting.gross_notional_exposure == pytest.approx(1.0)
    assert accounting.net_notional_exposure == pytest.approx(0.2)
    assert accounting.funding_cost_notional == pytest.approx(0.00046)
    assert accounting.borrow_cost_notional == pytest.approx(0.00032)
    assert accounting.roll_cost_notional == pytest.approx(0.00012)
    assert accounting.cost_notional == pytest.approx(0.00408)
    assert accounting.net_pnl_notional == pytest.approx(0.13592)


def test_run_decision_backtest_charges_turnover_cost():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series(
                        {"2026-03-24": 0.5, "2026-03-25": 0.45},
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        {"2026-03-24": 0.1, "2026-03-25": 0.1},
                        dtype=float,
                    ),
                ),
            ),
            trading_environment=TradingEnvironment(turnover_cost_rate=0.1),
        )
    )

    second_step = _subject_step(result.steps[1], "BTC")

    assert result.steps[0].turnover == pytest.approx(0.5)
    assert result.steps[0].traded_notional == pytest.approx(0.5)
    assert result.steps[0].cost == pytest.approx(0.05)
    assert result.steps[0].cost_notional == pytest.approx(0.05)
    assert second_step.target_weight == pytest.approx(0.45)
    assert second_step.position_delta == pytest.approx(-0.1)
    assert result.steps[1].turnover == pytest.approx(0.1)
    assert result.steps[1].cost == pytest.approx(0.01)


def test_run_decision_backtest_charges_execution_fee_bps():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
            trading_environment=TradingEnvironment(fee_bps=10.0),
        )
    )

    assert result.steps[0].traded_notional == pytest.approx(0.5)
    assert result.steps[0].cost_notional == pytest.approx(0.0005)
    assert result.cost_notional_total == pytest.approx(0.0005)


def test_run_decision_backtest_charges_bid_ask_spread_bps():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
            trading_environment=TradingEnvironment(bid_ask_spread_bps=20.0),
        )
    )

    assert result.steps[0].traded_notional == pytest.approx(0.5)
    assert result.steps[0].cost_notional == pytest.approx(0.001)
    assert result.cost_notional_total == pytest.approx(0.001)


def test_run_decision_backtest_charges_funding_on_gross_exposure():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
            trading_environment=TradingEnvironment(funding_bps_per_step=10.0),
        )
    )

    assert result.steps[0].gross_notional_exposure == pytest.approx(0.5)
    assert result.steps[0].cost_notional == pytest.approx(0.0005)
    assert result.steps[0].funding_cost_notional == pytest.approx(0.0005)
    assert result.funding_cost_notional_total == pytest.approx(0.0005)
    assert result.steps[0].borrow_cost_notional == pytest.approx(0.0)
    assert result.steps[0].roll_cost_notional == pytest.approx(0.0)
    assert result.steps[0].short_notional_exposure == pytest.approx(0.0)
    assert result.borrow_cost_notional_total == pytest.approx(0.0)


def test_run_decision_backtest_tracks_subject_specific_cost_breakdown():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="ES",
                    signal_series=pd.Series({"2026-03-24": -0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                    funding_cost_bps_series=pd.Series({"2026-03-24": 4.0}, dtype=float),
                    borrow_fee_bps_series=pd.Series({"2026-03-24": 6.0}, dtype=float),
                    roll_cost_bps_series=pd.Series({"2026-03-24": 8.0}, dtype=float),
                ),
            ),
        )
    )

    step = result.steps[0]

    assert step.long_notional_exposure == pytest.approx(0.0)
    assert step.short_notional_exposure == pytest.approx(0.5)
    assert step.funding_cost_notional == pytest.approx(-0.0002)
    assert step.borrow_cost_notional == pytest.approx(0.0003)
    assert step.roll_cost_notional == pytest.approx(0.0004)
    assert result.funding_cost_notional_total == pytest.approx(-0.0002)
    assert result.borrow_cost_notional_total == pytest.approx(0.0003)
    assert result.roll_cost_notional_total == pytest.approx(0.0004)


def test_run_decision_backtest_charges_borrow_fee_on_short_exposure():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series({"2026-03-24": -0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
            trading_environment=TradingEnvironment(borrow_fee_bps_per_step=10.0),
        )
    )

    assert result.steps[0].gross_notional_exposure == pytest.approx(0.5)
    assert result.steps[0].cost_notional == pytest.approx(0.0005)


def test_run_decision_backtest_tracks_drawdown_and_risk_scaling():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
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
                ),
            ),
            portfolio_construction=PortfolioConstructionSpec(gross_exposure_cap=1.0),
        )
    )

    first_step = _subject_step(result.steps[0], "BTC")
    second_step = _subject_step(result.steps[1], "BTC")

    assert first_step.risk_scale == pytest.approx(1.0)
    assert second_step.risk_scale == pytest.approx(0.5)
    assert second_step.target_weight == pytest.approx(0.5)
    assert result.max_drawdown == pytest.approx(0.1)


def test_constrained_targets_by_subject_respects_gross_leverage_cap():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget
    
    targets = (
        PortfolioTarget(
            subject_id="A",
            target_weight=0.8,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
        PortfolioTarget(
            subject_id="B",
            target_weight=-0.6,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
    )

    constrained = constrained_targets_by_subject(
        targets,
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=1.0,
        net_exposure_target=None,
        top_k=None,
        active_weight_budget=0.0,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["A"].target_weight == pytest.approx(4.0 / 7.0)
    assert constrained["B"].target_weight == pytest.approx(-3.0 / 7.0)
    assert sum(abs(item.target_weight) for item in constrained.values()) == pytest.approx(1.0)


def test_portfolio_construction_pipeline_returns_stage_trace():
    from alpha_os.portfolio_construction_pipeline import (
        build_portfolio_construction_request,
        construct_portfolio_targets,
    )
    from alpha_os.portfolio_decision import PortfolioTarget
    
    request = build_portfolio_construction_request(
        targets=(
            PortfolioTarget(subject_id="A", target_weight=0.10, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=-0.10, position_delta=0.0),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=0.40,
        net_exposure_target=None,
        top_k=None,
        active_weight_budget=0.0,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    result = construct_portfolio_targets(request)
    traces = {item.stage_name: item for item in result.trace}

    assert sum(abs(item.target_weight) for item in result.targets.values()) == pytest.approx(0.20)
    assert traces["gross_exposure_cap"].after.gross_exposure == pytest.approx(0.20)


def test_constrained_targets_by_subject_can_shift_to_net_exposure_target():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    targets = (
        PortfolioTarget(
            subject_id="A",
            target_weight=0.4,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
        PortfolioTarget(
            subject_id="B",
            target_weight=-0.4,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
    )

    constrained = constrained_targets_by_subject(
        targets,
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=1.0,
        net_exposure_target=0.4,
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["A"].target_weight == pytest.approx(0.4)
    assert constrained["B"].target_weight == pytest.approx(0.0)
    assert sum(item.target_weight for item in constrained.values()) == pytest.approx(0.4)
    assert sum(abs(item.target_weight) for item in constrained.values()) == pytest.approx(0.4)


def test_constrained_targets_by_subject_can_filter_to_short_only_top_k():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget
    
    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="LONG", target_weight=0.8, position_delta=0.0),
            PortfolioTarget(subject_id="BIG_SHORT", target_weight=-0.6, position_delta=0.0),
            PortfolioTarget(subject_id="SMALL_SHORT", target_weight=-0.2, position_delta=0.0),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        top_k=1,
        active_weight_budget=0.0,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
        direction_mode="short_only",
    )

    assert constrained["LONG"].target_weight == 0.0
    assert constrained["BIG_SHORT"].target_weight == pytest.approx(-0.6)
    assert constrained["SMALL_SHORT"].target_weight == 0.0


def test_constrained_targets_by_subject_long_short_top_k_uses_absolute_conviction():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget
    
    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="SMALL_LONG", target_weight=0.2, position_delta=0.0),
            PortfolioTarget(subject_id="BIG_SHORT", target_weight=-0.8, position_delta=0.0),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        top_k=1,
        active_weight_budget=0.0,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
        direction_mode="long_short",
    )

    assert constrained["SMALL_LONG"].target_weight == 0.0
    assert constrained["BIG_SHORT"].target_weight == pytest.approx(-0.8)


def test_constrained_targets_by_subject_scales_to_target_vol():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    targets = (
        PortfolioTarget(
            subject_id="A",
            target_weight=0.5,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
        PortfolioTarget(
            subject_id="B",
            target_weight=0.5,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
    )

    constrained = constrained_targets_by_subject(
        targets,
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        target_vol=0.1,
        risk_by_subject={"A": 0.2, "B": 0.2},
        direction_mode="long_only",
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["A"].target_weight == pytest.approx(0.3535533906)
    assert constrained["B"].target_weight == pytest.approx(0.3535533906)


def test_constrained_targets_by_subject_uses_constraint_boundary_for_stage_order():
    from alpha_os.contract_boundaries import PortfolioConstraintBoundary
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    targets = (
        PortfolioTarget(
            subject_id="A",
            target_weight=0.5,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
        PortfolioTarget(
            subject_id="B",
            target_weight=0.5,
            position_delta=0.0,
            target_notional=None,
            entry_allowed=True,
            risk_scale=1.0,
        ),
    )

    constrained = constrained_targets_by_subject(
        targets,
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        target_vol=0.1,
        risk_by_subject={"A": 0.2, "B": 0.2},
        constraint_boundary=PortfolioConstraintBoundary(
            sizing_time_fields=(),
            post_sizing_normalization_fields=(
                "direction_mode",
                "gross_exposure_cap",
                "gross_leverage_cap",
                "net_exposure_target",
                "asset_class_weight_caps",
                "cluster_weight_caps",
            ),
        ),
        direction_mode="long_only",
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["A"].target_weight == pytest.approx(0.5)
    assert constrained["B"].target_weight == pytest.approx(0.5)


def test_run_decision_backtest_feeds_state_into_next_decision():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series(
                        {"2026-03-24": 1.0, "2026-03-25": 1.0},
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        {"2026-03-24": -0.2, "2026-03-25": 0.1},
                        dtype=float,
                    ),
                ),
            ),
        )
    )

    first_step = _subject_step(result.steps[0], "BTC")
    second_step = _subject_step(result.steps[1], "BTC")

    assert first_step.target_weight == pytest.approx(1.0)
    assert second_step.risk_scale < 1.0
    assert second_step.target_weight < first_step.target_weight


def test_run_decision_backtest_solves_multi_subject_portfolio():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )
    from alpha_os.portfolio_sizing_policy import ConstrainedOptimizerSizingPolicy

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC_spot",
                    signal_series=pd.Series({"2026-03-24": 1.0}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                    risk_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="ETH_spot",
                    signal_series=pd.Series({"2026-03-24": 1.0}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                    risk_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                ),
            ),
            portfolio_construction=PortfolioConstructionSpec(gross_exposure_cap=1.0),
        ),
        sizing_policy=ConstrainedOptimizerSizingPolicy(dependence_aversion=2.0),
    )

    btc_step = _subject_step(result.steps[0], "BTC_spot")
    eth_step = _subject_step(result.steps[0], "ETH_spot")

    assert result.subject_ids == ("BTC_spot", "ETH_spot")
    assert btc_step.target_weight >= 0.0
    assert eth_step.target_weight >= 0.0
    assert result.steps[0].turnover == pytest.approx(
        abs(btc_step.position_delta) + abs(eth_step.position_delta)
    )
    assert result.steps[0].traded_notional == pytest.approx(
        btc_step.traded_notional + eth_step.traded_notional
    )


def test_run_decision_backtest_respects_asset_class_weight_caps():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            asset_class_by_subject={
                "ES_future": "equity_index",
                "NQ_future": "equity_index",
                "ZN_future": "rates",
            },
            portfolio_construction=PortfolioConstructionSpec(
                asset_class_weight_caps={"equity_index": 0.25}
            ),
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="ES_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="NQ_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="ZN_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
            ),
        )
    )

    first_step = result.steps[0]
    equity_weight = (
        _subject_step(first_step, "ES_future").target_weight
        + _subject_step(first_step, "NQ_future").target_weight
    )
    assert equity_weight == pytest.approx(0.25)
    assert _subject_step(first_step, "ZN_future").target_weight == pytest.approx(0.2)


def test_run_decision_backtest_respects_cluster_weight_caps():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            cluster_by_subject={
                "ES_future": "eq_us",
                "RTY_future": "eq_us",
                "ZN_future": "rates_us",
            },
            portfolio_construction=PortfolioConstructionSpec(
                cluster_weight_caps={"eq_us": 0.18}
            ),
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="ES_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="RTY_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
                SubjectBacktestSeries(
                    subject_id="ZN_future",
                    signal_series=pd.Series({"2026-03-24": 0.2}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                ),
            ),
        )
    )

    first_step = result.steps[0]
    eq_us_weight = (
        _subject_step(first_step, "ES_future").target_weight
        + _subject_step(first_step, "RTY_future").target_weight
    )
    assert eq_us_weight == pytest.approx(0.18)
    assert _subject_step(first_step, "ZN_future").target_weight == pytest.approx(0.2)


def test_run_decision_backtest_applies_subject_specific_cost_series_and_contract_multiplier():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="ES_future",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                    funding_cost_bps_series=pd.Series({"2026-03-24": 5.0}, dtype=float),
                    borrow_fee_bps_series=pd.Series({"2026-03-24": 7.0}, dtype=float),
                    roll_cost_bps_series=pd.Series({"2026-03-24": 3.0}, dtype=float),
                    contract_multiplier=50.0,
                ),
                SubjectBacktestSeries(
                    subject_id="BTCUSDT_perp",
                    signal_series=pd.Series({"2026-03-24": -0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": -0.05}, dtype=float),
                    funding_cost_bps_series=pd.Series({"2026-03-24": 4.0}, dtype=float),
                    borrow_fee_bps_series=pd.Series({"2026-03-24": 6.0}, dtype=float),
                    roll_cost_bps_series=pd.Series({"2026-03-24": 2.0}, dtype=float),
                    contract_multiplier=100.0,
                ),
            ),
        )
    )

    es_step = _subject_step(result.steps[0], "ES_future")
    btc_step = _subject_step(result.steps[0], "BTCUSDT_perp")

    assert es_step.target_contracts == pytest.approx(0.01)
    assert es_step.traded_contracts == pytest.approx(0.01)
    assert btc_step.target_contracts == pytest.approx(-0.005)
    assert btc_step.traded_contracts == pytest.approx(0.005)
    assert result.steps[0].cost_notional == pytest.approx(0.0006)


def test_run_decision_backtest_tracks_notional_with_initial_capital_base():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            initial_capital_base=2.5,
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="BTC",
                    signal_series=pd.Series({"2026-03-24": 0.5}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.1}, dtype=float),
                ),
            ),
        )
    )

    step = result.steps[0]
    subject_step = _subject_step(step, "BTC")

    assert subject_step.target_weight == pytest.approx(0.5)
    assert subject_step.target_notional == pytest.approx(1.25)
    assert subject_step.traded_notional == pytest.approx(1.25)
    assert step.gross_pnl_notional == pytest.approx(0.125)
    assert step.traded_notional == pytest.approx(1.25)
    assert step.gross_equity == pytest.approx(2.625)


def test_run_decision_backtest_uses_prior_history_for_skfolio_policy():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )
    from alpha_os.portfolio_sizing_policy import HistoricalModelSizingPolicy

    volatile_history = pd.Series(
        {
            "2026-03-17": 0.20,
            "2026-03-18": -0.20,
            "2026-03-19": 0.18,
            "2026-03-20": -0.18,
            "2026-03-23": 0.16,
        },
        dtype=float,
    )
    stable_history = pd.Series(
        {
            "2026-03-17": 0.01,
            "2026-03-18": 0.01,
            "2026-03-19": 0.011,
            "2026-03-20": 0.009,
            "2026-03-23": 0.01,
        },
        dtype=float,
    )

    result = run_decision_backtest(
        DecisionBacktestInput(
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="VOL",
                    signal_series=pd.Series({"2026-03-24": 0.6}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.02}, dtype=float),
                    historical_return_series=volatile_history,
                ),
                SubjectBacktestSeries(
                    subject_id="STABLE",
                    signal_series=pd.Series({"2026-03-24": 0.6}, dtype=float),
                    realized_return_series=pd.Series({"2026-03-24": 0.01}, dtype=float),
                    historical_return_series=stable_history,
                ),
            ),
            portfolio_construction=PortfolioConstructionSpec(
                gross_exposure_cap=1.0,
                active_weight_budget=0.0,
            ),
        ),
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="minimum_variance",
            min_history_steps=5,
        ),
    )

    first_step = result.steps[0]
    volatile_step = _subject_step(first_step, "VOL")
    stable_step = _subject_step(first_step, "STABLE")

    assert stable_step.target_weight > volatile_step.target_weight
    assert stable_step.target_weight > 0.9


def test_run_decision_backtest_can_rebalance_weekly_long_only_top_k():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
        run_decision_backtest,
    )

    index = [
        "2026-03-24",
        "2026-03-25",
        "2026-03-26",
        "2026-03-27",
    ]
    result = run_decision_backtest(
        DecisionBacktestInput(
            portfolio_construction=PortfolioConstructionSpec(
                gross_exposure_cap=1.0,
                rebalance_interval_steps=2,
                direction_mode="long_only",
            ),
            top_k=2,
            subject_series=(
                SubjectBacktestSeries(
                    subject_id="EWJ_etf",
                    signal_series=pd.Series(
                        [0.9, 0.1, 0.8, 0.2],
                        index=index,
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        [0.01, 0.01, 0.01, 0.01],
                        index=index,
                        dtype=float,
                    ),
                ),
                SubjectBacktestSeries(
                    subject_id="EWZ_etf",
                    signal_series=pd.Series(
                        [0.8, 0.2, 0.7, 0.1],
                        index=index,
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        [0.02, 0.02, 0.02, 0.02],
                        index=index,
                        dtype=float,
                    ),
                ),
                SubjectBacktestSeries(
                    subject_id="GDX_etf",
                    signal_series=pd.Series(
                        [-0.4, 0.9, -0.1, 0.8],
                        index=index,
                        dtype=float,
                    ),
                    realized_return_series=pd.Series(
                        [-0.01, -0.01, -0.01, -0.01],
                        index=index,
                        dtype=float,
                    ),
                ),
            ),
        )
    )

    first_step = result.steps[0]
    second_step = result.steps[1]
    third_step = result.steps[2]

    assert len(result.steps) == 4
    assert first_step.turnover > 0.0
    assert second_step.turnover == pytest.approx(0.0)
    assert third_step.turnover > 0.0

    first_weights = {
        step.subject_id: step.target_weight
        for step in first_step.subject_steps
    }
    assert first_weights["EWJ_etf"] > 0.0
    assert first_weights["EWZ_etf"] > 0.0
    assert first_weights["GDX_etf"] == pytest.approx(0.0)
    assert 0.0 < sum(first_weights.values()) <= 1.0
    assert result.gross_return_total > 0.0


def test_constrained_targets_target_vol_does_not_releverage_below_cap():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="ES", target_weight=0.10, position_delta=0.0),
            PortfolioTarget(subject_id="CL", target_weight=-0.10, position_delta=0.0),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        target_vol=0.50,
        risk_by_subject={"ES": 0.10, "CL": 0.10},
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["ES"].target_weight == pytest.approx(0.10)
    assert constrained["CL"].target_weight == pytest.approx(-0.10)


def test_constrained_targets_target_vol_scales_down_above_cap():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="ES", target_weight=1.00, position_delta=0.0),
            PortfolioTarget(subject_id="CL", target_weight=-1.00, position_delta=0.0),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=None,
        target_vol=0.10,
        risk_by_subject={"ES": 0.10, "CL": 0.10},
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert abs(constrained["ES"].target_weight) < 1.00
    assert abs(constrained["CL"].target_weight) < 1.00


def test_constrained_targets_net_target_does_not_open_new_exposure():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="ES", target_weight=0.20, position_delta=0.0),
            PortfolioTarget(
                subject_id="CL",
                target_weight=0.00,
                position_delta=0.0,
                entry_allowed=False,
            ),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=0.40,
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["ES"].target_weight == pytest.approx(0.20)
    assert constrained["CL"].target_weight == pytest.approx(0.00)
    assert not constrained["CL"].entry_allowed


def test_constrained_targets_net_target_reduces_existing_exposure_only():
    from alpha_os.decision_backtest import constrained_targets_by_subject
    from alpha_os.portfolio_decision import PortfolioTarget

    constrained = constrained_targets_by_subject(
        (
            PortfolioTarget(subject_id="ES", target_weight=0.30, position_delta=0.0),
            PortfolioTarget(
                subject_id="CL",
                target_weight=0.00,
                position_delta=0.0,
                entry_allowed=False,
            ),
        ),
        current_weights={},
        capital_base=1.0,
        gross_exposure_cap=None,
        gross_leverage_cap=None,
        net_exposure_target=0.00,
        top_k=None,
        asset_class_by_subject={},
        cluster_by_subject={},
        asset_class_weight_caps={},
        cluster_weight_caps={},
    )

    assert constrained["ES"].target_weight == pytest.approx(0.00)
    assert constrained["CL"].target_weight == pytest.approx(0.00)
