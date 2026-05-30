from __future__ import annotations

import pandas as pd
import pytest

from alpha_os.evaluation_cost_config import TradingEnvironment
from alpha_os.portfolio_decision import (
    PortfolioDecisionInput,
    PortfolioDecisionOutput,
    PortfolioTarget,
)


def _subject_step(step, subject_id: str):
    return step.subject_step_by_subject[subject_id]


class SignalTargetStrategy:
    def decide(self, strategy_input: PortfolioDecisionInput) -> PortfolioDecisionOutput:
        current_weights = strategy_input.portfolio_state.weights_by_subject
        targets = tuple(
            PortfolioTarget(
                subject_id=signal.subject_id,
                target_weight=float(signal.value),
                position_delta=float(signal.value)
                - current_weights.get(signal.subject_id, 0.0),
                target_notional=float(signal.value)
                * strategy_input.portfolio_state.capital_base,
                entry_allowed=abs(float(signal.value)) > 0.0,
                risk_scale=1.0,
            )
            for signal in strategy_input.predictive_signals
        )
        return PortfolioDecisionOutput(
            portfolio_id=strategy_input.portfolio_id,
            as_of=strategy_input.as_of,
            targets=targets,
        )


def _run_decision_backtest(backtest_input, *, strategy=None):
    from alpha_os.decision_backtest import run_decision_backtest

    return run_decision_backtest(
        backtest_input,
        strategy=SignalTargetStrategy() if strategy is None else strategy,
    )


def test_run_decision_backtest_replays_signal_into_equity_curve():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
        DecisionBacktestInput(
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
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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


def test_run_decision_backtest_tracks_drawdown():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
    )

    result = _run_decision_backtest(
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
        )
    )

    first_step = _subject_step(result.steps[0], "BTC")
    second_step = _subject_step(result.steps[1], "BTC")

    assert first_step.risk_scale == pytest.approx(1.0)
    assert second_step.risk_scale == pytest.approx(1.0)
    assert second_step.target_weight == pytest.approx(1.0)
    assert result.max_drawdown == pytest.approx(0.2)


def test_run_decision_backtest_feeds_state_into_next_decision():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
    )

    result = _run_decision_backtest(
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
    assert second_step.risk_scale == pytest.approx(1.0)
    assert second_step.target_weight == pytest.approx(1.0)


def test_run_decision_backtest_applies_subject_specific_cost_series_and_contract_multiplier():
    from alpha_os.decision_backtest import (
        DecisionBacktestInput,
        SubjectBacktestSeries,
    )

    result = _run_decision_backtest(
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
    )

    result = _run_decision_backtest(
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
