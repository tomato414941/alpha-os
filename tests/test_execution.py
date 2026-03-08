"""Tests for execution layer (paper trading, executor interface)."""
import pytest

from alpha_os.execution.constraints import apply_venue_constraints
from alpha_os.execution.costs import ExecutionCostModel
from alpha_os.execution.executor import Order
from alpha_os.execution.paper import PaperExecutor
from alpha_os.execution.planning import (
    build_execution_intent,
    build_target_position,
    plan_execution_intent,
)


class TestPaperExecutor:
    def test_buy(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        assert fill is not None
        assert fill.qty == 10
        assert fill.price == 100.0
        assert pe.get_position("NVDA") == 10
        assert pe.get_cash() == 9000.0

    def test_sell(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        pe.submit_order(Order(symbol="NVDA", side="sell", qty=5))
        assert pe.get_position("NVDA") == 5
        assert pe.get_cash() == 9500.0

    def test_sell_rejected_in_long_only_mode(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="sell", qty=1))
        assert fill is None
        assert pe.get_position("NVDA") == 0
        assert pe.get_cash() == 10000.0

    def test_sell_allows_net_short_when_enabled(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            supports_short=True,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="sell", qty=1))
        assert fill is not None
        assert pe.get_position("NVDA") == -1
        assert pe.get_cash() == 10100.0

    def test_insufficient_cash(self):
        pe = PaperExecutor(
            initial_cash=100.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 200.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert fill is None
        assert pe.get_position("NVDA") == 0

    def test_no_price(self):
        pe = PaperExecutor(cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0))
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert fill is None

    def test_portfolio_value(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        pe.set_price("NVDA", 110.0)
        assert pe.portfolio_value == pytest.approx(10100.0)  # 9000 + 10*110

    def test_rebalance(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        pe.set_price("AAPL", 150.0)

        pe.submit_order(Order(symbol="NVDA", side="buy", qty=20))

        fills = pe.rebalance({"NVDA": 10, "AAPL": 5})
        assert len(fills) == 2
        assert pe.get_position("NVDA") == 10
        assert pe.get_position("AAPL") == 5

    def test_all_fills(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=5))
        pe.submit_order(Order(symbol="NVDA", side="sell", qty=2))
        assert len(pe.all_fills) == 2

    def test_all_positions(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_price("NVDA", 100.0)
        pe.set_price("AAPL", 150.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=5))
        pe.submit_order(Order(symbol="AAPL", side="buy", qty=3))
        pos = pe.all_positions
        assert pos["NVDA"] == 5
        assert pos["AAPL"] == 3

    def test_set_prices_batch(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.0),
        )
        pe.set_prices({"NVDA": 100.0, "AAPL": 150.0})
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert pe.get_cash() == 9900.0

    def test_buy_deducts_modeled_execution_costs(self):
        pe = PaperExecutor(
            initial_cash=10000.0,
            cost_model=ExecutionCostModel(commission_pct=0.10, modeled_slippage_pct=0.05),
        )
        pe.set_price("BTC", 100.0)

        fill = pe.submit_order(Order(symbol="BTC", side="buy", qty=10))

        assert fill is not None
        assert fill.costs.commission == pytest.approx(1.0)
        assert fill.costs.modeled_slippage == pytest.approx(0.5)
        assert fill.execution_cost == pytest.approx(1.5)
        assert pe.get_cash() == pytest.approx(8998.5)


class TestExecutionPlanning:
    def test_build_target_position_clamps_short_in_long_only_mode(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=-0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=False,
        )

        assert target.qty == 0.0
        assert target.reference_price == 100.0
        assert target.dollar_target == pytest.approx(-5000.0)

    def test_build_execution_intent_uses_target_gap(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=True,
        )

        intent = build_execution_intent(target, current_qty=20.0)

        assert intent is not None
        assert intent.side == "buy"
        assert intent.qty == pytest.approx(30.0)
        assert intent.target_qty == pytest.approx(50.0)
        assert intent.notional_value == pytest.approx(3000.0)

    def test_build_execution_intent_skips_zero_gap(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=True,
        )

        intent = build_execution_intent(target, current_qty=50.0)

        assert intent is None

    def test_build_execution_intent_skips_small_rebalance_under_deadband(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=True,
        )

        intent = build_execution_intent(
            target,
            current_qty=49.95,
            rebalance_deadband_usd=10.0,
        )

        assert intent is None

    def test_plan_execution_intent_reports_deadband_skip_reason(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=True,
        )

        decision = plan_execution_intent(
            target,
            current_qty=49.95,
            rebalance_deadband_usd=10.0,
        )

        assert decision.intent is None
        assert decision.skip_reason == "deadband"

    def test_apply_venue_constraints_rejects_below_min_notional(self):
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.004,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=True,
        )
        intent = build_execution_intent(target, current_qty=0.0)
        assert intent is not None

        result = apply_venue_constraints(
            intent,
            min_notional=50.0,
            min_notional_buffer=1.02,
        )

        assert result.order is None
        assert result.rejection_reason == "below_min_notional"

    def test_execute_intent_submits_only_after_constraints(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("BTC", 100.0)
        target = build_target_position(
            symbol="BTC",
            adjusted_signal=0.5,
            portfolio_value=10000.0,
            current_price=100.0,
            max_position_pct=1.0,
            min_trade_usd=10.0,
            supports_short=False,
        )
        intent = build_execution_intent(target, current_qty=0.0)
        assert intent is not None

        fill = pe.execute_intent(intent)

        assert fill is not None
        assert pe.get_position("BTC") == pytest.approx(50.0)
