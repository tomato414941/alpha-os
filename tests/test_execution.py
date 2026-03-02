"""Tests for execution layer (paper trading, executor interface)."""
import pytest

from alpha_os.execution.executor import Order
from alpha_os.execution.paper import PaperExecutor


class TestPaperExecutor:
    def test_buy(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        assert fill is not None
        assert fill.qty == 10
        assert fill.price == 100.0
        assert pe.get_position("NVDA") == 10
        assert pe.get_cash() == 9000.0

    def test_sell(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        pe.submit_order(Order(symbol="NVDA", side="sell", qty=5))
        assert pe.get_position("NVDA") == 5
        assert pe.get_cash() == 9500.0

    def test_insufficient_cash(self):
        pe = PaperExecutor(initial_cash=100.0)
        pe.set_price("NVDA", 200.0)
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert fill is None
        assert pe.get_position("NVDA") == 0

    def test_no_price(self):
        pe = PaperExecutor()
        fill = pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert fill is None

    def test_portfolio_value(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=10))
        pe.set_price("NVDA", 110.0)
        assert pe.portfolio_value == pytest.approx(10100.0)  # 9000 + 10*110

    def test_rebalance(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        pe.set_price("AAPL", 150.0)

        pe.submit_order(Order(symbol="NVDA", side="buy", qty=20))

        fills = pe.rebalance({"NVDA": 10, "AAPL": 5})
        assert len(fills) == 2
        assert pe.get_position("NVDA") == 10
        assert pe.get_position("AAPL") == 5

    def test_all_fills(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=5))
        pe.submit_order(Order(symbol="NVDA", side="sell", qty=2))
        assert len(pe.all_fills) == 2

    def test_all_positions(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_price("NVDA", 100.0)
        pe.set_price("AAPL", 150.0)
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=5))
        pe.submit_order(Order(symbol="AAPL", side="buy", qty=3))
        pos = pe.all_positions
        assert pos["NVDA"] == 5
        assert pos["AAPL"] == 3

    def test_set_prices_batch(self):
        pe = PaperExecutor(initial_cash=10000.0)
        pe.set_prices({"NVDA": 100.0, "AAPL": 150.0})
        pe.submit_order(Order(symbol="NVDA", side="buy", qty=1))
        assert pe.get_cash() == 9900.0
