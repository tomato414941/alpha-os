"""Tests for expanded venue + universe abstractions."""
from __future__ import annotations

import pytest

from alpha_os.data.universe import (
    ETFS,
    STOCKS,
    CRYPTO,
    is_crypto,
    is_equity,
    is_etf,
    is_stock,
    is_polymarket,
    infer_venue,
    register_polymarket_market,
    POLYMARKET,
)
from alpha_os.alpha.combiner import cross_asset_neutralize
from alpha_os.risk.manager import (
    BinaryOutcomeRiskConfig,
    BinaryOutcomeRiskManager,
)


class TestAssetClassification:
    def test_stock_detection(self):
        assert is_stock("NVDA")
        assert is_stock("JPM")
        assert not is_stock("SPY")
        assert not is_stock("BTC")

    def test_etf_detection(self):
        assert is_etf("SPY")
        assert is_etf("QQQ")
        assert is_etf("GLD")
        assert is_etf("TLT")
        assert not is_etf("NVDA")
        assert not is_etf("BTC")

    def test_equity_covers_stocks_and_etfs(self):
        assert is_equity("NVDA")
        assert is_equity("SPY")
        assert is_equity("GLD")
        assert not is_equity("BTC")
        assert not is_equity("ETH")

    def test_crypto_detection(self):
        assert is_crypto("BTC")
        assert is_crypto("ETH")
        assert is_crypto("SOL")
        assert not is_crypto("SPY")

    def test_etf_catalog_size(self):
        assert len(ETFS) >= 10

    def test_stock_catalog_expanded(self):
        assert "JPM" in STOCKS
        assert "XOM" in STOCKS
        assert "UNH" in STOCKS


class TestVenueInference:
    def test_crypto_routes_to_binance(self):
        assert infer_venue("BTC") == "binance"
        assert infer_venue("ETH") == "binance"

    def test_equity_routes_to_alpaca(self):
        assert infer_venue("NVDA") == "alpaca"
        assert infer_venue("SPY") == "alpaca"
        assert infer_venue("GLD") == "alpaca"

    def test_unknown_routes_to_paper(self):
        assert infer_venue("UNKNOWN_ASSET_XYZ") == "paper"


class TestPolymarketRegistry:
    def test_register_and_detect(self):
        register_polymarket_market("cond-test-001", "poly_test_signal")
        assert is_polymarket("cond-test-001")
        assert POLYMARKET["cond-test-001"] == "poly_test_signal"
        # cleanup
        POLYMARKET.pop("cond-test-001", None)

    def test_polymarket_routes_to_polymarket(self):
        POLYMARKET["cond-test-002"] = "poly_test_002"
        assert infer_venue("cond-test-002") == "polymarket"
        POLYMARKET.pop("cond-test-002", None)


class TestKellyRiskManager:
    def test_positive_edge_buy_yes(self):
        rm = BinaryOutcomeRiskManager()
        sizing = rm.kelly_size(model_prob=0.70, market_price=0.50)
        assert sizing.edge == pytest.approx(0.20)
        assert sizing.fraction > 0

    def test_negative_edge_buy_no(self):
        rm = BinaryOutcomeRiskManager()
        sizing = rm.kelly_size(model_prob=0.30, market_price=0.50)
        assert sizing.edge == pytest.approx(-0.20)
        assert sizing.fraction < 0

    def test_no_edge_zero_fraction(self):
        rm = BinaryOutcomeRiskManager(BinaryOutcomeRiskConfig(min_edge=0.05))
        sizing = rm.kelly_size(model_prob=0.52, market_price=0.50)
        assert sizing.fraction == 0.0

    def test_fraction_capped_at_max(self):
        rm = BinaryOutcomeRiskManager(BinaryOutcomeRiskConfig(max_fraction=0.10))
        sizing = rm.kelly_size(model_prob=0.95, market_price=0.10)
        assert abs(sizing.fraction) <= 0.10 + 1e-8

    def test_position_usd(self):
        rm = BinaryOutcomeRiskManager()
        pos = rm.position_usd(
            model_prob=0.70,
            market_price=0.50,
            bankroll=10000.0,
            max_position_usd=100.0,
        )
        assert 0 < pos <= 100.0

    def test_position_usd_no_edge(self):
        rm = BinaryOutcomeRiskManager(BinaryOutcomeRiskConfig(min_edge=0.10))
        pos = rm.position_usd(
            model_prob=0.55,
            market_price=0.50,
            bankroll=10000.0,
        )
        assert pos == 0.0

    def test_half_kelly(self):
        rm_full = BinaryOutcomeRiskManager(BinaryOutcomeRiskConfig(kelly_fraction=1.0, max_fraction=1.0))
        rm_half = BinaryOutcomeRiskManager(BinaryOutcomeRiskConfig(kelly_fraction=0.5, max_fraction=1.0))
        full = rm_full.kelly_size(0.70, 0.50)
        half = rm_half.kelly_size(0.70, 0.50)
        assert abs(half.fraction) == pytest.approx(abs(full.fraction) / 2, rel=0.01)


class TestCrossAssetNeutralize:
    def test_subtracts_mean(self):
        signals = {"BTC": 0.3, "ETH": 0.5, "SOL": 0.1}
        result = cross_asset_neutralize(signals)
        assert result["BTC"] == pytest.approx(0.0)
        assert result["ETH"] == pytest.approx(0.2)
        assert result["SOL"] == pytest.approx(-0.2)

    def test_sums_to_zero(self):
        signals = {"BTC": 0.3, "ETH": 0.5, "SOL": 0.1}
        result = cross_asset_neutralize(signals)
        assert sum(result.values()) == pytest.approx(0.0)

    def test_single_asset_unchanged(self):
        signals = {"BTC": 0.5}
        result = cross_asset_neutralize(signals)
        assert result["BTC"] == 0.5

    def test_handles_nan(self):
        signals = {"BTC": float("nan"), "ETH": 0.5, "SOL": 0.1}
        result = cross_asset_neutralize(signals)
        assert result["BTC"] == 0.0
        assert result["ETH"] == pytest.approx(0.2)
        assert result["SOL"] == pytest.approx(-0.2)

    def test_all_same_becomes_zero(self):
        signals = {"BTC": 0.5, "ETH": 0.5, "SOL": 0.5}
        result = cross_asset_neutralize(signals)
        assert all(v == pytest.approx(0.0) for v in result.values())
