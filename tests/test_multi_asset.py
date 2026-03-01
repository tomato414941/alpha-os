"""Tests for multi-asset sequential execution via --assets flag."""
from __future__ import annotations

import argparse

from alpha_os.cli import _resolve_asset_list
from alpha_os.data.universe import is_crypto


class TestResolveAssetList:
    def test_single_asset_default(self):
        args = argparse.Namespace(asset="BTC", assets=None)
        assert _resolve_asset_list(args) == ["BTC"]

    def test_single_asset_explicit(self):
        args = argparse.Namespace(asset="ETH", assets=None)
        assert _resolve_asset_list(args) == ["ETH"]

    def test_multi_assets(self):
        args = argparse.Namespace(asset="BTC", assets="BTC,ETH,SOL")
        assert _resolve_asset_list(args) == ["BTC", "ETH", "SOL"]

    def test_multi_assets_normalizes_case(self):
        args = argparse.Namespace(asset="BTC", assets="btc,eth,sol")
        assert _resolve_asset_list(args) == ["BTC", "ETH", "SOL"]

    def test_multi_assets_strips_whitespace(self):
        args = argparse.Namespace(asset="BTC", assets=" BTC , ETH , SOL ")
        assert _resolve_asset_list(args) == ["BTC", "ETH", "SOL"]

    def test_all_seven_crypto(self):
        args = argparse.Namespace(
            asset="BTC", assets="BTC,ETH,SOL,BNB,XRP,ADA,DOGE"
        )
        result = _resolve_asset_list(args)
        assert len(result) == 7
        assert result[0] == "BTC"
        assert result[-1] == "DOGE"

    def test_assets_overrides_asset(self):
        args = argparse.Namespace(asset="ETH", assets="BTC,SOL")
        assert _resolve_asset_list(args) == ["BTC", "SOL"]

    def test_mixed_crypto_and_stocks(self):
        args = argparse.Namespace(asset="BTC", assets="BTC,NVDA,ETH,AAPL")
        result = _resolve_asset_list(args)
        assert result == ["BTC", "NVDA", "ETH", "AAPL"]


class TestIsCrypto:
    def test_crypto_assets(self):
        for asset in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]:
            assert is_crypto(asset) is True

    def test_stock_assets(self):
        for asset in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD"]:
            assert is_crypto(asset) is False

    def test_case_insensitive(self):
        assert is_crypto("btc") is True
        assert is_crypto("nvda") is False

    def test_unknown_asset(self):
        assert is_crypto("UNKNOWN") is False
