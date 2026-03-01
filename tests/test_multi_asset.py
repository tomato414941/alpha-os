"""Tests for multi-asset sequential execution via --assets flag."""
from __future__ import annotations

import argparse

from alpha_os.cli import _resolve_asset_list


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
