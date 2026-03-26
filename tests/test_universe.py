from __future__ import annotations

import json

import pytest

from alpha_os.data import universe


@pytest.fixture(autouse=True)
def _reset_caches():
    universe._daily_signal_cache = None
    universe._signal_catalog_cache = None
    yield
    universe._daily_signal_cache = None
    universe._signal_catalog_cache = None


class _FakeClient:
    def list_signals(self):
        return [
            {"name": "sig_60", "interval": 60, "signal_type": "scalar"},
            {"name": "sig_3600", "interval": 3600, "signal_type": "scalar"},
            {"name": "sig_86400", "interval": 86400, "signal_type": "ohlcv"},
        ]


def test_load_daily_signals_from_client(monkeypatch):
    client = _FakeClient()
    names = universe.load_daily_signals(client)
    assert names == ["sig_86400"]
    status = universe.signal_catalog_status()
    assert status.source == "api"
    assert status.signal_count == 3


def test_load_price_signals_from_client():
    client = _FakeClient()
    prices = universe.load_price_signals(client)
    assert prices == ["sig_86400"]


def test_load_daily_signals_falls_back_to_cached_catalog(monkeypatch, tmp_path):
    class _BrokenClient:
        def list_signals(self):
            raise TimeoutError("api timeout")

    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps([
            {"name": "sig_cached_a", "interval": 60, "signal_type": "scalar"},
            {"name": "sig_cached_b", "interval": 86400, "signal_type": "ohlcv"},
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", cache_path)

    names = universe.load_daily_signals(_BrokenClient())
    assert names == ["sig_cached_b"]
    status = universe.signal_catalog_status()
    assert status.source == "cache"
    assert status.api_error_kind == "timeout"


def test_load_daily_signals_falls_back_to_static(monkeypatch, tmp_path):
    class _AuthBrokenClient:
        def list_signals(self):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", tmp_path / "missing.json")

    names = universe.load_daily_signals(_AuthBrokenClient())
    assert names == sorted(universe.MACRO_SIGNALS)
    status = universe.signal_catalog_status()
    assert status.source == "static"
    assert status.api_error_kind == "auth"


def test_load_daily_signals_no_client_uses_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps([{"name": "cached_sig", "interval": 86400}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", cache_path)

    names = universe.load_daily_signals()  # no client
    assert names == ["cached_sig"]


def test_load_daily_signals_prefer_cache_skips_client(monkeypatch, tmp_path):
    class _ExplodingClient:
        def list_signals(self):
            raise AssertionError("client should not be called")

    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps([{"name": "cached_sig", "interval": 86400}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", cache_path)

    names = universe.load_daily_signals(_ExplodingClient(), prefer_cache=True)

    assert names == ["cached_sig"]


def test_load_daily_signals_refresh_bypasses_process_cache(monkeypatch):
    calls = {"n": 0}

    class _Client:
        def list_signals(self):
            calls["n"] += 1
            if calls["n"] == 1:
                return [{"name": "cached_sig", "interval": 86400}]
            return [{"name": "fresh_sig", "interval": 86400}]

    first = universe.load_daily_signals(_Client())
    second = universe.load_daily_signals(_Client(), refresh=True)

    assert first == ["cached_sig"]
    assert second == ["fresh_sig"]


def test_init_universe():
    universe.CROSS_ASSET_UNIVERSE = []
    universe.FEATURE_CATALOG = []
    client = _FakeClient()
    result = universe.init_universe(client)
    # Only OHLCV signals in universe
    assert result == ["sig_86400"]
    assert universe.CROSS_ASSET_UNIVERSE == ["sig_86400"]
    # Daily-only signals in feature catalog
    assert universe.FEATURE_CATALOG == ["sig_86400"]


def test_build_feature_list_adds_daily_derived_features_for_macro_onchain_derivatives_and_backing_price(monkeypatch):
    monkeypatch.setattr(
        universe,
        "load_daily_signals",
        lambda *args, **kwargs: [
            "btc_ohlcv",
            "sol_usdt",
            "btc_hashrate",
            "funding_rate_btc",
            "fear_greed",
        ],
    )

    features = universe.build_feature_list("BTC")

    assert "btc_hashrate" in features
    assert "funding_rate_btc" in features
    assert "fear_greed" in features
    assert "delta_1__btc_hashrate" in features
    assert "roc_5__btc_hashrate" in features
    assert "zscore_20__btc_hashrate" in features
    assert "delta_1__funding_rate_btc" in features
    assert "roc_5__funding_rate_btc" in features
    assert "zscore_20__funding_rate_btc" in features
    assert "delta_1__fear_greed" in features
    assert "roc_5__fear_greed" in features
    assert "zscore_20__fear_greed" in features
    assert "delta_1__btc_ohlcv" in features
    assert "roc_5__btc_ohlcv" in features
    assert "zscore_20__btc_ohlcv" in features
    assert "delta_1__sol_usdt" not in features


def test_derived_feature_names_resolve_to_raw_family_and_signal():
    assert universe.parse_derived_feature_name("roc_5__funding_rate_btc") == (
        "roc",
        5,
        "funding_rate_btc",
    )
    assert universe.base_signal_name("roc_5__funding_rate_btc") == "funding_rate_btc"
    assert universe.infer_feature_family("roc_5__funding_rate_btc") == "derivatives"
    assert universe.required_raw_signals(
        ["roc_5__funding_rate_btc", "zscore_20__btc_hashrate", "fear_greed"]
    ) == ["btc_hashrate", "fear_greed", "funding_rate_btc"]


def test_infer_feature_family_treats_futures_and_basis_like_derivatives():
    assert universe.infer_feature_family("futures_taker_ratio_btc") == "derivatives"
    assert universe.infer_feature_family("basis_btc_annualized") == "derivatives"
    assert universe.infer_feature_family("premium_eth_perp") == "derivatives"
