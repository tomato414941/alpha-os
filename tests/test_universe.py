from __future__ import annotations

import json

from alpha_os.data import universe


class _FakeClient:
    def list_signals(self):
        return [
            {"name": "sig_60", "interval": 60},
            {"name": "sig_3600", "interval": 3600},
            {"name": "sig_86400", "interval": 86400},
        ]


def test_load_daily_signals_from_client(monkeypatch):
    universe._daily_signal_cache = None
    client = _FakeClient()
    names = universe.load_daily_signals(client)
    assert names == ["sig_3600", "sig_60", "sig_86400"]
    status = universe.signal_catalog_status()
    assert status.source == "api"
    assert status.signal_count == 3


def test_load_daily_signals_falls_back_to_cached_catalog(monkeypatch, tmp_path):
    universe._daily_signal_cache = None

    class _BrokenClient:
        def list_signals(self):
            raise TimeoutError("api timeout")

    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps([
            {"name": "sig_cached_a", "interval": 60},
            {"name": "sig_cached_b", "interval": 86400},
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", cache_path)

    names = universe.load_daily_signals(_BrokenClient())
    assert names == ["sig_cached_a", "sig_cached_b"]
    status = universe.signal_catalog_status()
    assert status.source == "cache"
    assert status.api_error_kind == "timeout"


def test_load_daily_signals_falls_back_to_static(monkeypatch, tmp_path):
    universe._daily_signal_cache = None

    class _AuthBrokenClient:
        def list_signals(self):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", tmp_path / "missing.json")

    names = universe.load_daily_signals(_AuthBrokenClient())
    assert names == universe.MACRO_SIGNALS
    status = universe.signal_catalog_status()
    assert status.source == "static"
    assert status.api_error_kind == "auth"


def test_load_daily_signals_no_client_uses_cache(monkeypatch, tmp_path):
    universe._daily_signal_cache = None
    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps([{"name": "cached_sig", "interval": 86400}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_CATALOG_CACHE_PATH", cache_path)

    names = universe.load_daily_signals()  # no client
    assert names == ["cached_sig"]


def test_init_universe(monkeypatch):
    universe._daily_signal_cache = None
    universe.CROSS_ASSET_UNIVERSE = []
    client = _FakeClient()
    result = universe.init_universe(client)
    assert len(result) == 3
    assert universe.CROSS_ASSET_UNIVERSE == result
