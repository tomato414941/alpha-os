from __future__ import annotations

import json

from alpha_os.data import universe


class _FakeClient:
    def __init__(self, base_url: str, timeout: int, api_key: str | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key

    def list_signals(self):
        return [
            {"name": "sig_60", "interval": 60},
            {"name": "sig_3600", "interval": 3600},
            {"name": "sig_86400", "interval": 86400},
        ]


def test_load_daily_signals_all_intervals(monkeypatch):
    universe._daily_signal_cache = None
    monkeypatch.delenv("ALPHA_OS_SIGNAL_INTERVALS", raising=False)
    monkeypatch.setattr("alpha_os.data.signal_client.SignalClient", _FakeClient)

    names = universe.load_daily_signals()
    assert names == ["sig_3600", "sig_60", "sig_86400"]
    status = universe.signal_catalog_status()
    assert status.source == "api"
    assert status.signal_count == 3
    assert status.intervals is None


def test_load_daily_signals_interval_filter(monkeypatch):
    universe._daily_signal_cache = None
    monkeypatch.setenv("ALPHA_OS_SIGNAL_INTERVALS", "3600,86400")
    monkeypatch.setattr("alpha_os.data.signal_client.SignalClient", _FakeClient)

    names = universe.load_daily_signals()
    assert names == ["sig_3600", "sig_86400"]
    status = universe.signal_catalog_status()
    assert status.source == "api"
    assert status.intervals == (3600, 86400)


def test_load_daily_signals_falls_back_to_cached_catalog(monkeypatch, tmp_path):
    universe._daily_signal_cache = None
    monkeypatch.delenv("ALPHA_OS_SIGNAL_INTERVALS", raising=False)

    class _BrokenClient:
        def __init__(self, base_url: str, timeout: int, api_key: str | None = None):
            self.base_url = base_url
            self.timeout = timeout
            self.api_key = api_key

        def list_signals(self):
            raise TimeoutError("api timeout")

    monkeypatch.setattr("alpha_os.data.signal_client.SignalClient", _BrokenClient)
    cache_path = tmp_path / "signal_catalog.json"
    cache_path.write_text(
        json.dumps(
            [
                {"name": "sig_cached_a", "interval": 60},
                {"name": "sig_cached_b", "interval": 86400},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "_signal_catalog_cache_path", cache_path)

    names = universe.load_daily_signals()
    assert names == ["sig_cached_a", "sig_cached_b"]
    status = universe.signal_catalog_status()
    assert status.source == "cache"
    assert status.api_error_kind == "timeout"


def test_load_daily_signals_falls_back_to_static_when_api_and_cache_fail(monkeypatch, tmp_path):
    universe._daily_signal_cache = None
    monkeypatch.delenv("ALPHA_OS_SIGNAL_INTERVALS", raising=False)

    class _AuthBrokenClient:
        def __init__(self, base_url: str, timeout: int, api_key: str | None = None):
            self.base_url = base_url
            self.timeout = timeout
            self.api_key = api_key

        def list_signals(self):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("alpha_os.data.signal_client.SignalClient", _AuthBrokenClient)
    monkeypatch.setattr(universe, "_signal_catalog_cache_path", tmp_path / "missing.json")

    names = universe.load_daily_signals()

    assert names == universe.MACRO_SIGNALS
    status = universe.signal_catalog_status()
    assert status.source == "static"
    assert status.api_error_kind == "auth"
