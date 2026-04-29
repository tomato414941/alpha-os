from __future__ import annotations


def test_build_signal_client_uses_remote_defaults(monkeypatch):
    import alpha_os.signal_client as signal_client

    captured: dict[str, object] = {}

    class FakeSignalClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(signal_client, "SignalClient", FakeSignalClient)

    signal_client.build_signal_client(base_url="https://signal-noise.example.test")

    assert captured["base_url"] == "https://signal-noise.example.test"
    assert captured["timeout"] == 90
    assert captured["retry_count"] == 2
    assert captured["retry_backoff"] == 1.0


def test_build_signal_client_uses_local_defaults(monkeypatch):
    import alpha_os.signal_client as signal_client

    captured: dict[str, object] = {}

    class FakeSignalClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(signal_client, "SignalClient", FakeSignalClient)

    signal_client.build_signal_client(base_url="http://127.0.0.1:8000")

    assert captured["timeout"] == 30


def test_build_signal_client_allows_env_overrides(monkeypatch):
    import alpha_os.signal_client as signal_client

    captured: dict[str, object] = {}

    class FakeSignalClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(signal_client, "SignalClient", FakeSignalClient)
    monkeypatch.setenv(signal_client.SIGNAL_NOISE_TIMEOUT_ENV, "120")
    monkeypatch.setenv(signal_client.SIGNAL_NOISE_RETRY_COUNT_ENV, "4")
    monkeypatch.setenv(signal_client.SIGNAL_NOISE_RETRY_BACKOFF_ENV, "2.5")

    signal_client.build_signal_client(base_url="https://signal-noise.example.test")

    assert captured["timeout"] == 120
    assert captured["retry_count"] == 4
    assert captured["retry_backoff"] == 2.5


def test_build_signal_client_reports_missing_optional_dependency(monkeypatch):
    import pytest

    import alpha_os.signal_client as signal_client

    monkeypatch.setattr(signal_client, "SignalClient", None)

    with pytest.raises(RuntimeError, match="signal-noise is required"):
        signal_client.build_signal_client()
