from __future__ import annotations

from alpha_os.data import signal_client


class _FakeClient:
    def __init__(self, *, base_url: str, timeout: int, api_key: str | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key


def test_build_signal_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ALPHA_OS_SIGNAL_NOISE_API_KEY", "secret-token")
    monkeypatch.setattr(signal_client, "SignalClient", _FakeClient)

    client = signal_client.build_signal_client(base_url="https://signals.example", timeout=15)

    assert client.base_url == "https://signals.example"
    assert client.timeout == 15
    assert client.api_key == "secret-token"


def test_build_signal_client_omits_empty_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPHA_OS_SIGNAL_NOISE_API_KEY", raising=False)
    monkeypatch.setattr(signal_client, "_SECRETS_FILE", tmp_path / "nonexistent")
    monkeypatch.setattr(signal_client, "SignalClient", _FakeClient)

    client = signal_client.build_signal_client(base_url="https://signals.example", timeout=15)

    assert client.api_key is None
