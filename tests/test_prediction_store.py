"""Tests for prediction store."""
import tempfile
from pathlib import Path

from alpha_os_recovery.predictions.store import Prediction, PredictionStore, SignalMeta


class TestPredictionStore:
    def _make_store(self):
        tmp = tempfile.mkdtemp()
        return PredictionStore(db_path=Path(tmp) / "predictions.db")

    def test_write_and_read(self):
        store = self._make_store()
        store.write([
            Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.5, horizon=1),
            Prediction("sig_1", "2026-03-20", "eth_btc", -0.3, horizon=1),
            Prediction("sig_2", "2026-03-20", "btc_ohlcv", 0.1, horizon=5),
        ])
        latest = store.read_latest("2026-03-20")
        assert "sig_1" in latest
        assert latest["sig_1"]["btc_ohlcv"] == 0.5
        assert latest["sig_1"]["eth_btc"] == -0.3
        assert latest["sig_2"]["btc_ohlcv"] == 0.1
        store.close()

    def test_read_with_asset_filter(self):
        store = self._make_store()
        store.write([
            Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.5),
            Prediction("sig_1", "2026-03-20", "eth_btc", -0.3),
        ])
        latest = store.read_latest("2026-03-20", assets=["btc_ohlcv"])
        assert latest["sig_1"]["btc_ohlcv"] == 0.5
        assert "eth_btc" not in latest["sig_1"]
        store.close()

    def test_register_signal(self):
        store = self._make_store()
        store.register_signal(SignalMeta(
            signal_id="sig_1", source="gp",
            definition="(neg (zscore sp500))", horizon=20,
        ))
        signals = store.list_signals()
        assert len(signals) == 1
        assert signals[0].signal_id == "sig_1"
        assert signals[0].source == "gp"
        assert signals[0].horizon == 20
        store.close()

    def test_register_signal_upsert(self):
        store = self._make_store()
        store.register_signal(SignalMeta("sig_1", "gp", "(neg x)", 1))
        store.register_signal(SignalMeta("sig_1", "gp", "(neg y)", 5))
        signals = store.list_signals()
        assert len(signals) == 1
        assert signals[0].definition == "(neg y)"
        assert signals[0].horizon == 5
        store.close()

    def test_signal_history(self):
        store = self._make_store()
        store.write([
            Prediction("sig_1", "2026-03-18", "btc_ohlcv", 0.1),
            Prediction("sig_1", "2026-03-19", "btc_ohlcv", 0.2),
            Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.3),
        ])
        history = store.read_signal_history("sig_1", "btc_ohlcv", n_days=2)
        assert len(history) == 2
        assert history[0][0] == "2026-03-20"
        assert history[0][1] == 0.3
        store.close()

    def test_counts(self):
        store = self._make_store()
        store.register_signal(SignalMeta("sig_1", "gp", "(neg x)"))
        store.register_signal(SignalMeta("sig_2", "ml", "model_v1"))
        store.write([
            Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.5),
            Prediction("sig_2", "2026-03-20", "btc_ohlcv", 0.1),
        ])
        assert store.signal_count() == 2
        assert store.prediction_count() == 2
        assert store.prediction_count("2026-03-20") == 2
        assert store.prediction_count("2026-03-19") == 0
        store.close()

    def test_write_replaces_on_conflict(self):
        store = self._make_store()
        store.write([Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.5)])
        store.write([Prediction("sig_1", "2026-03-20", "btc_ohlcv", 0.9)])
        latest = store.read_latest("2026-03-20")
        assert latest["sig_1"]["btc_ohlcv"] == 0.9
        store.close()

    def test_list_signals_by_source(self):
        store = self._make_store()
        store.register_signal(SignalMeta("sig_1", "gp", "(neg x)"))
        store.register_signal(SignalMeta("sig_2", "classical", "rsi_14"))
        assert len(store.list_signals(source="gp")) == 1
        assert len(store.list_signals(source="classical")) == 1
        assert len(store.list_signals()) == 2
        store.close()
