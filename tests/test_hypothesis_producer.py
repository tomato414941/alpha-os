import numpy as np
import pandas as pd
import pytest

from alpha_os_recovery.hypotheses import (
    HypothesisKind,
    HypothesisRecord,
    HypothesisStore,
)
from alpha_os_recovery.hypotheses.producer import (
    _quick_healthcheck,
    _resolve_asset_series_name,
    _should_sync_required_features,
    collect_required_features,
    compute_hypothesis_prediction,
    produce_active_hypothesis_predictions,
    write_hypothesis_predictions,
)
from alpha_os_recovery.predictions.store import PredictionStore
from alpha_os_recovery.config import Config


def _sample_data():
    n = 80
    x = np.linspace(1.0, 2.0, n)
    return {
        "asset_a": x,
        "asset_b": x[::-1],
        "fear_greed": np.linspace(10.0, 90.0, n),
        "dxy": np.linspace(100.0, 90.0, n),
        "gold": np.linspace(1800.0, 1900.0, n),
        "sp500": np.linspace(4000.0, 4300.0, n),
        "nasdaq": np.linspace(12000.0, 13000.0, n),
        "vix_close": np.linspace(25.0, 15.0, n),
        "russell2000": np.linspace(1800.0, 2000.0, n),
    }


def test_collect_required_features_merges_assets_and_runtime_inputs():
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(sub fear_greed dxy)"},
        ),
        HypothesisRecord(
            hypothesis_id="h2",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "rsi_reversion", "inputs": ["core_universe_1000"]},
        ),
        HypothesisRecord(
            hypothesis_id="h3",
            kind=HypothesisKind.ML,
            definition={"features": ["sp500", "nasdaq"]},
        ),
    ]

    features = collect_required_features(hypotheses, ["asset_a", "asset_b"])

    assert "asset_a" in features
    assert "asset_b" in features
    assert "fear_greed" in features
    assert "dxy" in features
    assert "nasdaq" in features
    assert "sp500" in features
    assert "core_universe_1000" not in features


def test_collect_required_features_normalizes_known_assets():
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 20}},
        ),
    ]

    features = collect_required_features(hypotheses, ["BTC"])

    assert "btc_ohlcv" in features
    assert "BTC" not in features


def test_collect_required_features_ignores_temporal_operator_tokens():
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(corr_10 (lag_30 fear_greed) dxy)"},
        ),
    ]

    features = collect_required_features(hypotheses, ["BTC"])

    assert "btc_ohlcv" in features
    assert "fear_greed" in features
    assert "dxy" in features
    assert "corr_10" not in features
    assert "lag_30" not in features


def test_compute_hypothesis_prediction_for_dsl():
    record = HypothesisRecord(
        hypothesis_id="h1",
        kind=HypothesisKind.DSL,
        definition={"expression": "(sub fear_greed gold)"},
    )

    value = compute_hypothesis_prediction(record, data=_sample_data(), asset="asset_a")

    assert isinstance(value, float)
    assert value != 0.0
    assert abs(value) <= 1.0


def test_compute_hypothesis_prediction_for_technical():
    record = HypothesisRecord(
        hypothesis_id="h2",
        kind=HypothesisKind.TECHNICAL,
        definition={"indicator": "roc_momentum", "params": {"window": 20}},
    )

    value = compute_hypothesis_prediction(record, data=_sample_data(), asset="asset_a")

    assert value > 0
    assert abs(value) <= 1.0


def test_compute_hypothesis_prediction_for_known_asset_symbol():
    record = HypothesisRecord(
        hypothesis_id="h2",
        kind=HypothesisKind.TECHNICAL,
        definition={"indicator": "roc_momentum", "params": {"window": 20}},
    )
    data = _sample_data() | {"btc_ohlcv": np.linspace(1.0, 2.0, 80)}

    value = compute_hypothesis_prediction(record, data=data, asset="BTC")

    assert value > 0
    assert abs(value) <= 1.0


def test_compute_hypothesis_prediction_for_macd_is_bounded():
    record = HypothesisRecord(
        hypothesis_id="h-macd",
        kind=HypothesisKind.TECHNICAL,
        definition={"indicator": "macd_trend", "params": {"fast": 12, "slow": 26, "signal": 9}},
    )
    data = _sample_data() | {"asset_a": np.linspace(100.0, 200.0, 80)}

    value = compute_hypothesis_prediction(record, data=data, asset="asset_a")

    assert abs(value) <= 1.0
    assert value > 0.0


def test_compute_hypothesis_prediction_for_ml():
    record = HypothesisRecord(
        hypothesis_id="h3",
        kind=HypothesisKind.ML,
        definition={
            "model_type": "linear",
            "model_ref": "models/ml_linear_residual_v1.json",
            "features": ["sp500", "nasdaq", "vix_close"],
        },
    )

    value = compute_hypothesis_prediction(record, data=_sample_data(), asset="asset_a")

    assert isinstance(value, float)
    assert abs(value) <= 1.0


def test_write_hypothesis_predictions_persists_rows(tmp_path):
    prediction_store = PredictionStore(tmp_path / "predictions.db")
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(sub fear_greed dxy)"},
            source="random_dsl",
            stake=1.0,
        ),
        HypothesisRecord(
            hypothesis_id="h2",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 20}},
            source="bootstrap_technical",
            stake=1.0,
        ),
    ]

    written = write_hypothesis_predictions(
        hypotheses,
        data=_sample_data(),
        assets=["asset_a", "asset_b"],
        prediction_store=prediction_store,
        date="2026-03-21",
    )
    rows = prediction_store.read_latest("2026-03-21")

    assert written == 4
    assert set(rows.keys()) == {"h1", "h2"}
    assert set(rows["h1"].keys()) == {"asset_a", "asset_b"}
    assert set(rows["h2"].keys()) == {"asset_a", "asset_b"}
    prediction_store.close()


def test_write_hypothesis_predictions_skips_invalid_hypothesis(tmp_path):
    prediction_store = PredictionStore(tmp_path / "predictions.db")
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="bad",
            kind=HypothesisKind.DSL,
            definition={"expression": "(lag 1 3)"},
            source="random_dsl",
            stake=1.0,
        ),
        HypothesisRecord(
            hypothesis_id="good",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 20}},
            source="bootstrap_technical",
            stake=1.0,
        ),
    ]

    written = write_hypothesis_predictions(
        hypotheses,
        data=_sample_data(),
        assets=["asset_a", "asset_b"],
        prediction_store=prediction_store,
        date="2026-03-21",
    )
    rows = prediction_store.read_latest("2026-03-21")

    assert written == 2
    assert set(rows.keys()) == {"good"}
    assert set(rows["good"].keys()) == {"asset_a", "asset_b"}
    prediction_store.close()


def test_produce_active_hypothesis_predictions_includes_zero_stake_active_hypotheses(
    tmp_path,
    monkeypatch,
):
    from alpha_os_recovery.hypotheses.producer import produce_active_hypothesis_predictions

    class _FakeDataStore:
        def __init__(self, db_path, client):
            self.db_path = db_path
            self.client = client

        def get_matrix(self, features):
            return pd.DataFrame({name: np.linspace(1.0, 2.0, 80) for name in features})

        def close(self):
            return None

    monkeypatch.setattr("alpha_os_recovery.hypotheses.producer.DataStore", _FakeDataStore)
    monkeypatch.setattr("alpha_os_recovery.hypotheses.producer._should_sync_required_features", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("alpha_os_recovery.hypotheses.producer.load_cached_eval_universe", lambda: ["asset_a"])
    monkeypatch.setattr("alpha_os_recovery.hypotheses.producer.build_signal_client_from_config", lambda _cfg: object())

    cfg = Config()
    store = HypothesisStore(tmp_path / "hypotheses.db")
    prediction_store = PredictionStore(tmp_path / "predictions.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h_observe_only",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            status="active",
            stake=0.0,
            source="random_dsl",
        )
    )

    written = produce_active_hypothesis_predictions(
        cfg,
        today="2026-03-23",
        assets=["asset_a"],
        hypothesis_store=store,
        prediction_store=prediction_store,
    )

    prediction_store = PredictionStore(tmp_path / "predictions.db")
    rows = prediction_store.read_latest("2026-03-23")
    assert written == 1
    assert "h_observe_only" in rows
    prediction_store.close()


def test_compute_hypothesis_prediction_rejects_structurally_invalid_dsl():
    record = HypothesisRecord(
        hypothesis_id="bad-shape",
        kind=HypothesisKind.DSL,
        definition={"expression": "(lag_30 (sign 1.17))"},
    )

    with pytest.raises(ValueError, match="requires series input"):
        compute_hypothesis_prediction(record, data=_sample_data(), asset="asset_a")


def test_produce_active_hypothesis_predictions_syncs_needed_features(tmp_path, monkeypatch):
    calls = {}

    class _FakeClient:
        def health(self):
            return True

    class _FakeStore:
        def __init__(self, db_path, client):
            self.db_path = db_path
            self.client = client

        def sync(self, features):
            calls["features"] = list(features)

        def get_matrix(self, features):
            return pd.DataFrame(
                {
                    "asset_a": np.linspace(1.0, 2.0, 80),
                    "fear_greed": np.linspace(10.0, 90.0, 80),
                    "dxy": np.linspace(100.0, 90.0, 80),
                }
            ).loc[:, features]

        def close(self):
            return None

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.build_signal_client_from_config",
        lambda api: _FakeClient(),
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.init_universe",
        lambda client: None,
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.DataStore",
        _FakeStore,
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer._quick_healthcheck",
        lambda base_url, timeout=3.0: True,
    )

    hypothesis_store = HypothesisStore(tmp_path / "hypotheses.db")
    hypothesis_store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(sub fear_greed dxy)"},
            source="random_dsl",
            stake=1.0,
        )
    )

    written = produce_active_hypothesis_predictions(
        Config(),
        today="2026-03-21",
        assets=["asset_a"],
        hypothesis_store=hypothesis_store,
        prediction_store=PredictionStore(tmp_path / "predictions.db"),
    )

    assert written == 1
    assert set(calls["features"]) == {"asset_a", "fear_greed", "dxy"}


def test_produce_active_hypothesis_predictions_skips_universe_refresh_when_assets_given(
    tmp_path,
    monkeypatch,
):
    class _FakeStore:
        def __init__(self, db_path, client):
            self.db_path = db_path
            self.client = client

        def sync(self, features):
            return None

        def get_matrix(self, features):
            return pd.DataFrame(
                {
                    "asset_a": np.linspace(1.0, 2.0, 80),
                    "fear_greed": np.linspace(10.0, 90.0, 80),
                    "dxy": np.linspace(100.0, 90.0, 80),
                }
            ).loc[:, features]

        def close(self):
            return None

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.build_signal_client_from_config",
        lambda api: object(),
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.init_universe",
        lambda client: (_ for _ in ()).throw(AssertionError("init_universe should not be called")),
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer.DataStore",
        _FakeStore,
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.producer._quick_healthcheck",
        lambda base_url, timeout=3.0: False,
    )

    hypothesis_store = HypothesisStore(tmp_path / "hypotheses.db")
    hypothesis_store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(sub fear_greed dxy)"},
            source="random_dsl",
            stake=1.0,
        )
    )

    written = produce_active_hypothesis_predictions(
        Config(),
        today="2026-03-21",
        assets=["asset_a"],
        hypothesis_store=hypothesis_store,
        prediction_store=PredictionStore(tmp_path / "predictions.db"),
    )

    assert written == 1


def test_should_sync_required_features_uses_cache_depth(tmp_path):
    from alpha_os_recovery.data.store import DataStore

    store = DataStore(tmp_path / "cache.db")
    store._conn.executemany(
        "INSERT OR REPLACE INTO signals (name, date, value, resolution) VALUES (?, ?, ?, ?)",
        [
            ("btc_ohlcv", f"2026-03-{day:02d}", float(day), "1d")
            for day in range(1, 70)
        ],
    )
    store._conn.commit()

    hypotheses = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 20}},
            stake=1.0,
        )
    ]

    assert _should_sync_required_features(store, hypotheses, {"btc_ohlcv"}) is False
    assert _should_sync_required_features(store, hypotheses, {"btc_ohlcv", "fear_greed"}) is True
    store.close()


def test_resolve_asset_series_name_known_and_unknown_assets():
    assert _resolve_asset_series_name("BTC") == "btc_ohlcv"
    assert _resolve_asset_series_name("asset_a") == "asset_a"


def test_quick_healthcheck_returns_false_for_unreachable_host():
    assert _quick_healthcheck("http://127.0.0.1:9", timeout=0.01) is False
