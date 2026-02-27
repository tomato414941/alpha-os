"""Shared fixtures for alpha-os tests."""
from __future__ import annotations

import numpy as np
import pytest

from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from alpha_os.data.store import DataStore


FEATURES = ["f1", "f2", "f3"]


@pytest.fixture
def synthetic_data():
    """Generate reproducible synthetic price data for 3 features."""
    rng = np.random.default_rng(42)
    n_days = 300
    data: dict[str, np.ndarray] = {}
    for feat in FEATURES:
        drift = rng.uniform(-0.0005, 0.001)
        vol = rng.uniform(0.005, 0.03)
        returns = rng.normal(drift, vol, n_days)
        data[feat] = 100.0 * np.cumprod(1.0 + returns)
    return FEATURES, data, data["f1"], n_days


@pytest.fixture
def registry(tmp_path):
    """Fresh AlphaRegistry backed by a temp DB."""
    reg = AlphaRegistry(db_path=tmp_path / "test_registry.db")
    yield reg
    reg.close()


@pytest.fixture
def populated_registry(tmp_path):
    """AlphaRegistry with a few ACTIVE alphas for integration tests."""
    reg = AlphaRegistry(db_path=tmp_path / "pop_registry.db")
    alphas = [
        AlphaRecord(
            alpha_id="alpha_neg_f1",
            expression="(neg f1)",
            state=AlphaState.ACTIVE,
            fitness=0.8,
            oos_sharpe=0.6,
        ),
        AlphaRecord(
            alpha_id="alpha_roll_f2",
            expression="(ts_mean f2 10)",
            state=AlphaState.ACTIVE,
            fitness=0.5,
            oos_sharpe=0.5,
        ),
        AlphaRecord(
            alpha_id="alpha_dormant",
            expression="(neg f3)",
            state=AlphaState.DORMANT,
            fitness=0.3,
            oos_sharpe=0.2,
        ),
    ]
    for a in alphas:
        reg.register(a)
    yield reg
    reg.close()


@pytest.fixture
def data_store(tmp_path, synthetic_data):
    """DataStore pre-populated with synthetic signal data."""
    features, data, prices, n_days = synthetic_data
    store = DataStore(tmp_path / "test_cache.db")

    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    for feat in features:
        for i, d in enumerate(dates):
            store._conn.execute(
                "INSERT OR REPLACE INTO signals (name, date, value) VALUES (?, ?, ?)",
                (feat, d.strftime("%Y-%m-%d"), float(data[feat][i])),
            )
    store._conn.commit()
    yield store
    store.close()
