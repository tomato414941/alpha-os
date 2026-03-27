"""Shared fixtures for alpha-os tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alpha_os_recovery.legacy.managed_alphas import AlphaRecord, ManagedAlphaStore, AlphaState
from alpha_os_recovery.data.store import DataStore


FEATURES = ["f1", "f2", "f3"]

LEGACY_TEST_MODULES = {
    "test_admission_replay.py",
    "test_alpha.py",
    "test_alpha_generator.py",
    "test_asset_isolation.py",
    "test_funnel.py",
    "test_legacy_lifecycle.py",
    "test_pipeline.py",
}

RESEARCH_TEST_MODULES = {
    "test_backtest.py",
    "test_diversity.py",
    "test_event_driven.py",
    "test_evolution.py",
    "test_handcrafted.py",
    "test_replay_experiment.py",
    "test_replay_matrix.py",
    "test_simulator.py",
    "test_tactical.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in LEGACY_TEST_MODULES:
            item.add_marker(pytest.mark.legacy)
        elif filename in RESEARCH_TEST_MODULES:
            item.add_marker(pytest.mark.research)
        else:
            item.add_marker(pytest.mark.current)


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
    """Fresh ManagedAlphaStore backed by a temp DB."""
    reg = ManagedAlphaStore(db_path=tmp_path / "test_registry.db")
    yield reg
    reg.close()


@pytest.fixture
def populated_registry(tmp_path):
    """ManagedAlphaStore with a few ACTIVE alphas for integration tests."""
    reg = ManagedAlphaStore(db_path=tmp_path / "pop_registry.db")
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
