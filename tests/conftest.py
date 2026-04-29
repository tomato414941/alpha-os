"""Shared fixtures for alpha-os tests."""
from __future__ import annotations

import numpy as np
import pytest


FEATURES = ["f1", "f2", "f3"]

def pytest_collection_modifyitems(config, items):
    for item in items:
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

