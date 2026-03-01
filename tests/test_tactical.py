"""Tests for TacticalTrader (Layer 2 hourly alpha)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from alpha_os.data.store import DataStore
from alpha_os.paper.tactical import TacticalTrader, TacticalSignal


@pytest.fixture
def l2_registry(tmp_path):
    reg = AlphaRegistry(db_path=tmp_path / "l2_registry.db")
    yield reg
    reg.close()


@pytest.fixture
def l2_store(tmp_path):
    store = DataStore(tmp_path / "l2_cache.db")
    # Populate with synthetic hourly data
    features = ["btc_ohlcv", "funding_rate_btc", "oi_btc_1h"]
    rng = np.random.default_rng(42)
    n = 100
    for feat in features:
        for i in range(n):
            store._conn.execute(
                "INSERT OR REPLACE INTO signals (name, date, value, resolution)"
                " VALUES (?, ?, ?, ?)",
                (feat, f"2024-01-01T{i:02d}:00:00", float(rng.uniform(0.5, 1.5)), "1h"),
            )
    store._conn.commit()
    yield store
    store.close()


class TestNoL2Alphas:
    def test_returns_strategic_bias_when_no_alphas(self, tmp_path, l2_store):
        cfg = MagicMock()
        cfg.forward.degradation_window = 63
        cfg.api.base_url = "http://localhost:8000"
        cfg.api.timeout = 10

        reg = AlphaRegistry(db_path=tmp_path / "empty_reg.db")
        trader = TacticalTrader(
            asset="BTC", config=cfg, registry=reg, store=l2_store,
        )

        result = trader.run_cycle(strategic_bias=0.5)

        assert isinstance(result, TacticalSignal)
        assert result.tactical_score == 0.0
        assert result.combined_signal == 0.5
        assert result.n_alphas_evaluated == 0
        assert result.confidence == 0.0

        reg.close()


class TestAgreementAmplifies:
    def test_same_direction_amplifies(self):
        combined = TacticalTrader._modulate(bias=0.5, tactical=0.3)
        # 0.5 * (1 + 0.3 * 0.5) = 0.5 * 1.15 = 0.575
        assert combined == pytest.approx(0.575)
        assert abs(combined) > abs(0.5)

    def test_negative_agreement(self):
        combined = TacticalTrader._modulate(bias=-0.4, tactical=-0.6)
        # -0.4 * (1 + 0.6 * 0.5) = -0.4 * 1.3 = -0.52
        assert combined == pytest.approx(-0.52)
        assert abs(combined) > abs(-0.4)


class TestDisagreementAttenuates:
    def test_opposite_direction_attenuates(self):
        combined = TacticalTrader._modulate(bias=0.5, tactical=-0.3)
        # 0.5 * (1 - 0.3 * 0.5) = 0.5 * 0.85 = 0.425
        assert combined == pytest.approx(0.425)
        assert abs(combined) < abs(0.5)

    def test_strong_disagreement(self):
        combined = TacticalTrader._modulate(bias=0.8, tactical=-1.0)
        # 0.8 * (1 - 1.0 * 0.5) = 0.8 * 0.5 = 0.4
        assert combined == pytest.approx(0.4)
        assert abs(combined) < abs(0.8)


class TestModulateEdgeCases:
    def test_zero_tactical_returns_bias(self):
        assert TacticalTrader._modulate(0.5, 0.0) == 0.5

    def test_zero_bias_stays_zero(self):
        assert TacticalTrader._modulate(0.0, 0.8) == 0.0

    def test_clipped_to_bounds(self):
        combined = TacticalTrader._modulate(bias=0.99, tactical=0.99)
        assert -1.0 <= combined <= 1.0


class TestWithL2Alphas:
    def test_evaluates_registered_alphas(self, tmp_path, l2_store, l2_registry):
        cfg = MagicMock()
        cfg.forward.degradation_window = 63
        cfg.api.base_url = "http://localhost:8000"
        cfg.api.timeout = 10

        # Register a simple L2 alpha
        l2_registry.register(AlphaRecord(
            alpha_id="l2_alpha_1",
            expression="(neg funding_rate_btc)",
            state=AlphaState.ACTIVE,
            fitness=0.5,
            oos_sharpe=0.3,
        ))

        trader = TacticalTrader(
            asset="BTC", config=cfg, registry=l2_registry, store=l2_store,
        )

        result = trader.run_cycle(strategic_bias=0.5)
        assert result.n_alphas_evaluated == 1
        assert result.tactical_score != 0.0
        # Combined should be modulated from strategic bias
        assert result.combined_signal != 0.0


class TestTraderIntegration:
    def test_trader_without_tactical(self, tmp_path):
        """Verify Trader works without tactical (default None)."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config

        cfg = Config.load()
        store = DataStore(tmp_path / "cache.db")

        # Populate minimal data
        rng = np.random.default_rng(42)
        n = 100
        features = ["btc_ohlcv", "vix_close"]
        for feat in features:
            for i in range(n):
                store._conn.execute(
                    "INSERT OR REPLACE INTO signals (name, date, value)"
                    " VALUES (?, ?, ?)",
                    (feat, f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}",
                     float(rng.uniform(40000, 50000) if feat == "btc_ohlcv" else rng.uniform(10, 30))),
                )
        store._conn.commit()

        trader = Trader(
            asset="BTC", config=cfg, store=store,
        )
        # tactical is None → no L2 modulation
        assert trader.tactical is None
        store.close()
