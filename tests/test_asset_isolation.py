"""Tests for Phase 6 per-asset data isolation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_os.config import DATA_DIR


@pytest.fixture(autouse=True)
def _patch_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to tmp_path for all tests."""
    monkeypatch.setattr("alpha_os.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("alpha_os.config._BTC_MIGRATED", False)
    yield


class TestAssetDataDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        d = asset_data_dir("ETH")
        assert d == tmp_path / "ETH"
        assert d.is_dir()
        assert (d / "metrics").is_dir()
        assert (d / "logs").is_dir()

    def test_normalizes_case(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        assert asset_data_dir("btc") == asset_data_dir("BTC")
        assert asset_data_dir("Eth") == tmp_path / "ETH"

    def test_different_assets_different_dirs(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        btc = asset_data_dir("BTC")
        eth = asset_data_dir("ETH")
        assert btc != eth
        assert btc.name == "BTC"
        assert eth.name == "ETH"


class TestBtcMigration:
    def test_migrates_flat_files(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        # Create flat files mimicking existing state
        (tmp_path / "alpha_registry.db").write_text("registry")
        (tmp_path / "paper_trading.db").write_text("paper")
        (tmp_path / "metrics").mkdir()
        (tmp_path / "metrics" / "circuit_breaker.json").write_text("{}")

        d = asset_data_dir("BTC")

        assert (d / "alpha_registry.db").read_text() == "registry"
        assert (d / "paper_trading.db").read_text() == "paper"
        assert (d / "metrics" / "circuit_breaker.json").read_text() == "{}"
        # Originals moved
        assert not (tmp_path / "alpha_registry.db").exists()
        assert not (tmp_path / "paper_trading.db").exists()

    def test_no_migration_if_already_in_subdir(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        # Already-migrated state: file in BTC/ but not in flat data/
        btc_dir = tmp_path / "BTC"
        btc_dir.mkdir()
        (btc_dir / "alpha_registry.db").write_text("existing")

        d = asset_data_dir("BTC")
        assert (d / "alpha_registry.db").read_text() == "existing"

    def test_migration_skips_non_btc(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        (tmp_path / "alpha_registry.db").write_text("registry")
        d = asset_data_dir("ETH")
        # ETH should NOT trigger BTC migration
        assert not (d / "alpha_registry.db").exists()
        # Flat file should still be there
        assert (tmp_path / "alpha_registry.db").exists()


class TestRegistryIsolation:
    def test_two_assets_isolated(self, tmp_path, monkeypatch):
        from alpha_os.config import asset_data_dir
        from alpha_os.alpha.registry import AlphaRegistry, AlphaRecord, AlphaState

        btc_dir = asset_data_dir("BTC")
        eth_dir = asset_data_dir("ETH")

        reg_btc = AlphaRegistry(db_path=btc_dir / "alpha_registry.db")
        reg_eth = AlphaRegistry(db_path=eth_dir / "alpha_registry.db")

        reg_btc.register(AlphaRecord(
            alpha_id="btc_1", expression="(neg btc_ohlcv)",
            state=AlphaState.ACTIVE, fitness=1.0,
        ))
        reg_eth.register(AlphaRecord(
            alpha_id="eth_1", expression="(neg eth_btc)",
            state=AlphaState.ACTIVE, fitness=0.8,
        ))

        assert reg_btc.count() == 1
        assert reg_eth.count() == 1
        assert reg_btc.get("eth_1") is None
        assert reg_eth.get("btc_1") is None

        reg_btc.close()
        reg_eth.close()


class TestCircuitBreakerPath:
    def test_saves_to_asset_path(self, tmp_path):
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cb_path = tmp_path / "BTC" / "metrics" / "circuit_breaker.json"
        cb = CircuitBreaker.load(path=cb_path)
        cb.record_trade(-100)
        assert cb_path.exists()

        data = json.loads(cb_path.read_text())
        assert data["daily_pnl"] == -100

    def test_load_preserves_path(self, tmp_path):
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        path1 = tmp_path / "BTC" / "metrics" / "cb.json"
        path2 = tmp_path / "ETH" / "metrics" / "cb.json"

        cb1 = CircuitBreaker.load(path=path1)
        cb1.record_trade(-50)

        cb2 = CircuitBreaker.load(path=path2)
        cb2.record_trade(-200)

        # Reload and verify isolation
        cb1_reloaded = CircuitBreaker.load(path=path1)
        cb2_reloaded = CircuitBreaker.load(path=path2)
        assert cb1_reloaded._daily_pnl == -50
        assert cb2_reloaded._daily_pnl == -200
