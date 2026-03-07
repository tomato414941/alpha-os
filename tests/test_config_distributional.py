from __future__ import annotations

import pytest

from alpha_os.config import Config
from alpha_os.config import TRADING_MODE_FUTURES_LONG_SHORT, TradingConfig


def test_unknown_toml_keys_ignored(tmp_path):
    """Config.load() should silently ignore removed TOML keys."""
    p = tmp_path / "cfg.toml"
    p.write_text(
        "\n".join(
            [
                "[paper]",
                "max_position_pct = 0.25",
                "signal_compression = 0.5",  # removed field
                "exit_enabled = true",        # removed field
                "[distributional]",           # removed section
                "enabled = true",
                "kelly_fraction = 0.25",
            ]
        )
    )
    cfg = Config.load(p)
    assert cfg.paper.max_position_pct == 0.25


def test_override_file_inherits_default_toml(tmp_path, monkeypatch):
    default_dir = tmp_path / "config"
    default_dir.mkdir()
    (default_dir / "default.toml").write_text(
        "\n".join(
            [
                'fitness_metric = "log_growth"',
                "[paper]",
                'combine_mode = "consensus"',
                "max_trading_alphas = 30",
            ]
        )
    )
    override = tmp_path / "prod.toml"
    override.write_text(
        "\n".join(
            [
                "[paper]",
                "max_position_pct = 0.25",
            ]
        )
    )

    monkeypatch.setattr("alpha_os.config.CONFIG_DIR", default_dir)

    cfg = Config.load(override)
    assert cfg.fitness_metric == "log_growth"
    assert cfg.paper.combine_mode == "consensus"
    assert cfg.paper.max_trading_alphas == 30
    assert cfg.paper.max_position_pct == 0.25


def test_generation_penalties_load_from_toml(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(
        "\n".join(
            [
                "[generation]",
                "max_depth = 4",
                "bloat_penalty = 0.02",
                "depth_penalty = 0.005",
                "similarity_penalty = 0.03",
            ]
        )
    )
    cfg = Config.load(p)
    assert cfg.generation.max_depth == 4
    assert cfg.generation.bloat_penalty == 0.02
    assert cfg.generation.depth_penalty == 0.005
    assert cfg.generation.similarity_penalty == 0.03


def test_trading_mode_loads_from_toml(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(
        "\n".join(
            [
                "[trading]",
                'mode = "futures_long_short"',
            ]
        )
    )
    cfg = Config.load(p)
    assert cfg.trading.mode == TRADING_MODE_FUTURES_LONG_SHORT
    assert cfg.trading.supports_short is True


def test_unknown_trading_mode_raises_on_capability_check():
    cfg = TradingConfig(mode="invalid")
    with pytest.raises(ValueError, match="Unknown trading mode"):
        _ = cfg.supports_short


def test_config_runtime_helpers_follow_current_settings():
    cfg = Config()
    cfg.forward.degradation_window = 42
    cfg.live_quality.min_observations = 11
    cfg.live_quality.full_weight_observations = 50
    cfg.live_quality.weight_confidence_floor = 0.25
    cfg.live_quality.weight_confidence_power = 2.0
    cfg.lifecycle.active_quality_min = 0.12

    monitor_cfg = cfg.to_monitor_config()
    lifecycle_cfg = cfg.to_lifecycle_config()
    estimate = cfg.estimate_alpha_quality(0.8, [])

    assert monitor_cfg.rolling_window == 42
    assert monitor_cfg.min_observations == 11
    assert lifecycle_cfg.active_quality_min == pytest.approx(0.12)
    assert estimate.blended_quality == pytest.approx(0.8)
    assert estimate.confidence == pytest.approx(0.0)
    assert cfg.live_quality.weight_confidence_floor == pytest.approx(0.25)
    assert cfg.live_quality.weight_confidence_power == pytest.approx(2.0)
