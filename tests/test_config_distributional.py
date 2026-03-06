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
