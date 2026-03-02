from __future__ import annotations

from alpha_os.config import Config


def test_distributional_defaults_disabled():
    cfg = Config()
    assert not cfg.distributional.enabled
    assert cfg.distributional.window == 63


def test_distributional_load_from_toml(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(
        "\n".join(
            [
                "[distributional]",
                "enabled = true",
                "window = 42",
                "min_samples = 15",
                "tail_sigma = 1.5",
                "cvar_alpha = 0.1",
                "max_left_tail_prob = 0.2",
                "max_cvar_abs = 0.04",
                "kelly_fraction = 0.25",
                "max_kelly_leverage = 0.8",
            ]
        )
    )
    cfg = Config.load(p)
    assert cfg.distributional.enabled
    assert cfg.distributional.window == 42
    assert cfg.distributional.min_samples == 15
    assert cfg.distributional.max_kelly_leverage == 0.8


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
