from __future__ import annotations

from alpha_os.config import Config


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
