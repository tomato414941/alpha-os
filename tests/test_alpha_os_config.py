from __future__ import annotations

from pathlib import Path


def test_load_runtime_config_defaults_to_persistent_workspace_db():
    from alpha_os.config import DEFAULT_DB_PATH, load_runtime_config

    cfg = load_runtime_config()

    assert cfg.db_path == DEFAULT_DB_PATH
    assert cfg.db_path == Path.home() / ".local" / "share" / "alpha-os" / "db" / "workspace.db"


def test_load_runtime_config_expands_user_db_path():
    from alpha_os.config import load_runtime_config

    cfg = load_runtime_config(db_path="~/alpha-os-test.db")

    assert cfg.db_path == Path.home() / "alpha-os-test.db"
