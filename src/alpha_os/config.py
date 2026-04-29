from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .targets import residual_return_target_definition

DEFAULT_RUNTIME_HOME = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "alpha-os"
)
DEFAULT_DB_PATH = DEFAULT_RUNTIME_HOME / "db" / "workspace.db"
DEFAULT_SUBJECT_ID = "BTC"
DEFAULT_HORIZON_DAYS = 3
DEFAULT_SIGNAL_NOISE_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TARGET_DEFINITION = residual_return_target_definition(DEFAULT_HORIZON_DAYS)
DEFAULT_TARGET = DEFAULT_TARGET_DEFINITION.target_id


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    default_subject_id: str = DEFAULT_SUBJECT_ID
    target_id: str = DEFAULT_TARGET


def default_runtime_asset(subject_id: str | None = None) -> str:
    return DEFAULT_SUBJECT_ID


def load_runtime_config(*, db_path: str | None = None) -> RuntimeConfig:
    path = DEFAULT_DB_PATH if db_path is None else Path(db_path).expanduser()
    return RuntimeConfig(
        db_path=path,
        default_subject_id=DEFAULT_SUBJECT_ID,
        target_id=DEFAULT_TARGET,
    )
