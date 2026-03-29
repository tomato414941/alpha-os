from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .targets import residual_return_target_definition

DEFAULT_DB_PATH = Path("data") / "v1" / "runtime.db"
DEFAULT_ASSET = "BTC"
DEFAULT_HORIZON_DAYS = 3
DEFAULT_SIGNAL_NOISE_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PRICE_SIGNAL = "btc_ohlcv"
DEFAULT_TARGET_DEFINITION = residual_return_target_definition(DEFAULT_HORIZON_DAYS)
DEFAULT_TARGET = DEFAULT_TARGET_DEFINITION.target_id


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    asset: str = DEFAULT_ASSET
    target: str = DEFAULT_TARGET


def load_runtime_config(*, db_path: str | None = None) -> RuntimeConfig:
    path = DEFAULT_DB_PATH if db_path is None else Path(db_path)
    return RuntimeConfig(db_path=path, asset=DEFAULT_ASSET, target=DEFAULT_TARGET)
