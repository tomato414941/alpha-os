from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = Path("data") / "v1" / "runtime.db"
DEFAULT_ASSET = "BTC"
DEFAULT_TARGET = "residual_return_1d"
DEFAULT_SIGNAL_NOISE_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PRICE_SIGNAL = "btc_ohlcv"


@dataclass(frozen=True)
class V1Config:
    db_path: Path
    asset: str = DEFAULT_ASSET
    target: str = DEFAULT_TARGET


def build_config(*, db_path: str | None = None) -> V1Config:
    path = DEFAULT_DB_PATH if db_path is None else Path(db_path)
    return V1Config(db_path=path, asset=DEFAULT_ASSET, target=DEFAULT_TARGET)
