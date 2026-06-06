from __future__ import annotations

from pathlib import Path


LOCAL_DATASET_DIR = Path(__file__).resolve().parent / "market_data" / "daily_close"
DEFAULT_SYMBOLS = ("SPY", "QQQ", "GLD", "TLT", "BTCUSDT", "ETHUSDT")
