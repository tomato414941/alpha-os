from __future__ import annotations

from pathlib import Path


LOCAL_DATASET_DIR = Path(__file__).resolve().parent / "market_data" / "yahoo_daily"
DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
