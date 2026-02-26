from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_os.data.client import SignalClient

log = logging.getLogger(__name__)

SIGNAL_NOISE_DB = Path.home() / "projects" / "signal-noise" / "data" / "signals.db"

STOCKS: dict[str, str] = {
    "NVDA": "nvda",
    "AAPL": "aapl",
    "MSFT": "msft",
    "GOOGL": "googl",
    "AMZN": "amzn",
    "META": "meta",
    "TSLA": "tsla",
    "AMD": "amd",
}

CRYPTO: dict[str, str] = {
    "BTC": "btc_ohlcv",
    "ETH": "eth_btc",
    "SOL": "sol_usdt",
    "BNB": "bnb_usdt",
    "XRP": "xrp_usdt",
    "ADA": "ada_usdt",
    "DOGE": "doge_usdt",
}

MACRO_SIGNALS = [
    "vix_close",
    "fear_greed",
    "dxy",
    "gold",
    "sp500",
    "nasdaq",
    "tsy_yield_10y",
    "tsy_yield_2y",
    "oil_wti",
    "russell2000",
]

EXTRA_SIGNALS = [
    "earthquake_count",
    "sunspot_number",
    "enso_oni",
    "btc_mempool_size",
    "btc_hashrate",
    "github_pushes",
    "steam_online",
]

_ALL_ASSETS = {**STOCKS, **CRYPTO}


def all_signals() -> list[str]:
    """Return all signal names used for alpha generation."""
    return list(_ALL_ASSETS.values()) + MACRO_SIGNALS


def price_signal(asset: str) -> str:
    """Return the price signal name for a given asset."""
    if asset in _ALL_ASSETS:
        return _ALL_ASSETS[asset]
    lower = asset.lower()
    for signal in _ALL_ASSETS.values():
        if signal == lower:
            return signal
    raise KeyError(f"Unknown asset: {asset}")


_daily_signal_cache: list[str] | None = None


def load_daily_signals(db_path: Path | None = None) -> list[str]:
    """Load all daily signal names from signal-noise DB.

    Falls back to MACRO_SIGNALS if DB is unavailable.
    """
    global _daily_signal_cache
    if _daily_signal_cache is not None:
        return _daily_signal_cache

    path = db_path or SIGNAL_NOISE_DB
    if not path.exists():
        log.warning("signal-noise DB not found at %s, using static list", path)
        _daily_signal_cache = MACRO_SIGNALS
        return _daily_signal_cache

    try:
        conn = sqlite3.connect(str(path))
        cur = conn.execute(
            "SELECT name FROM signal_meta WHERE interval = 86400 ORDER BY name"
        )
        names = [row[0] for row in cur]
        conn.close()
        if names:
            log.info("Loaded %d daily signals from signal-noise", len(names))
            _daily_signal_cache = names
            return _daily_signal_cache
    except Exception as e:
        log.warning("Failed to read signal-noise DB: %s, using static list", e)

    _daily_signal_cache = MACRO_SIGNALS
    return _daily_signal_cache


def discover_signals(
    client: SignalClient,
    domain: str | None = None,
    category: str | None = None,
) -> list[str]:
    """Discover signal names from signal-noise metadata API.

    Falls back to static list if API is unavailable.
    """
    try:
        signals = client.list_signals(domain=domain, category=category)
        return [s["name"] for s in signals]
    except Exception as e:
        log.warning("Failed to discover signals from API: %s. Using static list.", e)
        return all_signals()
