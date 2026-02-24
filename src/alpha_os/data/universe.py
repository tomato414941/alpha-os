from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_os.data.client import SignalClient

log = logging.getLogger(__name__)

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
