from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signal_noise.client import SignalClient

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

EXTRA_SIGNALS = [
    "earthquake_count",
    "sunspot_number",
    "enso_oni",
    "btc_mempool_size",
    "btc_hashrate",
    "github_pushes",
    "steam_online",
]

HOURLY_SIGNALS = [
    "funding_rate_btc", "funding_rate_eth", "funding_rate_sol",
    "liq_ratio_btc_1h", "liq_ratio_eth_1h",
    "oi_btc_1h", "oi_eth_1h", "oi_sol_1h",
    "ls_ratio_global_btc", "ls_ratio_top_btc", "ls_position_ratio_btc",
]

_ALL_ASSETS = {**STOCKS, **CRYPTO}


def is_crypto(asset: str) -> bool:
    """Return True if asset is a cryptocurrency (traded on Binance)."""
    return asset.upper() in CRYPTO


def all_signals() -> list[str]:
    """Return all signal names used for alpha generation."""
    return list(_ALL_ASSETS.values()) + MACRO_SIGNALS


def build_feature_list(asset: str) -> list[str]:
    """Build deduplicated feature list: price signal first, then all daily signals."""
    try:
        price = price_signal(asset)
    except KeyError:
        price = asset.lower()
    daily = load_daily_signals()
    seen = {price}
    result = [price]
    for s in daily:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


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


def load_daily_signals() -> list[str]:
    """Load daily + hourly signal names from signal-noise REST API.

    Falls back to MACRO_SIGNALS if API is unavailable.
    """
    global _daily_signal_cache
    if _daily_signal_cache is not None:
        return _daily_signal_cache

    try:
        from signal_noise.client import SignalClient
        client = SignalClient(base_url="http://127.0.0.1:8000", timeout=10)
        signals = client.list_signals()
        names = sorted(
            s["name"] for s in signals
            if s.get("interval") in (3600, 86400)
        )
        if names:
            log.info("Loaded %d daily signals from API", len(names))
            _daily_signal_cache = names
            return _daily_signal_cache
    except Exception as e:
        log.warning("Failed to load signals from API: %s, using static list", e)

    _daily_signal_cache = MACRO_SIGNALS
    return _daily_signal_cache


def build_hourly_feature_list(asset: str) -> list[str]:
    """Layer 2 feature list: price signal + hourly derivatives signals."""
    try:
        price = price_signal(asset)
    except KeyError:
        price = asset.lower()
    seen = {price}
    result = [price]
    for s in HOURLY_SIGNALS:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


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
