from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import DATA_DIR

if TYPE_CHECKING:
    from signal_noise.client import SignalClient

log = logging.getLogger(__name__)

# ── Asset maps (for venue inference and price signal lookup) ──

STOCKS: dict[str, str] = {
    "NVDA": "nvda", "AAPL": "aapl", "MSFT": "msft", "GOOGL": "googl",
    "AMZN": "amzn", "META": "meta", "TSLA": "tsla", "AMD": "amd",
    "JPM": "jpm", "V": "v", "UNH": "unh", "JNJ": "jnj",
    "XOM": "xom", "PG": "pg",
}

ETFS: dict[str, str] = {
    "SPY": "spy", "QQQ": "qqq", "IWM": "iwm", "DIA": "dia",
    "GLD": "gld", "TLT": "tlt", "XLF": "xlf", "XLE": "xle",
    "XLK": "xlk", "XLV": "xlv", "EEM": "eem", "HYG": "hyg",
    "LQD": "lqd", "VXX": "vxx",
}

CRYPTO: dict[str, str] = {
    "BTC": "btc_ohlcv", "ETH": "eth_btc", "SOL": "sol_usdt",
    "BNB": "bnb_usdt", "XRP": "xrp_usdt", "ADA": "ada_usdt",
    "DOGE": "doge_usdt",
}

MACRO_SIGNALS = [
    "vix_close", "fear_greed", "dxy", "gold", "sp500", "nasdaq",
    "tsy_yield_10y", "tsy_yield_2y", "oil_wti", "russell2000",
]

EXTRA_SIGNALS = [
    "earthquake_count", "sunspot_number", "enso_oni",
    "btc_mempool_size", "btc_hashrate", "github_pushes", "steam_online",
]

MICROSTRUCTURE_SIGNALS = [
    "book_imbalance_btc", "book_depth_ratio_btc", "spread_bps_btc",
    "trade_flow_btc", "vpin_btc", "large_trade_count_btc",
]

HOURLY_SIGNALS = [
    "funding_rate_btc", "funding_rate_eth", "funding_rate_sol",
    "liq_ratio_btc_1h", "liq_ratio_eth_1h",
    "oi_btc_1h", "oi_eth_1h", "oi_sol_1h",
    "ls_ratio_global_btc", "ls_ratio_top_btc", "ls_position_ratio_btc",
    "iv_atm_btc_30d", "iv_atm_btc_7d", "iv_skew_btc_7d",
    "put_call_ratio_btc", "max_pain_btc", "gamma_exposure_btc",
    "spread_binance_bybit_btc", "spread_binance_okx_btc",
    "volume_dominance_btc", "lead_lag_btc",
]

POLYMARKET: dict[str, str] = {}

_ALL_ASSETS = {**STOCKS, **ETFS, **CRYPTO}

# ── Signal catalog (API-driven with local cache) ──

_CATALOG_CACHE_PATH = DATA_DIR / "signal_catalog.json"
_daily_signal_cache: list[str] | None = None
_signal_catalog_cache: list[dict] | None = None


@dataclass(frozen=True)
class SignalCatalogStatus:
    source: str
    signal_count: int
    intervals: tuple[int, ...] | None
    api_error_kind: str = ""
    api_error_message: str = ""


_signal_catalog_status = SignalCatalogStatus(
    source="uninitialized", signal_count=0, intervals=None,
)


def signal_catalog_status() -> SignalCatalogStatus:
    return _signal_catalog_status


def _classify_error(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timeout" in message:
        return "timeout"
    if isinstance(exc, ConnectionError) or "connection" in name or "connect" in message:
        return "connection"
    if "401" in message or "403" in message or "unauthorized" in message:
        return "auth"
    return "api_error"


def _write_catalog_cache(signals: list[dict]) -> None:
    try:
        _CATALOG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CATALOG_CACHE_PATH.write_text(json.dumps(signals), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to write signal catalog cache: %s", e)


def _load_catalog_cache() -> list[dict] | None:
    if not _CATALOG_CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(_CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else None
    except Exception:
        return None


def _load_signal_catalog(client: SignalClient | None = None) -> list[dict]:
    """Load full signal catalog (list of dicts) from API, cache, or static fallback."""
    global _signal_catalog_cache, _signal_catalog_status
    if _signal_catalog_cache is not None:
        return _signal_catalog_cache

    # Try API
    if client is not None:
        try:
            signals = client.list_signals()
            if signals:
                _write_catalog_cache(signals)
                _signal_catalog_status = SignalCatalogStatus(
                    source="api", signal_count=len(signals), intervals=None,
                )
                log.info("Signal catalog source=api count=%d", len(signals))
                _signal_catalog_cache = signals
                return signals
        except Exception as e:
            error_kind = _classify_error(e)
            if error_kind == "auth":
                # Auth errors are permanent — log as ERROR, don't cache fallback
                log.error("Signal catalog API auth failed: %s — check ALPHA_OS_SIGNAL_NOISE_API_KEY", e)
            else:
                log.warning("Signal catalog API failed: %s (%s)", error_kind, e)
            _signal_catalog_status = SignalCatalogStatus(
                source="cache", signal_count=0, intervals=None,
                api_error_kind=error_kind, api_error_message=str(e),
            )

    # Try local cache (don't save to process cache — retry API next call)
    cached = _load_catalog_cache()
    if cached:
        _signal_catalog_status = SignalCatalogStatus(
            source="cache", signal_count=len(cached), intervals=None,
            api_error_kind=_signal_catalog_status.api_error_kind,
            api_error_message=_signal_catalog_status.api_error_message,
        )
        log.warning("Signal catalog source=cache count=%d", len(cached))
        return cached

    # Static fallback
    static = [{"name": s, "signal_type": "scalar"} for s in MACRO_SIGNALS]
    _signal_catalog_status = SignalCatalogStatus(
        source="static", signal_count=len(static), intervals=None,
        api_error_kind=_signal_catalog_status.api_error_kind,
        api_error_message=_signal_catalog_status.api_error_message,
    )
    log.error("Signal catalog source=static count=%d", len(static))
    return static


def load_daily_signals(client: SignalClient | None = None) -> list[str]:
    """Load available signal names from signal-noise API.

    Falls back to local cache, then to static list.
    The client parameter should come from Config — no hardcoded URLs.
    """
    global _daily_signal_cache
    if _daily_signal_cache is not None:
        return _daily_signal_cache
    catalog = _load_signal_catalog(client)
    names = sorted(s["name"] for s in catalog)
    _daily_signal_cache = names
    return names


def load_price_signals(client: SignalClient | None = None) -> list[str]:
    """Return only price (OHLCV) signal names from the catalog."""
    catalog = _load_signal_catalog(client)
    return sorted(
        s["name"] for s in catalog
        if s.get("signal_type") == "ohlcv"
    )


def load_cross_asset_universe(client: SignalClient | None = None) -> list[str]:
    """Build cross-asset universe from signal-noise API.

    Returns only price (OHLCV) signals — these are tradeable assets.
    """
    return load_price_signals(client)


# Module-level universes — populated lazily or by explicit init
CROSS_ASSET_UNIVERSE: list[str] = []
FEATURE_CATALOG: list[str] = []


def init_universe(client: SignalClient | None = None) -> list[str]:
    """Initialize the cross-asset universe from API. Call once at startup."""
    global CROSS_ASSET_UNIVERSE, FEATURE_CATALOG
    CROSS_ASSET_UNIVERSE = load_cross_asset_universe(client)
    FEATURE_CATALOG = load_daily_signals(client)
    log.info(
        "Cross-asset universe: %d price signals, feature catalog: %d total",
        len(CROSS_ASSET_UNIVERSE), len(FEATURE_CATALOG),
    )
    return CROSS_ASSET_UNIVERSE


# ── Asset classification ──

def is_stock(asset: str) -> bool:
    return asset.upper() in STOCKS

def is_etf(asset: str) -> bool:
    return asset.upper() in ETFS

def is_equity(asset: str) -> bool:
    return asset.upper() in STOCKS or asset.upper() in ETFS

def is_crypto(asset: str) -> bool:
    return asset.upper() in CRYPTO

def is_polymarket(asset: str) -> bool:
    return asset in POLYMARKET

def register_polymarket_market(condition_id: str, signal_name: str) -> None:
    POLYMARKET[condition_id] = signal_name

def infer_venue(asset: str) -> str:
    if is_crypto(asset):
        return "binance"
    if is_equity(asset):
        return "alpaca"
    if is_polymarket(asset):
        return "polymarket"
    return "paper"


# ── Feature building ──

def all_signals() -> list[str]:
    return list(_ALL_ASSETS.values()) + MACRO_SIGNALS


def price_signal(asset: str) -> str:
    if asset in _ALL_ASSETS:
        return _ALL_ASSETS[asset]
    lower = asset.lower()
    for signal in _ALL_ASSETS.values():
        if signal == lower:
            return signal
    raise KeyError(f"Unknown asset: {asset}")


def build_feature_list(asset: str, client: SignalClient | None = None) -> list[str]:
    """Build deduplicated feature list: price signal first, then all daily signals."""
    try:
        price = price_signal(asset)
    except KeyError:
        price = asset.lower()
    daily = load_daily_signals(client)
    seen = {price}
    result = [price]
    for s in daily:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def build_hourly_feature_list(asset: str) -> list[str]:
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


def build_microstructure_feature_list(asset: str) -> list[str]:
    try:
        price = price_signal(asset)
    except KeyError:
        price = asset.lower()
    seen = {price}
    result = [price]
    for s in MICROSTRUCTURE_SIGNALS:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def infer_feature_family(name: str) -> str:
    lowered = name.lower()
    if lowered in {"sp500", "nasdaq", "vix_close", "fear_greed", "dxy",
                   "gold", "oil_wti", "russell2000", "tsy_yield_10y", "tsy_yield_2y"}:
        return "macro"
    if lowered.startswith(("book_", "spread_", "trade_flow_", "vpin_", "large_trade_")):
        return "microstructure"
    if lowered.startswith(("funding_", "oi_", "liq_", "ls_", "iv_", "put_call_", "max_pain_", "gamma_")):
        return "derivatives"
    if lowered.startswith(("btc_mempool_", "btc_hashrate", "btc_active_", "btc_difficulty")):
        return "onchain"
    if lowered.startswith(("earthquake_", "sunspot_", "enso_", "github_", "steam_")):
        return "alt"
    if lowered.startswith(("gdelt_", "eonet_", "fr24_", "glad_", "nasa_")):
        return "event"
    if lowered.endswith(("_ohlcv", "_usdt", "_btc")) or lowered in {
        "nvda", "aapl", "msft", "googl", "amzn", "meta", "tsla", "amd",
    }:
        return "price"
    return "other"


def stratified_feature_subset(
    features: list[str], *, k: int, seed: int | None = None,
) -> frozenset[str]:
    if k <= 0 or not features:
        return frozenset()
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(infer_feature_family(feature), []).append(feature)
    for bucket in grouped.values():
        rng.shuffle(bucket)
    family_order = list(grouped)
    rng.shuffle(family_order)
    selected: list[str] = []
    while len(selected) < min(k, len(features)):
        added = False
        for family in family_order:
            bucket = grouped[family]
            if not bucket:
                continue
            selected.append(bucket.pop())
            added = True
            if len(selected) >= min(k, len(features)):
                break
        if not added:
            break
    return frozenset(selected)


def discover_signals(
    client: SignalClient,
    domain: str | None = None,
    category: str | None = None,
) -> list[str]:
    try:
        signals = client.list_signals(domain=domain, category=category)
        return [s["name"] for s in signals]
    except Exception as e:
        log.warning("Failed to discover signals: %s", e)
        return all_signals()
