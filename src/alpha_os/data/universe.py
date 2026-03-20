from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import DATA_DIR
from .signal_client import build_signal_client

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
    "JPM": "jpm",
    "V": "v",
    "UNH": "unh",
    "JNJ": "jnj",
    "XOM": "xom",
    "PG": "pg",
}

ETFS: dict[str, str] = {
    "SPY": "spy",
    "QQQ": "qqq",
    "IWM": "iwm",
    "DIA": "dia",
    "GLD": "gld",
    "TLT": "tlt",
    "XLF": "xlf",
    "XLE": "xle",
    "XLK": "xlk",
    "XLV": "xlv",
    "EEM": "eem",
    "HYG": "hyg",
    "LQD": "lqd",
    "VXX": "vxx",
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

MICROSTRUCTURE_SIGNALS = [
    "book_imbalance_btc",
    "book_depth_ratio_btc",
    "spread_bps_btc",
    "trade_flow_btc",
    "vpin_btc",
    "large_trade_count_btc",
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

def _load_cross_asset_universe() -> list[str]:
    """Load cross-asset universe from signal-noise tickers.csv if available."""
    import csv
    csv_path = Path.home() / "projects" / "signal-noise" / "data" / "universe" / "tickers.csv"
    if not csv_path.exists():
        # Fallback: hardcoded core universe
        return [
            "btc_ohlcv", "eth_btc", "sol_usdt", "bnb_usdt", "xrp_usdt", "ada_usdt", "doge_usdt",
            "nvda", "aapl", "msft", "googl", "amzn", "meta", "tsla", "amd", "jpm", "xom",
            "sp500", "nasdaq", "gold", "russell2000", "tlt", "eem", "hyg", "oil_wti",
        ]
    names = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
    return names


CROSS_ASSET_UNIVERSE: list[str] = _load_cross_asset_universe()

POLYMARKET: dict[str, str] = {}

_ALL_ASSETS = {**STOCKS, **ETFS, **CRYPTO}


def is_stock(asset: str) -> bool:
    """Return True if asset is an individual stock."""
    return asset.upper() in STOCKS


def is_etf(asset: str) -> bool:
    """Return True if asset is an ETF."""
    return asset.upper() in ETFS


def is_equity(asset: str) -> bool:
    """Return True if asset is a stock or ETF (traded on Alpaca/traditional broker)."""
    return asset.upper() in STOCKS or asset.upper() in ETFS


def is_crypto(asset: str) -> bool:
    """Return True if asset is a cryptocurrency (traded on Binance)."""
    return asset.upper() in CRYPTO


def is_polymarket(asset: str) -> bool:
    """Return True if asset is a Polymarket prediction market."""
    return asset in POLYMARKET


def register_polymarket_market(condition_id: str, signal_name: str) -> None:
    """Register a Polymarket market for use in the signal universe."""
    POLYMARKET[condition_id] = signal_name
    log.info("Registered Polymarket market: %s -> %s", condition_id, signal_name)


def infer_venue(asset: str) -> str:
    """Infer the default trading venue for an asset.

    Returns 'binance' for crypto, 'alpaca' for stocks/ETFs,
    'polymarket' for prediction markets, or 'paper' as fallback.
    """
    if is_crypto(asset):
        return "binance"
    if is_equity(asset):
        return "alpaca"
    if is_polymarket(asset):
        return "polymarket"
    return "paper"


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
_signal_catalog_cache_path = DATA_DIR / "signal_catalog.json"


@dataclass(frozen=True)
class SignalCatalogStatus:
    source: str
    signal_count: int
    intervals: tuple[int, ...] | None
    api_error_kind: str = ""
    api_error_message: str = ""


_signal_catalog_status = SignalCatalogStatus(
    source="uninitialized",
    signal_count=0,
    intervals=None,
)


def _interval_filter() -> set[int] | None:
    """Parse interval filter from env.

    ALPHA_OS_SIGNAL_INTERVALS:
      - unset / "all" -> include all intervals
      - comma-separated seconds, e.g. "60,300,3600,86400"
    """
    raw = os.getenv("ALPHA_OS_SIGNAL_INTERVALS", "all").strip().lower()
    if not raw or raw == "all":
        return None
    values: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            log.warning("Ignoring invalid interval token: %s", token)
    return values or None


def _filter_signal_names(signals: list[dict], intervals: set[int] | None) -> list[str]:
    if intervals is None:
        return sorted(s["name"] for s in signals)
    return sorted(
        s["name"] for s in signals
        if s.get("interval") in intervals
    )


def _status_intervals(intervals: set[int] | None) -> tuple[int, ...] | None:
    if intervals is None:
        return None
    return tuple(sorted(intervals))


def _set_signal_catalog_status(
    *,
    source: str,
    signal_count: int,
    intervals: set[int] | None,
    api_error_kind: str = "",
    api_error_message: str = "",
) -> None:
    global _signal_catalog_status
    _signal_catalog_status = SignalCatalogStatus(
        source=source,
        signal_count=signal_count,
        intervals=_status_intervals(intervals),
        api_error_kind=api_error_kind,
        api_error_message=api_error_message,
    )


def signal_catalog_status() -> SignalCatalogStatus:
    return _signal_catalog_status


def _classify_signal_catalog_error(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "401" in message or "403" in message or "unauthorized" in message or "forbidden" in message:
        return "auth"
    if "timeout" in name or "timeout" in message:
        return "timeout"
    if "connection" in name or "connect" in message:
        return "connection"
    return "api_error"


def _write_signal_catalog_cache(signals: list[dict]) -> None:
    try:
        _signal_catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _signal_catalog_cache_path.write_text(json.dumps(signals), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to write signal catalog cache: %s", e)


def _load_signal_catalog_cache() -> list[dict] | None:
    if not _signal_catalog_cache_path.exists():
        return None
    try:
        raw = json.loads(_signal_catalog_cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    except Exception as e:
        log.warning("Failed to load cached signal catalog: %s", e)
        return None
    log.warning("Ignoring malformed cached signal catalog: %s", _signal_catalog_cache_path)
    return None


def _cached_signal_catalog_names(intervals: set[int] | None) -> list[str] | None:
    cached = _load_signal_catalog_cache()
    if not cached:
        return None
    names = _filter_signal_names(cached, intervals)
    return names or None


def load_daily_signals() -> list[str]:
    """Load runtime signal names from signal-noise REST API.

    Default includes all available intervals.
    Falls back to alpha-os cached signal catalog if API is unavailable.
    """
    global _daily_signal_cache
    if _daily_signal_cache is not None:
        return _daily_signal_cache

    intervals = _interval_filter()
    api_error_kind = ""
    api_error_message = ""
    try:
        base_url = os.getenv("ALPHA_OS_SIGNAL_NOISE_URL", "http://127.0.0.1:8000")
        client = build_signal_client(base_url=base_url, timeout=10)
        signals = client.list_signals()
        names = _filter_signal_names(signals, intervals)
        if names:
            _write_signal_catalog_cache(signals)
            _set_signal_catalog_status(
                source="api",
                signal_count=len(names),
                intervals=intervals,
            )
            if intervals is None:
                log.info("Signal catalog source=api count=%d intervals=all", len(names))
            else:
                log.info(
                    "Signal catalog source=api count=%d intervals=%s",
                    len(names),
                    sorted(intervals),
                )
            _daily_signal_cache = names
            return _daily_signal_cache
    except Exception as e:
        api_error_kind = _classify_signal_catalog_error(e)
        api_error_message = str(e)
        log.warning(
            "Signal catalog API failed kind=%s base_url=%s error=%s",
            api_error_kind,
            os.getenv("ALPHA_OS_SIGNAL_NOISE_URL", "http://127.0.0.1:8000"),
            e,
        )

    names = _cached_signal_catalog_names(intervals)
    if names:
        _set_signal_catalog_status(
            source="cache",
            signal_count=len(names),
            intervals=intervals,
            api_error_kind=api_error_kind,
            api_error_message=api_error_message,
        )
        if intervals is None:
            log.warning(
                "Signal catalog source=cache count=%d intervals=all after_api_failure=%s",
                len(names),
                api_error_kind or "none",
            )
        else:
            log.warning(
                "Signal catalog source=cache count=%d intervals=%s after_api_failure=%s",
                len(names),
                sorted(intervals),
                api_error_kind or "none",
            )
        _daily_signal_cache = names
        return _daily_signal_cache

    _set_signal_catalog_status(
        source="static",
        signal_count=len(MACRO_SIGNALS),
        intervals=intervals,
        api_error_kind=api_error_kind,
        api_error_message=api_error_message,
    )
    log.error(
        "Signal catalog source=static count=%d after_api_failure=%s cache=miss",
        len(MACRO_SIGNALS),
        api_error_kind or "none",
    )

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


def build_microstructure_feature_list(asset: str) -> list[str]:
    """Layer 1 feature list: price signal + microstructure signals."""
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
    """Infer a coarse family label from a raw feature name."""
    lowered = name.lower()

    if lowered in {
        "sp500",
        "nasdaq",
        "vix_close",
        "fear_greed",
        "dxy",
        "gold",
        "oil_wti",
        "russell2000",
        "tsy_yield_10y",
        "tsy_yield_2y",
    }:
        return "macro"
    if lowered.startswith(("book_", "spread_", "trade_flow_", "vpin_", "large_trade_")):
        return "microstructure"
    if lowered.startswith(
        ("funding_", "oi_", "liq_", "ls_", "iv_", "put_call_", "max_pain_", "gamma_")
    ):
        return "derivatives"
    if lowered.startswith(("btc_mempool_", "btc_hashrate", "btc_active_", "btc_difficulty")):
        return "onchain"
    if lowered.startswith(("earthquake_", "sunspot_", "enso_", "github_", "steam_")):
        return "alt"
    if lowered.startswith(("gdelt_", "eonet_", "fr24_", "glad_", "nasa_")):
        return "event"
    if lowered.endswith(("_ohlcv", "_usdt", "_btc")) or lowered in {
        "nvda",
        "aapl",
        "msft",
        "googl",
        "amzn",
        "meta",
        "tsla",
        "amd",
    }:
        return "price"
    return "other"


def stratified_feature_subset(
    features: list[str],
    *,
    k: int,
    seed: int | None = None,
) -> frozenset[str]:
    """Sample a feature subset while spreading picks across coarse families."""
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
        added_this_round = False
        for family in family_order:
            bucket = grouped[family]
            if not bucket:
                continue
            selected.append(bucket.pop())
            added_this_round = True
            if len(selected) >= min(k, len(features)):
                break
        if not added_this_round:
            break

    return frozenset(selected)


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
