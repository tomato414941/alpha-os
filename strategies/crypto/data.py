from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "datasets"
    / "ds_crypto_btc_eth_daily_2024_2025"
)
LOCAL_DATASET_DIR = (
    Path(__file__).resolve().parent / "market_data" / "binance_spot_daily"
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
EXPANDED_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "FILUSDT",
    "INJUSDT",
    "ETCUSDT",
    "ATOMUSDT",
    "AAVEUSDT",
    "RUNEUSDT",
    "SEIUSDT",
    "WIFUSDT",
    "PEPEUSDT",
    "HBARUSDT",
    "FETUSDT",
)


@dataclass(frozen=True)
class DailyClose:
    timestamp: str
    close: float


@dataclass(frozen=True)
class DailyMarketBar:
    timestamp: str
    closes: dict[str, float]


def load_daily_market_bars(
    *,
    dataset_dir: Path = DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> tuple[DailyMarketBar, ...]:
    closes_by_symbol: dict[str, list[DailyClose]] = {}
    for symbol in symbols:
        path = dataset_dir / f"{symbol}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            closes_by_symbol[symbol] = [
                DailyClose(
                    timestamp=str(row["timestamp"]),
                    close=float(row["close"]),
                )
                for row in reader
            ]
    return align_daily_closes(closes_by_symbol)


def align_daily_closes(
    closes_by_symbol: dict[str, list[DailyClose]],
) -> tuple[DailyMarketBar, ...]:
    symbol_closes_by_timestamp = {
        symbol: {row.timestamp: row.close for row in closes}
        for symbol, closes in closes_by_symbol.items()
    }
    timestamps = sorted(
        {
            timestamp
            for closes_by_timestamp in symbol_closes_by_timestamp.values()
            for timestamp in closes_by_timestamp
        }
    )
    return tuple(
        DailyMarketBar(
            timestamp=timestamp,
            closes={
                symbol: closes_by_timestamp[timestamp]
                for symbol, closes_by_timestamp in symbol_closes_by_timestamp.items()
                if timestamp in closes_by_timestamp
            },
        )
        for timestamp in timestamps
    )
