from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


LOCAL_DATASET_DIR = (
    Path(__file__).resolve().parent / "market_data" / "binance_um_futures"
)
DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
)


@dataclass(frozen=True)
class MarketStructureDay:
    timestamp: str
    close: float
    volume: float
    taker_buy_volume: float
    funding_rate_sum: float
    funding_rate_mean: float
    premium_close: float


def load_market_structure_days(
    *,
    dataset_dir: Path = LOCAL_DATASET_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> dict[str, tuple[MarketStructureDay, ...]]:
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]] = {}
    for symbol in symbols:
        path = dataset_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows_by_symbol[symbol] = tuple(
                MarketStructureDay(
                    timestamp=str(row["timestamp"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    taker_buy_volume=float(row["taker_buy_volume"]),
                    funding_rate_sum=float(row["funding_rate_sum"]),
                    funding_rate_mean=float(row["funding_rate_mean"]),
                    premium_close=float(row["premium_close"]),
                )
                for row in reader
            )
    return rows_by_symbol
