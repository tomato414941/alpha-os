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
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
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
    shared_timestamps = set.intersection(
        *(
            set(closes_by_timestamp)
            for closes_by_timestamp in symbol_closes_by_timestamp.values()
        )
    )
    return tuple(
        DailyMarketBar(
            timestamp=timestamp,
            closes={
                symbol: closes_by_timestamp[timestamp]
                for symbol, closes_by_timestamp in symbol_closes_by_timestamp.items()
            },
        )
        for timestamp in sorted(shared_timestamps)
    )
