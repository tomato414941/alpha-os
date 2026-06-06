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


def load_daily_closes(
    *,
    dataset_dir: Path = DATASET_DIR,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> dict[str, list[DailyClose]]:
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
    return closes_by_symbol
