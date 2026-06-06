from __future__ import annotations

import argparse
import csv
from pathlib import Path

from strategies.cross_asset_rotation.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR
from strategies.crypto.data import LOCAL_DATASET_DIR as CRYPTO_DATASET_DIR
from strategies.equity_index.data import LOCAL_DATASET_DIR as EQUITY_DATASET_DIR


SYMBOL_SOURCE_DIRS = {
    "BTCUSDT": CRYPTO_DATASET_DIR,
    "ETHUSDT": CRYPTO_DATASET_DIR,
    "SPY": EQUITY_DATASET_DIR,
    "QQQ": EQUITY_DATASET_DIR,
    "GLD": EQUITY_DATASET_DIR,
    "TLT": EQUITY_DATASET_DIR,
}


def prepare_cross_asset_market_data(
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    output_dir: Path = LOCAL_DATASET_DIR,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_symbol = {
        symbol: _read_normalized_daily_close(
            SYMBOL_SOURCE_DIRS[symbol] / f"{symbol}.csv"
        )
        for symbol in symbols
    }
    common_dates = sorted(
        set.intersection(
            *(set(rows_by_date) for rows_by_date in rows_by_symbol.values())
        )
    )
    written_paths = []
    for symbol in symbols:
        written_paths.append(_write_daily_close(symbol, rows_by_symbol[symbol], common_dates, output_dir))
    return tuple(written_paths)


def _read_normalized_daily_close(source_path: Path) -> dict[str, float]:
    with source_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return {
            str(row["timestamp"])[:10]: float(row["close"])
            for row in reader
            if row.get("close") not in (None, "")
        }


def _write_daily_close(
    symbol: str,
    rows_by_date: dict[str, float],
    common_dates: list[str],
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{symbol}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("timestamp", "close"))
        for timestamp in common_dates:
            writer.writerow((timestamp, rows_by_date[timestamp]))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DATASET_DIR)
    args = parser.parse_args()

    for path in prepare_cross_asset_market_data(
        symbols=tuple(args.symbols),
        output_dir=args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
