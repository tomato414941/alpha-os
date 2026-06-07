from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests

from strategies.crypto.data import EXPANDED_SYMBOLS, LOCAL_DATASET_DIR


BINANCE_PUBLIC_DATA_URL = "https://data.binance.vision/data/spot/monthly"


@dataclass(frozen=True)
class BinanceDailyRow:
    timestamp: str
    close: float
    volume: float


def fetch_binance_spot_daily_rows(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
) -> tuple[BinanceDailyRow, ...]:
    rows: list[BinanceDailyRow] = []
    for month in _months(start_date, end_date):
        payload = _download_zip_csv(
            _archive_url(
                symbol,
                month,
                filename=f"{symbol}-1d-{month:%Y-%m}.zip",
            )
        )
        for item in payload:
            if item and item[0] == "open_time":
                continue
            open_time_ms = _timestamp_to_ms(int(item[0]))
            row_date = _timestamp_to_date(_ms_to_timestamp(open_time_ms))
            if not start_date <= row_date < end_date:
                continue
            rows.append(
                BinanceDailyRow(
                    timestamp=_ms_to_timestamp(open_time_ms),
                    close=float(item[4]),
                    volume=float(item[5]),
                )
            )

    return tuple(rows)


def write_daily_rows(
    symbol: str,
    rows: tuple[BinanceDailyRow, ...],
    *,
    output_dir: Path = LOCAL_DATASET_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "close", "volume", "funding_rate", "open_interest"))
        for row in rows:
            writer.writerow((row.timestamp, row.close, row.volume, "", ""))
    return path


def fetch_market_data(
    *,
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
    output_dir: Path = LOCAL_DATASET_DIR,
) -> tuple[Path, ...]:
    with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as executor:
        paths = executor.map(
            lambda symbol: _fetch_and_write_symbol(
                symbol,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
            ),
            symbols,
        )
    return tuple(path for path in paths if path is not None)


def _fetch_and_write_symbol(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    output_dir: Path,
) -> Path | None:
    rows = fetch_binance_spot_daily_rows(
        symbol,
        start_date=start_date,
        end_date=end_date,
    )
    if not rows:
        return None
    return write_daily_rows(symbol, rows, output_dir=output_dir)


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _timestamp_to_ms(value: int) -> int:
    if value > 10_000_000_000_000:
        return value // 1000
    return value


def _timestamp_to_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _months(start_date: date, end_date: date):
    cursor = date(start_date.year, start_date.month, 1)
    while cursor < end_date:
        yield cursor
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def _archive_url(symbol: str, month: date, *, filename: str) -> str:
    return f"{BINANCE_PUBLIC_DATA_URL}/klines/{symbol}/1d/{filename}"


def _download_zip_csv(url: str) -> tuple[list[str], ...]:
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as raw_handle:
            reader = csv.reader(TextIOWrapper(raw_handle, encoding="utf-8"))
            return tuple(reader)


def _default_end_date() -> date:
    return datetime.now(UTC).date()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(EXPANDED_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2024, 1, 1))
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=_default_end_date(),
        help="Exclusive UTC date. The default excludes today's still-open daily bar.",
    )
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DATASET_DIR)
    args = parser.parse_args()

    for path in fetch_market_data(
        symbols=tuple(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
