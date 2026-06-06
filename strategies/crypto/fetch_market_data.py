from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from strategies.crypto.data import EXPANDED_SYMBOLS, LOCAL_DATASET_DIR


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
ONE_DAY_MS = 24 * 60 * 60 * 1000


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
    url: str = BINANCE_KLINES_URL,
) -> tuple[BinanceDailyRow, ...]:
    rows: list[BinanceDailyRow] = []
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)

    while start_ms < end_ms:
        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break

        for item in payload:
            open_time_ms = int(item[0])
            if open_time_ms >= end_ms:
                continue
            rows.append(
                BinanceDailyRow(
                    timestamp=_ms_to_timestamp(open_time_ms),
                    close=float(item[4]),
                    volume=float(item[5]),
                )
            )

        next_start_ms = int(payload[-1][0]) + ONE_DAY_MS
        if next_start_ms <= start_ms:
            break
        start_ms = next_start_ms

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
    written_paths = []
    for symbol in symbols:
        rows = fetch_binance_spot_daily_rows(
            symbol,
            start_date=start_date,
            end_date=end_date,
        )
        written_paths.append(write_daily_rows(symbol, rows, output_dir=output_dir))
    return tuple(written_paths)


def _date_to_ms(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


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
