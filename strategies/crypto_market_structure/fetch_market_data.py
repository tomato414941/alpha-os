from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests

from strategies.crypto_market_structure.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR


BINANCE_PUBLIC_DATA_URL = "https://data.binance.vision/data/futures/um/monthly"


@dataclass(frozen=True)
class FuturesKlineRow:
    timestamp: str
    close: float
    volume: float
    taker_buy_volume: float


@dataclass(frozen=True)
class FundingDay:
    funding_rate_sum: float
    funding_rate_mean: float


@dataclass(frozen=True)
class PremiumIndexDay:
    premium_close: float


def fetch_market_data(
    *,
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
    output_dir: Path = LOCAL_DATASET_DIR,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    months = tuple(_months(start_date, end_date))
    with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as executor:
        paths = executor.map(
            lambda symbol: _fetch_and_write_symbol(
                symbol,
                months=months,
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
    months: tuple[date, ...],
    start_date: date,
    end_date: date,
    output_dir: Path,
) -> Path | None:
    klines = _fetch_klines(symbol, months=months)
    if not klines:
        return None
    funding_by_date = _fetch_funding(symbol, months=months)
    premium_by_date = _fetch_premium_index(symbol, months=months)
    rows = [
        row
        for row in klines
        if start_date <= _timestamp_to_date(row.timestamp) < end_date
    ]
    return _write_symbol_rows(
        symbol,
        rows=tuple(rows),
        funding_by_date=funding_by_date,
        premium_by_date=premium_by_date,
        output_dir=output_dir,
    )


def _fetch_klines(
    symbol: str,
    *,
    months: tuple[date, ...],
) -> tuple[FuturesKlineRow, ...]:
    rows: list[FuturesKlineRow] = []
    for month in months:
        payload = _download_zip_csv(
            _archive_url(
                "klines",
                symbol,
                month,
                interval="1d",
                filename=f"{symbol}-1d-{month:%Y-%m}.zip",
            )
        )
        for item in payload:
            if item and item[0] == "open_time":
                continue
            rows.append(
                FuturesKlineRow(
                    timestamp=_ms_to_timestamp(int(item[0])),
                    close=float(item[4]),
                    volume=float(item[5]),
                    taker_buy_volume=float(item[9]),
                )
            )
    return tuple(rows)


def _fetch_funding(
    symbol: str,
    *,
    months: tuple[date, ...],
) -> dict[str, FundingDay]:
    rates_by_date: dict[str, list[float]] = defaultdict(list)
    for month in months:
        payload = _download_zip_csv(
            _archive_url(
                "fundingRate",
                symbol,
                month,
                filename=f"{symbol}-fundingRate-{month:%Y-%m}.zip",
            )
        )
        for item in payload:
            if item and item[0] == "calc_time":
                continue
            day = _ms_to_day(int(item[0]))
            rates_by_date[day].append(float(item[2]))
    return {
        day: FundingDay(
            funding_rate_sum=sum(rates),
            funding_rate_mean=sum(rates) / len(rates),
        )
        for day, rates in rates_by_date.items()
        if rates
    }


def _fetch_premium_index(
    symbol: str,
    *,
    months: tuple[date, ...],
) -> dict[str, PremiumIndexDay]:
    rows_by_date: dict[str, PremiumIndexDay] = {}
    for month in months:
        payload = _download_zip_csv(
            _archive_url(
                "premiumIndexKlines",
                symbol,
                month,
                interval="1d",
                filename=f"{symbol}-1d-{month:%Y-%m}.zip",
            )
        )
        for item in payload:
            if item and item[0] == "open_time":
                continue
            rows_by_date[_ms_to_day(int(item[0]))] = PremiumIndexDay(
                premium_close=float(item[4])
            )
    return rows_by_date


def _write_symbol_rows(
    symbol: str,
    *,
    rows: tuple[FuturesKlineRow, ...],
    funding_by_date: dict[str, FundingDay],
    premium_by_date: dict[str, PremiumIndexDay],
    output_dir: Path,
) -> Path:
    path = output_dir / f"{symbol}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "close",
                "volume",
                "taker_buy_volume",
                "funding_rate_sum",
                "funding_rate_mean",
                "premium_close",
            )
        )
        for row in rows:
            day = row.timestamp[:10]
            funding = funding_by_date.get(day, FundingDay(0.0, 0.0))
            premium = premium_by_date.get(day, PremiumIndexDay(0.0))
            writer.writerow(
                (
                    row.timestamp,
                    row.close,
                    row.volume,
                    row.taker_buy_volume,
                    funding.funding_rate_sum,
                    funding.funding_rate_mean,
                    premium.premium_close,
                )
            )
    return path


def _download_zip_csv(url: str) -> tuple[list[str], ...]:
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as handle:
            reader = csv.reader(TextIOWrapper(handle, encoding="utf-8"))
            return tuple(list(row) for row in reader)


def _archive_url(
    data_type: str,
    symbol: str,
    month: date,
    *,
    filename: str,
    interval: str | None = None,
) -> str:
    parts = [BINANCE_PUBLIC_DATA_URL, data_type, symbol]
    if interval is not None:
        parts.append(interval)
    return "/".join(parts + [filename])


def _months(start_date: date, end_date: date) -> tuple[date, ...]:
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    months = []
    while current <= end_month:
        months.append(current)
        current = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
    return tuple(months)


def _timestamp_to_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _ms_to_day(value: int) -> str:
    return _ms_to_timestamp(value)[:10]


def _default_end_date() -> date:
    return datetime.now(UTC).date()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
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
