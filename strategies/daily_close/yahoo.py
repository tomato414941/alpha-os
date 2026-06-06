from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests


YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class YahooDailyRow:
    timestamp: str
    close: float


def fetch_yahoo_daily_rows(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
) -> tuple[YahooDailyRow, ...]:
    response = requests.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={
            "period1": _date_to_seconds(start_date),
            "period2": _date_to_seconds(end_date),
            "interval": "1d",
            "events": "history",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    timestamps = payload["timestamp"]
    closes = payload["indicators"]["quote"][0]["close"]
    rows = []
    for timestamp, close in zip(timestamps, closes, strict=True):
        if close is None:
            continue
        rows.append(
            YahooDailyRow(
                timestamp=datetime.fromtimestamp(timestamp, tz=UTC).isoformat(),
                close=float(close),
            )
        )
    return tuple(rows)


def write_daily_rows(
    symbol: str,
    rows: tuple[YahooDailyRow, ...],
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "close"))
        for row in rows:
            writer.writerow((row.timestamp, row.close))
    return path


def _date_to_seconds(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())
