from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests


BINANCE_UM_DAILY_AGGTRADES_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "market_data" / "binance_um_aggtrades_5m"


@dataclass(frozen=True)
class EventFlowBar:
    timestamp: str
    symbol: str
    close: float
    volume: float
    taker_buy_volume: float
    taker_sell_volume: float
    trade_count: int


def fetch_event_flow_sample(
    *,
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for symbol in symbols:
        bars: list[EventFlowBar] = []
        for day in _days(start_date, end_date):
            bars.extend(_fetch_daily_aggtrade_bars(symbol, day))
        path = output_dir / f"{symbol}.csv"
        _write_bars(tuple(bars), output_path=path)
        paths.append(path)
    return tuple(paths)


def _fetch_daily_aggtrade_bars(symbol: str, day: date) -> tuple[EventFlowBar, ...]:
    url = f"{BINANCE_UM_DAILY_AGGTRADES_URL}/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    bars: dict[int, dict[str, float]] = {}
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            reader = csv.DictReader(TextIOWrapper(handle, encoding="utf-8"))
            for row in reader:
                timestamp_ms = int(row["transact_time"])
                bucket_ms = timestamp_ms - (timestamp_ms % (5 * 60 * 1000))
                quantity = float(row["quantity"])
                price = float(row["price"])
                bar = bars.setdefault(
                    bucket_ms,
                    {
                        "close": price,
                        "volume": 0.0,
                        "taker_buy_volume": 0.0,
                        "taker_sell_volume": 0.0,
                        "trade_count": 0.0,
                    },
                )
                bar["close"] = price
                bar["volume"] += quantity
                if str(row["is_buyer_maker"]).lower() == "true":
                    bar["taker_sell_volume"] += quantity
                else:
                    bar["taker_buy_volume"] += quantity
                bar["trade_count"] += 1
    return tuple(
        EventFlowBar(
            timestamp=_ms_to_timestamp(bucket_ms),
            symbol=symbol,
            close=values["close"],
            volume=values["volume"],
            taker_buy_volume=values["taker_buy_volume"],
            taker_sell_volume=values["taker_sell_volume"],
            trade_count=int(values["trade_count"]),
        )
        for bucket_ms, values in sorted(bars.items())
    )


def _write_bars(bars: tuple[EventFlowBar, ...], *, output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "close",
                "volume",
                "taker_buy_volume",
                "taker_sell_volume",
                "trade_count",
            )
        )
        for bar in bars:
            writer.writerow(
                (
                    bar.timestamp,
                    bar.symbol,
                    f"{bar.close:.12f}",
                    f"{bar.volume:.12f}",
                    f"{bar.taker_buy_volume:.12f}",
                    f"{bar.taker_sell_volume:.12f}",
                    bar.trade_count,
                )
            )
    return output_path


def _days(start_date: date, end_date: date) -> tuple[date, ...]:
    days = []
    current = start_date
    while current < end_date:
        days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=date(2024, 1, 4))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for path in fetch_event_flow_sample(
        symbols=tuple(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
