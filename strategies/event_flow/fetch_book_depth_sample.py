from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests


BINANCE_UM_DAILY_BOOK_DEPTH_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth"
BINANCE_UM_DAILY_BASE_URL = "https://data.binance.vision/data/futures/um/daily"
DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "NEARUSDT",
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "market_data" / "binance_um_book_depth_liquidity"


@dataclass(frozen=True)
class BookDepthLiquiditySnapshot:
    timestamp: str
    symbol: str
    bid_notional_1pct: float
    ask_notional_1pct: float
    bid_notional_5pct: float
    ask_notional_5pct: float
    imbalance_1pct: float
    imbalance_5pct: float
    premium_index_1m: float
    mark_index_basis_1m: float
    open_interest_value_5m: float
    top_trader_long_short_ratio_5m: float
    account_long_short_ratio_5m: float
    taker_long_short_volume_ratio_5m: float
    close_1m: float
    next_1m_return: float


def fetch_book_depth_sample(
    *,
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = 8,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    days = _days(start_date, end_date)
    for symbol in symbols:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            daily_snapshots = tuple(
                executor.map(
                    lambda day: _fetch_daily_book_depth_liquidity(symbol, day),
                    days,
                )
            )
        snapshots = tuple(snapshot for rows in daily_snapshots for snapshot in rows)
        path = output_dir / f"{symbol}.csv"
        _write_snapshots(snapshots, output_path=path)
        paths.append(path)
    return tuple(paths)


def _fetch_daily_book_depth_liquidity(symbol: str, day: date) -> tuple[BookDepthLiquiditySnapshot, ...]:
    depth_rows = _fetch_depth_rows(symbol, day)
    if not depth_rows:
        return ()
    close_by_minute = _fetch_1m_closes(symbol, day, data_type="klines")
    premium_by_minute = _fetch_1m_closes(symbol, day, data_type="premiumIndexKlines")
    mark_by_minute = _fetch_1m_closes(symbol, day, data_type="markPriceKlines")
    index_by_minute = _fetch_1m_closes(symbol, day, data_type="indexPriceKlines")
    metrics_by_bucket = _fetch_5m_metrics(symbol, day)
    grouped: dict[str, dict[int, float]] = {}
    for row in depth_rows:
        grouped.setdefault(row["timestamp"], {})[int(float(row["percentage"]))] = float(row["notional"])
    snapshots = []
    for timestamp, buckets in sorted(grouped.items()):
        minute_ms = _timestamp_to_minute_ms(timestamp)
        close = close_by_minute.get(minute_ms)
        next_close = close_by_minute.get(minute_ms + 60_000)
        if close is None or next_close is None or close <= 0.0:
            continue
        bid_1 = buckets.get(-1, 0.0)
        ask_1 = buckets.get(1, 0.0)
        bid_5 = sum(notional for pct, notional in buckets.items() if pct < 0)
        ask_5 = sum(notional for pct, notional in buckets.items() if pct > 0)
        index_price = index_by_minute.get(minute_ms, 0.0)
        mark_price = mark_by_minute.get(minute_ms, 0.0)
        metrics = metrics_by_bucket.get(_five_minute_bucket_ms(minute_ms), {})
        snapshots.append(
            BookDepthLiquiditySnapshot(
                timestamp=_timestamp_to_iso(timestamp),
                symbol=symbol,
                bid_notional_1pct=bid_1,
                ask_notional_1pct=ask_1,
                bid_notional_5pct=bid_5,
                ask_notional_5pct=ask_5,
                imbalance_1pct=_imbalance(bid_1, ask_1),
                imbalance_5pct=_imbalance(bid_5, ask_5),
                premium_index_1m=premium_by_minute.get(minute_ms, 0.0),
                mark_index_basis_1m=(mark_price / index_price) - 1.0 if index_price > 0.0 else 0.0,
                open_interest_value_5m=metrics.get("sum_open_interest_value", 0.0),
                top_trader_long_short_ratio_5m=metrics.get("sum_toptrader_long_short_ratio", 0.0),
                account_long_short_ratio_5m=metrics.get("count_long_short_ratio", 0.0),
                taker_long_short_volume_ratio_5m=metrics.get("sum_taker_long_short_vol_ratio", 0.0),
                close_1m=close,
                next_1m_return=(next_close / close) - 1.0,
            )
        )
    return tuple(snapshots)


def _fetch_depth_rows(symbol: str, day: date) -> tuple[dict[str, str], ...]:
    url = f"{BINANCE_UM_DAILY_BOOK_DEPTH_URL}/{symbol}/{symbol}-bookDepth-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            return tuple(csv.DictReader(TextIOWrapper(handle, encoding="utf-8")))


def _fetch_1m_closes(symbol: str, day: date, *, data_type: str) -> dict[int, float]:
    url = f"{BINANCE_UM_DAILY_BASE_URL}/{data_type}/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    closes: dict[int, float] = {}
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            for row in csv.DictReader(TextIOWrapper(handle, encoding="utf-8")):
                closes[int(row["open_time"])] = float(row["close"])
    return closes


def _fetch_5m_metrics(symbol: str, day: date) -> dict[int, dict[str, float]]:
    url = f"{BINANCE_UM_DAILY_BASE_URL}/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    metrics: dict[int, dict[str, float]] = {}
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            for row in csv.DictReader(TextIOWrapper(handle, encoding="utf-8")):
                bucket_ms = _timestamp_to_minute_ms(row["create_time"])
                metrics[bucket_ms] = {
                    "sum_open_interest_value": _float(row.get("sum_open_interest_value")),
                    "sum_toptrader_long_short_ratio": _float(row.get("sum_toptrader_long_short_ratio")),
                    "count_long_short_ratio": _float(row.get("count_long_short_ratio")),
                    "sum_taker_long_short_vol_ratio": _float(row.get("sum_taker_long_short_vol_ratio")),
                }
    return metrics


def _write_snapshots(snapshots: tuple[BookDepthLiquiditySnapshot, ...], *, output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "bid_notional_1pct",
                "ask_notional_1pct",
                "bid_notional_5pct",
                "ask_notional_5pct",
                "imbalance_1pct",
                "imbalance_5pct",
                "premium_index_1m",
                "mark_index_basis_1m",
                "open_interest_value_5m",
                "top_trader_long_short_ratio_5m",
                "account_long_short_ratio_5m",
                "taker_long_short_volume_ratio_5m",
                "close_1m",
                "next_1m_return",
            )
        )
        for snapshot in snapshots:
            writer.writerow(
                (
                    snapshot.timestamp,
                    snapshot.symbol,
                    f"{snapshot.bid_notional_1pct:.8f}",
                    f"{snapshot.ask_notional_1pct:.8f}",
                    f"{snapshot.bid_notional_5pct:.8f}",
                    f"{snapshot.ask_notional_5pct:.8f}",
                    f"{snapshot.imbalance_1pct:.10f}",
                    f"{snapshot.imbalance_5pct:.10f}",
                    f"{snapshot.premium_index_1m:.10f}",
                    f"{snapshot.mark_index_basis_1m:.10f}",
                    f"{snapshot.open_interest_value_5m:.8f}",
                    f"{snapshot.top_trader_long_short_ratio_5m:.10f}",
                    f"{snapshot.account_long_short_ratio_5m:.10f}",
                    f"{snapshot.taker_long_short_volume_ratio_5m:.10f}",
                    f"{snapshot.close_1m:.12f}",
                    f"{snapshot.next_1m_return:.10f}",
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


def _timestamp_to_minute_ms(value: str) -> int:
    timestamp = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    minute = timestamp.replace(second=0, microsecond=0)
    return int(minute.timestamp() * 1000)


def _five_minute_bucket_ms(value: int) -> int:
    return value - (value % (5 * 60 * 1000))


def _timestamp_to_iso(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).isoformat()


def _imbalance(bid_notional: float, ask_notional: float) -> float:
    denominator = bid_notional + ask_notional
    if denominator <= 0.0:
        return 0.0
    return (bid_notional - ask_notional) / denominator


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2026, 6, 1))
    parser.add_argument("--end-date", type=_parse_date, default=date(2026, 6, 8))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    for path in fetch_book_depth_sample(
        symbols=tuple(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    ):
        print(path)


if __name__ == "__main__":
    main()
