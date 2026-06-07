from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import requests


BINANCE_UM_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


@dataclass(frozen=True)
class MetricsRow:
    timestamp: str
    symbol: str
    sum_open_interest: float
    sum_open_interest_value: float
    count_top_long_short_ratio: float
    sum_top_long_short_ratio: float
    count_long_short_ratio: float
    sum_taker_long_short_vol_ratio: float


@dataclass(frozen=True)
class PremiumMinuteRow:
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class DerivativesDaySummary:
    date: str
    symbol: str
    metrics_rows: int
    premium_rows: int
    close: float
    oi_value_change: float
    next_return: float
    mean_open_interest_value: float
    mean_sum_taker_long_short_vol_ratio: float
    mean_premium_close: float
    max_abs_premium_close: float


@dataclass(frozen=True)
class SignalSummary:
    feature: str
    observations: int
    correlation_to_next_return: float
    low_bucket_mean_next_return: float
    low_bucket_hit_rate: float
    high_bucket_mean_next_return: float
    high_bucket_hit_rate: float


def inspect_binance_derivatives_history(
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start_date: date,
    days: int,
) -> tuple[DerivativesDaySummary, ...]:
    partial_rows: list[dict[str, object]] = []
    for day_offset in range(days):
        day = start_date + timedelta(days=day_offset)
        for symbol in symbols:
            metrics_rows = _fetch_metrics_rows(symbol, day)
            premium_rows = _fetch_premium_rows(symbol, day)
            partial_rows.append(
                {
                    "date": day.isoformat(),
                    "symbol": symbol,
                    "metrics_rows": len(metrics_rows),
                    "premium_rows": len(premium_rows),
                    "close": _fetch_daily_close(symbol, day),
                    "mean_open_interest_value": _mean(
                        row.sum_open_interest_value for row in metrics_rows
                    ),
                    "mean_sum_taker_long_short_vol_ratio": _mean(
                        row.sum_taker_long_short_vol_ratio for row in metrics_rows
                    ),
                    "mean_premium_close": _mean(row.close for row in premium_rows),
                    "max_abs_premium_close": max(
                        (abs(row.close) for row in premium_rows),
                        default=0.0,
                    ),
                }
            )
    return _build_labeled_summaries(tuple(partial_rows))


def write_derivatives_history_summaries(
    summaries: tuple[DerivativesDaySummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "date",
                "symbol",
                "metrics_rows",
                "premium_rows",
                "close",
                "oi_value_change",
                "next_return",
                "mean_open_interest_value",
                "mean_sum_taker_long_short_vol_ratio",
                "mean_premium_close",
                "max_abs_premium_close",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.date,
                    summary.symbol,
                    summary.metrics_rows,
                    summary.premium_rows,
                    f"{summary.close:.8f}",
                    f"{summary.oi_value_change:.8f}",
                    f"{summary.next_return:.12f}",
                    f"{summary.mean_open_interest_value:.8f}",
                    f"{summary.mean_sum_taker_long_short_vol_ratio:.8f}",
                    f"{summary.mean_premium_close:.12f}",
                    f"{summary.max_abs_premium_close:.12f}",
                )
            )
    return output_path


def summarize_derivatives_signals(
    summaries: tuple[DerivativesDaySummary, ...],
) -> tuple[SignalSummary, ...]:
    labeled_rows = tuple(summary for summary in summaries if summary.next_return != 0.0)
    return tuple(
        _signal_summary(labeled_rows, feature=feature)
        for feature in (
            "mean_premium_close",
            "max_abs_premium_close",
            "oi_value_change",
            "mean_sum_taker_long_short_vol_ratio",
        )
    )


def write_signal_summaries(
    summaries: tuple[SignalSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "feature",
                "observations",
                "correlation_to_next_return",
                "low_bucket_mean_next_return",
                "low_bucket_hit_rate",
                "high_bucket_mean_next_return",
                "high_bucket_hit_rate",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.feature,
                    summary.observations,
                    f"{summary.correlation_to_next_return:.8f}",
                    f"{summary.low_bucket_mean_next_return:.8f}",
                    f"{summary.low_bucket_hit_rate:.8f}",
                    f"{summary.high_bucket_mean_next_return:.8f}",
                    f"{summary.high_bucket_hit_rate:.8f}",
                )
            )
    return output_path


def write_schema_sample(
    *,
    symbol: str,
    day: date,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = _download_zip_csv(_metrics_url(symbol, day))
    premium_payload = _download_zip_csv(_premium_url(symbol, day))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives History Schema\n\n")
        handle.write(f"Symbol: `{symbol}`\n\n")
        handle.write(f"Date: `{day.isoformat()}`\n\n")
        handle.write("## Metrics\n\n")
        handle.write(f"- URL: `{_metrics_url(symbol, day)}`\n")
        handle.write(f"- Header: `{metrics_payload[0] if metrics_payload else ()}`\n")
        handle.write(f"- First row: `{metrics_payload[1] if len(metrics_payload) > 1 else ()}`\n\n")
        handle.write("## Premium Index Klines\n\n")
        handle.write(f"- URL: `{_premium_url(symbol, day)}`\n")
        handle.write(f"- Header: `{premium_payload[0] if premium_payload else ()}`\n")
        handle.write(f"- First row: `{premium_payload[1] if len(premium_payload) > 1 else ()}`\n")
    return output_path


def _fetch_metrics_rows(symbol: str, day: date) -> tuple[MetricsRow, ...]:
    rows = []
    for item in _download_zip_csv(_metrics_url(symbol, day)):
        if not item or item[0] == "create_time":
            continue
        rows.append(
            MetricsRow(
                timestamp=_parse_binance_time(item[0]),
                symbol=str(item[1]),
                sum_open_interest=float(item[2]),
                sum_open_interest_value=float(item[3]),
                count_top_long_short_ratio=float(item[4]),
                sum_top_long_short_ratio=float(item[5]),
                count_long_short_ratio=float(item[6]),
                sum_taker_long_short_vol_ratio=float(item[7]),
            )
        )
    return tuple(rows)


def _fetch_premium_rows(symbol: str, day: date) -> tuple[PremiumMinuteRow, ...]:
    rows = []
    for item in _download_zip_csv(_premium_url(symbol, day)):
        if not item or item[0] == "open_time":
            continue
        rows.append(
            PremiumMinuteRow(
                timestamp=_ms_to_timestamp(int(item[0])),
                symbol=symbol,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
            )
        )
    return tuple(rows)


def _fetch_daily_close(symbol: str, day: date) -> float:
    payload = _download_zip_csv(_daily_kline_url(symbol, day))
    for item in payload:
        if not item or item[0] == "open_time":
            continue
        return float(item[4])
    return 0.0


def _build_labeled_summaries(
    partial_rows: tuple[dict[str, object], ...],
) -> tuple[DerivativesDaySummary, ...]:
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in partial_rows:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)

    summaries: list[DerivativesDaySummary] = []
    for symbol_rows in by_symbol.values():
        ordered_rows = sorted(symbol_rows, key=lambda row: str(row["date"]))
        previous_oi_value = 0.0
        for index, row in enumerate(ordered_rows):
            close = float(row["close"])
            next_close = (
                float(ordered_rows[index + 1]["close"])
                if index + 1 < len(ordered_rows)
                else 0.0
            )
            mean_oi_value = float(row["mean_open_interest_value"])
            summaries.append(
                DerivativesDaySummary(
                    date=str(row["date"]),
                    symbol=str(row["symbol"]),
                    metrics_rows=int(row["metrics_rows"]),
                    premium_rows=int(row["premium_rows"]),
                    close=close,
                    oi_value_change=(
                        (mean_oi_value / previous_oi_value) - 1.0
                        if previous_oi_value > 0.0
                        else 0.0
                    ),
                    next_return=(
                        (next_close / close) - 1.0
                        if close > 0.0 and next_close > 0.0
                        else 0.0
                    ),
                    mean_open_interest_value=mean_oi_value,
                    mean_sum_taker_long_short_vol_ratio=float(
                        row["mean_sum_taker_long_short_vol_ratio"]
                    ),
                    mean_premium_close=float(row["mean_premium_close"]),
                    max_abs_premium_close=float(row["max_abs_premium_close"]),
                )
            )
            previous_oi_value = mean_oi_value
    return tuple(sorted(summaries, key=lambda summary: (summary.date, summary.symbol)))


def _signal_summary(
    summaries: tuple[DerivativesDaySummary, ...],
    *,
    feature: str,
) -> SignalSummary:
    values = tuple(float(getattr(summary, feature)) for summary in summaries)
    next_returns = tuple(summary.next_return for summary in summaries)
    if not values:
        return SignalSummary(feature, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sorted_values = sorted(values)
    low_threshold = sorted_values[int(len(sorted_values) * 0.25)]
    high_threshold = sorted_values[int(len(sorted_values) * 0.75)]
    low_returns = tuple(
        summary.next_return
        for summary in summaries
        if float(getattr(summary, feature)) <= low_threshold
    )
    high_returns = tuple(
        summary.next_return
        for summary in summaries
        if float(getattr(summary, feature)) >= high_threshold
    )
    return SignalSummary(
        feature=feature,
        observations=len(summaries),
        correlation_to_next_return=_correlation(values, next_returns),
        low_bucket_mean_next_return=_mean(low_returns),
        low_bucket_hit_rate=_hit_rate(low_returns),
        high_bucket_mean_next_return=_mean(high_returns),
        high_bucket_hit_rate=_hit_rate(high_returns),
    )


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = (
        sum((left_value - left_mean) ** 2 for left_value in left)
        * sum((right_value - right_mean) ** 2 for right_value in right)
    ) ** 0.5
    return numerator / denominator if denominator > 0.0 else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return mean(1.0 if value > 0.0 else 0.0 for value in values) if values else 0.0


def _download_zip_csv(url: str) -> tuple[list[str], ...]:
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            reader = csv.reader(TextIOWrapper(handle, encoding="utf-8"))
            return tuple(list(row) for row in reader)


def _metrics_url(symbol: str, day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"


def _premium_url(symbol: str, day: date) -> str:
    return (
        f"{BINANCE_UM_DAILY_URL}/premiumIndexKlines/{symbol}/1m/"
        f"{symbol}-1m-{day:%Y-%m-%d}.zip"
    )


def _daily_kline_url(symbol: str, day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/klines/{symbol}/1d/{symbol}-1d-{day:%Y-%m-%d}.zip"


def _mean(values: object) -> float:
    items = tuple(values)
    return mean(items) if items else 0.0


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _parse_binance_time(value: str) -> str:
    if value.isdigit():
        return _ms_to_timestamp(int(value))
    return datetime.fromisoformat(value).replace(tzinfo=UTC).isoformat()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2024, 1, 1))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "binance_derivatives_history.csv",
    )
    parser.add_argument(
        "--schema-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "binance_derivatives_schema.md",
    )
    parser.add_argument(
        "--signal-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "binance_derivatives_signal_summary.csv",
    )
    args = parser.parse_args()

    symbols = tuple(args.symbols)
    summaries = inspect_binance_derivatives_history(
        symbols=symbols,
        start_date=args.start_date,
        days=args.days,
    )
    write_derivatives_history_summaries(summaries, output_path=args.output_path)
    signal_summaries = summarize_derivatives_signals(summaries)
    write_signal_summaries(signal_summaries, output_path=args.signal_output_path)
    write_schema_sample(
        symbol=symbols[0],
        day=args.start_date,
        output_path=args.schema_output_path,
    )
    for summary in summaries:
        print(
            summary.date,
            summary.symbol,
            f"metrics={summary.metrics_rows}",
            f"premium={summary.premium_rows}",
            f"next={summary.next_return:.6f}",
            f"oi_change={summary.oi_value_change:.6f}",
            f"oi_value={summary.mean_open_interest_value:.2f}",
            f"premium={summary.mean_premium_close:.8f}",
        )
    for signal_summary in signal_summaries:
        print(
            signal_summary.feature,
            f"corr={signal_summary.correlation_to_next_return:.6f}",
            f"low={signal_summary.low_bucket_mean_next_return:.6f}",
            f"high={signal_summary.high_bucket_mean_next_return:.6f}",
        )


if __name__ == "__main__":
    main()
