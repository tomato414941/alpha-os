from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import requests


ROOT = Path(__file__).resolve().parent
BINANCE_UM_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"
DEFAULT_SYMBOLS = (
    "ARBUSDT",
    "NEARUSDT",
    "BCHUSDT",
    "OPUSDT",
    "UNIUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "ADAUSDT",
)
FEATURES = (
    "oi_value_change",
    "sum_top_long_short_ratio",
    "count_top_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
    "premium_close",
    "abs_premium_close",
)


@dataclass(frozen=True)
class IntradayFeatureLabel:
    timestamp: str
    symbol: str
    close: float
    next_1h_return: float
    oi_value_change: float
    sum_top_long_short_ratio: float
    count_top_long_short_ratio: float
    count_long_short_ratio: float
    sum_taker_long_short_vol_ratio: float
    premium_close: float
    abs_premium_close: float


@dataclass(frozen=True)
class IntradayFeatureCandidate:
    symbol: str
    feature: str
    status: str
    preferred_bucket: str
    observations: int
    low_bucket_mean_next_1h_return: float
    low_bucket_hit_rate: float
    high_bucket_mean_next_1h_return: float
    high_bucket_hit_rate: float
    correlation_to_next_1h_return: float
    edge_score: float
    next_step: str


def build_intraday_feature_labels(
    *,
    symbols: tuple[str, ...],
    start_date: date,
    days: int,
    max_workers: int,
) -> tuple[IntradayFeatureLabel, ...]:
    tasks = tuple(
        (symbol, start_date + timedelta(days=offset))
        for offset in range(days)
        for symbol in symbols
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows_by_task = tuple(executor.map(lambda task: _build_day_labels(*task), tasks))
    return tuple(row for rows in rows_by_task for row in rows)


def build_intraday_feature_candidates(
    labels: tuple[IntradayFeatureLabel, ...],
) -> tuple[IntradayFeatureCandidate, ...]:
    by_symbol: dict[str, list[IntradayFeatureLabel]] = {}
    for label in labels:
        if label.next_1h_return != 0.0:
            by_symbol.setdefault(label.symbol, []).append(label)
    candidates = [
        _candidate_for_feature(symbol=symbol, feature=feature, labels=tuple(symbol_labels))
        for symbol, symbol_labels in sorted(by_symbol.items())
        if len(symbol_labels) >= 200
        for feature in FEATURES
    ]
    return tuple(sorted(candidates, key=lambda row: row.edge_score, reverse=True))


def write_intraday_feature_labels_csv(
    labels: tuple[IntradayFeatureLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "close",
                "next_1h_return",
                "oi_value_change",
                "sum_top_long_short_ratio",
                "count_top_long_short_ratio",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
                "premium_close",
                "abs_premium_close",
            )
        )
        for label in labels:
            writer.writerow(
                (
                    label.timestamp,
                    label.symbol,
                    f"{label.close:.12f}",
                    f"{label.next_1h_return:.12f}",
                    f"{label.oi_value_change:.12f}",
                    f"{label.sum_top_long_short_ratio:.8f}",
                    f"{label.count_top_long_short_ratio:.8f}",
                    f"{label.count_long_short_ratio:.8f}",
                    f"{label.sum_taker_long_short_vol_ratio:.8f}",
                    f"{label.premium_close:.12f}",
                    f"{label.abs_premium_close:.12f}",
                )
            )
    return output_path


def write_intraday_feature_candidates_csv(
    candidates: tuple[IntradayFeatureCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "feature",
                "status",
                "preferred_bucket",
                "observations",
                "low_bucket_mean_next_1h_return",
                "low_bucket_hit_rate",
                "high_bucket_mean_next_1h_return",
                "high_bucket_hit_rate",
                "correlation_to_next_1h_return",
                "edge_score",
                "next_step",
            )
        )
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.symbol,
                    candidate.feature,
                    candidate.status,
                    candidate.preferred_bucket,
                    candidate.observations,
                    f"{candidate.low_bucket_mean_next_1h_return:.12f}",
                    f"{candidate.low_bucket_hit_rate:.8f}",
                    f"{candidate.high_bucket_mean_next_1h_return:.12f}",
                    f"{candidate.high_bucket_hit_rate:.8f}",
                    f"{candidate.correlation_to_next_1h_return:.8f}",
                    f"{candidate.edge_score:.8f}",
                    candidate.next_step,
                )
            )
    return output_path


def write_intraday_feature_candidates_md(
    candidates: tuple[IntradayFeatureCandidate, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives Intraday Feature Labels\n\n")
        handle.write(
            "This screen joins Binance USD-M 5m metrics, 5m premium-index klines, and 5m price klines. "
            "It tests whether current derivatives features separate the next 1h return. "
            "It is a research label screen, not a trade list.\n\n"
        )
        handle.write(
            "| symbol | feature | status | bucket | obs | low mean 1h | low hit | high mean 1h | high hit | corr | score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for candidate in candidates[:top]:
            handle.write(
                f"| {candidate.symbol} | {candidate.feature} | {candidate.status} | "
                f"{candidate.preferred_bucket} | {candidate.observations} | "
                f"{candidate.low_bucket_mean_next_1h_return:.6f} | "
                f"{candidate.low_bucket_hit_rate:.4f} | "
                f"{candidate.high_bucket_mean_next_1h_return:.6f} | "
                f"{candidate.high_bucket_hit_rate:.4f} | "
                f"{candidate.correlation_to_next_1h_return:.4f} | "
                f"{candidate.edge_score:.4f} | {candidate.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High rows are intraday label candidates. They still need fees, spread, fill probability, "
            "funding PnL, and repeat-window checks before promotion.\n"
        )
    return output_path


def _build_day_labels(symbol: str, day: date) -> tuple[IntradayFeatureLabel, ...]:
    metric_rows = _fetch_metrics_rows(symbol, day)
    close_rows = _fetch_5m_close_rows(symbol, day)
    premium_rows = _fetch_5m_premium_rows(symbol, day)
    close_by_ts = {row["timestamp"]: row["close"] for row in close_rows}
    premium_by_ts = {row["timestamp"]: row["premium_close"] for row in premium_rows}
    ordered_metrics = sorted(metric_rows, key=lambda row: row["timestamp"])
    labels: list[IntradayFeatureLabel] = []
    previous_oi_value = 0.0
    for row in ordered_metrics:
        timestamp = row["timestamp"]
        close = close_by_ts.get(timestamp, 0.0)
        next_close = close_by_ts.get(_timestamp_after_5m(timestamp, steps=12), 0.0)
        oi_value = row["sum_open_interest_value"]
        premium_close = premium_by_ts.get(timestamp, 0.0)
        if close <= 0.0 or next_close <= 0.0:
            previous_oi_value = oi_value
            continue
        labels.append(
            IntradayFeatureLabel(
                timestamp=timestamp,
                symbol=symbol,
                close=close,
                next_1h_return=(next_close / close) - 1.0,
                oi_value_change=(oi_value / previous_oi_value) - 1.0 if previous_oi_value > 0.0 else 0.0,
                sum_top_long_short_ratio=row["sum_top_long_short_ratio"],
                count_top_long_short_ratio=row["count_top_long_short_ratio"],
                count_long_short_ratio=row["count_long_short_ratio"],
                sum_taker_long_short_vol_ratio=row["sum_taker_long_short_vol_ratio"],
                premium_close=premium_close,
                abs_premium_close=abs(premium_close),
            )
        )
        previous_oi_value = oi_value
    return tuple(labels)


def _candidate_for_feature(
    *,
    symbol: str,
    feature: str,
    labels: tuple[IntradayFeatureLabel, ...],
) -> IntradayFeatureCandidate:
    values = tuple(float(getattr(label, feature)) for label in labels)
    next_returns = tuple(label.next_1h_return for label in labels)
    sorted_values = sorted(values)
    low_threshold = sorted_values[int(len(sorted_values) * 0.25)]
    high_threshold = sorted_values[int(len(sorted_values) * 0.75)]
    low_returns = tuple(
        label.next_1h_return
        for label in labels
        if float(getattr(label, feature)) <= low_threshold
    )
    high_returns = tuple(
        label.next_1h_return
        for label in labels
        if float(getattr(label, feature)) >= high_threshold
    )
    low_mean = _mean(low_returns)
    high_mean = _mean(high_returns)
    low_hit = _hit_rate(low_returns)
    high_hit = _hit_rate(high_returns)
    correlation = _correlation(values, next_returns)
    preferred_bucket = _preferred_bucket(low_mean=low_mean, low_hit=low_hit, high_mean=high_mean, high_hit=high_hit)
    edge_score = _edge_score(
        low_mean=low_mean,
        low_hit=low_hit,
        high_mean=high_mean,
        high_hit=high_hit,
        correlation=correlation,
        observations=len(labels),
    )
    return IntradayFeatureCandidate(
        symbol=symbol,
        feature=feature,
        status=_status(edge_score=edge_score, observations=len(labels)),
        preferred_bucket=preferred_bucket,
        observations=len(labels),
        low_bucket_mean_next_1h_return=low_mean,
        low_bucket_hit_rate=low_hit,
        high_bucket_mean_next_1h_return=high_mean,
        high_bucket_hit_rate=high_hit,
        correlation_to_next_1h_return=correlation,
        edge_score=edge_score,
        next_step=(
            f"repeat {symbol} {feature} 5m-to-1h label on a fresh window, "
            "then add fees, spread, and fill assumptions"
        ),
    )


def _fetch_metrics_rows(symbol: str, day: date) -> tuple[dict[str, float | str], ...]:
    rows = []
    for item in _download_zip_csv(_metrics_url(symbol, day)):
        if not item or item[0] == "create_time":
            continue
        rows.append(
            {
                "timestamp": _parse_binance_time(item[0]),
                "sum_open_interest_value": float(item[3]),
                "count_top_long_short_ratio": float(item[4]),
                "sum_top_long_short_ratio": float(item[5]),
                "count_long_short_ratio": float(item[6]),
                "sum_taker_long_short_vol_ratio": float(item[7]),
            }
        )
    return tuple(rows)


def _fetch_5m_close_rows(symbol: str, day: date) -> tuple[dict[str, float | str], ...]:
    return tuple(
        {"timestamp": _ms_to_timestamp(int(item[0])), "close": float(item[4])}
        for item in _download_zip_csv(_kline_5m_url(symbol, day))
        if item and item[0] != "open_time"
    )


def _fetch_5m_premium_rows(symbol: str, day: date) -> tuple[dict[str, float | str], ...]:
    return tuple(
        {"timestamp": _ms_to_timestamp(int(item[0])), "premium_close": float(item[4])}
        for item in _download_zip_csv(_premium_5m_url(symbol, day))
        if item and item[0] != "open_time"
    )


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


def _kline_5m_url(symbol: str, day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/klines/{symbol}/5m/{symbol}-5m-{day:%Y-%m-%d}.zip"


def _premium_5m_url(symbol: str, day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/premiumIndexKlines/{symbol}/5m/{symbol}-5m-{day:%Y-%m-%d}.zip"


def _timestamp_after_5m(timestamp: str, *, steps: int) -> str:
    value = datetime.fromisoformat(timestamp)
    return (value + timedelta(minutes=5 * steps)).isoformat()


def _parse_binance_time(value: str) -> str:
    if value.isdigit():
        return _ms_to_timestamp(int(value))
    return datetime.fromisoformat(value).replace(tzinfo=UTC).isoformat()


def _ms_to_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _preferred_bucket(*, low_mean: float, low_hit: float, high_mean: float, high_hit: float) -> str:
    if high_mean > low_mean and high_hit >= low_hit:
        return "high"
    if low_mean > high_mean and low_hit >= high_hit:
        return "low"
    if high_mean > low_mean:
        return "high_mean_only"
    return "low_mean_only"


def _edge_score(
    *,
    low_mean: float,
    low_hit: float,
    high_mean: float,
    high_hit: float,
    correlation: float,
    observations: int,
) -> float:
    mean_component = abs(high_mean - low_mean) * 100_000.0
    hit_component = abs(high_hit - low_hit) * 30.0
    corr_component = abs(correlation) * 20.0
    sample_component = min(observations / 200.0, 15.0)
    return mean_component + hit_component + corr_component + sample_component


def _status(*, edge_score: float, observations: int) -> str:
    if observations < 1_000:
        return "thin_intraday_history"
    if edge_score >= 140.0:
        return "intraday_feature_priority"
    if edge_score >= 80.0:
        return "intraday_feature_watch"
    return "weak_intraday_feature_context"


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


def _mean(values: tuple[float, ...]) -> float:
    return mean(values) if values else 0.0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2026, 5, 20))
    parser.add_argument("--days", type=int, default=18)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--labels-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_labels.csv",
    )
    parser.add_argument(
        "--candidates-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_candidates.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_candidates.md",
    )
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    labels = build_intraday_feature_labels(
        symbols=tuple(args.symbols),
        start_date=args.start_date,
        days=args.days,
        max_workers=args.max_workers,
    )
    candidates = build_intraday_feature_candidates(labels)
    write_intraday_feature_labels_csv(labels, output_path=args.labels_output_path)
    write_intraday_feature_candidates_csv(candidates, output_path=args.candidates_output_path)
    write_intraday_feature_candidates_md(candidates, output_path=args.markdown_output_path, top=args.top)
    for candidate in candidates[: args.top]:
        print(
            candidate.symbol,
            candidate.feature,
            candidate.status,
            candidate.preferred_bucket,
            f"score={candidate.edge_score:.4f}",
        )


if __name__ == "__main__":
    main()
