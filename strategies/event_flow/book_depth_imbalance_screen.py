from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.event_flow.fetch_book_depth_sample import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class BookDepthSample:
    timestamp: str
    symbol: str
    imbalance_1pct: float
    imbalance_5pct: float
    premium_index_1m: float
    mark_index_basis_1m: float
    open_interest_value_5m: float
    top_trader_long_short_ratio_5m: float
    account_long_short_ratio_5m: float
    taker_long_short_volume_ratio_5m: float
    next_1m_return: float


@dataclass(frozen=True)
class BucketResult:
    feature: str
    bucket: str
    count: int
    mean_next_return: float
    hit_rate: float


def load_book_depth_samples(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[BookDepthSample, ...]:
    samples = []
    for path in sorted(dataset_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            samples.extend(
                BookDepthSample(
                    timestamp=str(row["timestamp"]),
                    symbol=str(row["symbol"]),
                    imbalance_1pct=float(row["imbalance_1pct"]),
                    imbalance_5pct=float(row["imbalance_5pct"]),
                    premium_index_1m=float(row.get("premium_index_1m") or 0.0),
                    mark_index_basis_1m=float(row.get("mark_index_basis_1m") or 0.0),
                    open_interest_value_5m=float(row.get("open_interest_value_5m") or 0.0),
                    top_trader_long_short_ratio_5m=float(row.get("top_trader_long_short_ratio_5m") or 0.0),
                    account_long_short_ratio_5m=float(row.get("account_long_short_ratio_5m") or 0.0),
                    taker_long_short_volume_ratio_5m=float(row.get("taker_long_short_volume_ratio_5m") or 0.0),
                    next_1m_return=float(row["next_1m_return"]),
                )
                for row in reader
            )
    return tuple(samples)


def screen_book_depth_imbalance(samples: tuple[BookDepthSample, ...]) -> tuple[BucketResult, ...]:
    rows = []
    for feature in (
        "imbalance_1pct",
        "imbalance_5pct",
        "premium_index_1m",
        "mark_index_basis_1m",
        "open_interest_value_5m",
        "top_trader_long_short_ratio_5m",
        "account_long_short_ratio_5m",
        "taker_long_short_volume_ratio_5m",
    ):
        feature_samples = tuple((getattr(sample, feature), sample.next_1m_return) for sample in samples)
        rows.extend(_bucket_results(feature, feature_samples))
    return tuple(rows)


def write_bucket_results(rows: tuple[BucketResult, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("feature", "bucket", "count", "mean_next_return", "hit_rate"))
        for row in rows:
            writer.writerow(
                (
                    row.feature,
                    row.bucket,
                    row.count,
                    f"{row.mean_next_return:.10f}",
                    f"{row.hit_rate:.6f}",
                )
            )
    return output_path


def write_bucket_results_md(rows: tuple[BucketResult, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Book Depth Context Screen\n\n")
        handle.write(
            "This checks whether futures bookDepth liquidity imbalance and perp basis context have a simple next-1m return edge. "
            "It is a data-path and diagnostic check, not a deployable strategy.\n\n"
        )
        handle.write("| feature | bucket | count | mean next return | hit rate |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row.feature} | "
                f"{row.bucket} | "
                f"{row.count} | "
                f"{row.mean_next_return:.10f} | "
                f"{row.hit_rate:.6f} |\n"
            )
    return output_path


def _bucket_results(feature: str, samples: tuple[tuple[float, float], ...]) -> tuple[BucketResult, ...]:
    if not samples:
        return ()
    values = sorted(value for value, _ in samples)
    low = values[int(len(values) * 0.2)]
    high = values[int(len(values) * 0.8)]
    buckets = {
        "bottom_20": [next_return for value, next_return in samples if value <= low],
        "middle_60": [next_return for value, next_return in samples if low < value < high],
        "top_20": [next_return for value, next_return in samples if value >= high],
    }
    return tuple(_bucket_result(feature, bucket, values) for bucket, values in buckets.items() if values)


def _bucket_result(feature: str, bucket: str, values: list[float]) -> BucketResult:
    return BucketResult(
        feature=feature,
        bucket=bucket,
        count=len(values),
        mean_next_return=mean(values),
        hit_rate=mean(1.0 if value > 0.0 else 0.0 for value in values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "book_depth_imbalance_screen.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "book_depth_imbalance_screen.md",
    )
    args = parser.parse_args()

    rows = screen_book_depth_imbalance(load_book_depth_samples(args.dataset_dir))
    write_bucket_results(rows, output_path=args.output_path)
    write_bucket_results_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.feature, row.bucket, row.count, f"{row.mean_next_return:.8f}", f"{row.hit_rate:.4f}")


if __name__ == "__main__":
    main()
