from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)


@dataclass(frozen=True)
class Observation:
    symbol: str
    timestamp: str
    next_return: float
    funding_rate_sum: float
    premium_close: float
    taker_buy_imbalance: float
    volume_ratio_20d: float


@dataclass(frozen=True)
class BucketSummary:
    feature: str
    bucket: str
    count: int
    mean_next_return: float
    hit_rate: float


def build_observations(
    rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for symbol, rows in rows_by_symbol.items():
        for index, row in enumerate(rows[:-1]):
            next_close = rows[index + 1].close
            if row.close <= 0.0 or row.volume <= 0.0:
                continue
            prior_volumes = tuple(item.volume for item in rows[max(0, index - 19) : index + 1])
            volume_ratio = row.volume / mean(prior_volumes) if prior_volumes else 1.0
            observations.append(
                Observation(
                    symbol=symbol,
                    timestamp=row.timestamp,
                    next_return=(next_close / row.close) - 1.0,
                    funding_rate_sum=row.funding_rate_sum,
                    premium_close=row.premium_close,
                    taker_buy_imbalance=(row.taker_buy_volume / row.volume) - 0.5,
                    volume_ratio_20d=volume_ratio,
                )
            )
    return tuple(observations)


def summarize_feature_buckets(
    observations: tuple[Observation, ...],
) -> tuple[BucketSummary, ...]:
    summaries: list[BucketSummary] = []
    for feature in (
        "funding_rate_sum",
        "premium_close",
        "taker_buy_imbalance",
        "volume_ratio_20d",
    ):
        summaries.extend(_summarize_quantile_feature(observations, feature=feature))
    return tuple(summaries)


def write_bucket_summaries(
    summaries: tuple[BucketSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature", "bucket", "count", "mean_next_return", "hit_rate"))
        for summary in summaries:
            writer.writerow(
                (
                    summary.feature,
                    summary.bucket,
                    summary.count,
                    f"{summary.mean_next_return:.10f}",
                    f"{summary.hit_rate:.6f}",
                )
            )
    return output_path


def _summarize_quantile_feature(
    observations: tuple[Observation, ...],
    *,
    feature: str,
) -> tuple[BucketSummary, ...]:
    values = sorted(getattr(observation, feature) for observation in observations)
    if len(values) < 5:
        return ()
    low = values[int(len(values) * 0.2)]
    high = values[int(len(values) * 0.8)]
    buckets = {
        "bottom_20": [
            observation
            for observation in observations
            if getattr(observation, feature) <= low
        ],
        "middle_60": [
            observation
            for observation in observations
            if low < getattr(observation, feature) < high
        ],
        "top_20": [
            observation
            for observation in observations
            if getattr(observation, feature) >= high
        ],
    }
    return tuple(
        _summary(feature=feature, bucket=bucket, observations=tuple(bucket_observations))
        for bucket, bucket_observations in buckets.items()
        if bucket_observations
    )


def _summary(
    *,
    feature: str,
    bucket: str,
    observations: tuple[Observation, ...],
) -> BucketSummary:
    return BucketSummary(
        feature=feature,
        bucket=bucket,
        count=len(observations),
        mean_next_return=mean(observation.next_return for observation in observations),
        hit_rate=mean(1.0 if observation.next_return > 0.0 else 0.0 for observation in observations),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=LOCAL_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "market_structure_diagnostics.csv",
    )
    args = parser.parse_args()

    rows_by_symbol = load_market_structure_days(
        dataset_dir=args.dataset_dir,
        symbols=tuple(args.symbols),
    )
    observations = build_observations(rows_by_symbol)
    summaries = summarize_feature_buckets(observations)
    write_bucket_summaries(summaries, output_path=args.output_path)
    for summary in summaries:
        print(
            summary.feature,
            summary.bucket,
            summary.count,
            f"{summary.mean_next_return:.6f}",
            f"{summary.hit_rate:.3f}",
        )


if __name__ == "__main__":
    main()
