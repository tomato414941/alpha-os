from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean

from strategies.event_flow.fetch_book_depth_sample import DEFAULT_OUTPUT_DIR


FEATURES = (
    "imbalance_1pct",
    "imbalance_5pct",
    "premium_index_1m",
    "mark_index_basis_1m",
    "open_interest_value_5m",
    "top_trader_long_short_ratio_5m",
    "account_long_short_ratio_5m",
    "taker_long_short_volume_ratio_5m",
)


@dataclass(frozen=True)
class ContextSample:
    date: date
    feature_values: dict[str, float]
    next_1m_return: float


@dataclass(frozen=True)
class WalkForwardResult:
    feature: str
    bucket: str
    action: str
    train_count: int
    train_mean_bps: float
    test_count: int
    test_gross_bps: float
    test_net_bps: float
    test_hit_rate: float
    decision: str


def load_context_samples(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[ContextSample, ...]:
    samples = []
    for path in sorted(dataset_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                samples.append(
                    ContextSample(
                        date=date.fromisoformat(row["timestamp"][:10]),
                        feature_values={feature: _float(row.get(feature)) for feature in FEATURES},
                        next_1m_return=_float(row.get("next_1m_return")),
                    )
                )
    return tuple(samples)


def walk_forward_check(
    samples: tuple[ContextSample, ...],
    *,
    cost_bps: float = 8.0,
    purge_days: int = 1,
) -> tuple[WalkForwardResult, ...]:
    dates = sorted({sample.date for sample in samples})
    if len(dates) < 4:
        return ()
    split = max(2, len(dates) // 2)
    train_dates = set(dates[:split])
    test_dates = set(dates[split + purge_days :])
    train_samples = tuple(sample for sample in samples if sample.date in train_dates)
    test_samples = tuple(sample for sample in samples if sample.date in test_dates)
    rows = []
    for feature in FEATURES:
        thresholds = _thresholds(train_samples, feature=feature)
        if thresholds is None:
            continue
        for bucket in ("bottom_20", "top_20"):
            train_returns = _bucket_returns(train_samples, feature=feature, bucket=bucket, thresholds=thresholds)
            if not train_returns:
                continue
            action = "paper_long" if mean(train_returns) >= 0.0 else "paper_short"
            test_returns = _bucket_returns(test_samples, feature=feature, bucket=bucket, thresholds=thresholds)
            rows.append(
                _result(
                    feature=feature,
                    bucket=bucket,
                    action=action,
                    train_returns=train_returns,
                    test_returns=test_returns,
                    cost_bps=cost_bps,
                )
            )
    return tuple(sorted(rows, key=lambda row: row.test_net_bps, reverse=True))


def write_walk_forward_csv(rows: tuple[WalkForwardResult, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "feature",
                "bucket",
                "action",
                "train_count",
                "train_mean_bps",
                "test_count",
                "test_gross_bps",
                "test_net_bps",
                "test_hit_rate",
                "decision",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.feature,
                    row.bucket,
                    row.action,
                    row.train_count,
                    f"{row.train_mean_bps:.8f}",
                    row.test_count,
                    f"{row.test_gross_bps:.8f}",
                    f"{row.test_net_bps:.8f}",
                    f"{row.test_hit_rate:.6f}",
                    row.decision,
                )
            )
    return output_path


def write_walk_forward_md(rows: tuple[WalkForwardResult, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Book Depth Walk Forward Check\n\n")
        handle.write(
            "This uses train-side buckets to choose direction, skips a purge window, "
            "then checks test-side next-1m returns after an explicit round-trip cost. "
            "It is a diagnostic gate, not a deployable strategy.\n\n"
        )
        handle.write("| feature | bucket | action | train n | train bps | test n | gross bps | net bps | hit | decision |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.feature} | "
                f"{row.bucket} | "
                f"{row.action} | "
                f"{row.train_count} | "
                f"{row.train_mean_bps:.4f} | "
                f"{row.test_count} | "
                f"{row.test_gross_bps:.4f} | "
                f"{row.test_net_bps:.4f} | "
                f"{row.test_hit_rate:.6f} | "
                f"{row.decision} |\n"
            )
    return output_path


def _thresholds(samples: tuple[ContextSample, ...], *, feature: str) -> tuple[float, float] | None:
    values = sorted(sample.feature_values[feature] for sample in samples)
    if not values:
        return None
    return values[int(len(values) * 0.2)], values[int(len(values) * 0.8)]


def _bucket_returns(
    samples: tuple[ContextSample, ...],
    *,
    feature: str,
    bucket: str,
    thresholds: tuple[float, float],
) -> list[float]:
    low, high = thresholds
    if bucket == "bottom_20":
        return [sample.next_1m_return for sample in samples if sample.feature_values[feature] <= low]
    return [sample.next_1m_return for sample in samples if sample.feature_values[feature] >= high]


def _result(
    *,
    feature: str,
    bucket: str,
    action: str,
    train_returns: list[float],
    test_returns: list[float],
    cost_bps: float,
) -> WalkForwardResult:
    side = 1.0 if action == "paper_long" else -1.0
    directional_test = [side * value for value in test_returns]
    gross_bps = mean(directional_test) * 10_000.0 if directional_test else 0.0
    train_bps = mean(train_returns) * 10_000.0 * side if train_returns else 0.0
    return WalkForwardResult(
        feature=feature,
        bucket=bucket,
        action=action,
        train_count=len(train_returns),
        train_mean_bps=train_bps,
        test_count=len(test_returns),
        test_gross_bps=gross_bps,
        test_net_bps=gross_bps - cost_bps if test_returns else 0.0,
        test_hit_rate=mean(1.0 if value > 0.0 else 0.0 for value in directional_test) if directional_test else 0.0,
        decision=_decision(gross_bps=gross_bps, net_bps=gross_bps - cost_bps, test_count=len(test_returns)),
    )


def _decision(*, gross_bps: float, net_bps: float, test_count: int) -> str:
    if test_count == 0:
        return "no_test_samples"
    if net_bps > 0.0:
        return "cost_adjusted_candidate"
    if gross_bps > 0.0:
        return "gross_only_candidate"
    return "reject_after_walk_forward"


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cost-bps", type=float, default=8.0)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "book_depth_walk_forward_check.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "book_depth_walk_forward_check.md",
    )
    args = parser.parse_args()

    rows = walk_forward_check(
        load_context_samples(args.dataset_dir),
        cost_bps=args.cost_bps,
        purge_days=args.purge_days,
    )
    write_walk_forward_csv(rows, output_path=args.output_path)
    write_walk_forward_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.feature, row.bucket, row.action, f"{row.test_net_bps:.4f}", row.decision)


if __name__ == "__main__":
    main()
