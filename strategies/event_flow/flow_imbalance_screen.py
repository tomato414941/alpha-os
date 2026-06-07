from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.event_flow.fetch_aggtrade_sample import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class FlowBar:
    timestamp: str
    symbol: str
    close: float
    volume: float
    taker_buy_volume: float
    taker_sell_volume: float
    trade_count: int

    @property
    def imbalance(self) -> float:
        denominator = self.taker_buy_volume + self.taker_sell_volume
        if denominator <= 0.0:
            return 0.0
        return (self.taker_buy_volume - self.taker_sell_volume) / denominator


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    count: int
    mean_next_return: float
    hit_rate: float


def load_flow_bars(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[FlowBar, ...]:
    bars = []
    for path in sorted(dataset_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            bars.extend(
                FlowBar(
                    timestamp=str(row["timestamp"]),
                    symbol=str(row["symbol"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    taker_buy_volume=float(row["taker_buy_volume"]),
                    taker_sell_volume=float(row["taker_sell_volume"]),
                    trade_count=int(row["trade_count"]),
                )
                for row in reader
            )
    return tuple(sorted(bars, key=lambda bar: (bar.symbol, bar.timestamp)))


def screen_flow_imbalance(bars: tuple[FlowBar, ...]) -> tuple[BucketResult, ...]:
    samples: list[tuple[float, float]] = []
    bars_by_symbol: dict[str, list[FlowBar]] = {}
    for bar in bars:
        bars_by_symbol.setdefault(bar.symbol, []).append(bar)
    for symbol_bars in bars_by_symbol.values():
        for current, next_bar in zip(symbol_bars, symbol_bars[1:]):
            if current.close <= 0.0:
                continue
            samples.append((current.imbalance, (next_bar.close / current.close) - 1.0))
    if not samples:
        return ()
    imbalances = sorted(imbalance for imbalance, _ in samples)
    low = imbalances[int(len(imbalances) * 0.2)]
    high = imbalances[int(len(imbalances) * 0.8)]
    buckets = {
        "bottom_20": [next_return for imbalance, next_return in samples if imbalance <= low],
        "middle_60": [
            next_return for imbalance, next_return in samples if low < imbalance < high
        ],
        "top_20": [next_return for imbalance, next_return in samples if imbalance >= high],
    }
    return tuple(_bucket_result(bucket, values) for bucket, values in buckets.items() if values)


def write_bucket_results(results: tuple[BucketResult, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bucket", "count", "mean_next_return", "hit_rate"))
        for result in results:
            writer.writerow(
                (
                    result.bucket,
                    result.count,
                    f"{result.mean_next_return:.10f}",
                    f"{result.hit_rate:.6f}",
                )
            )
    return output_path


def _bucket_result(bucket: str, values: list[float]) -> BucketResult:
    return BucketResult(
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
        default=Path(__file__).resolve().parent / "flow_imbalance_screen.csv",
    )
    args = parser.parse_args()

    results = screen_flow_imbalance(load_flow_bars(args.dataset_dir))
    write_bucket_results(results, output_path=args.output_path)
    for result in results:
        print(
            result.bucket,
            result.count,
            f"{result.mean_next_return:.8f}",
            f"{result.hit_rate:.4f}",
        )


if __name__ == "__main__":
    main()
