from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean

from strategies.event_flow.fetch_book_depth_sample import DEFAULT_OUTPUT_DIR


ROOT = Path(__file__).resolve().parent
FEATURES = (
    "imbalance_1pct_persistence_5",
    "imbalance_1pct_delta_5",
    "imbalance_5pct_persistence_5",
    "imbalance_5pct_delta_5",
    "bid_liquidity_change_5",
    "ask_liquidity_change_5",
    "shallow_deep_imbalance_gap",
    "basis_delta_5",
    "premium_delta_5",
    "taker_pressure_delta_5",
    "oi_value_change_15",
)
EXECUTION_COSTS_BPS = (
    ("market_order", 8.0),
    ("low_fee_cross", 2.0),
    ("maker_or_internalized_limit", 0.5),
    ("zero_cost_representation", 0.0),
)


@dataclass(frozen=True)
class LobSequenceStateSample:
    timestamp: str
    sample_date: date
    symbol: str
    features: dict[str, float]
    next_1m_return: float


@dataclass(frozen=True)
class LobSequenceStateProbeRow:
    feature: str
    bucket: str
    signal_action: str
    train_count: int
    train_signal_bps: float
    test_count: int
    execution_mode: str
    cost_bps: float
    test_gross_bps: float
    test_net_bps: float
    test_hit_rate: float
    decision: str
    next_step: str


def load_lob_sequence_state_samples(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[LobSequenceStateSample, ...]:
    samples: list[LobSequenceStateSample] = []
    for path in sorted(dataset_dir.glob("*.csv")):
        rows = _read_numeric_rows(path)
        for index, row in enumerate(rows):
            previous_5 = rows[max(0, index - 5) : index]
            previous_15 = rows[max(0, index - 15) : index]
            if len(previous_5) < 5 or len(previous_15) < 15:
                continue
            samples.append(
                LobSequenceStateSample(
                    timestamp=str(row["timestamp"]),
                    sample_date=date.fromisoformat(str(row["timestamp"])[:10]),
                    symbol=str(row["symbol"]),
                    features=_sequence_features(row=row, previous_5=previous_5, previous_15=previous_15),
                    next_1m_return=float(row["next_1m_return"]),
                )
            )
    return tuple(samples)


def build_lob_sequence_state_probe(
    samples: tuple[LobSequenceStateSample, ...],
    *,
    purge_days: int = 1,
) -> tuple[LobSequenceStateProbeRow, ...]:
    train_samples, test_samples = _train_test_split(samples, purge_days=purge_days)
    rows = [
        _probe_row(
            feature=feature,
            bucket=bucket,
            train_returns=train_returns,
            test_returns=test_returns,
            execution_mode=execution_mode,
            cost_bps=cost_bps,
        )
        for feature in FEATURES
        for bucket, train_returns, test_returns in _feature_bucket_returns(
            train_samples,
            test_samples,
            feature=feature,
        )
        for execution_mode, cost_bps in EXECUTION_COSTS_BPS
    ]
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_lob_sequence_state_probe_csv(
    rows: tuple[LobSequenceStateProbeRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "feature",
                "bucket",
                "signal_action",
                "train_count",
                "train_signal_bps",
                "test_count",
                "execution_mode",
                "cost_bps",
                "test_gross_bps",
                "test_net_bps",
                "test_hit_rate",
                "decision",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.feature,
                    row.bucket,
                    row.signal_action,
                    row.train_count,
                    f"{row.train_signal_bps:.8f}",
                    row.test_count,
                    row.execution_mode,
                    f"{row.cost_bps:.8f}",
                    f"{row.test_gross_bps:.8f}",
                    f"{row.test_net_bps:.8f}",
                    f"{row.test_hit_rate:.6f}",
                    row.decision,
                    row.next_step,
                )
            )
    return output_path


def write_lob_sequence_state_probe_md(
    rows: tuple[LobSequenceStateProbeRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# LOB Sequence State Probe\n\n")
        handle.write(
            "This turns book-depth snapshots into rolling state features before evaluating the next-1m label. "
            "It is a representation probe: positive zero-cost rows are not alpha unless they survive an execution mode.\n\n"
        )
        handle.write(
            "| feature | bucket | signal | mode | train n | train bps | test n | gross bps | cost bps | net bps | hit | decision | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.feature} | "
                f"{row.bucket} | "
                f"{row.signal_action} | "
                f"{row.execution_mode} | "
                f"{row.train_count} | "
                f"{row.train_signal_bps:.4f} | "
                f"{row.test_count} | "
                f"{row.test_gross_bps:.4f} | "
                f"{row.cost_bps:.2f} | "
                f"{row.test_net_bps:.4f} | "
                f"{row.test_hit_rate:.6f} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary(rows))
    return output_path


def _read_numeric_rows(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, str | float]] = []
        for row in csv.DictReader(handle):
            numeric_row: dict[str, str | float] = {
                "timestamp": row.get("timestamp", ""),
                "symbol": row.get("symbol", ""),
            }
            for key, value in row.items():
                if key not in {"timestamp", "symbol"}:
                    numeric_row[key] = _float(value)
            rows.append(numeric_row)
        return sorted(rows, key=lambda row: str(row["timestamp"]))


def _sequence_features(
    *,
    row: dict[str, str | float],
    previous_5: list[dict[str, str | float]],
    previous_15: list[dict[str, str | float]],
) -> dict[str, float]:
    imbalance_1pct = float(row["imbalance_1pct"])
    imbalance_5pct = float(row["imbalance_5pct"])
    previous_imbalance_1pct = _mean(previous_5, "imbalance_1pct")
    previous_imbalance_5pct = _mean(previous_5, "imbalance_5pct")
    return {
        "imbalance_1pct_persistence_5": imbalance_1pct * previous_imbalance_1pct,
        "imbalance_1pct_delta_5": imbalance_1pct - previous_imbalance_1pct,
        "imbalance_5pct_persistence_5": imbalance_5pct * previous_imbalance_5pct,
        "imbalance_5pct_delta_5": imbalance_5pct - previous_imbalance_5pct,
        "bid_liquidity_change_5": _relative_change(float(row["bid_notional_1pct"]), _mean(previous_5, "bid_notional_1pct")),
        "ask_liquidity_change_5": _relative_change(float(row["ask_notional_1pct"]), _mean(previous_5, "ask_notional_1pct")),
        "shallow_deep_imbalance_gap": imbalance_1pct - imbalance_5pct,
        "basis_delta_5": float(row["mark_index_basis_1m"]) - _mean(previous_5, "mark_index_basis_1m"),
        "premium_delta_5": float(row["premium_index_1m"]) - _mean(previous_5, "premium_index_1m"),
        "taker_pressure_delta_5": (
            float(row["taker_long_short_volume_ratio_5m"])
            - _mean(previous_5, "taker_long_short_volume_ratio_5m")
        ),
        "oi_value_change_15": _relative_change(float(row["open_interest_value_5m"]), _mean(previous_15, "open_interest_value_5m")),
    }


def _feature_bucket_returns(
    train_samples: tuple[LobSequenceStateSample, ...],
    test_samples: tuple[LobSequenceStateSample, ...],
    *,
    feature: str,
) -> tuple[tuple[str, list[float], list[float]], ...]:
    thresholds = _thresholds(train_samples, feature=feature)
    if thresholds is None:
        return ()
    return (
        (
            "bottom_20",
            _bucket_returns(train_samples, feature=feature, bucket="bottom_20", thresholds=thresholds),
            _bucket_returns(test_samples, feature=feature, bucket="bottom_20", thresholds=thresholds),
        ),
        (
            "top_20",
            _bucket_returns(train_samples, feature=feature, bucket="top_20", thresholds=thresholds),
            _bucket_returns(test_samples, feature=feature, bucket="top_20", thresholds=thresholds),
        ),
    )


def _probe_row(
    *,
    feature: str,
    bucket: str,
    train_returns: list[float],
    test_returns: list[float],
    execution_mode: str,
    cost_bps: float,
) -> LobSequenceStateProbeRow:
    side = 1.0 if mean(train_returns) >= 0.0 else -1.0
    signal_action = "paper_long" if side > 0.0 else "paper_short"
    directional_test = [side * value for value in test_returns]
    train_signal_bps = mean(train_returns) * side * 10_000.0 if train_returns else 0.0
    gross_bps = mean(directional_test) * 10_000.0 if directional_test else 0.0
    net_bps = gross_bps - cost_bps if test_returns else 0.0
    hit_rate = mean(1.0 if value > 0.0 else 0.0 for value in directional_test) if directional_test else 0.0
    decision = _decision(execution_mode=execution_mode, net_bps=net_bps, hit_rate=hit_rate, test_count=len(test_returns))
    return LobSequenceStateProbeRow(
        feature=feature,
        bucket=bucket,
        signal_action=signal_action,
        train_count=len(train_returns),
        train_signal_bps=train_signal_bps,
        test_count=len(test_returns),
        execution_mode=execution_mode,
        cost_bps=cost_bps,
        test_gross_bps=gross_bps,
        test_net_bps=net_bps,
        test_hit_rate=hit_rate,
        decision=decision,
        next_step=_next_step(decision),
    )


def _decision(*, execution_mode: str, net_bps: float, hit_rate: float, test_count: int) -> str:
    if test_count == 0:
        return "no_test_samples"
    if net_bps <= 0.0:
        return "reject_after_cost"
    if execution_mode == "zero_cost_representation":
        return "representation_only"
    if execution_mode == "maker_or_internalized_limit":
        if hit_rate >= 0.5:
            return "maker_sequence_candidate"
        return "maker_sequence_tail_candidate"
    if execution_mode == "low_fee_cross":
        if hit_rate >= 0.5:
            return "low_fee_sequence_candidate"
        return "low_fee_sequence_tail_candidate"
    if hit_rate >= 0.5:
        return "market_sequence_candidate"
    return "market_sequence_tail_candidate"


def _next_step(decision: str) -> str:
    if decision == "maker_sequence_candidate":
        return "build queue/fill labels for this rolling state before any paper execution"
    if decision == "maker_sequence_tail_candidate":
        return "inspect tail dependence and adverse selection before trusting maker execution"
    if decision == "low_fee_sequence_candidate":
        return "verify real low-fee route, spread, funding, and stop behavior"
    if decision == "low_fee_sequence_tail_candidate":
        return "check whether low-fee reward is tail-only before paper probing"
    if decision == "market_sequence_candidate":
        return "paper-probe only after spread, funding, and adverse-excursion checks"
    if decision == "market_sequence_tail_candidate":
        return "inspect tail and stop risk before using market orders"
    if decision == "representation_only":
        return "keep as a feature for a model; it does not survive execution costs"
    if decision == "no_test_samples":
        return "extend sample window"
    return "reject this rolling state under the current execution cost"


def _train_test_split(
    samples: tuple[LobSequenceStateSample, ...],
    *,
    purge_days: int,
) -> tuple[tuple[LobSequenceStateSample, ...], tuple[LobSequenceStateSample, ...]]:
    dates = sorted({sample.sample_date for sample in samples})
    if len(dates) < 4:
        return (), ()
    split = max(2, len(dates) // 2)
    train_dates = set(dates[:split])
    test_dates = set(dates[split + purge_days :])
    return (
        tuple(sample for sample in samples if sample.sample_date in train_dates),
        tuple(sample for sample in samples if sample.sample_date in test_dates),
    )


def _thresholds(samples: tuple[LobSequenceStateSample, ...], *, feature: str) -> tuple[float, float] | None:
    values = sorted(sample.features[feature] for sample in samples)
    if not values:
        return None
    return values[int(len(values) * 0.2)], values[int(len(values) * 0.8)]


def _bucket_returns(
    samples: tuple[LobSequenceStateSample, ...],
    *,
    feature: str,
    bucket: str,
    thresholds: tuple[float, float],
) -> list[float]:
    low, high = thresholds
    if bucket == "bottom_20":
        return [sample.next_1m_return for sample in samples if sample.features[feature] <= low]
    return [sample.next_1m_return for sample in samples if sample.features[feature] >= high]


def _sort_key(row: LobSequenceStateProbeRow) -> tuple[int, float, float]:
    decision_rank = {
        "market_sequence_candidate": 900,
        "low_fee_sequence_candidate": 800,
        "maker_sequence_candidate": 700,
        "market_sequence_tail_candidate": 600,
        "low_fee_sequence_tail_candidate": 500,
        "maker_sequence_tail_candidate": 400,
        "representation_only": 300,
        "reject_after_cost": 100,
        "no_test_samples": 0,
    }.get(row.decision, 0)
    return decision_rank, row.test_net_bps, row.test_hit_rate


def _mean(rows: list[dict[str, str | float]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _relative_change(current: float, previous: float) -> float:
    if previous <= 0.0:
        return 0.0
    return (current / previous) - 1.0


def _summary(rows: tuple[LobSequenceStateProbeRow, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.decision] = counts.get(row.decision, 0) + 1
    lines = [f"- {decision}: {count}" for decision, count in sorted(counts.items())]
    best = next((row for row in rows if row.decision != "reject_after_cost"), None)
    if best:
        lines.append(
            "- best non-reject: "
            f"{best.feature}/{best.bucket}/{best.signal_action}/{best.execution_mode} "
            f"net={best.test_net_bps:.8f}bps hit={best.test_hit_rate:.6f} decision={best.decision}"
        )
    if not lines:
        lines.append("- no sequence-state rows")
    return "\n".join(lines) + "\n"


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_lob_sequence_state_probe.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_lob_sequence_state_probe.md")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_lob_sequence_state_probe(
        load_lob_sequence_state_samples(args.dataset_dir),
        purge_days=args.purge_days,
    )
    write_lob_sequence_state_probe_csv(rows, output_path=args.output_path)
    write_lob_sequence_state_probe_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.feature,
            row.bucket,
            row.signal_action,
            row.execution_mode,
            row.decision,
            f"{row.test_net_bps:.4f}",
        )


if __name__ == "__main__":
    main()
