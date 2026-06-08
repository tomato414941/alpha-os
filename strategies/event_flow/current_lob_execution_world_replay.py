from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.event_flow.book_depth_walk_forward_check import (
    FEATURES,
    ContextSample,
    load_context_samples,
)
from strategies.event_flow.fetch_book_depth_sample import DEFAULT_OUTPUT_DIR


ROOT = Path(__file__).resolve().parent

EXECUTION_ACTIONS = (
    ("hold", "no_order", 0.0),
    ("market_order", "full_fill", 8.0),
    ("low_fee_cross", "full_fill_low_fee_or_internalized", 2.0),
    ("maker_or_internalized_limit", "optimistic_full_fill_not_queue_verified", 0.5),
)


@dataclass(frozen=True)
class LobExecutionWorldReplayRow:
    feature: str
    bucket: str
    signal_action: str
    train_count: int
    train_signal_bps: float
    test_count: int
    execution_action: str
    fill_assumption: str
    cost_bps: float
    gross_reward_bps: float
    net_reward_bps: float
    hit_rate: float
    beats_hold: str
    decision: str
    next_step: str


def build_lob_execution_world_replay(
    samples: tuple[ContextSample, ...],
    *,
    purge_days: int = 1,
) -> tuple[LobExecutionWorldReplayRow, ...]:
    train_samples, test_samples = _train_test_split(samples, purge_days=purge_days)
    rows = [
        _replay_row(
            feature=feature,
            bucket=bucket,
            train_returns=train_returns,
            test_returns=test_returns,
            execution_action=execution_action,
            fill_assumption=fill_assumption,
            cost_bps=cost_bps,
        )
        for feature in FEATURES
        for bucket, train_returns, test_returns in _feature_bucket_returns(
            train_samples,
            test_samples,
            feature=feature,
        )
        for execution_action, fill_assumption, cost_bps in EXECUTION_ACTIONS
    ]
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_lob_execution_world_replay_csv(
    rows: tuple[LobExecutionWorldReplayRow, ...],
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
                "execution_action",
                "fill_assumption",
                "cost_bps",
                "gross_reward_bps",
                "net_reward_bps",
                "hit_rate",
                "beats_hold",
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
                    row.execution_action,
                    row.fill_assumption,
                    f"{row.cost_bps:.8f}",
                    f"{row.gross_reward_bps:.8f}",
                    f"{row.net_reward_bps:.8f}",
                    f"{row.hit_rate:.6f}",
                    row.beats_hold,
                    row.decision,
                    row.next_step,
                )
            )
    return output_path


def write_lob_execution_world_replay_md(
    rows: tuple[LobExecutionWorldReplayRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# LOB Execution World Replay\n\n")
        handle.write(
            "This is a tiny RL-shaped replay over existing book-depth samples. "
            "The state is a train-side feature bucket, the action is hold/market/low-fee/maker-like execution, "
            "and the reward is next-1m directional return after explicit cost. "
            "Maker/internalized rows are optimistic full-fill diagnostics, not executable claims.\n\n"
        )
        handle.write(
            "| feature | bucket | signal | execution action | fill assumption | train bps | test n | gross bps | cost bps | net bps | hit | decision | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.feature} | "
                f"{row.bucket} | "
                f"{row.signal_action} | "
                f"{row.execution_action} | "
                f"{row.fill_assumption} | "
                f"{row.train_signal_bps:.4f} | "
                f"{row.test_count} | "
                f"{row.gross_reward_bps:.4f} | "
                f"{row.cost_bps:.2f} | "
                f"{row.net_reward_bps:.4f} | "
                f"{row.hit_rate:.6f} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary(rows))
    return output_path


def _feature_bucket_returns(
    train_samples: tuple[ContextSample, ...],
    test_samples: tuple[ContextSample, ...],
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


def _replay_row(
    *,
    feature: str,
    bucket: str,
    train_returns: list[float],
    test_returns: list[float],
    execution_action: str,
    fill_assumption: str,
    cost_bps: float,
) -> LobExecutionWorldReplayRow:
    signal_side = 1.0 if mean(train_returns) >= 0.0 else -1.0
    signal_action = "paper_long" if signal_side > 0.0 else "paper_short"
    train_signal_bps = mean(train_returns) * signal_side * 10_000.0 if train_returns else 0.0
    gross_rewards = [signal_side * value * 10_000.0 for value in test_returns]
    gross_reward_bps = mean(gross_rewards) if gross_rewards else 0.0
    net_reward_bps = 0.0 if execution_action == "hold" else gross_reward_bps - cost_bps
    hit_rate = mean(1.0 if value > 0.0 else 0.0 for value in gross_rewards) if gross_rewards else 0.0
    decision = _decision(
        execution_action=execution_action,
        net_reward_bps=net_reward_bps,
        hit_rate=hit_rate,
        test_count=len(test_returns),
    )
    return LobExecutionWorldReplayRow(
        feature=feature,
        bucket=bucket,
        signal_action=signal_action,
        train_count=len(train_returns),
        train_signal_bps=train_signal_bps,
        test_count=len(test_returns),
        execution_action=execution_action,
        fill_assumption=fill_assumption,
        cost_bps=cost_bps,
        gross_reward_bps=gross_reward_bps,
        net_reward_bps=net_reward_bps,
        hit_rate=hit_rate,
        beats_hold="yes" if net_reward_bps > 0.0 else "no",
        decision=decision,
        next_step=_next_step(decision),
    )


def _decision(*, execution_action: str, net_reward_bps: float, hit_rate: float, test_count: int) -> str:
    if execution_action == "hold":
        return "hold_baseline"
    if test_count == 0:
        return "no_test_samples"
    if net_reward_bps <= 0.0:
        return "worse_than_hold"
    if execution_action == "market_order":
        return "market_action_candidate" if hit_rate >= 0.5 else "market_tail_candidate"
    if execution_action == "low_fee_cross":
        return "low_fee_action_candidate" if hit_rate >= 0.5 else "low_fee_tail_candidate"
    if hit_rate >= 0.5:
        return "maker_fill_model_needed"
    return "maker_tail_or_queue_research"


def _next_step(decision: str) -> str:
    if decision == "market_action_candidate":
        return "open only a tiny paper label after checking spread, funding, stop, and adverse excursion"
    if decision == "market_tail_candidate":
        return "inspect payoff tails and stop behavior before any market-order probe"
    if decision == "low_fee_action_candidate":
        return "validate real fee tier, spread, funding, and whether low-fee execution is available"
    if decision == "low_fee_tail_candidate":
        return "check whether positive reward is tail-only before treating low-fee execution as alpha"
    if decision == "maker_fill_model_needed":
        return "build maker fill probability, queue position, cancel, and adverse-selection labels"
    if decision == "maker_tail_or_queue_research":
        return "treat as representation research until queue/fill and tail risk are measured"
    if decision == "worse_than_hold":
        return "do not act under this execution action; hold beats the replay reward"
    if decision == "no_test_samples":
        return "extend the replay window before evaluating this state/action"
    return "hold is the baseline action"


def _train_test_split(
    samples: tuple[ContextSample, ...],
    *,
    purge_days: int,
) -> tuple[tuple[ContextSample, ...], tuple[ContextSample, ...]]:
    dates = sorted({sample.date for sample in samples})
    if len(dates) < 4:
        return (), ()
    split = max(2, len(dates) // 2)
    train_dates = set(dates[:split])
    test_dates = set(dates[split + purge_days :])
    return (
        tuple(sample for sample in samples if sample.date in train_dates),
        tuple(sample for sample in samples if sample.date in test_dates),
    )


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


def _sort_key(row: LobExecutionWorldReplayRow) -> tuple[int, float, float]:
    decision_rank = {
        "market_action_candidate": 800,
        "low_fee_action_candidate": 700,
        "maker_fill_model_needed": 600,
        "market_tail_candidate": 500,
        "low_fee_tail_candidate": 400,
        "maker_tail_or_queue_research": 300,
        "hold_baseline": 200,
        "worse_than_hold": 100,
        "no_test_samples": 0,
    }.get(row.decision, 0)
    return decision_rank, row.net_reward_bps, row.hit_rate


def _summary(rows: tuple[LobExecutionWorldReplayRow, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.decision] = counts.get(row.decision, 0) + 1
    lines = [f"- {decision}: {count}" for decision, count in sorted(counts.items())]
    best = next((row for row in rows if row.decision != "hold_baseline"), None)
    if best:
        lines.append(
            "- best non-hold action: "
            f"{best.feature}/{best.bucket}/{best.signal_action}/{best.execution_action} "
            f"net={best.net_reward_bps:.8f}bps hit={best.hit_rate:.6f} decision={best.decision}"
        )
    if not lines:
        lines.append("- no replay rows")
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_lob_execution_world_replay.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_lob_execution_world_replay.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_lob_execution_world_replay(
        load_context_samples(args.dataset_dir),
        purge_days=args.purge_days,
    )
    write_lob_execution_world_replay_csv(rows, output_path=args.output_path)
    write_lob_execution_world_replay_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.feature,
            row.bucket,
            row.signal_action,
            row.execution_action,
            row.decision,
            f"{row.net_reward_bps:.4f}",
        )


if __name__ == "__main__":
    main()
