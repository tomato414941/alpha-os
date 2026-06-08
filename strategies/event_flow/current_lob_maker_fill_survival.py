from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from strategies.event_flow.book_depth_walk_forward_check import (
    ContextSample,
    load_context_samples,
)
from strategies.event_flow.current_lob_sequence_state_probe import (
    LobSequenceStateSample,
    load_lob_sequence_state_samples,
)
from strategies.event_flow.fetch_book_depth_sample import DEFAULT_OUTPUT_DIR


ROOT = Path(__file__).resolve().parent
MAKER_OFFSET_BPS = 0.5
MAKER_COST_BPS = 0.5


@dataclass(frozen=True)
class LobMakerFillSurvival:
    candidate_id: str
    state_family: str
    source_probe: str
    feature: str
    bucket: str
    signal_action: str
    execution_mode: str
    survival_status: str
    survival_score: float
    train_count: int
    test_state_count: int
    fill_count: int
    fill_rate: float
    filled_mark_reward_bps: float
    all_state_reward_bps: float
    adverse_fill_rate: float
    optimistic_net_bps: float
    maker_offset_bps: float
    maker_cost_bps: float
    reason: str
    next_step: str


def build_lob_maker_fill_survival_rows(
    *,
    policy_path: Path = ROOT / "current_lob_policy_candidate_survival.csv",
    dataset_dir: Path = DEFAULT_OUTPUT_DIR,
    purge_days: int = 1,
) -> tuple[LobMakerFillSurvival, ...]:
    context_train, context_test = _split_by_date(load_context_samples(dataset_dir), purge_days=purge_days)
    sequence_train, sequence_test = _split_by_date(load_lob_sequence_state_samples(dataset_dir), purge_days=purge_days)
    rows: list[LobMakerFillSurvival] = []
    for candidate in _read_rows(policy_path):
        if candidate.get("execution_mode") != "maker_or_internalized_limit":
            continue
        if candidate.get("survival_status") not in {
            "lob_world_execution_with_sequence_representation",
            "lob_world_execution_probe",
            "lob_sequence_execution_probe",
            "lob_policy_consensus_execution_probe",
        }:
            continue
        rows.extend(
            _rows_for_candidate(
                candidate=candidate,
                context_train=context_train,
                context_test=context_test,
                sequence_train=sequence_train,
                sequence_test=sequence_test,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_lob_maker_fill_survival_csv(rows: tuple[LobMakerFillSurvival, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "state_family",
                "source_probe",
                "feature",
                "bucket",
                "signal_action",
                "execution_mode",
                "survival_status",
                "survival_score",
                "train_count",
                "test_state_count",
                "fill_count",
                "fill_rate",
                "filled_mark_reward_bps",
                "all_state_reward_bps",
                "adverse_fill_rate",
                "optimistic_net_bps",
                "maker_offset_bps",
                "maker_cost_bps",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.state_family,
                    row.source_probe,
                    row.feature,
                    row.bucket,
                    row.signal_action,
                    row.execution_mode,
                    row.survival_status,
                    f"{row.survival_score:.8f}",
                    row.train_count,
                    row.test_state_count,
                    row.fill_count,
                    f"{row.fill_rate:.8f}",
                    f"{row.filled_mark_reward_bps:.8f}",
                    f"{row.all_state_reward_bps:.8f}",
                    f"{row.adverse_fill_rate:.8f}",
                    f"{row.optimistic_net_bps:.8f}",
                    f"{row.maker_offset_bps:.8f}",
                    f"{row.maker_cost_bps:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_lob_maker_fill_survival_md(rows: tuple[LobMakerFillSurvival, ...], *, output_path: Path, top: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current LOB Maker Fill Survival\n\n")
        handle.write(
            "This is a conservative maker-fill proxy over existing book-depth samples. "
            "A passive order is considered filled only when the next 1m mark crosses the passive offset; "
            "the reward is the filled mark reward after maker cost. It is a fill/adverse-selection gate, "
            "not a live execution model.\n\n"
        )
        handle.write(
            "| candidate | source | status | score | test n | fills | fill rate | filled bps | all-state bps | adverse | optimistic net | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.source_probe} | "
                f"{row.survival_status} | "
                f"{row.survival_score:.4f} | "
                f"{row.test_state_count} | "
                f"{row.fill_count} | "
                f"{row.fill_rate:.6f} | "
                f"{row.filled_mark_reward_bps:.4f} | "
                f"{row.all_state_reward_bps:.4f} | "
                f"{row.adverse_fill_rate:.6f} | "
                f"{row.optimistic_net_bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _rows_for_candidate(
    *,
    candidate: dict[str, str],
    context_train: tuple[ContextSample, ...],
    context_test: tuple[ContextSample, ...],
    sequence_train: tuple[LobSequenceStateSample, ...],
    sequence_test: tuple[LobSequenceStateSample, ...],
) -> tuple[LobMakerFillSurvival, ...]:
    rows: list[LobMakerFillSurvival] = []
    world_feature = candidate.get("world_feature", "")
    if world_feature:
        rows.append(
            _build_row(
                candidate=candidate,
                source_probe="world_replay",
                feature=world_feature,
                bucket=candidate.get("world_bucket", ""),
                train_values=_context_values(context_train, feature=world_feature),
                test_values=_context_values(context_test, feature=world_feature),
                optimistic_net_bps=_float(candidate.get("world_net_bps")),
            )
        )
    sequence_feature = candidate.get("sequence_feature", "")
    if sequence_feature:
        rows.append(
            _build_row(
                candidate=candidate,
                source_probe="sequence_state",
                feature=sequence_feature,
                bucket=candidate.get("sequence_bucket", ""),
                train_values=_sequence_values(sequence_train, feature=sequence_feature),
                test_values=_sequence_values(sequence_test, feature=sequence_feature),
                optimistic_net_bps=_float(candidate.get("sequence_net_bps")),
            )
        )
    return tuple(rows)


def _build_row(
    *,
    candidate: dict[str, str],
    source_probe: str,
    feature: str,
    bucket: str,
    train_values: tuple[tuple[float, float], ...],
    test_values: tuple[tuple[float, float], ...],
    optimistic_net_bps: float,
) -> LobMakerFillSurvival:
    thresholds = _thresholds(tuple(value for value, _ in train_values))
    state_returns = _state_returns(test_values, bucket=bucket, thresholds=thresholds)
    rewards = _maker_rewards(state_returns, signal_action=candidate.get("signal_action", ""))
    filled_rewards = tuple(value for value in rewards if value is not None)
    all_state_rewards = tuple(value if value is not None else 0.0 for value in rewards)
    filled_mark_reward = mean(filled_rewards) if filled_rewards else 0.0
    all_state_reward = mean(all_state_rewards) if all_state_rewards else 0.0
    adverse_fill_rate = (
        mean(1.0 if value < 0.0 else 0.0 for value in filled_rewards)
        if filled_rewards
        else 0.0
    )
    fill_rate = len(filled_rewards) / len(rewards) if rewards else 0.0
    status = _survival_status(
        test_state_count=len(rewards),
        fill_count=len(filled_rewards),
        fill_rate=fill_rate,
        all_state_reward_bps=all_state_reward,
        adverse_fill_rate=adverse_fill_rate,
    )
    candidate_id = (
        f"{candidate.get('state_family', '')}_{source_probe}_{feature}_"
        f"{bucket}_{candidate.get('signal_action', '')}".lower()
    )
    return LobMakerFillSurvival(
        candidate_id=candidate_id,
        state_family=candidate.get("state_family", ""),
        source_probe=source_probe,
        feature=feature,
        bucket=bucket,
        signal_action=candidate.get("signal_action", ""),
        execution_mode=candidate.get("execution_mode", ""),
        survival_status=status,
        survival_score=_survival_score(
            status=status,
            fill_rate=fill_rate,
            filled_mark_reward_bps=filled_mark_reward,
            all_state_reward_bps=all_state_reward,
            adverse_fill_rate=adverse_fill_rate,
            optimistic_net_bps=optimistic_net_bps,
        ),
        train_count=len(train_values),
        test_state_count=len(rewards),
        fill_count=len(filled_rewards),
        fill_rate=fill_rate,
        filled_mark_reward_bps=filled_mark_reward,
        all_state_reward_bps=all_state_reward,
        adverse_fill_rate=adverse_fill_rate,
        optimistic_net_bps=optimistic_net_bps,
        maker_offset_bps=MAKER_OFFSET_BPS,
        maker_cost_bps=MAKER_COST_BPS,
        reason=_reason(status),
        next_step=_next_step(status),
    )


def _context_values(samples: tuple[ContextSample, ...], *, feature: str) -> tuple[tuple[float, float], ...]:
    return tuple((sample.feature_values[feature], sample.next_1m_return) for sample in samples)


def _sequence_values(samples: tuple[LobSequenceStateSample, ...], *, feature: str) -> tuple[tuple[float, float], ...]:
    return tuple((sample.features[feature], sample.next_1m_return) for sample in samples)


def _split_by_date(samples: tuple, *, purge_days: int) -> tuple[tuple, tuple]:
    dates = sorted({sample.sample_date if hasattr(sample, "sample_date") else sample.date for sample in samples})
    if len(dates) < 4:
        return (), ()
    split = max(2, len(dates) // 2)
    train_dates = set(dates[:split])
    test_dates = set(dates[split + purge_days :])
    return (
        tuple(sample for sample in samples if (sample.sample_date if hasattr(sample, "sample_date") else sample.date) in train_dates),
        tuple(sample for sample in samples if (sample.sample_date if hasattr(sample, "sample_date") else sample.date) in test_dates),
    )


def _thresholds(values: tuple[float, ...]) -> tuple[float, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return sorted_values[int(len(sorted_values) * 0.2)], sorted_values[int(len(sorted_values) * 0.8)]


def _state_returns(
    values: tuple[tuple[float, float], ...],
    *,
    bucket: str,
    thresholds: tuple[float, float] | None,
) -> tuple[float, ...]:
    if thresholds is None:
        return ()
    low, high = thresholds
    if bucket == "bottom_20":
        return tuple(next_return for value, next_return in values if value <= low)
    if bucket == "top_20":
        return tuple(next_return for value, next_return in values if value >= high)
    return ()


def _maker_rewards(next_returns: tuple[float, ...], *, signal_action: str) -> tuple[float | None, ...]:
    offset_return = MAKER_OFFSET_BPS / 10_000.0
    rewards: list[float | None] = []
    for next_return in next_returns:
        next_bps = next_return * 10_000.0
        if signal_action == "paper_short":
            if next_return < offset_return:
                rewards.append(None)
                continue
            rewards.append(MAKER_OFFSET_BPS - next_bps - MAKER_COST_BPS)
            continue
        if signal_action == "paper_long":
            if next_return > -offset_return:
                rewards.append(None)
                continue
            rewards.append(next_bps + MAKER_OFFSET_BPS - MAKER_COST_BPS)
            continue
        rewards.append(None)
    return tuple(rewards)


def _survival_status(
    *,
    test_state_count: int,
    fill_count: int,
    fill_rate: float,
    all_state_reward_bps: float,
    adverse_fill_rate: float,
) -> str:
    if test_state_count == 0:
        return "maker_fill_no_test_state"
    if fill_count < 30:
        return "maker_fill_too_few_fills"
    if fill_rate < 0.02:
        return "maker_fill_too_rare"
    if all_state_reward_bps > 0.0 and adverse_fill_rate < 0.55:
        return "maker_fill_survival_candidate"
    if all_state_reward_bps > 0.0:
        return "maker_fill_tail_or_selection_watch"
    if adverse_fill_rate >= 0.60:
        return "maker_adverse_selection_blocked"
    return "maker_fill_rejects_policy"


def _survival_score(
    *,
    status: str,
    fill_rate: float,
    filled_mark_reward_bps: float,
    all_state_reward_bps: float,
    adverse_fill_rate: float,
    optimistic_net_bps: float,
) -> float:
    base = {
        "maker_fill_survival_candidate": 120.0,
        "maker_fill_tail_or_selection_watch": 70.0,
        "maker_fill_too_rare": 20.0,
        "maker_fill_too_few_fills": 10.0,
        "maker_fill_rejects_policy": -40.0,
        "maker_adverse_selection_blocked": -90.0,
        "maker_fill_no_test_state": -120.0,
    }.get(status, 0.0)
    return (
        base
        + all_state_reward_bps * 30.0
        + filled_mark_reward_bps * 5.0
        + fill_rate * 50.0
        + max(optimistic_net_bps, 0.0) * 20.0
        - adverse_fill_rate * 80.0
    )


def _reason(status: str) -> str:
    if status == "maker_fill_survival_candidate":
        return "passive fill proxy keeps positive reward without excessive adverse fills"
    if status == "maker_fill_tail_or_selection_watch":
        return "all-state reward is positive, but adverse fill rate is high"
    if status == "maker_fill_too_rare":
        return "state appears too rarely filled by the passive offset proxy"
    if status == "maker_fill_too_few_fills":
        return "not enough filled passive-order proxy samples"
    if status == "maker_adverse_selection_blocked":
        return "passive fills are mostly adverse after the mark crosses the order"
    if status == "maker_fill_rejects_policy":
        return "optimistic maker replay does not survive the passive fill proxy"
    return "no usable test-state samples"


def _next_step(status: str) -> str:
    if status == "maker_fill_survival_candidate":
        return "repeat with real best bid/ask, queue position, partial fills, cancel rule, and post-fill horizon"
    if status == "maker_fill_tail_or_selection_watch":
        return "inspect tails and split fill timing before any maker policy promotion"
    if status in {"maker_fill_too_rare", "maker_fill_too_few_fills"}:
        return "extend snapshots or relax/tune the passive offset before judging"
    if status == "maker_adverse_selection_blocked":
        return "reject or require a cancel rule before the order is crossed"
    return "do not promote this maker policy under the current fill proxy"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_lob_maker_fill_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_lob_maker_fill_survival.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_lob_maker_fill_survival_rows(dataset_dir=args.dataset_dir, purge_days=args.purge_days)
    write_lob_maker_fill_survival_csv(rows, output_path=args.output_path)
    write_lob_maker_fill_survival_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.survival_status, row.candidate_id, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
