from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.policy_learning.current_action_preference_candidates import (
    _context_family,
    _read_rows,
    _sample_reward,
    _usable_sample,
)


LOCAL_ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = LOCAL_ROOT / "current_policy_learning_samples.csv"
CANDIDATES_PATH = LOCAL_ROOT / "current_action_preference_candidates.csv"

TRAIN_SOURCES = {"paper", "symbol_lane"}
TEST_SOURCES = {"repeat", "second_repeat", "symbol_lane_repeat"}


@dataclass(frozen=True)
class ActionPreferenceOosCheck:
    candidate_id: str
    context: str
    asset: str
    action: str
    train_samples: int
    train_hit_rate: float
    train_mean_reward_bps: float
    test_samples: int
    test_hit_rate: float
    test_mean_reward_bps: float
    oos_score: float
    decision: str
    evidence: str
    next_step: str


def build_action_preference_oos_checks(
    *,
    samples_path: Path = SAMPLES_PATH,
    candidates_path: Path = CANDIDATES_PATH,
) -> tuple[ActionPreferenceOosCheck, ...]:
    samples = tuple(row for row in _read_rows(samples_path) if _usable_sample(row))
    candidates = tuple(
        row
        for row in _read_rows(candidates_path)
        if row.get("decision") in {"promote_action_preference_candidate", "watch_action_preference_candidate"}
    )
    checks = tuple(_check_candidate(candidate, samples) for candidate in candidates)
    return tuple(sorted(checks, key=lambda row: _sort_key(row), reverse=True))


def write_action_preference_oos_checks_csv(
    rows: tuple[ActionPreferenceOosCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "context",
                "asset",
                "action",
                "train_samples",
                "train_hit_rate",
                "train_mean_reward_bps",
                "test_samples",
                "test_hit_rate",
                "test_mean_reward_bps",
                "oos_score",
                "decision",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.context,
                    row.asset,
                    row.action,
                    row.train_samples,
                    f"{row.train_hit_rate:.8f}",
                    f"{row.train_mean_reward_bps:.8f}",
                    row.test_samples,
                    f"{row.test_hit_rate:.8f}",
                    f"{row.test_mean_reward_bps:.8f}",
                    f"{row.oos_score:.8f}",
                    row.decision,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_action_preference_oos_checks_md(
    rows: tuple[ActionPreferenceOosCheck, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Action Preference OOS Check\n\n")
        handle.write(
            "This checks whether action preferences found in initial paper samples survive repeat samples. "
            "It is an OOS-shaped guardrail, not a final backtest or trained policy.\n\n"
        )
        handle.write(
            "| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.context} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.train_samples} | "
                f"{row.train_mean_reward_bps:.2f} | "
                f"{row.test_samples} | "
                f"{row.test_mean_reward_bps:.2f} | "
                f"{row.test_hit_rate:.3f} | "
                f"{row.oos_score:.2f} | "
                f"{row.decision} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A passing row means the same context/action preference has repeat-sample support. "
            "A failing row means the apparent edge is likely first-sample overfit, timing luck, "
            "or missing execution/friction modeling.\n"
        )
    return output_path


def _check_candidate(
    candidate: dict[str, str],
    samples: tuple[dict[str, str], ...],
) -> ActionPreferenceOosCheck:
    context = candidate.get("context", "")
    asset = candidate.get("asset", "")
    action = candidate.get("action", "")
    matched = tuple(
        sample
        for sample in samples
        if sample.get("action") == action
        and _context_family(sample.get("opportunity", "")) == context
        and (not asset or sample.get("asset") == asset)
    )
    train_rewards = tuple(_sample_reward(sample) for sample in matched if sample.get("source") in TRAIN_SOURCES)
    test_rewards = tuple(_sample_reward(sample) for sample in matched if sample.get("source") in TEST_SOURCES)
    train_hit = _hit_rate(train_rewards)
    test_hit = _hit_rate(test_rewards)
    train_mean = _mean(train_rewards)
    test_mean = _mean(test_rewards)
    oos_score = _oos_score(
        train_samples=len(train_rewards),
        train_hit_rate=train_hit,
        train_mean_reward=train_mean,
        test_samples=len(test_rewards),
        test_hit_rate=test_hit,
        test_mean_reward=test_mean,
    )
    decision = _decision(
        train_samples=len(train_rewards),
        test_samples=len(test_rewards),
        test_hit_rate=test_hit,
        test_mean_reward=test_mean,
    )
    return ActionPreferenceOosCheck(
        candidate_id=candidate.get("candidate_id", ""),
        context=context,
        asset=asset,
        action=action,
        train_samples=len(train_rewards),
        train_hit_rate=train_hit,
        train_mean_reward_bps=train_mean,
        test_samples=len(test_rewards),
        test_hit_rate=test_hit,
        test_mean_reward_bps=test_mean,
        oos_score=oos_score,
        decision=decision,
        evidence=(
            f"train={len(train_rewards)} mean={train_mean:.2f} hit={train_hit:.3f}; "
            f"repeat={len(test_rewards)} mean={test_mean:.2f} hit={test_hit:.3f}"
        ),
        next_step=_next_step(
            decision=decision,
            candidate_id=candidate.get("candidate_id", ""),
            context=context,
            asset=asset,
            action=action,
        ),
    )


def _decision(
    *,
    train_samples: int,
    test_samples: int,
    test_hit_rate: float,
    test_mean_reward: float,
) -> str:
    if train_samples < 2 or test_samples < 2:
        return "needs_repeat_oos"
    if test_hit_rate >= 0.5 and test_mean_reward > 5.0:
        return "oos_supported_action_preference"
    if test_hit_rate < 0.5 or test_mean_reward < 0.0:
        return "oos_failed_action_preference"
    return "mixed_oos_action_preference"


def _next_step(*, decision: str, candidate_id: str, context: str, asset: str, action: str) -> str:
    label = candidate_id or f"{asset} {context} {action}"
    if decision == "oos_supported_action_preference":
        return f"paper-check {label} with explicit stop, fill, and reward attribution before any model training"
    if decision == "oos_failed_action_preference":
        return f"deprioritize {label}; repeat samples do not support the initial action preference"
    if decision == "mixed_oos_action_preference":
        return f"split {label} by failure regime, venue, or execution mode before training"
    return f"collect repeat samples for {label} before treating it as an action preference"


def _oos_score(
    *,
    train_samples: int,
    train_hit_rate: float,
    train_mean_reward: float,
    test_samples: int,
    test_hit_rate: float,
    test_mean_reward: float,
) -> float:
    repeat_weight = min(test_samples / 3.0, 1.0)
    train_weight = min(train_samples / 3.0, 1.0)
    return (
        test_mean_reward * repeat_weight
        + test_hit_rate * 30.0
        + min(test_samples * 3.0, 12.0)
        + train_mean_reward * 0.15 * train_weight
        + train_hit_rate * 5.0
    )


def _hit_rate(rewards: tuple[float, ...]) -> float:
    if not rewards:
        return 0.0
    return sum(1 for reward in rewards if reward > 0.0) / len(rewards)


def _mean(rewards: tuple[float, ...]) -> float:
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def _sort_key(row: ActionPreferenceOosCheck) -> tuple[int, float, int]:
    rank = {
        "oos_supported_action_preference": 4,
        "mixed_oos_action_preference": 3,
        "needs_repeat_oos": 2,
        "oos_failed_action_preference": 1,
    }.get(row.decision, 0)
    return (rank, row.oos_score, row.test_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=Path, default=SAMPLES_PATH)
    parser.add_argument("--candidates-path", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_action_preference_oos_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_action_preference_oos_check.md")
    args = parser.parse_args()
    rows = build_action_preference_oos_checks(
        samples_path=args.samples_path,
        candidates_path=args.candidates_path,
    )
    write_action_preference_oos_checks_csv(rows, output_path=args.output_path)
    write_action_preference_oos_checks_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.candidate_id, row.decision, row.test_samples, f"{row.oos_score:.4f}")


if __name__ == "__main__":
    main()
