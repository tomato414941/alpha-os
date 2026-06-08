from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parent
DATASET_PATH = LOCAL_ROOT / "current_observation_action_reward_dataset.csv"


@dataclass(frozen=True)
class PolicyContextFrontierRow:
    context: str
    records: int
    initial_records: int
    repeat_records: int
    cost_known_records: int
    hit_rate: float
    mean_reward_bps: float
    repeat_hit_rate: float
    repeat_mean_reward_bps: float
    worst_reward_bps: float
    best_reward_bps: float
    frontier_score: float
    decision: str
    evidence: str
    next_step: str


def build_policy_context_frontier(
    dataset_path: Path = DATASET_PATH,
) -> tuple[PolicyContextFrontierRow, ...]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(dataset_path):
        context = row.get("context", "")
        if context:
            grouped.setdefault(context, []).append(row)
    rows = tuple(_frontier_row(context=context, records=records) for context, records in grouped.items())
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_policy_context_frontier_csv(
    rows: tuple[PolicyContextFrontierRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "context",
                "records",
                "initial_records",
                "repeat_records",
                "cost_known_records",
                "hit_rate",
                "mean_reward_bps",
                "repeat_hit_rate",
                "repeat_mean_reward_bps",
                "worst_reward_bps",
                "best_reward_bps",
                "frontier_score",
                "decision",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.context,
                    row.records,
                    row.initial_records,
                    row.repeat_records,
                    row.cost_known_records,
                    f"{row.hit_rate:.8f}",
                    f"{row.mean_reward_bps:.8f}",
                    f"{row.repeat_hit_rate:.8f}",
                    f"{row.repeat_mean_reward_bps:.8f}",
                    f"{row.worst_reward_bps:.8f}",
                    f"{row.best_reward_bps:.8f}",
                    f"{row.frontier_score:.8f}",
                    row.decision,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_policy_context_frontier_md(
    rows: tuple[PolicyContextFrontierRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Policy Context Frontier\n\n")
        handle.write(
            "This summarizes observation/action/reward records by context. "
            "It is a prioritization board for alpha exploration, not a trained policy.\n\n"
        )
        handle.write(
            "| context | decision | records | repeat | hit | mean | repeat mean | worst | best | score | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.context} | "
                f"{row.decision} | "
                f"{row.records} | "
                f"{row.repeat_records} | "
                f"{row.hit_rate:.3f} | "
                f"{row.mean_reward_bps:.2f} | "
                f"{row.repeat_mean_reward_bps:.2f} | "
                f"{row.worst_reward_bps:.2f} | "
                f"{row.best_reward_bps:.2f} | "
                f"{row.frontier_score:.2f} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`expand_context_now` means repeat samples are positive enough to open more labels. "
            "`expand_with_failure_split` means repeat samples are positive, but the worst loss is large enough "
            "that failure regimes must be separated before increasing confidence. "
            "`collect_repeat_samples` means the context is promising but under-tested. "
            "`shrink_or_rework_context` means the current paper evidence is net negative and should not receive more blind expansion.\n"
        )
    return output_path


def _frontier_row(*, context: str, records: list[dict[str, str]]) -> PolicyContextFrontierRow:
    rewards = tuple(_float(row.get("reward_used_bps")) for row in records)
    repeat_rewards = tuple(_float(row.get("reward_used_bps")) for row in records if row.get("split") == "repeat")
    initial_records = sum(1 for row in records if row.get("split") == "initial")
    cost_known_records = sum(1 for row in records if row.get("cost_known") == "yes")
    hit_rate = _hit_rate(rewards)
    repeat_hit_rate = _hit_rate(repeat_rewards)
    mean_reward = _mean(rewards)
    repeat_mean = _mean(repeat_rewards)
    worst = min(rewards)
    best = max(rewards)
    frontier_score = _frontier_score(
        records=len(records),
        repeat_records=len(repeat_rewards),
        cost_known_records=cost_known_records,
        hit_rate=hit_rate,
        mean_reward=mean_reward,
        repeat_hit_rate=repeat_hit_rate,
        repeat_mean=repeat_mean,
        worst=worst,
    )
    decision = _decision(
        records=len(records),
        repeat_records=len(repeat_rewards),
        hit_rate=hit_rate,
        mean_reward=mean_reward,
        repeat_hit_rate=repeat_hit_rate,
        repeat_mean=repeat_mean,
        worst=worst,
    )
    return PolicyContextFrontierRow(
        context=context,
        records=len(records),
        initial_records=initial_records,
        repeat_records=len(repeat_rewards),
        cost_known_records=cost_known_records,
        hit_rate=hit_rate,
        mean_reward_bps=mean_reward,
        repeat_hit_rate=repeat_hit_rate,
        repeat_mean_reward_bps=repeat_mean,
        worst_reward_bps=worst,
        best_reward_bps=best,
        frontier_score=frontier_score,
        decision=decision,
        evidence=(
            f"records={len(records)} initial={initial_records} repeat={len(repeat_rewards)} "
            f"cost_known={cost_known_records} hit={hit_rate:.3f} repeat_hit={repeat_hit_rate:.3f}"
        ),
        next_step=_next_step(
            context=context,
            decision=decision,
            repeat_records=len(repeat_rewards),
            repeat_mean=repeat_mean,
            worst=worst,
        ),
    )


def _decision(
    *,
    records: int,
    repeat_records: int,
    hit_rate: float,
    mean_reward: float,
    repeat_hit_rate: float,
    repeat_mean: float,
    worst: float,
) -> str:
    if repeat_records >= 2 and repeat_hit_rate >= 0.6 and repeat_mean > 20.0 and mean_reward > 20.0:
        if worst < -200.0:
            return "expand_with_failure_split"
        return "expand_context_now"
    if records >= 3 and mean_reward > 20.0 and hit_rate >= 0.6:
        return "collect_repeat_samples"
    if records >= 3 and (mean_reward < 0.0 or (repeat_records >= 2 and repeat_mean < 0.0)):
        return "shrink_or_rework_context"
    return "watch_context"


def _next_step(*, context: str, decision: str, repeat_records: int, repeat_mean: float, worst: float) -> str:
    if decision == "expand_context_now":
        return f"open more {context} paper labels across fresh assets, venues, and failure regimes"
    if decision == "expand_with_failure_split":
        return f"open more {context} labels, but split the failure regime before increasing size or confidence"
    if decision == "collect_repeat_samples":
        return f"collect repeat {context} samples before treating the context as durable"
    if decision == "shrink_or_rework_context":
        return f"stop blind {context} expansion; isolate why repeat_mean={repeat_mean:.2f} and worst={worst:.2f}"
    return f"keep {context} on watch until repeat sample count exceeds {repeat_records}"


def _frontier_score(
    *,
    records: int,
    repeat_records: int,
    cost_known_records: int,
    hit_rate: float,
    mean_reward: float,
    repeat_hit_rate: float,
    repeat_mean: float,
    worst: float,
) -> float:
    repeat_weight = min(repeat_records / 4.0, 1.0)
    cost_weight = min(cost_known_records / max(records, 1), 1.0)
    downside_penalty = max(-worst / 10.0, 0.0)
    return (
        mean_reward * 0.25
        + repeat_mean * 0.55 * repeat_weight
        + hit_rate * 20.0
        + repeat_hit_rate * 30.0 * repeat_weight
        + min(records * 0.5, 12.0)
        + cost_weight * 8.0
        - downside_penalty
    )


def _sort_key(row: PolicyContextFrontierRow) -> tuple[int, float]:
    decision_rank = {
        "expand_context_now": 3,
        "expand_with_failure_split": 3,
        "collect_repeat_samples": 2,
        "watch_context": 1,
        "shrink_or_rework_context": 0,
    }.get(row.decision, 0)
    return (decision_rank, row.frontier_score)


def _hit_rate(rewards: tuple[float, ...]) -> float:
    if not rewards:
        return 0.0
    return sum(1 for reward in rewards if reward > 0.0) / len(rewards)


def _mean(rewards: tuple[float, ...]) -> float:
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_policy_context_frontier.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_policy_context_frontier.md")
    args = parser.parse_args()
    rows = build_policy_context_frontier(dataset_path=args.dataset_path)
    write_policy_context_frontier_csv(rows, output_path=args.output_path)
    write_policy_context_frontier_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.context, row.decision, f"{row.frontier_score:.4f}", row.next_step)


if __name__ == "__main__":
    main()
