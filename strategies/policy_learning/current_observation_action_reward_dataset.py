from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.policy_learning.current_action_preference_candidates import _context_family


LOCAL_ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = LOCAL_ROOT / "current_policy_learning_samples.csv"

INITIAL_SOURCES = {"paper", "symbol_lane"}
REPEAT_SOURCES = {"repeat", "second_repeat", "symbol_lane_repeat"}


@dataclass(frozen=True)
class ObservationActionRewardRecord:
    record_id: str
    split: str
    source: str
    asset: str
    venue: str
    context: str
    opportunity: str
    observation_key: str
    action: str
    action_side: str
    reward_bps: float
    cost_adjusted_reward_bps: str
    reward_used_bps: float
    reward_status: str
    cost_known: str
    terminal: str
    elapsed_minutes: str
    next_step: str


def build_observation_action_reward_dataset(
    samples_path: Path = SAMPLES_PATH,
) -> tuple[ObservationActionRewardRecord, ...]:
    records = tuple(_record_from_sample(row) for row in _read_rows(samples_path) if _usable(row))
    return tuple(sorted(records, key=_sort_key, reverse=True))


def write_observation_action_reward_dataset_csv(
    rows: tuple[ObservationActionRewardRecord, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "record_id",
                "split",
                "source",
                "asset",
                "venue",
                "context",
                "opportunity",
                "observation_key",
                "action",
                "action_side",
                "reward_bps",
                "cost_adjusted_reward_bps",
                "reward_used_bps",
                "reward_status",
                "cost_known",
                "terminal",
                "elapsed_minutes",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.record_id,
                    row.split,
                    row.source,
                    row.asset,
                    row.venue,
                    row.context,
                    row.opportunity,
                    row.observation_key,
                    row.action,
                    row.action_side,
                    f"{row.reward_bps:.8f}",
                    row.cost_adjusted_reward_bps,
                    f"{row.reward_used_bps:.8f}",
                    row.reward_status,
                    row.cost_known,
                    row.terminal,
                    row.elapsed_minutes,
                    row.next_step,
                )
            )
    return output_path


def write_observation_action_reward_dataset_md(
    rows: tuple[ObservationActionRewardRecord, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = _counts(row.split for row in rows)
    context_counts = _counts(row.context for row in rows)
    cost_known = sum(1 for row in rows if row.cost_known == "yes")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Observation Action Reward Dataset\n\n")
        handle.write(
            "This is a dataset-shaped view over paper outcomes for policy research. "
            "It is not a trained policy, not a simulator, and not a trade list.\n\n"
        )
        handle.write("## Coverage\n\n")
        handle.write(f"- records: {len(rows)}\n")
        handle.write(f"- initial split: {split_counts.get('initial', 0)}\n")
        handle.write(f"- repeat split: {split_counts.get('repeat', 0)}\n")
        handle.write(f"- cost-known records: {cost_known}\n")
        handle.write(f"- contexts: {len(context_counts)}\n\n")
        handle.write("## Top Records\n\n")
        handle.write(
            "| record | split | asset | context | action | reward used | status | cost known | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.record_id} | "
                f"{row.split} | "
                f"{row.asset} | "
                f"{row.context} | "
                f"{row.action} | "
                f"{row.reward_used_bps:.2f} | "
                f"{row.reward_status} | "
                f"{row.cost_known} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The useful next step is not to train immediately. The dataset first needs "
            "a cleaner observation state, explicit action constraints, stop/adverse-excursion "
            "fields, and a cost/fill model that is not inferred from paper mark moves alone.\n"
        )
    return output_path


def _record_from_sample(row: dict[str, str]) -> ObservationActionRewardRecord:
    cost_adjusted = row.get("cost_adjusted_reward_bps", "")
    reward = _float(row.get("reward_bps"))
    reward_used = _float(cost_adjusted) if cost_adjusted else reward
    opportunity = row.get("opportunity", "")
    return ObservationActionRewardRecord(
        record_id=row.get("sample_id", ""),
        split=_split(row.get("source", "")),
        source=row.get("source", ""),
        asset=row.get("asset", ""),
        venue=row.get("venue", ""),
        context=_context_family(opportunity),
        opportunity=opportunity,
        observation_key=_observation_key(row),
        action=row.get("action", ""),
        action_side=_action_side(row.get("action", "")),
        reward_bps=reward,
        cost_adjusted_reward_bps=cost_adjusted,
        reward_used_bps=reward_used,
        reward_status=row.get("reward_status", ""),
        cost_known="yes" if cost_adjusted else "no",
        terminal="yes" if row.get("checkpoint_status") == "ready" else "no",
        elapsed_minutes=row.get("elapsed_minutes", ""),
        next_step=row.get("next_step", ""),
    )


def _usable(row: dict[str, str]) -> bool:
    return (
        row.get("checkpoint_status") == "ready"
        and row.get("action", "") in {"paper_long", "paper_short"}
        and row.get("reward_status", "") not in {"pending", "missing_mark"}
    )


def _split(source: str) -> str:
    if source in INITIAL_SOURCES:
        return "initial"
    if source in REPEAT_SOURCES:
        return "repeat"
    return "other"


def _observation_key(row: dict[str, str]) -> str:
    parts = (
        row.get("asset", ""),
        _context_family(row.get("opportunity", "")),
        row.get("venue", "") or "any_venue",
    )
    return "/".join(part for part in parts if part)


def _action_side(action: str) -> str:
    if action == "paper_long":
        return "long"
    if action == "paper_short":
        return "short"
    return "other"


def _sort_key(row: ObservationActionRewardRecord) -> tuple[int, int, float]:
    split_rank = {"repeat": 2, "initial": 1}.get(row.split, 0)
    status_rank = {
        "cost_adjusted_win": 4,
        "mark_win_without_cost": 3,
        "depth_too_thin_for_probe": 2,
        "cost_adjusted_edge_failed": 1,
        "mark_loss": 0,
    }.get(row.reward_status, 0)
    return (split_rank, status_rank, row.reward_used_bps)


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    parser.add_argument("--samples-path", type=Path, default=SAMPLES_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_observation_action_reward_dataset.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=LOCAL_ROOT / "current_observation_action_reward_dataset.md",
    )
    args = parser.parse_args()
    rows = build_observation_action_reward_dataset(samples_path=args.samples_path)
    write_observation_action_reward_dataset_csv(rows, output_path=args.output_path)
    write_observation_action_reward_dataset_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.record_id, row.split, row.asset, row.context, row.action, f"{row.reward_used_bps:.4f}")


if __name__ == "__main__":
    main()
