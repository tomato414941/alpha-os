from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PolicyLearningSample:
    sample_id: str
    source: str
    asset: str
    venue: str
    opportunity: str
    observation_summary: str
    action: str
    reward_bps: str
    cost_adjusted_reward_bps: str
    reward_status: str
    checkpoint_status: str
    elapsed_minutes: str
    next_step: str


OUTCOME_PATHS = (
    ("paper", ROOT / "current_paper_ticket_outcomes.csv"),
    ("broad_alpha_paper", ROOT / "current_broad_alpha_paper_outcomes.csv"),
    ("broad_alpha_repeat", ROOT / "current_broad_alpha_repeat_outcomes.csv"),
    ("ofi", ROOT / "event_flow" / "current_ofi_paper_outcomes.csv"),
    ("repeat", ROOT / "current_promoted_ticket_repeat_outcomes.csv"),
    ("second_repeat", ROOT / "current_second_promoted_ticket_repeat_outcomes.csv"),
    ("symbol_lane", ROOT / "current_symbol_lane_paper_outcomes.csv"),
    ("symbol_lane_repeat", ROOT / "current_symbol_lane_promoted_repeat_outcomes.csv"),
)

FILL_RISK_PATHS = (
    ROOT / "current_paper_ticket_fill_risk_check.csv",
    ROOT / "current_broad_alpha_paper_fill_risk_check.csv",
    ROOT / "current_broad_alpha_repeat_fill_risk_check.csv",
    ROOT / "event_flow" / "current_ofi_paper_fill_risk_check.csv",
    ROOT / "current_promoted_ticket_repeat_fill_risk_check.csv",
    ROOT / "current_second_promoted_ticket_repeat_fill_risk_check.csv",
    ROOT / "current_symbol_lane_paper_fill_risk_check.csv",
    ROOT / "current_symbol_lane_promoted_repeat_fill_risk_check.csv",
)


def build_policy_learning_samples(
    *,
    outcome_paths: tuple[tuple[str, Path], ...] = OUTCOME_PATHS,
    fill_risk_paths: tuple[Path, ...] = FILL_RISK_PATHS,
) -> tuple[PolicyLearningSample, ...]:
    fill_risk = _fill_risk_by_ticket(fill_risk_paths)
    samples: list[PolicyLearningSample] = []
    for source, path in outcome_paths:
        for row in _read_rows(path):
            samples.append(_sample_from_outcome(source=source, row=row, fill_risk=fill_risk))
    return tuple(sorted(samples, key=_sample_sort_key, reverse=True))


def write_policy_learning_samples_csv(rows: tuple[PolicyLearningSample, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_id",
                "source",
                "asset",
                "venue",
                "opportunity",
                "observation_summary",
                "action",
                "reward_bps",
                "cost_adjusted_reward_bps",
                "reward_status",
                "checkpoint_status",
                "elapsed_minutes",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.sample_id,
                    row.source,
                    row.asset,
                    row.venue,
                    row.opportunity,
                    row.observation_summary,
                    row.action,
                    row.reward_bps,
                    row.cost_adjusted_reward_bps,
                    row.reward_status,
                    row.checkpoint_status,
                    row.elapsed_minutes,
                    row.next_step,
                )
            )
    return output_path


def write_policy_learning_samples_md(
    rows: tuple[PolicyLearningSample, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Policy Learning Samples\n\n")
        handle.write(
            "This is an RL-shaped observation/action/reward view over current paper outcomes. "
            "It is not a trained model and not a trade list.\n\n"
        )
        handle.write(
            "| sample | source | asset | action | reward | cost-adjusted | status | checkpoint | observation | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.sample_id} | "
                f"{row.source} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.reward_bps} | "
                f"{row.cost_adjusted_reward_bps} | "
                f"{row.reward_status} | "
                f"{row.checkpoint_status} | "
                f"{_escape(row.observation_summary)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _sample_from_outcome(
    *,
    source: str,
    row: dict[str, str],
    fill_risk: dict[str, dict[str, str]],
) -> PolicyLearningSample:
    ticket_id = row.get("ticket_id", "")
    risk_row = fill_risk.get(ticket_id, {})
    reward = row.get("directional_return_bps", "")
    cost_adjusted = risk_row.get("estimated_net_after_cost_bps", "")
    reward_status = _reward_status(row=row, risk_row=risk_row)
    observation_summary = (
        f"{row.get('opportunity', '')}; "
        f"entry={row.get('entry_mark', '')}; "
        f"current={row.get('current_mark', '')}; "
        f"current_source={row.get('current_source', '')}"
    )
    return PolicyLearningSample(
        sample_id=ticket_id,
        source=source,
        asset=row.get("asset", ""),
        venue=row.get("venue", ""),
        opportunity=row.get("opportunity", ""),
        observation_summary=observation_summary,
        action=row.get("decision", ""),
        reward_bps=reward,
        cost_adjusted_reward_bps=cost_adjusted,
        reward_status=reward_status,
        checkpoint_status=row.get("checkpoint_status", ""),
        elapsed_minutes=row.get("elapsed_minutes", ""),
        next_step=risk_row.get("next_step") or row.get("next_step", ""),
    )


def _reward_status(*, row: dict[str, str], risk_row: dict[str, str]) -> str:
    if row.get("checkpoint_status") != "ready":
        return "pending"
    if row.get("decision") == "paper_observe":
        return "context_only"
    if row.get("outcome") == "missing_current_mark":
        return "missing_mark"
    risk_action = risk_row.get("risk_action", "")
    if risk_action == "cost_adjusted_paper_probe":
        return "cost_adjusted_win"
    if risk_action in {"cost_adjusted_edge_failed", "depth_too_thin_for_probe", "missing_execution_context"}:
        return risk_action
    if row.get("outcome") == "paper_mark_win":
        return "mark_win_without_cost"
    if row.get("outcome") == "paper_mark_loss":
        return "mark_loss"
    return row.get("outcome", "")


def _fill_risk_by_ticket(paths: tuple[Path, ...]) -> dict[str, dict[str, str]]:
    by_ticket: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in _read_rows(path):
            ticket_id = row.get("ticket_id", "")
            if ticket_id:
                by_ticket[ticket_id] = row
    return by_ticket


def _sample_sort_key(row: PolicyLearningSample) -> tuple[int, float]:
    status_rank = {
        "cost_adjusted_win": 5,
        "mark_win_without_cost": 4,
        "cost_adjusted_edge_failed": 3,
        "depth_too_thin_for_probe": 2,
        "mark_loss": 1,
    }.get(row.reward_status, 0)
    return (status_rank, _float(row.cost_adjusted_reward_bps or row.reward_bps))


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
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_policy_learning_samples.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_policy_learning_samples.md")
    args = parser.parse_args()
    rows = build_policy_learning_samples()
    write_policy_learning_samples_csv(rows, output_path=args.output_path)
    write_policy_learning_samples_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.sample_id, row.reward_status, row.asset, row.action, row.cost_adjusted_reward_bps or row.reward_bps)


if __name__ == "__main__":
    main()
