from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.policy_learning.current_action_preference_candidates import _context_family, _read_rows


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent
OOS_PATH = LOCAL_ROOT / "current_action_preference_oos_check.csv"
CANDIDATES_PATH = LOCAL_ROOT / "current_action_preference_candidates.csv"
LANE_REVIEW_PATH = ROOT / "current_symbol_lane_split_review.csv"
LANE_TICKETS_PATH = ROOT / "current_symbol_lane_paper_tickets.csv"


@dataclass(frozen=True)
class PolicyExpansionTarget:
    target_id: str
    seed_id: str
    seed_type: str
    context: str
    source_asset: str
    target_asset: str
    target_opportunity: str
    action: str
    support_state: str
    lane_priority: float
    expansion_score: float
    decision: str
    reason: str
    next_step: str


def build_policy_expansion_targets(
    *,
    oos_path: Path = OOS_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    lane_review_path: Path = LANE_REVIEW_PATH,
    lane_tickets_path: Path = LANE_TICKETS_PATH,
) -> tuple[PolicyExpansionTarget, ...]:
    seeds = _seed_rows(oos_path=oos_path, candidates_path=candidates_path)
    lane_rows = tuple(row for row in _lane_rows(lane_review_path, lane_tickets_path) if _action_from_lane(row))
    output: list[PolicyExpansionTarget] = []
    seen: set[tuple[str, str, str]] = set()
    for seed in seeds:
        for lane in lane_rows:
            if not _matches_seed(seed, lane):
                continue
            key = (seed["seed_id"], lane.get("symbol", ""), lane.get("opportunity", ""))
            if key in seen:
                continue
            seen.add(key)
            output.append(_target_from_seed_lane(seed=seed, lane=lane))
    return tuple(sorted(output, key=lambda row: row.expansion_score, reverse=True))


def write_policy_expansion_targets_csv(
    rows: tuple[PolicyExpansionTarget, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "target_id",
                "seed_id",
                "seed_type",
                "context",
                "source_asset",
                "target_asset",
                "target_opportunity",
                "action",
                "support_state",
                "lane_priority",
                "expansion_score",
                "decision",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.target_id,
                    row.seed_id,
                    row.seed_type,
                    row.context,
                    row.source_asset,
                    row.target_asset,
                    row.target_opportunity,
                    row.action,
                    row.support_state,
                    f"{row.lane_priority:.8f}",
                    f"{row.expansion_score:.8f}",
                    row.decision,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_policy_expansion_targets_md(
    rows: tuple[PolicyExpansionTarget, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Policy Expansion Targets\n\n")
        handle.write(
            "This expands paper-supported action preferences into adjacent current lanes. "
            "It is not a model, not a strategy implementation, and not a trade list.\n\n"
        )
        handle.write(
            "| target | seed | context | source | target | action | support | score | decision | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.target_id} | "
                f"{row.seed_id} | "
                f"{row.context} | "
                f"{row.source_asset or 'family'} | "
                f"{row.target_asset}/{row.target_opportunity} | "
                f"{row.action} | "
                f"{row.support_state} | "
                f"{row.expansion_score:.2f} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A row means a currently observed lane resembles a paper-supported action preference. "
            "The next work is to collect new labels and execution evidence, not to hard-code the preference.\n"
        )
    return output_path


def _seed_rows(*, oos_path: Path, candidates_path: Path) -> tuple[dict[str, str], ...]:
    seeds: list[dict[str, str]] = []
    for row in _read_rows(oos_path):
        if row.get("decision") not in {"oos_supported_action_preference", "mixed_oos_action_preference"}:
            continue
        seeds.append(
            {
                "seed_id": row.get("candidate_id", ""),
                "seed_type": row.get("decision", ""),
                "context": row.get("context", ""),
                "asset": row.get("asset", ""),
                "action": row.get("action", ""),
                "score": row.get("oos_score", ""),
                "evidence": row.get("evidence", ""),
            }
        )
    for row in _read_rows(candidates_path):
        if row.get("decision") != "collect_more_labels":
            continue
        if row.get("context") == "unclassified":
            continue
        if _float(row.get("score")) < 40.0 or _float(row.get("mean_reward_bps")) <= 20.0:
            continue
        seeds.append(
            {
                "seed_id": row.get("candidate_id", ""),
                "seed_type": "high_reward_seed_needs_repeat",
                "context": row.get("context", ""),
                "asset": row.get("asset", ""),
                "action": row.get("action", ""),
                "score": row.get("score", ""),
                "evidence": row.get("evidence", ""),
            }
        )
    return tuple(seeds)


def _matches_seed(seed: dict[str, str], lane: dict[str, str]) -> bool:
    action = _action_from_lane(lane)
    if action != seed.get("action"):
        return False
    lane_context = _context_family(lane.get("opportunity", ""))
    if lane_context != seed.get("context"):
        return False
    source_asset = seed.get("asset", "")
    if source_asset and source_asset == lane.get("symbol", ""):
        return False
    if lane.get("support_state") in {"feature_source_blocked", "mechanics_unvalidated"}:
        return False
    return True


def _target_from_seed_lane(*, seed: dict[str, str], lane: dict[str, str]) -> PolicyExpansionTarget:
    lane_priority = _lane_priority(lane)
    seed_score = _float(seed.get("score"))
    support_bonus = _support_bonus(lane.get("support_state", ""))
    expansion_score = lane_priority * 0.45 + seed_score * 0.45 + support_bonus
    symbol = lane.get("symbol", "")
    opportunity = lane.get("opportunity", "")
    target_id = f"{symbol.lower()}_{opportunity}_from_{seed.get('seed_id', '')}"
    decision = _decision(expansion_score=expansion_score, seed_type=seed.get("seed_type", ""))
    return PolicyExpansionTarget(
        target_id=target_id,
        seed_id=seed.get("seed_id", ""),
        seed_type=seed.get("seed_type", ""),
        context=seed.get("context", ""),
        source_asset=seed.get("asset", ""),
        target_asset=symbol,
        target_opportunity=opportunity,
        action=seed.get("action", ""),
        support_state=lane.get("support_state", ""),
        lane_priority=lane_priority,
        expansion_score=expansion_score,
        decision=decision,
        reason=(
            f"seed={seed.get('seed_id', '')}; seed_evidence={seed.get('evidence', '')}; "
            f"lane_support={lane.get('support_state', '')}"
        ),
        next_step=_next_step(
            decision=decision,
            symbol=symbol,
            opportunity=opportunity,
            context=seed.get("context", ""),
            action=seed.get("action", ""),
        ),
    )


def _action_from_lane(row: dict[str, str]) -> str:
    lane_bias = row.get("lane_bias", "")
    side = row.get("side", "")
    decision = row.get("decision", "")
    if lane_bias == "long" or side.startswith("long_") or side.startswith("paper_long") or decision == "paper_long":
        return "paper_long"
    if lane_bias == "short" or side.startswith("short_") or side.startswith("paper_short") or decision == "paper_short":
        return "paper_short"
    return ""


def _lane_rows(lane_review_path: Path, lane_tickets_path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    rows.extend(_read_rows(lane_review_path))
    for row in _read_rows(lane_tickets_path):
        rows.append(
            {
                "symbol": row.get("symbol", "") or row.get("asset", ""),
                "opportunity": row.get("opportunity", ""),
                "lane_bias": row.get("lane_bias", ""),
                "side": row.get("decision", ""),
                "decision": row.get("decision", ""),
                "support_state": row.get("support_state", ""),
                "priority_score": _ticket_priority(row),
            }
        )
    return tuple(rows)


def _ticket_priority(row: dict[str, str]) -> str:
    support_state = row.get("support_state", "")
    base = {
        "paper_execution_gated": 90.0,
        "paper_1h_supported": 86.0,
        "paper_15m_supported": 82.0,
        "pending_label": 75.0,
        "unlabeled": 72.0,
    }.get(support_state, 70.0)
    if row.get("decision") in {"paper_long", "paper_short"}:
        base += 4.0
    return f"{base:.8f}"


def _lane_priority(row: dict[str, str]) -> float:
    return _float(row.get("priority_score"))


def _decision(*, expansion_score: float, seed_type: str) -> str:
    if seed_type == "high_reward_seed_needs_repeat":
        return "repeat_seed_before_expansion"
    if expansion_score >= 95.0:
        return "expand_supported_preference_now"
    if expansion_score >= 75.0:
        return "collect_expansion_labels"
    return "watch_expansion_target"


def _next_step(*, decision: str, symbol: str, opportunity: str, context: str, action: str) -> str:
    if decision == "expand_supported_preference_now":
        return (
            f"open a small paper label for {symbol}/{opportunity} as {action}, then compare reward to "
            f"the existing {context} preference"
        )
    if decision == "repeat_seed_before_expansion":
        return (
            f"repeat-label {symbol}/{opportunity} before using this high-reward seed as a broader policy preference"
        )
    if decision == "collect_expansion_labels":
        return f"collect fresh labels for {symbol}/{opportunity} and include fill, funding, stop, and failure regime"
    return f"keep {symbol}/{opportunity} on the expansion watch list until stronger support appears"


def _support_bonus(support_state: str) -> float:
    return {
        "paper_execution_gated": 12.0,
        "paper_cost_supported": 10.0,
        "paper_15m_supported": 8.0,
        "pending_label": 5.0,
        "unlabeled": 3.0,
    }.get(support_state, 0.0)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-path", type=Path, default=OOS_PATH)
    parser.add_argument("--candidates-path", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--lane-review-path", type=Path, default=LANE_REVIEW_PATH)
    parser.add_argument("--lane-tickets-path", type=Path, default=LANE_TICKETS_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_policy_expansion_targets.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_policy_expansion_targets.md")
    args = parser.parse_args()
    rows = build_policy_expansion_targets(
        oos_path=args.oos_path,
        candidates_path=args.candidates_path,
        lane_review_path=args.lane_review_path,
        lane_tickets_path=args.lane_tickets_path,
    )
    write_policy_expansion_targets_csv(rows, output_path=args.output_path)
    write_policy_expansion_targets_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.target_id, row.decision, f"{row.expansion_score:.4f}")


if __name__ == "__main__":
    main()
