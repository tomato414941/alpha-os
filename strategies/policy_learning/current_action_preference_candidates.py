from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = LOCAL_ROOT / "current_policy_learning_samples.csv"


@dataclass(frozen=True)
class ActionPreferenceCandidate:
    candidate_id: str
    scope: str
    context: str
    asset: str
    action: str
    samples: int
    hit_rate: float
    mean_reward_bps: float
    median_reward_bps: float
    best_reward_bps: float
    worst_reward_bps: float
    score: float
    decision: str
    evidence: str
    next_step: str


def build_action_preference_candidates(
    samples_path: Path = SAMPLES_PATH,
) -> tuple[ActionPreferenceCandidate, ...]:
    rows = tuple(row for row in _read_rows(samples_path) if _usable_sample(row))
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        family = _context_family(row.get("opportunity", ""))
        action = row.get("action", "")
        asset = row.get("asset", "")
        _append_group(groups, ("family_action", family, "", action), row)
        if asset:
            _append_group(groups, ("asset_family_action", family, asset, action), row)
    candidates = tuple(_candidate_from_group(key, group) for key, group in groups.items())
    return tuple(sorted(candidates, key=_sort_key, reverse=True))


def write_action_preference_candidates_csv(
    rows: tuple[ActionPreferenceCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "scope",
                "context",
                "asset",
                "action",
                "samples",
                "hit_rate",
                "mean_reward_bps",
                "median_reward_bps",
                "best_reward_bps",
                "worst_reward_bps",
                "score",
                "decision",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.scope,
                    row.context,
                    row.asset,
                    row.action,
                    row.samples,
                    f"{row.hit_rate:.8f}",
                    f"{row.mean_reward_bps:.8f}",
                    f"{row.median_reward_bps:.8f}",
                    f"{row.best_reward_bps:.8f}",
                    f"{row.worst_reward_bps:.8f}",
                    f"{row.score:.8f}",
                    row.decision,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_action_preference_candidates_md(
    rows: tuple[ActionPreferenceCandidate, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Action Preference Candidates\n\n")
        handle.write(
            "This aggregates RL-shaped paper samples into context/action preferences. "
            "It is not a trained policy and not a deployable strategy.\n\n"
        )
        handle.write(
            "| candidate | scope | context | asset | action | samples | hit | mean | median | worst | score | decision |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.scope} | "
                f"{row.context} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.samples} | "
                f"{row.hit_rate:.3f} | "
                f"{row.mean_reward_bps:.2f} | "
                f"{row.median_reward_bps:.2f} | "
                f"{row.worst_reward_bps:.2f} | "
                f"{row.score:.2f} | "
                f"{row.decision} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A candidate means the current paper logs suggest an action preference for a context. "
            "It still needs a leakage-safe split, more samples, explicit cost/fill assumptions, and a "
            "clear policy evaluation protocol before it can influence live decisions.\n"
        )
    return output_path


def _append_group(
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]],
    key: tuple[str, str, str, str],
    row: dict[str, str],
) -> None:
    groups.setdefault(key, []).append(row)


def _candidate_from_group(
    key: tuple[str, str, str, str],
    group: list[dict[str, str]],
) -> ActionPreferenceCandidate:
    scope, context, asset, action = key
    rewards = tuple(_sample_reward(row) for row in group)
    wins = tuple(reward for reward in rewards if reward > 0.0)
    mean_reward = sum(rewards) / len(rewards)
    hit_rate = len(wins) / len(rewards)
    worst_reward = min(rewards)
    best_reward = max(rewards)
    median_reward = median(rewards)
    raw_score = mean_reward + hit_rate * 25.0 + min(len(rewards) * 2.5, 15.0) + min(median_reward, 20.0)
    reliability = min(len(rewards) / 3.0, 1.0)
    score = raw_score * reliability
    decision = _decision(samples=len(rewards), hit_rate=hit_rate, mean_reward=mean_reward)
    label = asset.lower() + "_" if asset else ""
    candidate_id = f"{label}{context}_{action}"
    return ActionPreferenceCandidate(
        candidate_id=candidate_id,
        scope=scope,
        context=context,
        asset=asset,
        action=action,
        samples=len(rewards),
        hit_rate=hit_rate,
        mean_reward_bps=mean_reward,
        median_reward_bps=median_reward,
        best_reward_bps=best_reward,
        worst_reward_bps=worst_reward,
        score=score,
        decision=decision,
        evidence=_evidence(group, rewards),
        next_step=_next_step(
            decision=decision,
            context=context,
            asset=asset,
            action=action,
            samples=len(rewards),
        ),
    )


def _decision(*, samples: int, hit_rate: float, mean_reward: float) -> str:
    if samples >= 3 and hit_rate >= 0.6 and mean_reward > 10.0:
        return "promote_action_preference_candidate"
    if samples >= 2 and hit_rate >= 0.5 and mean_reward > 5.0:
        return "watch_action_preference_candidate"
    if samples >= 2 and hit_rate <= 0.4 and mean_reward < -5.0:
        return "reject_action_preference_candidate"
    return "collect_more_labels"


def _next_step(*, decision: str, context: str, asset: str, action: str, samples: int) -> str:
    target = f"{asset} {context}" if asset else context
    if decision == "promote_action_preference_candidate":
        return (
            f"turn {target}/{action} into a leakage-safe policy split and rerun with explicit costs, "
            "fills, and failure regimes"
        )
    if decision == "watch_action_preference_candidate":
        return f"collect more {target}/{action} samples before treating this as a policy rule"
    if decision == "reject_action_preference_candidate":
        return f"deprioritize {target}/{action} unless a new independent feature changes the reward profile"
    return f"collect more labels for {target}/{action}; current sample count is {samples}"


def _evidence(group: list[dict[str, str]], rewards: tuple[float, ...]) -> str:
    ids = ", ".join(row.get("sample_id", "") for row in group[:5])
    return (
        f"samples={len(group)}, reward_range={min(rewards):.2f}..{max(rewards):.2f}, "
        f"examples={ids}"
    )


def _usable_sample(row: dict[str, str]) -> bool:
    if row.get("checkpoint_status") != "ready":
        return False
    if row.get("reward_status") in {"pending", "context_only", "missing_mark"}:
        return False
    if row.get("action") not in {"paper_long", "paper_short"}:
        return False
    return bool(row.get("reward_bps") or row.get("cost_adjusted_reward_bps"))


def _sample_reward(row: dict[str, str]) -> float:
    return _float(row.get("cost_adjusted_reward_bps") or row.get("reward_bps"))


def _context_family(opportunity: str) -> str:
    value = opportunity.lower()
    family_tokens = (
        ("microstructure_flow", ("microstructure", "l2_imbalance", "flow_probe")),
        ("volume_price_dislocation", ("volume_price_dislocation", "volume-dislocation")),
        ("repeat_execution", ("repeat_execution_gate", "repeat-execution")),
        ("liquidation_intensity", ("liquidation_intensity", "liquidation")),
        ("intraday_derivatives", ("intraday_derivatives", "long_short_ratio", "premium_close")),
        ("protocol_fee", ("protocol_fee", "fee_growth")),
        ("token_unlock", ("unlock",)),
        ("event_pressure", ("event_pressure",)),
        ("event_crypto_hedge", ("event_crypto_hedge",)),
        ("news_event", ("news_event", "attention_price", "narrative_event", "institutional_flow")),
        ("wallet_entity_flow", ("wallet_flow", "wallet-entity-flow")),
        ("sector_rotation", ("sector_rotation", "sector-rotation")),
        ("options_volatility", ("volatility", "delta_hedge", "straddle")),
        ("basis_term_structure", ("basis",)),
        ("stablecoin_migration", ("stablecoin_migration",)),
        ("execution_edge", ("execution_edge", "maker_or_low_fee", "taker_small")),
    )
    for family, tokens in family_tokens:
        if any(token in value for token in tokens):
            return family
    return "unclassified"


def _sort_key(row: ActionPreferenceCandidate) -> tuple[int, float, int]:
    decision_rank = {
        "promote_action_preference_candidate": 4,
        "watch_action_preference_candidate": 3,
        "collect_more_labels": 2,
        "reject_action_preference_candidate": 1,
    }.get(row.decision, 0)
    return (decision_rank, row.score, row.samples)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=Path, default=SAMPLES_PATH)
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_action_preference_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_action_preference_candidates.md")
    args = parser.parse_args()
    rows = build_action_preference_candidates(samples_path=args.samples_path)
    write_action_preference_candidates_csv(rows, output_path=args.output_path)
    write_action_preference_candidates_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.candidate_id, row.decision, row.samples, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
