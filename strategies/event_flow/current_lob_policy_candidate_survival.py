from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LobPolicyCandidateSurvival:
    state_family: str
    signal_action: str
    execution_mode: str
    survival_status: str
    survival_score: float
    world_feature: str
    world_bucket: str
    world_net_bps: float
    world_hit_rate: float
    world_decision: str
    sequence_feature: str
    sequence_bucket: str
    sequence_net_bps: float
    sequence_hit_rate: float
    sequence_decision: str
    sequence_zero_cost_feature: str
    sequence_zero_cost_net_bps: float
    reason: str
    next_step: str


def build_lob_policy_candidate_survival_rows(
    *,
    world_path: Path = ROOT / "current_lob_execution_world_replay.csv",
    sequence_path: Path = ROOT / "current_lob_sequence_state_probe.csv",
) -> tuple[LobPolicyCandidateSurvival, ...]:
    world_rows = _best_rows(
        rows=_read_rows(world_path),
        feature_key="feature",
        action_key="signal_action",
        mode_key="execution_action",
        net_key="net_reward_bps",
        hit_key="hit_rate",
        decision_key="decision",
    )
    sequence_rows = _best_rows(
        rows=_read_rows(sequence_path),
        feature_key="feature",
        action_key="signal_action",
        mode_key="execution_mode",
        net_key="test_net_bps",
        hit_key="test_hit_rate",
        decision_key="decision",
    )
    sequence_zero_rows = _best_sequence_zero_cost_rows(_read_rows(sequence_path))
    keys = sorted(set(world_rows) | set(sequence_rows))
    output = tuple(
        _build_row(
            state_family=state_family,
            signal_action=signal_action,
            execution_mode=execution_mode,
            world=world_rows.get((state_family, signal_action, execution_mode), {}),
            sequence=sequence_rows.get((state_family, signal_action, execution_mode), {}),
            sequence_zero=sequence_zero_rows.get((state_family, signal_action), {}),
        )
        for state_family, signal_action, execution_mode in keys
        if execution_mode != "hold"
    )
    return tuple(sorted(output, key=lambda row: row.survival_score, reverse=True))


def write_lob_policy_candidate_survival_csv(
    rows: tuple[LobPolicyCandidateSurvival, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "state_family",
                "signal_action",
                "execution_mode",
                "survival_status",
                "survival_score",
                "world_feature",
                "world_bucket",
                "world_net_bps",
                "world_hit_rate",
                "world_decision",
                "sequence_feature",
                "sequence_bucket",
                "sequence_net_bps",
                "sequence_hit_rate",
                "sequence_decision",
                "sequence_zero_cost_feature",
                "sequence_zero_cost_net_bps",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.state_family,
                    row.signal_action,
                    row.execution_mode,
                    row.survival_status,
                    f"{row.survival_score:.8f}",
                    row.world_feature,
                    row.world_bucket,
                    f"{row.world_net_bps:.8f}",
                    f"{row.world_hit_rate:.6f}",
                    row.world_decision,
                    row.sequence_feature,
                    row.sequence_bucket,
                    f"{row.sequence_net_bps:.8f}",
                    f"{row.sequence_hit_rate:.6f}",
                    row.sequence_decision,
                    row.sequence_zero_cost_feature,
                    f"{row.sequence_zero_cost_net_bps:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_lob_policy_candidate_survival_md(
    rows: tuple[LobPolicyCandidateSurvival, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current LOB Policy Candidate Survival\n\n")
        handle.write(
            "This compares static LOB world replay with rolling sequence-state probes. "
            "A row is a policy-candidate diagnostic only when an action survives execution costs; "
            "zero-cost representation rows are kept separate.\n\n"
        )
        handle.write(
            "| state family | action | mode | status | score | world net | seq net | zero-cost seq | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.state_family} | "
                f"{row.signal_action} | "
                f"{row.execution_mode} | "
                f"{row.survival_status} | "
                f"{row.survival_score:.4f} | "
                f"{row.world_net_bps:.4f} | "
                f"{row.sequence_net_bps:.4f} | "
                f"{row.sequence_zero_cost_net_bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _best_rows(
    *,
    rows: tuple[dict[str, str], ...],
    feature_key: str,
    action_key: str,
    mode_key: str,
    net_key: str,
    hit_key: str,
    decision_key: str,
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (_state_family(row.get(feature_key, "")), row.get(action_key, ""), row.get(mode_key, ""))
        if not all(key):
            continue
        prepared = dict(row)
        prepared["_net_bps"] = row.get(net_key, "")
        prepared["_hit_rate"] = row.get(hit_key, "")
        prepared["_decision"] = row.get(decision_key, "")
        current = output.get(key)
        if current is None or _float(prepared.get("_net_bps")) > _float(current.get("_net_bps")):
            output[key] = prepared
    return output


def _best_sequence_zero_cost_rows(rows: tuple[dict[str, str], ...]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("execution_mode") != "zero_cost_representation":
            continue
        key = (_state_family(row.get("feature", "")), row.get("signal_action", ""))
        current = output.get(key)
        if current is None or _float(row.get("test_net_bps")) > _float(current.get("test_net_bps")):
            output[key] = row
    return output


def _build_row(
    *,
    state_family: str,
    signal_action: str,
    execution_mode: str,
    world: dict[str, str],
    sequence: dict[str, str],
    sequence_zero: dict[str, str],
) -> LobPolicyCandidateSurvival:
    world_net = _float(world.get("_net_bps"))
    world_hit = _float(world.get("_hit_rate"))
    sequence_net = _float(sequence.get("_net_bps"))
    sequence_hit = _float(sequence.get("_hit_rate"))
    sequence_zero_net = _float(sequence_zero.get("test_net_bps"))
    status = _survival_status(
        execution_mode=execution_mode,
        world_net=world_net,
        sequence_net=sequence_net,
        sequence_zero_net=sequence_zero_net,
    )
    return LobPolicyCandidateSurvival(
        state_family=state_family,
        signal_action=signal_action,
        execution_mode=execution_mode,
        survival_status=status,
        survival_score=_survival_score(
            status=status,
            world_net=world_net,
            world_hit=world_hit,
            sequence_net=sequence_net,
            sequence_hit=sequence_hit,
            sequence_zero_net=sequence_zero_net,
        ),
        world_feature=world.get("feature", ""),
        world_bucket=world.get("bucket", ""),
        world_net_bps=world_net,
        world_hit_rate=world_hit,
        world_decision=world.get("_decision", ""),
        sequence_feature=sequence.get("feature", ""),
        sequence_bucket=sequence.get("bucket", ""),
        sequence_net_bps=sequence_net,
        sequence_hit_rate=sequence_hit,
        sequence_decision=sequence.get("_decision", ""),
        sequence_zero_cost_feature=sequence_zero.get("feature", ""),
        sequence_zero_cost_net_bps=sequence_zero_net,
        reason=_reason(status),
        next_step=_next_step(status),
    )


def _survival_status(*, execution_mode: str, world_net: float, sequence_net: float, sequence_zero_net: float) -> str:
    if execution_mode == "zero_cost_representation":
        if max(sequence_net, sequence_zero_net) > 0.0:
            return "lob_representation_only"
        return "lob_policy_rejected_after_cost"
    if world_net > 0.0 and sequence_net > 0.0:
        return "lob_policy_consensus_execution_probe"
    if world_net > 0.0 and sequence_zero_net > 0.0:
        return "lob_world_execution_with_sequence_representation"
    if sequence_net > 0.0:
        return "lob_sequence_execution_probe"
    if world_net > 0.0:
        return "lob_world_execution_probe"
    return "lob_policy_rejected_after_cost"


def _survival_score(
    *,
    status: str,
    world_net: float,
    world_hit: float,
    sequence_net: float,
    sequence_hit: float,
    sequence_zero_net: float,
) -> float:
    base = {
        "lob_policy_consensus_execution_probe": 130.0,
        "lob_world_execution_with_sequence_representation": 95.0,
        "lob_sequence_execution_probe": 70.0,
        "lob_world_execution_probe": 60.0,
        "lob_representation_only": 20.0,
        "lob_policy_rejected_after_cost": -50.0,
    }.get(status, 0.0)
    return (
        base
        + max(world_net, 0.0) * 45.0
        + max(sequence_net, 0.0) * 55.0
        + max(sequence_zero_net, 0.0) * 12.0
        + max(world_hit - 0.5, 0.0) * 100.0
        + max(sequence_hit - 0.5, 0.0) * 100.0
    )


def _reason(status: str) -> str:
    if status == "lob_policy_consensus_execution_probe":
        return "static replay and rolling sequence state both survive execution costs"
    if status == "lob_world_execution_with_sequence_representation":
        return "static replay survives execution, and rolling sequence keeps representation value"
    if status == "lob_sequence_execution_probe":
        return "rolling sequence state survives execution costs, but static replay does not confirm it"
    if status == "lob_world_execution_probe":
        return "static replay survives execution costs, but rolling sequence does not confirm it"
    if status == "lob_representation_only":
        return "state representation has signal before execution costs only"
    return "state/action does not survive the current execution-cost check"


def _next_step(status: str) -> str:
    if status == "lob_policy_consensus_execution_probe":
        return "turn this into a tiny maker-fill paper policy with queue, cancel, and adverse-selection labels"
    if status == "lob_world_execution_with_sequence_representation":
        return "test whether the rolling state can choose the same action before maker-fill promotion"
    if status == "lob_sequence_execution_probe":
        return "repeat on fresh snapshots and add static replay confirmation"
    if status == "lob_world_execution_probe":
        return "add sequence-state confirmation before treating the replay as a policy"
    if status == "lob_representation_only":
        return "keep as model input, not a policy, until execution-cost survival appears"
    return "do not promote this state/action under current costs"


def _state_family(feature: str) -> str:
    if "liquidity" in feature:
        return "liquidity_change"
    if "imbalance" in feature:
        return "order_book_imbalance"
    if "basis" in feature or "premium" in feature:
        return "basis_premium"
    if "taker" in feature:
        return "taker_pressure"
    if "long_short" in feature or "open_interest" in feature or "oi_" in feature:
        return "positioning"
    return "other_lob_state"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_lob_policy_candidate_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_lob_policy_candidate_survival.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_lob_policy_candidate_survival_rows()
    write_lob_policy_candidate_survival_csv(rows, output_path=args.output_path)
    write_lob_policy_candidate_survival_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.survival_status, row.state_family, row.signal_action, row.execution_mode, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
