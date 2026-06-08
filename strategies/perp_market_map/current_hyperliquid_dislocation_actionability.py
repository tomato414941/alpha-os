from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HyperliquidDislocationActionabilityRow:
    asset: str
    source_status: str
    side: str
    status: str
    action: str
    score: float
    candidate_score: float
    monitor_observations: int
    monitor_action: str
    current_outcome_15m: str
    current_net_15m_bps: float
    current_outcome_1h: str
    current_net_1h_bps: float
    execution_gate: str
    candidate_size_usd: float
    conservative_net_15m_bps: float
    conservative_net_1h_bps: float
    spread_bps: float
    visible_depth_usage_10bps: float
    history_action: str
    history_observations: int
    history_win_1h: int
    history_mean_net_1h_bps: float
    reason: str
    next_step: str


def build_hyperliquid_dislocation_actionability_rows(root: Path = ROOT) -> tuple[HyperliquidDislocationActionabilityRow, ...]:
    candidates = _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_candidates.csv")
    labels = _latest_by_key(root / "perp_market_map" / "current_hyperliquid_dislocation_forward_labels.csv")
    monitors = _best_by_key(root / "perp_market_map" / "current_hyperliquid_dislocation_monitor_summary.csv", "mean_score")
    executions = _best_execution_by_key(root / "perp_market_map" / "current_hyperliquid_dislocation_execution_check.csv")
    histories = _best_by_key(root / "perp_market_map" / "current_hyperliquid_dislocation_label_history_summary.csv", "mean_net_1h_bps")
    repeat_queue = _best_by_key(root / "perp_market_map" / "current_hyperliquid_dislocation_repeat_label_queue.csv", "priority_score")

    output: list[HyperliquidDislocationActionabilityRow] = []
    for row in candidates:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not all(key):
            continue
        label = labels.get(key, {})
        monitor = monitors.get(key, {})
        execution = executions.get(key, {})
        history = histories.get(key, {})
        repeat = repeat_queue.get(key, {})
        status, action, reason = _status_action_reason(
            label=label,
            execution=execution,
            history=history,
            repeat=repeat,
        )
        output.append(
            HyperliquidDislocationActionabilityRow(
                asset=row.get("asset", ""),
                source_status=row.get("status", ""),
                side=row.get("side", ""),
                status=status,
                action=action,
                score=_score(
                    status=status,
                    candidate_score=_float(row.get("score")),
                    monitor_observations=_int(monitor.get("observations")),
                    current_net_1h_bps=_float(label.get("net_1h_bps")),
                    conservative_net_1h_bps=_float(execution.get("conservative_net_1h_bps")),
                    history_win_1h=_int(history.get("win_1h")),
                    history_mean_net_1h_bps=_float(history.get("mean_net_1h_bps")),
                    spread_bps=_float(execution.get("spread_bps")),
                    visible_depth_usage_10bps=_float(execution.get("visible_depth_usage_10bps")),
                ),
                candidate_score=_float(row.get("score")),
                monitor_observations=_int(monitor.get("observations")),
                monitor_action=monitor.get("monitor_action", ""),
                current_outcome_15m=label.get("outcome_15m", ""),
                current_net_15m_bps=_float(label.get("net_15m_bps")),
                current_outcome_1h=label.get("outcome_1h", ""),
                current_net_1h_bps=_float(label.get("net_1h_bps")),
                execution_gate=execution.get("gate_action", ""),
                candidate_size_usd=_float(execution.get("candidate_size_usd")),
                conservative_net_15m_bps=_float(execution.get("conservative_net_15m_bps")),
                conservative_net_1h_bps=_float(execution.get("conservative_net_1h_bps")),
                spread_bps=_float(execution.get("spread_bps")),
                visible_depth_usage_10bps=_float(execution.get("visible_depth_usage_10bps")),
                history_action=history.get("history_action", ""),
                history_observations=_int(history.get("observations")),
                history_win_1h=_int(history.get("win_1h")),
                history_mean_net_1h_bps=_float(history.get("mean_net_1h_bps")),
                reason=reason,
                next_step=_next_step(asset=row.get("asset", ""), status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_hyperliquid_dislocation_actionability_csv(
    rows: tuple[HyperliquidDislocationActionabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "source_status",
                "side",
                "status",
                "action",
                "score",
                "candidate_score",
                "monitor_observations",
                "monitor_action",
                "current_outcome_15m",
                "current_net_15m_bps",
                "current_outcome_1h",
                "current_net_1h_bps",
                "execution_gate",
                "candidate_size_usd",
                "conservative_net_15m_bps",
                "conservative_net_1h_bps",
                "spread_bps",
                "visible_depth_usage_10bps",
                "history_action",
                "history_observations",
                "history_win_1h",
                "history_mean_net_1h_bps",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.source_status,
                    row.side,
                    row.status,
                    row.action,
                    f"{row.score:.8f}",
                    f"{row.candidate_score:.8f}",
                    row.monitor_observations,
                    row.monitor_action,
                    row.current_outcome_15m,
                    f"{row.current_net_15m_bps:.8f}",
                    row.current_outcome_1h,
                    f"{row.current_net_1h_bps:.8f}",
                    row.execution_gate,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.conservative_net_15m_bps:.8f}",
                    f"{row.conservative_net_1h_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.visible_depth_usage_10bps:.8f}",
                    row.history_action,
                    row.history_observations,
                    row.history_win_1h,
                    f"{row.history_mean_net_1h_bps:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_hyperliquid_dislocation_actionability_md(
    rows: tuple[HyperliquidDislocationActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Actionability\n\n")
        handle.write(
            "This separates raw dislocation screens and short-window labels from candidates "
            "that deserve a repeated paper probe. It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| asset | side | status | action | score | current 15m | current 1h | gate | hist obs | hist 1h wins | hist mean 1h | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | {row.side} | {row.status} | {row.action} | {row.score:.4f} | "
                f"{row.current_outcome_15m} {row.current_net_15m_bps:.2f} | "
                f"{row.current_outcome_1h} {row.current_net_1h_bps:.2f} | "
                f"{row.execution_gate} | {row.history_observations} | {row.history_win_1h} | "
                f"{row.history_mean_net_1h_bps:.2f} | {_escape(row.reason)} |\n"
            )
    return output_path


def _status_action_reason(
    *,
    label: dict[str, str],
    execution: dict[str, str],
    history: dict[str, str],
    repeat: dict[str, str],
) -> tuple[str, str, str]:
    if repeat.get("queue_action") == "monitor_conflict_relabel":
        return (
            "dislocation_monitor_conflict_relabel",
            "relabel_before_probe",
            "repeated monitor conflicts with the latest paper label",
        )
    if history.get("history_action") == "deprioritize_until_fresh_snapshot":
        return (
            "dislocation_history_deprioritize",
            "deprioritize_until_fresh_snapshot",
            "label history is weak or negative despite the current snapshot",
        )
    if label.get("outcome_1h") == "paper_1h_loss" or execution.get("gate_action") == "failed_1h_confirmation":
        return "dislocation_failed_1h_confirmation", "wait_for_fresh_1h", "15m edge failed after the 1h label matured"
    if (
        history.get("history_action") == "repeat_paper_probe_priority"
        and execution.get("gate_action") == "paper_execution_probe"
        and _float(execution.get("conservative_net_1h_bps")) > 0.0
    ):
        return (
            "dislocation_repeat_execution_candidate",
            "repeat_paper_probe",
            "repeated 1h label history and current public-book execution gate both pass",
        )
    if history.get("history_action") == "repeat_paper_probe_priority":
        return (
            "dislocation_repeat_needs_execution_check",
            "refresh_execution_gate",
            "repeated 1h label history is positive but the current execution gate is missing or weak",
        )
    if label.get("outcome_1h") == "paper_1h_win" and _float(label.get("net_1h_bps")) > 0.0:
        return (
            "dislocation_single_snapshot_1h_watch",
            "repeat_before_probe",
            "current 1h label is positive, but repeat history is not strong enough",
        )
    if label.get("outcome_15m") == "paper_15m_win":
        return (
            "dislocation_15m_only_watch",
            "wait_for_1h_confirmation",
            "15m label is positive but not enough to establish persistence",
        )
    return "dislocation_deprioritize", "none", "no repeated positive label or current execution support"


def _score(
    *,
    status: str,
    candidate_score: float,
    monitor_observations: int,
    current_net_1h_bps: float,
    conservative_net_1h_bps: float,
    history_win_1h: int,
    history_mean_net_1h_bps: float,
    spread_bps: float,
    visible_depth_usage_10bps: float,
) -> float:
    if status == "dislocation_repeat_execution_candidate":
        return min(
            88.0,
            56.0
            + min(history_win_1h * 5.0, 15.0)
            + min(history_mean_net_1h_bps / 20.0, 10.0)
            + min(conservative_net_1h_bps / 50.0, 6.0)
            + min(monitor_observations / 4.0, 5.0)
            - max(spread_bps - 5.0, 0.0) * 0.4
            - max(visible_depth_usage_10bps - 0.05, 0.0) * 20.0,
        )
    if status == "dislocation_repeat_needs_execution_check":
        depth_penalty = 0.0
        if visible_depth_usage_10bps > 0.0:
            depth_penalty = max(visible_depth_usage_10bps - 0.05, 0.0) * 30.0
        return min(
            54.0,
            40.0
            + min(history_win_1h * 3.0, 6.0)
            + min(history_mean_net_1h_bps / 60.0, 5.0)
            - depth_penalty,
        )
    if status == "dislocation_single_snapshot_1h_watch":
        return min(58.0, 40.0 + min(current_net_1h_bps / 40.0, 8.0) + min(candidate_score / 8.0, 4.0))
    if status == "dislocation_15m_only_watch":
        return min(48.0, 34.0 + min(candidate_score / 8.0, 4.0))
    if status in {"dislocation_monitor_conflict_relabel", "dislocation_failed_1h_confirmation"}:
        return 34.0
    if status == "dislocation_history_deprioritize":
        return 28.0
    return 20.0


def _next_step(*, asset: str, status: str) -> str:
    if status == "dislocation_repeat_execution_candidate":
        return f"run a small repeated paper probe for {asset} and record 15m/1h/4h mark, funding, spread, and stop behavior"
    if status == "dislocation_repeat_needs_execution_check":
        return f"refresh {asset} public-book execution gate before any paper probe"
    if status == "dislocation_single_snapshot_1h_watch":
        return f"rerun {asset} on a fresh monitor window; do not promote from one positive 1h label"
    if status == "dislocation_15m_only_watch":
        return f"wait for {asset} 1h confirmation before any probe"
    if status == "dislocation_monitor_conflict_relabel":
        return f"relabel {asset} because monitor and latest label disagree"
    if status == "dislocation_failed_1h_confirmation":
        return f"deprioritize {asset} until a fresh 1h label supports the same lane"
    return f"deprioritize {asset} until repeated positive labels and an execution gate appear"


def _latest_by_key(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sorted(_read_rows(path), key=lambda item: item.get("timestamp", ""), reverse=True):
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if all(key) and key not in output:
            output[key] = row
    return output


def _best_by_key(path: Path, score_key: str) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sorted(_read_rows(path), key=lambda item: _float(item.get(score_key)), reverse=True):
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if all(key) and key not in output:
            output[key] = row
    return output


def _best_execution_by_key(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    gate_rank = {
        "paper_execution_probe": 4,
        "wide_spread_watch": 3,
        "no_edge_after_rough_cost": 2,
        "too_large_for_visible_depth": 1,
        "no_visible_depth": 0,
    }
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    rows = sorted(
        _read_rows(path),
        key=lambda item: (
            gate_rank.get(item.get("gate_action", ""), 0),
            _float(item.get("conservative_net_1h_bps")),
            -_float(item.get("candidate_size_usd")),
        ),
        reverse=True,
    )
    for row in rows:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if all(key) and key not in output:
            output[key] = row
    return output


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    try:
        return int(float(value)) if value else 0
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_hyperliquid_dislocation_actionability.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=LANE_ROOT / "current_hyperliquid_dislocation_actionability.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_hyperliquid_dislocation_actionability_rows()
    write_hyperliquid_dislocation_actionability_csv(rows, output_path=args.output_path)
    write_hyperliquid_dislocation_actionability_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.asset, row.side, f"score={row.score:.2f}")


if __name__ == "__main__":
    main()
