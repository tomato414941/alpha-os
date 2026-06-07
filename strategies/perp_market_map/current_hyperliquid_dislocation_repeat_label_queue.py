from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HyperliquidDislocationRepeatLabelQueueRow:
    asset: str
    status: str
    side: str
    queue_action: str
    priority_score: float
    observations: int
    mean_score: float
    last_outcome_15m: str
    last_net_15m_bps: str
    last_outcome_1h: str
    last_net_1h_bps: str
    execution_gate: str
    conservative_net_15m_bps: str
    reason: str
    next_step: str


def build_repeat_label_queue(
    *,
    monitor_summary_path: Path = ROOT / "current_hyperliquid_dislocation_monitor_summary.csv",
    label_path: Path = ROOT / "current_hyperliquid_dislocation_forward_labels.csv",
    execution_path: Path = ROOT / "current_hyperliquid_dislocation_execution_check.csv",
) -> tuple[HyperliquidDislocationRepeatLabelQueueRow, ...]:
    labels = _latest_by_key(_read_rows(label_path), timestamp_key="timestamp")
    executions = _best_execution_by_key(_read_rows(execution_path))
    rows: list[HyperliquidDislocationRepeatLabelQueueRow] = []
    for monitor in _read_rows(monitor_summary_path):
        if monitor.get("monitor_action") != "repeat_label_priority":
            continue
        key = (monitor.get("asset", ""), monitor.get("status", ""), monitor.get("side", ""))
        label = labels.get(key, {})
        execution = executions.get(key, {})
        rows.append(_build_queue_row(monitor=monitor, label=label, execution=execution))
    return tuple(sorted(rows, key=lambda row: row.priority_score, reverse=True))


def write_repeat_label_queue_csv(
    rows: tuple[HyperliquidDislocationRepeatLabelQueueRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "status",
                "side",
                "queue_action",
                "priority_score",
                "observations",
                "mean_score",
                "last_outcome_15m",
                "last_net_15m_bps",
                "last_outcome_1h",
                "last_net_1h_bps",
                "execution_gate",
                "conservative_net_15m_bps",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.status,
                    row.side,
                    row.queue_action,
                    f"{row.priority_score:.8f}",
                    row.observations,
                    f"{row.mean_score:.8f}",
                    row.last_outcome_15m,
                    row.last_net_15m_bps,
                    row.last_outcome_1h,
                    row.last_net_1h_bps,
                    row.execution_gate,
                    row.conservative_net_15m_bps,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_repeat_label_queue_md(
    rows: tuple[HyperliquidDislocationRepeatLabelQueueRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Repeat Label Queue\n\n")
        handle.write(
            "This queue turns repeated monitor observations into the next labeling "
            "or paper-probe actions. It is a workflow queue, not a strategy or trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | side | action | priority | obs | mean score | net15 | out15 | gate | net15 gated | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.queue_action} | "
                f"{row.priority_score:.4f} | "
                f"{row.observations} | "
                f"{row.mean_score:.4f} | "
                f"{row.last_net_15m_bps} | "
                f"{row.last_outcome_15m} | "
                f"{row.execution_gate} | "
                f"{row.conservative_net_15m_bps} | "
                f"{row.next_step} |\n"
            )
    return output_path


def _build_queue_row(
    *,
    monitor: dict[str, str],
    label: dict[str, str],
    execution: dict[str, str],
) -> HyperliquidDislocationRepeatLabelQueueRow:
    queue_action = _queue_action(label=label, execution=execution)
    priority_score = (
        _float(monitor.get("mean_score"))
        + min(_float(monitor.get("observations")) * 2.0, 12.0)
        + max(_float(label.get("net_15m_bps")) / 10.0, 0.0)
        + max(_float(execution.get("conservative_net_15m_bps")) / 10.0, 0.0)
    )
    return HyperliquidDislocationRepeatLabelQueueRow(
        asset=monitor.get("asset", ""),
        status=monitor.get("status", ""),
        side=monitor.get("side", ""),
        queue_action=queue_action,
        priority_score=priority_score,
        observations=int(monitor.get("observations") or "0"),
        mean_score=_float(monitor.get("mean_score")),
        last_outcome_15m=label.get("outcome_15m", ""),
        last_net_15m_bps=label.get("net_15m_bps", ""),
        last_outcome_1h=label.get("outcome_1h", ""),
        last_net_1h_bps=label.get("net_1h_bps", ""),
        execution_gate=execution.get("gate_action", ""),
        conservative_net_15m_bps=execution.get("conservative_net_15m_bps", ""),
        reason=_queue_reason(label=label, execution=execution),
        next_step=_queue_next_step(
            asset=monitor.get("asset", ""),
            label=label,
            execution=execution,
        ),
    )


def _queue_action(*, label: dict[str, str], execution: dict[str, str]) -> str:
    if execution.get("gate_action") == "paper_execution_probe":
        return "repeat_paper_probe_candidate"
    if label.get("outcome_15m") == "paper_15m_win" or label.get("outcome_1h") == "paper_1h_win":
        return "repeat_forward_label_priority"
    if label.get("outcome_15m") == "paper_15m_loss":
        return "monitor_conflict_relabel"
    return "fresh_forward_label_candidate"


def _queue_reason(*, label: dict[str, str], execution: dict[str, str]) -> str:
    if execution.get("gate_action") == "paper_execution_probe":
        return "repeated monitor candidate already passed the public-book paper gate"
    if label.get("outcome_15m") == "paper_15m_win" or label.get("outcome_1h") == "paper_1h_win":
        return "repeated monitor candidate has a positive prior paper label"
    if label.get("outcome_15m") == "paper_15m_loss":
        return "repeated monitor candidate conflicts with the latest paper label"
    return "repeated monitor candidate has not been labeled yet"


def _queue_next_step(
    *,
    asset: str,
    label: dict[str, str],
    execution: dict[str, str],
) -> str:
    if execution.get("gate_action") == "paper_execution_probe":
        return f"repeat {asset} paper probe on a fresh snapshot and record fill/outcome evidence"
    if label.get("outcome_1h") == "pending_1h":
        return f"wait for {asset} 1h label, then rerun execution check if still visible"
    return f"rerun {asset} forward label on a fresh repeated monitor window"


def _latest_by_key(
    rows: tuple[dict[str, str], ...],
    *,
    timestamp_key: str,
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: item.get(timestamp_key, ""), reverse=True):
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in output:
            continue
        output[key] = row
    return output


def _best_execution_by_key(rows: tuple[dict[str, str], ...]) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sorted(rows, key=_execution_sort_key, reverse=True):
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in output:
            continue
        output[key] = row
    return output


def _execution_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    gate_rank = {
        "paper_execution_probe": 4,
        "wide_spread_watch": 3,
        "no_edge_after_rough_cost": 2,
        "too_large_for_visible_depth": 1,
        "no_visible_depth": 0,
    }.get(row.get("gate_action", ""), 0)
    return (
        gate_rank,
        _float(row.get("conservative_net_15m_bps")),
        -_float(row.get("candidate_size_usd")),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-summary-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_monitor_summary.csv",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_forward_labels.csv",
    )
    parser.add_argument(
        "--execution-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_execution_check.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_repeat_label_queue.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_repeat_label_queue.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_repeat_label_queue(
        monitor_summary_path=args.monitor_summary_path,
        label_path=args.label_path,
        execution_path=args.execution_path,
    )
    write_repeat_label_queue_csv(rows, output_path=args.output_path)
    write_repeat_label_queue_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.status,
            row.side,
            row.queue_action,
            f"priority={row.priority_score:.4f}",
        )


if __name__ == "__main__":
    main()
