from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationPaperOutcome:
    event_timestamp: str
    asset: str
    action: str
    paper_direction: str
    candidate_size_usd: float
    conservative_cost_bps: float
    continuation_return_15m: float | None
    continuation_return_1h: float | None
    net_15m_bps: float | None
    net_1h_bps: float | None
    outcome_15m: str
    outcome_1h: str


def build_paper_outcomes(
    *,
    paper_gate_path: Path = ROOT / "current_okx_liquidation_paper_gate.csv",
    forward_label_path: Path = ROOT / "current_okx_liquidation_monitor_forward_labels.csv",
) -> tuple[LiquidationPaperOutcome, ...]:
    gate_rows = _selected_gate_rows(paper_gate_path)
    labels_by_key = _labels_by_key(forward_label_path)
    rows = tuple(
        _build_outcome(gate=row, label=label)
        for row in gate_rows
        for label in labels_by_key.get((row["asset"], row["action"]), ())
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.net_15m_bps is not None,
                row.net_15m_bps or -1_000_000.0,
                row.net_1h_bps is not None,
                row.net_1h_bps or -1_000_000.0,
            ),
            reverse=True,
        )
    )


def write_paper_outcomes(
    rows: tuple[LiquidationPaperOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "event_timestamp",
                "asset",
                "action",
                "paper_direction",
                "candidate_size_usd",
                "conservative_cost_bps",
                "continuation_return_15m",
                "continuation_return_1h",
                "net_15m_bps",
                "net_1h_bps",
                "outcome_15m",
                "outcome_1h",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.event_timestamp,
                    row.asset,
                    row.action,
                    row.paper_direction,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.conservative_cost_bps:.8f}",
                    (
                        ""
                        if row.continuation_return_15m is None
                        else f"{row.continuation_return_15m:.8f}"
                    ),
                    (
                        ""
                        if row.continuation_return_1h is None
                        else f"{row.continuation_return_1h:.8f}"
                    ),
                    "" if row.net_15m_bps is None else f"{row.net_15m_bps:.8f}",
                    "" if row.net_1h_bps is None else f"{row.net_1h_bps:.8f}",
                    row.outcome_15m,
                    row.outcome_1h,
                )
            )
    return output_path


def write_paper_outcome_md(
    rows: tuple[LiquidationPaperOutcome, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Paper Outcome\n\n")
        handle.write(
            "This joins paper-gate rows to monitor forward labels. It measures "
            "the paper result after the same conservative cost proxy used by the "
            "gate. It is still a retrospective observation, not a live fill.\n\n"
        )
        handle.write(
            "| event | asset | action | dir | size USD | cost bps | net15 bps | out15 | net1h bps | out1h |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.event_timestamp} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.paper_direction} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{row.conservative_cost_bps:.2f} | "
                f"{'' if row.net_15m_bps is None else f'{row.net_15m_bps:.2f}'} | "
                f"{row.outcome_15m} | "
                f"{'' if row.net_1h_bps is None else f'{row.net_1h_bps:.2f}'} | "
                f"{row.outcome_1h} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_15m_win` means the event label stayed positive after the rough "
            "cost proxy. A deployable strategy still needs fresh-event repetition, "
            "live fills, risk limits, and a rule for skipping crowded or stale events.\n"
        )
    return output_path


def _selected_gate_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") == "small_paper_probe"
        )
    if not rows:
        return ()
    best = max(
        rows,
        key=lambda row: (
            float(row.get("conservative_net_bps") or "-inf"),
            -float(row.get("visible_depth_usage") or "inf"),
            float(row.get("candidate_size_usd") or "0"),
        ),
    )
    return (best,)


def _labels_by_key(path: Path) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault((row["asset"], row["action"]), {})[row["timestamp"]] = row
    return {
        key: tuple(sorted(rows.values(), key=lambda row: row["timestamp"]))
        for key, rows in grouped.items()
    }


def _build_outcome(
    *,
    gate: dict[str, str],
    label: dict[str, str],
) -> LiquidationPaperOutcome:
    cost_bps = float(gate["conservative_cost_bps"])
    continuation_15m = _optional_float(label.get("continuation_return_15m", ""))
    continuation_1h = _optional_float(label.get("continuation_return_1h", ""))
    net_15m_bps = _net_bps(continuation_15m, cost_bps=cost_bps)
    net_1h_bps = _net_bps(continuation_1h, cost_bps=cost_bps)
    return LiquidationPaperOutcome(
        event_timestamp=label["timestamp"],
        asset=gate["asset"],
        action=gate["action"],
        paper_direction=_direction_for_action(gate["action"]),
        candidate_size_usd=float(gate["candidate_size_usd"]),
        conservative_cost_bps=cost_bps,
        continuation_return_15m=continuation_15m,
        continuation_return_1h=continuation_1h,
        net_15m_bps=net_15m_bps,
        net_1h_bps=net_1h_bps,
        outcome_15m=_outcome(net_15m_bps, horizon="15m"),
        outcome_1h=_outcome(net_1h_bps, horizon="1h"),
    )


def _net_bps(value: float | None, *, cost_bps: float) -> float | None:
    return None if value is None else (value * 10_000.0) - cost_bps


def _outcome(value: float | None, *, horizon: str) -> str:
    if value is None:
        return f"pending_{horizon}"
    if value > 0.0:
        return f"paper_{horizon}_win"
    return f"paper_{horizon}_loss"


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _direction_for_action(action: str) -> str:
    if action.startswith("long_liquidation"):
        return "short"
    if action.startswith("short_liquidation"):
        return "long"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-gate-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_gate.csv",
    )
    parser.add_argument(
        "--forward-label-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_forward_labels.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_outcome.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_outcome.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_paper_outcomes(
        paper_gate_path=args.paper_gate_path,
        forward_label_path=args.forward_label_path,
    )
    write_paper_outcomes(rows, output_path=args.output_path)
    write_paper_outcome_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            row.event_timestamp,
            f"net15={'' if row.net_15m_bps is None else f'{row.net_15m_bps:.2f}'}",
            row.outcome_15m,
            f"net1h={'' if row.net_1h_bps is None else f'{row.net_1h_bps:.2f}'}",
            row.outcome_1h,
        )


if __name__ == "__main__":
    main()
