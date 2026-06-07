from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationActionabilityRow:
    asset: str
    action: str
    monitor_observations: int
    monitor_mean_score: float
    continuation_return_15m: float | None
    spread_bps: float | None
    bid_depth_5bps: float | None
    ask_depth_5bps: float | None
    near_touch_depth_5bps: float | None
    actionability_score: float
    note: str


def build_actionability_rows(
    *,
    monitor_summary_path: Path = ROOT / "current_okx_liquidation_monitor_summary.csv",
    forward_label_path: Path = ROOT / "current_okx_liquidation_monitor_forward_label_summary.csv",
    depth_check_path: Path = ROOT / "current_okx_liquidation_depth_check.csv",
) -> tuple[LiquidationActionabilityRow, ...]:
    labels = {
        (row["asset"], row["action"]): row
        for row in _read_rows(forward_label_path)
    }
    depth = {row["asset"]: row for row in _read_rows(depth_check_path)}
    rows = tuple(
        _build_row(
            monitor=row,
            label=labels.get((row["asset"], row["action"])),
            depth=depth.get(row["asset"]),
        )
        for row in _read_rows(monitor_summary_path)
    )
    return tuple(sorted(rows, key=lambda row: row.actionability_score, reverse=True))


def write_actionability_rows(
    rows: tuple[LiquidationActionabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "action",
                "monitor_observations",
                "monitor_mean_score",
                "continuation_return_15m",
                "spread_bps",
                "bid_depth_5bps",
                "ask_depth_5bps",
                "near_touch_depth_5bps",
                "actionability_score",
                "note",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.monitor_observations,
                    f"{row.monitor_mean_score:.8f}",
                    "" if row.continuation_return_15m is None else f"{row.continuation_return_15m:.8f}",
                    "" if row.spread_bps is None else f"{row.spread_bps:.8f}",
                    "" if row.bid_depth_5bps is None else f"{row.bid_depth_5bps:.8f}",
                    "" if row.ask_depth_5bps is None else f"{row.ask_depth_5bps:.8f}",
                    "" if row.near_touch_depth_5bps is None else f"{row.near_touch_depth_5bps:.8f}",
                    f"{row.actionability_score:.8f}",
                    row.note,
                )
            )
    return output_path


def write_actionability_md(
    rows: tuple[LiquidationActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Actionability Review\n\n")
        handle.write(
            "This joins liquidation persistence, monitor-sample continuation "
            "labels, and visible near-touch depth. It is a triage view, not "
            "an order plan.\n\n"
        )
        handle.write(
            "| asset | action | obs | monitor score | cont15 | spread bps | near depth 5bps | score | note |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.monitor_observations} | "
                f"{row.monitor_mean_score:.6f} | "
                f"{'' if row.continuation_return_15m is None else f'{row.continuation_return_15m:.6f}'} | "
                f"{'' if row.spread_bps is None else f'{row.spread_bps:.4f}'} | "
                f"{'' if row.near_touch_depth_5bps is None else f'{row.near_touch_depth_5bps:.0f}'} | "
                f"{row.actionability_score:.6f} | "
                f"{row.note} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A high score means the candidate has some combination of persistent "
            "liquidation flow, positive monitor-sample continuation, and visible "
            "depth. Thin-depth high-signal names should be treated as small-size "
            "probes until better venue depth is found.\n"
        )
    return output_path


def _build_row(
    *,
    monitor: dict[str, str],
    label: dict[str, str] | None,
    depth: dict[str, str] | None,
) -> LiquidationActionabilityRow:
    continuation_return_15m = _continuation_return_15m(label)
    bid_depth_5bps = _float_or_none("" if depth is None else depth.get("bid_depth_5bps", ""))
    ask_depth_5bps = _float_or_none("" if depth is None else depth.get("ask_depth_5bps", ""))
    near_depth = (
        None
        if bid_depth_5bps is None or ask_depth_5bps is None
        else min(bid_depth_5bps, ask_depth_5bps)
    )
    spread_bps = _float_or_none("" if depth is None else depth.get("spread_bps", ""))
    monitor_score = float(monitor.get("mean_cascade_score") or 0.0)
    actionability_score = _actionability_score(
        monitor_score=monitor_score,
        continuation_return_15m=continuation_return_15m,
        near_depth=near_depth,
        spread_bps=spread_bps,
    )
    return LiquidationActionabilityRow(
        asset=monitor["asset"],
        action=monitor["action"],
        monitor_observations=int(monitor.get("observations") or "0"),
        monitor_mean_score=monitor_score,
        continuation_return_15m=continuation_return_15m,
        spread_bps=spread_bps,
        bid_depth_5bps=bid_depth_5bps,
        ask_depth_5bps=ask_depth_5bps,
        near_touch_depth_5bps=near_depth,
        actionability_score=actionability_score,
        note=_note(
            continuation_return_15m=continuation_return_15m,
            near_depth=near_depth,
            spread_bps=spread_bps,
        ),
    )


def _actionability_score(
    *,
    monitor_score: float,
    continuation_return_15m: float | None,
    near_depth: float | None,
    spread_bps: float | None,
) -> float:
    continuation_component = max(continuation_return_15m or 0.0, 0.0) * 20.0
    depth_component = min((near_depth or 0.0) / 10000.0, 2.0)
    spread_penalty = 1.0 / max(spread_bps or 10.0, 0.5)
    if continuation_return_15m is None:
        return (monitor_score * 0.5) + (min((near_depth or 0.0) / 10000.0, 1.0) * 0.05)
    if continuation_return_15m is not None and continuation_return_15m <= 0.0:
        return monitor_score * 0.1
    return monitor_score + continuation_component + depth_component * spread_penalty


def _note(
    *,
    continuation_return_15m: float | None,
    near_depth: float | None,
    spread_bps: float | None,
) -> str:
    if continuation_return_15m is None:
        return "waiting for matching forward label"
    if continuation_return_15m <= 0.0:
        return "continuation label weak"
    if near_depth is not None and near_depth < 5000.0:
        return "signal ok but visible depth thin"
    if spread_bps is not None and spread_bps > 5.0:
        return "spread is wide"
    return "first checks support follow-up"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def _continuation_return_15m(label: dict[str, str] | None) -> float | None:
    if label is None:
        return None
    return _float_or_none(
        label.get("mean_continuation_return_15m", "")
        or label.get("continuation_return_15m", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-summary-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_summary.csv",
    )
    parser.add_argument(
        "--forward-label-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_forward_label_summary.csv",
    )
    parser.add_argument(
        "--depth-check-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_depth_check.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_actionability_review.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_actionability_review.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_actionability_rows(
        monitor_summary_path=args.monitor_summary_path,
        forward_label_path=args.forward_label_path,
        depth_check_path=args.depth_check_path,
    )
    write_actionability_rows(rows, output_path=args.output_path)
    write_actionability_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"cont15={'' if row.continuation_return_15m is None else f'{row.continuation_return_15m:.4f}'}",
            f"near_depth={'' if row.near_touch_depth_5bps is None else f'{row.near_touch_depth_5bps:.0f}'}",
            f"score={row.actionability_score:.4f}",
            row.note,
        )


if __name__ == "__main__":
    main()
