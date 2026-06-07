from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SIZES_USD = (100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0)


@dataclass(frozen=True)
class LiquidationPaperGateRow:
    asset: str
    action: str
    candidate_size_usd: float
    gross_continuation_bps: float | None
    fee_bps_per_fill: float
    round_trip_fee_bps: float
    spread_cost_bps: float | None
    visible_depth_impact_bps: float | None
    conservative_cost_bps: float | None
    conservative_net_bps: float | None
    near_touch_depth_5bps: float | None
    visible_depth_usage: float | None
    gate_action: str
    reason: str


def build_paper_gate_rows(
    *,
    actionability_path: Path = ROOT / "current_okx_liquidation_actionability_review.csv",
    sizes_usd: tuple[float, ...] = DEFAULT_SIZES_USD,
    fee_bps_per_fill: float = 5.0,
) -> tuple[LiquidationPaperGateRow, ...]:
    rows = tuple(
        _build_row(row=row, size_usd=size, fee_bps_per_fill=fee_bps_per_fill)
        for row in _read_rows(actionability_path)
        for size in sizes_usd
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.gate_action == "small_paper_probe",
                row.conservative_net_bps or -1_000_000.0,
                -(row.visible_depth_usage or 1_000_000.0),
            ),
            reverse=True,
        )
    )


def write_paper_gate_rows(
    rows: tuple[LiquidationPaperGateRow, ...],
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
                "candidate_size_usd",
                "gross_continuation_bps",
                "fee_bps_per_fill",
                "round_trip_fee_bps",
                "spread_cost_bps",
                "visible_depth_impact_bps",
                "conservative_cost_bps",
                "conservative_net_bps",
                "near_touch_depth_5bps",
                "visible_depth_usage",
                "gate_action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    f"{row.candidate_size_usd:.2f}",
                    (
                        ""
                        if row.gross_continuation_bps is None
                        else f"{row.gross_continuation_bps:.8f}"
                    ),
                    f"{row.fee_bps_per_fill:.8f}",
                    f"{row.round_trip_fee_bps:.8f}",
                    "" if row.spread_cost_bps is None else f"{row.spread_cost_bps:.8f}",
                    (
                        ""
                        if row.visible_depth_impact_bps is None
                        else f"{row.visible_depth_impact_bps:.8f}"
                    ),
                    (
                        ""
                        if row.conservative_cost_bps is None
                        else f"{row.conservative_cost_bps:.8f}"
                    ),
                    (
                        ""
                        if row.conservative_net_bps is None
                        else f"{row.conservative_net_bps:.8f}"
                    ),
                    (
                        ""
                        if row.near_touch_depth_5bps is None
                        else f"{row.near_touch_depth_5bps:.8f}"
                    ),
                    (
                        ""
                        if row.visible_depth_usage is None
                        else f"{row.visible_depth_usage:.8f}"
                    ),
                    row.gate_action,
                    row.reason,
                )
            )
    return output_path


def write_paper_gate_md(
    rows: tuple[LiquidationPaperGateRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Paper Gate\n\n")
        handle.write(
            "This subtracts assumed round-trip taker fees, current spread, and a "
            "simple visible-depth impact proxy from the 15m monitor-sample "
            "continuation label. It is a sizing gate, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{'' if row.gross_continuation_bps is None else f'{row.gross_continuation_bps:.2f}'} | "
                f"{'' if row.conservative_cost_bps is None else f'{row.conservative_cost_bps:.2f}'} | "
                f"{'' if row.conservative_net_bps is None else f'{row.conservative_net_bps:.2f}'} | "
                f"{'' if row.near_touch_depth_5bps is None else f'{row.near_touch_depth_5bps:.0f}'} | "
                f"{'' if row.visible_depth_usage is None else f'{row.visible_depth_usage:.4f}'} | "
                f"{row.gate_action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`small_paper_probe` means the current short-window label survives this "
            "rough fee/spread/depth check at the listed notional. This still omits "
            "real account fees, order-type choice, live spread changes, queue "
            "position, and stop logic.\n"
        )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    size_usd: float,
    fee_bps_per_fill: float,
) -> LiquidationPaperGateRow:
    continuation_return = _optional_float(row.get("continuation_return_15m", ""))
    gross_bps = None if continuation_return is None else continuation_return * 10_000.0
    spread_bps = _optional_float(row.get("spread_bps", ""))
    near_depth = _optional_float(row.get("near_touch_depth_5bps", ""))
    round_trip_fee_bps = fee_bps_per_fill * 2.0
    visible_depth_usage = None if near_depth is None or near_depth <= 0.0 else size_usd / near_depth
    impact_bps = None
    conservative_cost_bps = None
    conservative_net_bps = None
    if spread_bps is not None and visible_depth_usage is not None:
        impact_bps = min(visible_depth_usage, 1.0) * 10.0
        conservative_cost_bps = round_trip_fee_bps + spread_bps + impact_bps
        conservative_net_bps = None if gross_bps is None else gross_bps - conservative_cost_bps
    gate_action, reason = _gate_action(
        gross_bps=gross_bps,
        near_depth=near_depth,
        visible_depth_usage=visible_depth_usage,
        conservative_net_bps=conservative_net_bps,
    )
    return LiquidationPaperGateRow(
        asset=row["asset"],
        action=row["action"],
        candidate_size_usd=size_usd,
        gross_continuation_bps=gross_bps,
        fee_bps_per_fill=fee_bps_per_fill,
        round_trip_fee_bps=round_trip_fee_bps,
        spread_cost_bps=spread_bps,
        visible_depth_impact_bps=impact_bps,
        conservative_cost_bps=conservative_cost_bps,
        conservative_net_bps=conservative_net_bps,
        near_touch_depth_5bps=near_depth,
        visible_depth_usage=visible_depth_usage,
        gate_action=gate_action,
        reason=reason,
    )


def _gate_action(
    *,
    gross_bps: float | None,
    near_depth: float | None,
    visible_depth_usage: float | None,
    conservative_net_bps: float | None,
) -> tuple[str, str]:
    if gross_bps is None:
        return "wait_for_label", "no positive-direction monitor label yet"
    if gross_bps <= 0.0:
        return "blocked_by_label", "monitor label is not positive"
    if near_depth is None or visible_depth_usage is None:
        return "wait_for_depth", "visible near-touch depth is missing"
    if visible_depth_usage > 0.25:
        return "too_large_for_visible_depth", "candidate size uses too much visible near-touch depth"
    if conservative_net_bps is None or conservative_net_bps <= 0.0:
        return "blocked_by_cost", "fee, spread, and impact proxy consume the short-window edge"
    return "small_paper_probe", "survives rough fee, spread, and visible-depth check"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actionability-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_actionability_review.csv",
    )
    parser.add_argument(
        "--sizes-usd",
        nargs="+",
        type=float,
        default=list(DEFAULT_SIZES_USD),
    )
    parser.add_argument("--fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_paper_gate.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_paper_gate_rows(
        actionability_path=args.actionability_path,
        sizes_usd=tuple(args.sizes_usd),
        fee_bps_per_fill=args.fee_bps_per_fill,
    )
    write_paper_gate_rows(rows, output_path=args.output_path)
    write_paper_gate_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"size={row.candidate_size_usd:.0f}",
            f"net={'' if row.conservative_net_bps is None else f'{row.conservative_net_bps:.2f}'}bps",
            row.gate_action,
        )


if __name__ == "__main__":
    main()
