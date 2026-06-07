from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SIZES_USD = (100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0)


@dataclass(frozen=True)
class L2ImbalancePaperGateRow:
    asset: str
    candidate_size_usd: float
    spread_bps: float
    imbalance_10_bps: float
    gross_directional_bps_15m: float | None
    gross_directional_bps_1h: float | None
    taker_fee_bps_per_fill: float
    conservative_cost_bps: float
    net_15m_bps: float | None
    net_1h_bps: float | None
    near_depth_10bps_notional: float
    visible_depth_usage: float
    gate_action: str
    reason: str


def build_l2_imbalance_paper_gate_rows(
    *,
    snapshot_path: Path = ROOT / "current_l2_snapshot.csv",
    label_path: Path = ROOT / "current_l2_imbalance_forward_labels.csv",
    sizes_usd: tuple[float, ...] = DEFAULT_SIZES_USD,
    taker_fee_bps_per_fill: float = 5.0,
) -> tuple[L2ImbalancePaperGateRow, ...]:
    snapshot_by_asset = {row["asset"]: row for row in _read_rows(snapshot_path)}
    rows = tuple(
        _build_row(
            label=label,
            snapshot=snapshot_by_asset[label["asset"]],
            size_usd=size,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for label in _read_rows(label_path)
        if label["asset"] in snapshot_by_asset
        for size in sizes_usd
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.gate_action == "small_paper_probe",
                row.net_15m_bps or -1_000_000.0,
                row.net_1h_bps or -1_000_000.0,
                -row.visible_depth_usage,
            ),
            reverse=True,
        )
    )


def write_l2_imbalance_paper_gate_rows(
    rows: tuple[L2ImbalancePaperGateRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "candidate_size_usd",
                "spread_bps",
                "imbalance_10_bps",
                "gross_directional_bps_15m",
                "gross_directional_bps_1h",
                "taker_fee_bps_per_fill",
                "conservative_cost_bps",
                "net_15m_bps",
                "net_1h_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage",
                "gate_action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.imbalance_10_bps:.8f}",
                    (
                        ""
                        if row.gross_directional_bps_15m is None
                        else f"{row.gross_directional_bps_15m:.8f}"
                    ),
                    (
                        ""
                        if row.gross_directional_bps_1h is None
                        else f"{row.gross_directional_bps_1h:.8f}"
                    ),
                    f"{row.taker_fee_bps_per_fill:.8f}",
                    f"{row.conservative_cost_bps:.8f}",
                    "" if row.net_15m_bps is None else f"{row.net_15m_bps:.8f}",
                    "" if row.net_1h_bps is None else f"{row.net_1h_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    row.gate_action,
                    row.reason,
                )
            )
    return output_path


def write_l2_imbalance_paper_gate_md(
    rows: tuple[L2ImbalancePaperGateRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current L2 Imbalance Paper Gate\n\n")
        handle.write(
            "This subtracts taker round-trip fees and current spread from the "
            "book-imbalance directional label, then checks visible 10 bps depth. "
            "It is a directional paper gate, not a maker-fill model.\n\n"
        )
        handle.write(
            "| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{row.imbalance_10_bps:.4f} | "
                f"{row.conservative_cost_bps:.2f} | "
                f"{'' if row.net_15m_bps is None else f'{row.net_15m_bps:.2f}'} | "
                f"{'' if row.net_1h_bps is None else f'{row.net_1h_bps:.2f}'} | "
                f"{row.near_depth_10bps_notional:.0f} | "
                f"{row.visible_depth_usage:.4f} | "
                f"{row.gate_action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`small_paper_probe` means the imbalance direction survived the rough "
            "fee/spread/depth check at that notional. This does not prove a market "
            "making edge because queue position, fill probability, rebates, and "
            "adverse selection are still unmodeled.\n"
        )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    snapshot: dict[str, str],
    size_usd: float,
    taker_fee_bps_per_fill: float,
) -> L2ImbalancePaperGateRow:
    spread_bps = float(label["spread_bps"])
    cost_bps = (taker_fee_bps_per_fill * 2.0) + spread_bps
    gross_15m = _optional_return_bps(label.get("directional_return_15m", ""))
    gross_1h = _optional_return_bps(label.get("directional_return_1h", ""))
    net_15m = None if gross_15m is None else gross_15m - cost_bps
    net_1h = None if gross_1h is None else gross_1h - cost_bps
    mid_price = float(snapshot["mid_price"])
    near_depth = min(
        float(snapshot["bid_depth_10_bps"]),
        float(snapshot["ask_depth_10_bps"]),
    ) * mid_price
    depth_usage = size_usd / near_depth if near_depth > 0.0 else float("inf")
    gate_action, reason = _gate_action(
        net_15m=net_15m,
        depth_usage=depth_usage,
    )
    return L2ImbalancePaperGateRow(
        asset=label["asset"],
        candidate_size_usd=size_usd,
        spread_bps=spread_bps,
        imbalance_10_bps=float(label["imbalance_10_bps"]),
        gross_directional_bps_15m=gross_15m,
        gross_directional_bps_1h=gross_1h,
        taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        conservative_cost_bps=cost_bps,
        net_15m_bps=net_15m,
        net_1h_bps=net_1h,
        near_depth_10bps_notional=near_depth,
        visible_depth_usage=depth_usage,
        gate_action=gate_action,
        reason=reason,
    )


def _gate_action(*, net_15m: float | None, depth_usage: float) -> tuple[str, str]:
    if net_15m is None:
        return "wait_for_label", "no 15m imbalance label yet"
    if net_15m <= 0.0:
        return "blocked_by_cost", "fee and spread consume the 15m directional label"
    if depth_usage > 0.25:
        return "too_large_for_visible_depth", "candidate size uses too much visible 10 bps depth"
    return "small_paper_probe", "survives rough fee, spread, and visible-depth check"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _optional_return_bps(value: str) -> float | None:
    return None if value == "" else float(value) * 10_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "current_l2_snapshot.csv",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_forward_labels.csv",
    )
    parser.add_argument("--sizes-usd", nargs="+", type=float, default=list(DEFAULT_SIZES_USD))
    parser.add_argument("--taker-fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_paper_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_paper_gate.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_l2_imbalance_paper_gate_rows(
        snapshot_path=args.snapshot_path,
        label_path=args.label_path,
        sizes_usd=tuple(args.sizes_usd),
        taker_fee_bps_per_fill=args.taker_fee_bps_per_fill,
    )
    write_l2_imbalance_paper_gate_rows(rows, output_path=args.output_path)
    write_l2_imbalance_paper_gate_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            f"size={row.candidate_size_usd:.0f}",
            f"net15={'' if row.net_15m_bps is None else f'{row.net_15m_bps:.2f}'}",
            f"net1h={'' if row.net_1h_bps is None else f'{row.net_1h_bps:.2f}'}",
            row.gate_action,
        )


if __name__ == "__main__":
    main()
