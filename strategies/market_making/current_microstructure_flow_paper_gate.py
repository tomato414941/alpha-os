from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SIZES_USD = (100.0, 250.0, 500.0, 1_000.0)


@dataclass(frozen=True)
class MicrostructureFlowPaperGateRow:
    asset: str
    action: str
    direction: int
    candidate_size_usd: float
    gross_15m_bps: float
    gross_1h_bps: float
    spread_bps: float
    depth_10bps_usd: float
    visible_depth_usage: float
    taker_fee_bps_per_fill: float
    conservative_cost_bps: float
    conservative_net_15m_bps: float
    conservative_net_1h_bps: float
    gate_action: str
    reason: str
    next_step: str


def build_microstructure_flow_paper_gate_rows(
    *,
    snapshot_path: Path = ROOT / "current_microstructure_flow_snapshot.csv",
    label_path: Path = ROOT / "current_microstructure_flow_forward_labels.csv",
    sizes_usd: tuple[float, ...] = DEFAULT_SIZES_USD,
    taker_fee_bps_per_fill: float = 5.0,
) -> tuple[MicrostructureFlowPaperGateRow, ...]:
    snapshots = {
        (row.get("asset", ""), row.get("action", "")): row
        for row in _read_rows(snapshot_path)
        if row.get("asset") and row.get("action")
    }
    rows = tuple(
        _build_row(
            label=label,
            snapshot=snapshots[(label["asset"], label["action"])],
            size_usd=size,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for label in _read_rows(label_path)
        if (label.get("asset", ""), label.get("action", "")) in snapshots
        for size in sizes_usd
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.gate_action == "microstructure_small_paper_probe",
                row.conservative_net_15m_bps,
                row.conservative_net_1h_bps,
                -row.visible_depth_usage,
            ),
            reverse=True,
        )
    )


def write_microstructure_flow_paper_gate_rows(
    rows: tuple[MicrostructureFlowPaperGateRow, ...],
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
                "direction",
                "candidate_size_usd",
                "gross_15m_bps",
                "gross_1h_bps",
                "spread_bps",
                "depth_10bps_usd",
                "visible_depth_usage",
                "taker_fee_bps_per_fill",
                "conservative_cost_bps",
                "conservative_net_15m_bps",
                "conservative_net_1h_bps",
                "gate_action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.direction,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.gross_15m_bps:.8f}",
                    f"{row.gross_1h_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.depth_10bps_usd:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    f"{row.taker_fee_bps_per_fill:.8f}",
                    f"{row.conservative_cost_bps:.8f}",
                    f"{row.conservative_net_15m_bps:.8f}",
                    f"{row.conservative_net_1h_bps:.8f}",
                    row.gate_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_microstructure_flow_paper_gate_md(
    rows: tuple[MicrostructureFlowPaperGateRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Microstructure Flow Paper Gate\n\n")
        handle.write(
            "This subtracts taker round-trip fees, current spread, and a rough "
            "visible-depth impact from microstructure flow labels. It is a small "
            "paper-probe gate, not a maker queue or fill model.\n\n"
        )
        handle.write(
            "| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.direction} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{row.gross_15m_bps:.2f} | "
                f"{row.conservative_net_15m_bps:.2f} | "
                f"{row.conservative_net_1h_bps:.2f} | "
                f"{row.spread_bps:.2f} | "
                f"{row.depth_10bps_usd:.0f} | "
                f"{row.visible_depth_usage:.4f} | "
                f"{row.gate_action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`microstructure_small_paper_probe` means the 15m and 1h directional "
            "labels survived a rough taker-fee, spread, and visible-depth check. "
            "It still needs real fill logs, queue/adverse-selection measurement, "
            "and repeat snapshots before it can be treated as a trading edge.\n"
        )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    snapshot: dict[str, str],
    size_usd: float,
    taker_fee_bps_per_fill: float,
) -> MicrostructureFlowPaperGateRow:
    spread_bps = float(snapshot["spread_bps"])
    depth_10bps_usd = float(snapshot["depth_10bps_usd"])
    depth_usage = size_usd / depth_10bps_usd if depth_10bps_usd > 0.0 else float("inf")
    gross_15m_bps = float(label.get("directional_return_15m") or "0") * 10_000.0
    gross_1h_bps = float(label.get("directional_return_1h") or "0") * 10_000.0
    depth_impact_bps = min(depth_usage, 1.0) * 10.0
    cost_bps = (taker_fee_bps_per_fill * 2.0) + spread_bps + depth_impact_bps
    net_15m_bps = gross_15m_bps - cost_bps
    net_1h_bps = gross_1h_bps - cost_bps
    gate_action, reason = _gate_action(
        label_status=label.get("label_status", ""),
        gross_15m_bps=gross_15m_bps,
        gross_1h_bps=gross_1h_bps,
        net_15m_bps=net_15m_bps,
        net_1h_bps=net_1h_bps,
        depth_usage=depth_usage,
    )
    asset = label["asset"]
    return MicrostructureFlowPaperGateRow(
        asset=asset,
        action=label["action"],
        direction=int(label["direction"]),
        candidate_size_usd=size_usd,
        gross_15m_bps=gross_15m_bps,
        gross_1h_bps=gross_1h_bps,
        spread_bps=spread_bps,
        depth_10bps_usd=depth_10bps_usd,
        visible_depth_usage=depth_usage,
        taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        conservative_cost_bps=cost_bps,
        conservative_net_15m_bps=net_15m_bps,
        conservative_net_1h_bps=net_1h_bps,
        gate_action=gate_action,
        reason=reason,
        next_step=_next_step(asset=asset, gate_action=gate_action),
    )


def _gate_action(
    *,
    label_status: str,
    gross_15m_bps: float,
    gross_1h_bps: float,
    net_15m_bps: float,
    net_1h_bps: float,
    depth_usage: float,
) -> tuple[str, str]:
    if label_status != "labeled_1h":
        return "wait_for_1h_label", "1h label is not mature yet"
    if gross_15m_bps <= 0.0 or gross_1h_bps <= 0.0:
        return "blocked_by_label", "15m and 1h directional labels do not both support the snapshot direction"
    if depth_usage > 0.25:
        return "too_large_for_visible_depth", "candidate size uses too much visible 10 bps depth"
    if net_15m_bps <= 0.0 or net_1h_bps <= 0.0:
        return "blocked_by_cost", "rough taker fee, spread, and visible-depth impact consume the label"
    return "microstructure_small_paper_probe", "survives rough fee, spread, 1h label, and visible-depth check"


def _next_step(*, asset: str, gate_action: str) -> str:
    if gate_action == "microstructure_small_paper_probe":
        return (
            f"paper-check {asset} microstructure flow with real fill, funding, "
            "queue, and adverse-selection logs"
        )
    return f"collect a fresh {asset} microstructure snapshot before any paper probe"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_snapshot.csv",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_forward_labels.csv",
    )
    parser.add_argument("--sizes-usd", nargs="+", type=float, default=list(DEFAULT_SIZES_USD))
    parser.add_argument("--taker-fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_paper_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_paper_gate.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_microstructure_flow_paper_gate_rows(
        snapshot_path=args.snapshot_path,
        label_path=args.label_path,
        sizes_usd=tuple(args.sizes_usd),
        taker_fee_bps_per_fill=args.taker_fee_bps_per_fill,
    )
    write_microstructure_flow_paper_gate_rows(rows, output_path=args.output_path)
    write_microstructure_flow_paper_gate_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"size={row.candidate_size_usd:.0f}",
            f"net15={row.conservative_net_15m_bps:.2f}",
            f"net1h={row.conservative_net_1h_bps:.2f}",
            row.gate_action,
        )


if __name__ == "__main__":
    main()
