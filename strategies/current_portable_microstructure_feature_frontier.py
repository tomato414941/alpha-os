from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_ASSETS = ("BTC", "ETH", "SOL", "HYPE")


@dataclass(frozen=True)
class PortableMicrostructureFeatureRow:
    asset: str
    status: str
    priority: float
    spread_bps: str
    depth_10bps_usd: str
    book_imbalance_10bps: str
    trade_imbalance: str
    pressure_score: str
    snapshot_action: str
    directional_return_15m: str
    directional_return_1h: str
    feature_state: str
    missing_link: str
    next_step: str


def build_portable_microstructure_feature_frontier(
    *,
    snapshot_path: Path = ROOT / "market_making" / "current_microstructure_flow_snapshot.csv",
    l2_label_path: Path = ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv",
) -> tuple[PortableMicrostructureFeatureRow, ...]:
    snapshots = {row.get("asset", ""): row for row in _read_rows(snapshot_path)}
    labels = {row.get("asset", ""): row for row in _read_rows(l2_label_path)}
    rows = tuple(
        _frontier_row(asset=asset, snapshot=snapshots.get(asset, {}), label=labels.get(asset, {}))
        for asset in TARGET_ASSETS
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_portable_microstructure_feature_frontier_csv(
    rows: tuple[PortableMicrostructureFeatureRow, ...],
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
                "priority",
                "spread_bps",
                "depth_10bps_usd",
                "book_imbalance_10bps",
                "trade_imbalance",
                "pressure_score",
                "snapshot_action",
                "directional_return_15m",
                "directional_return_1h",
                "feature_state",
                "missing_link",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.status,
                    f"{row.priority:.8f}",
                    row.spread_bps,
                    row.depth_10bps_usd,
                    row.book_imbalance_10bps,
                    row.trade_imbalance,
                    row.pressure_score,
                    row.snapshot_action,
                    row.directional_return_15m,
                    row.directional_return_1h,
                    row.feature_state,
                    row.missing_link,
                    row.next_step,
                )
            )
    return output_path


def write_portable_microstructure_feature_frontier_md(
    rows: tuple[PortableMicrostructureFeatureRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Portable Microstructure Feature Frontier\n\n")
        handle.write(
            "This checks whether the same microstructure feature shape can be compared across BTC, ETH, SOL, "
            "and HYPE. It is a feature frontier, not a trading strategy or execution instruction.\n\n"
        )
        handle.write(
            "| asset | status | priority | spread | depth | book imbalance | trade imbalance | 15m | 1h | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.priority:.4f} | "
                f"{row.spread_bps} | "
                f"{row.depth_10bps_usd} | "
                f"{row.book_imbalance_10bps} | "
                f"{row.trade_imbalance} | "
                f"{row.directional_return_15m} | "
                f"{row.directional_return_1h} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _frontier_row(
    *,
    asset: str,
    snapshot: dict[str, str],
    label: dict[str, str],
) -> PortableMicrostructureFeatureRow:
    status = _status(snapshot=snapshot, label=label)
    priority = _priority(status=status, snapshot=snapshot, label=label)
    feature_state = (
        f"spread={snapshot.get('spread_bps', '')}; "
        f"depth10={snapshot.get('depth_10bps_usd', '')}; "
        f"book_imbalance={snapshot.get('book_imbalance_10bps', '')}; "
        f"trade_imbalance={snapshot.get('trade_imbalance', '')}; "
        f"pressure={snapshot.get('pressure_score', '')}; "
        f"label15={label.get('directional_return_15m', '')}; "
        f"label1h={label.get('directional_return_1h', '')}"
    )
    return PortableMicrostructureFeatureRow(
        asset=asset,
        status=status,
        priority=priority,
        spread_bps=snapshot.get("spread_bps", ""),
        depth_10bps_usd=snapshot.get("depth_10bps_usd", ""),
        book_imbalance_10bps=snapshot.get("book_imbalance_10bps", ""),
        trade_imbalance=snapshot.get("trade_imbalance", ""),
        pressure_score=snapshot.get("pressure_score", ""),
        snapshot_action=snapshot.get("action", ""),
        directional_return_15m=label.get("directional_return_15m", ""),
        directional_return_1h=label.get("directional_return_1h", ""),
        feature_state=feature_state,
        missing_link=_missing_link(status),
        next_step=_next_step(status=status, asset=asset),
    )


def _status(*, snapshot: dict[str, str], label: dict[str, str]) -> str:
    if not snapshot:
        return "missing_snapshot"
    if not label:
        return "missing_forward_label"
    ret15 = _float(label.get("directional_return_15m"))
    ret1h = _float(label.get("directional_return_1h"))
    if ret15 > 0.0 and ret1h > 0.0:
        return "cross_horizon_supported"
    if ret15 > 0.0 and ret1h <= 0.0:
        return "short_horizon_only"
    if ret15 <= 0.0 and ret1h > 0.0:
        return "delayed_or_reversal_support"
    return "microstructure_label_failed"


def _priority(*, status: str, snapshot: dict[str, str], label: dict[str, str]) -> float:
    status_bonus = {
        "cross_horizon_supported": 110.0,
        "delayed_or_reversal_support": 90.0,
        "short_horizon_only": 70.0,
        "microstructure_label_failed": 30.0,
        "missing_forward_label": 20.0,
        "missing_snapshot": 0.0,
    }[status]
    depth_bonus = min(_float(snapshot.get("depth_10bps_usd")) / 100_000.0, 30.0)
    spread_penalty = min(_float(snapshot.get("spread_bps")) * 2.0, 20.0)
    label_bonus = abs(_float(label.get("directional_return_15m"))) * 500.0 + abs(
        _float(label.get("directional_return_1h"))
    ) * 300.0
    return status_bonus + depth_bonus + label_bonus - spread_penalty


def _missing_link(status: str) -> str:
    if status in {"cross_horizon_supported", "delayed_or_reversal_support", "short_horizon_only"}:
        return "same-feature repeat, maker/taker execution split, queue/adverse-selection stress, and cross-asset stability"
    if status == "microstructure_label_failed":
        return "failure-regime reason and whether the same feature works on another asset"
    return "snapshot and forward label coverage"


def _next_step(*, status: str, asset: str) -> str:
    if status == "cross_horizon_supported":
        return f"add {asset} to the shared microstructure feature table and test feature stability across assets"
    if status == "delayed_or_reversal_support":
        return f"split {asset} horizon handling before putting this feature into a shared table"
    if status == "short_horizon_only":
        return f"keep {asset} as a 15m-only feature candidate and reject 1h holding unless repeated"
    if status == "microstructure_label_failed":
        return f"use {asset} as a negative-control asset for the shared feature table"
    return f"collect {asset} snapshot and forward labels before shared-feature comparison"


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
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "market_making" / "current_microstructure_flow_snapshot.csv",
    )
    parser.add_argument(
        "--l2-label-path",
        type=Path,
        default=ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_portable_microstructure_feature_frontier.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_feature_frontier.md",
    )
    args = parser.parse_args()

    rows = build_portable_microstructure_feature_frontier(
        snapshot_path=args.snapshot_path,
        l2_label_path=args.l2_label_path,
    )
    write_portable_microstructure_feature_frontier_csv(rows, output_path=args.output_path)
    write_portable_microstructure_feature_frontier_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.status, row.asset, f"{row.priority:.4f}", row.next_step)


if __name__ == "__main__":
    main()
