from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_FLOW_ROOT = ROOT / "event_flow"


@dataclass(frozen=True)
class OfiExecutionSurvivalRow:
    asset: str
    status: str
    action: str
    execution_mode: str
    feature_route: str
    survival_score: float
    maker_net_bps: float
    current_pressure: float
    spread_bps: float
    near_depth_10bps_notional: float
    visible_depth_usage_100usd: float
    l2_net_15m_bps: float
    l2_net_1h_bps: float
    evidence: str
    missing_work: str
    next_probe: str


def build_ofi_execution_survival_rows(
    *,
    cost_sweep_path: Path = EVENT_FLOW_ROOT / "book_depth_execution_cost_sweep.csv",
    l2_monitor_path: Path = ROOT / "market_making" / "current_l2_imbalance_monitor_summary.csv",
    l2_paper_gate_path: Path = ROOT / "market_making" / "current_l2_imbalance_paper_gate.csv",
) -> tuple[OfiExecutionSurvivalRow, ...]:
    maker_route = _best_maker_route(cost_sweep_path)
    if not maker_route:
        return ()
    gates = _best_gate_by_asset(l2_paper_gate_path)
    rows = tuple(
        _build_row(route=maker_route, monitor=monitor, gate=gates.get(monitor.get("asset", "")))
        for monitor in _read_rows(l2_monitor_path)
        if _matches_route(route=maker_route, monitor=monitor)
    )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_ofi_execution_survival_csv(rows: tuple[OfiExecutionSurvivalRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "status",
                "action",
                "execution_mode",
                "feature_route",
                "survival_score",
                "maker_net_bps",
                "current_pressure",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage_100usd",
                "l2_net_15m_bps",
                "l2_net_1h_bps",
                "evidence",
                "missing_work",
                "next_probe",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.status,
                    row.action,
                    row.execution_mode,
                    row.feature_route,
                    f"{row.survival_score:.8f}",
                    f"{row.maker_net_bps:.8f}",
                    f"{row.current_pressure:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    f"{row.visible_depth_usage_100usd:.8f}",
                    f"{row.l2_net_15m_bps:.8f}",
                    f"{row.l2_net_1h_bps:.8f}",
                    row.evidence,
                    row.missing_work,
                    row.next_probe,
                )
            )
    return output_path


def write_ofi_execution_survival_md(rows: tuple[OfiExecutionSurvivalRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OFI Execution Survival\n\n")
        handle.write(
            "This joins the book-depth execution-cost sweep to current L2 imbalance assets. "
            "Rows are execution-survival probes, not standalone trading strategies.\n\n"
        )
        handle.write(
            "| asset | status | action | mode | score | maker net | pressure | spread | depth | net15 | net1h | next probe |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:30]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.action} | "
                f"{row.execution_mode} | "
                f"{row.survival_score:.4f} | "
                f"{row.maker_net_bps:.4f} | "
                f"{row.current_pressure:.4f} | "
                f"{row.spread_bps:.4f} | "
                f"{row.near_depth_10bps_notional:.2f} | "
                f"{row.l2_net_15m_bps:.4f} | "
                f"{row.l2_net_1h_bps:.4f} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _build_row(
    *,
    route: dict[str, str],
    monitor: dict[str, str],
    gate: dict[str, str] | None,
) -> OfiExecutionSurvivalRow:
    asset = monitor.get("asset", "")
    maker_net = _float(route.get("test_net_bps"))
    pressure = abs(_float(monitor.get("mean_imbalance_10_bps")))
    spread = _float(monitor.get("mean_spread_bps"))
    depth = _float(monitor.get("mean_near_depth_10bps_notional"))
    usage = 100.0 / depth if depth > 0.0 else 0.0
    net15 = _float((gate or {}).get("net_15m_bps"))
    net1h = _float((gate or {}).get("net_1h_bps"))
    status = _status(maker_net=maker_net, net15=net15, net1h=net1h, spread=spread, usage=usage)
    survival_score = _survival_score(
        maker_net=maker_net,
        pressure=pressure,
        spread=spread,
        depth=depth,
        usage=usage,
        net15=net15,
        net1h=net1h,
        status=status,
    )
    feature_route = f"{route.get('feature', '')}/{route.get('bucket', '')}/{route.get('action', '')}"
    evidence = (
        f"route_hit={route.get('test_hit_rate', '')}; "
        f"route_gross={route.get('test_gross_bps', '')}; "
        f"route_net={maker_net:.4f}; "
        f"monitor_direction_persistence={monitor.get('direction_persistence_rate', '')}; "
        f"l2_gate={(gate or {}).get('gate_action', '')}"
    )
    return OfiExecutionSurvivalRow(
        asset=asset,
        status=status,
        action=route.get("action", ""),
        execution_mode=route.get("execution_mode", ""),
        feature_route=feature_route,
        survival_score=survival_score,
        maker_net_bps=maker_net,
        current_pressure=pressure,
        spread_bps=spread,
        near_depth_10bps_notional=depth,
        visible_depth_usage_100usd=usage,
        l2_net_15m_bps=net15,
        l2_net_1h_bps=net1h,
        evidence=evidence,
        missing_work=_missing_work(status),
        next_probe=_next_probe(status=status, asset=asset),
    )


def _best_maker_route(path: Path) -> dict[str, str]:
    rows = tuple(
        row
        for row in _read_rows(path)
        if row.get("viability_status") == "maker_or_internalized_candidate"
        and row.get("execution_mode") == "maker_or_internalized"
    )
    return max(rows, key=lambda row: _float(row.get("viability_score")), default={})


def _best_gate_by_asset(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        asset = row.get("asset", "")
        if not asset:
            continue
        current = rows.get(asset)
        if current is None or _float(row.get("candidate_size_usd")) < _float(current.get("candidate_size_usd")):
            rows[asset] = row
    return rows


def _matches_route(*, route: dict[str, str], monitor: dict[str, str]) -> bool:
    action = route.get("action", "")
    imbalance = _float(monitor.get("mean_imbalance_10_bps"))
    if action == "paper_short":
        return imbalance < 0.0
    if action == "paper_long":
        return imbalance > 0.0
    return False


def _status(*, maker_net: float, net15: float, net1h: float, spread: float, usage: float) -> str:
    if maker_net <= 0.0:
        return "not_viable_after_cost"
    if spread > 5.0 or usage > 0.05:
        return "execution_world_blocks_ofi"
    if net15 > 0.0 and net1h <= 0.0:
        return "short_horizon_maker_probe_only"
    if net15 > 0.0 and net1h > 0.0:
        return "maker_ofi_survival_candidate"
    return "current_l2_label_missing_or_negative"


def _survival_score(
    *,
    maker_net: float,
    pressure: float,
    spread: float,
    depth: float,
    usage: float,
    net15: float,
    net1h: float,
    status: str,
) -> float:
    status_bonus = {
        "maker_ofi_survival_candidate": 180.0,
        "short_horizon_maker_probe_only": 130.0,
        "current_l2_label_missing_or_negative": 40.0,
        "execution_world_blocks_ofi": -80.0,
        "not_viable_after_cost": -120.0,
    }.get(status, 0.0)
    depth_score = min(depth / 100_000.0, 40.0)
    return status_bonus + maker_net * 80.0 + pressure * 60.0 + max(net15, 0.0) * 0.4 + max(net1h, 0.0) * 0.2 + depth_score - spread * 3.0 - usage * 300.0


def _missing_work(status: str) -> str:
    if status == "maker_ofi_survival_candidate":
        return "maker fill probability, queue position, cancellation, and adverse-selection model"
    if status == "short_horizon_maker_probe_only":
        return "1h reverses or decays; horizon, stop, and maker fill model are unresolved"
    if status == "execution_world_blocks_ofi":
        return "spread, depth, or size prevents this OFI state from being executable"
    return "current L2 label does not support promotion"


def _next_probe(*, status: str, asset: str) -> str:
    if status == "maker_ofi_survival_candidate":
        return f"paper-check {asset} OFI with maker fill, queue, cancel, and adverse-selection notes"
    if status == "short_horizon_maker_probe_only":
        return f"label {asset} OFI at 5m/15m only; do not hold to 1h without a separate rule"
    if status == "execution_world_blocks_ofi":
        return f"reject {asset} OFI for current size or wait for tighter spread and deeper book"
    return f"collect a fresh {asset} L2 label before using OFI as an alpha feature"


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
    parser.add_argument("--output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_execution_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=EVENT_FLOW_ROOT / "current_ofi_execution_survival.md")
    args = parser.parse_args()

    rows = build_ofi_execution_survival_rows()
    write_ofi_execution_survival_csv(rows, output_path=args.output_path)
    write_ofi_execution_survival_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.asset, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
