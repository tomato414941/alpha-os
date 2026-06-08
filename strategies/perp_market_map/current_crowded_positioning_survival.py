from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERP_ROOT = ROOT / "perp_market_map"


@dataclass(frozen=True)
class CrowdedPositioningSurvivalRow:
    asset: str
    status: str
    side: str
    action: str
    survival_score: float
    derivatives_score: float
    label_gate_score: float
    venue_count: int
    actionable_venue_count: int
    funding_rate: float
    oi_volume_ratio: float
    net_directional_return_1h_proxy: float
    positive_directional_1h_rate: float
    evidence: str
    missing_work: str
    next_probe: str


def build_crowded_positioning_survival_rows(
    *,
    derivatives_path: Path = ROOT / "derivatives_positioning" / "current_coingecko_derivatives_positioning.csv",
    label_gate_path: Path = PERP_ROOT / "current_crowding_unwind_label_gate.csv",
) -> tuple[CrowdedPositioningSurvivalRow, ...]:
    label_gates = _best_label_gate_by_asset(label_gate_path)
    rows = tuple(
        _build_row(derivatives=derivatives, label_gate=label_gates.get(derivatives.get("index_id", "")))
        for derivatives in _best_derivatives_by_asset(derivatives_path).values()
    )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_crowded_positioning_survival_csv(
    rows: tuple[CrowdedPositioningSurvivalRow, ...],
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
                "action",
                "survival_score",
                "derivatives_score",
                "label_gate_score",
                "venue_count",
                "actionable_venue_count",
                "funding_rate",
                "oi_volume_ratio",
                "net_directional_return_1h_proxy",
                "positive_directional_1h_rate",
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
                    row.side,
                    row.action,
                    f"{row.survival_score:.8f}",
                    f"{row.derivatives_score:.8f}",
                    f"{row.label_gate_score:.8f}",
                    row.venue_count,
                    row.actionable_venue_count,
                    f"{row.funding_rate:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.net_directional_return_1h_proxy:.8f}",
                    f"{row.positive_directional_1h_rate:.8f}",
                    row.evidence,
                    row.missing_work,
                    row.next_probe,
                )
            )
    return output_path


def write_crowded_positioning_survival_md(
    rows: tuple[CrowdedPositioningSurvivalRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowded Positioning Survival\n\n")
        handle.write(
            "This joins current derivatives-positioning candidates to the crowding-unwind label gate. "
            "It separates crowded context from return-supported unwind alpha.\n\n"
        )
        handle.write(
            "| asset | status | side | action | score | deriv score | gate score | venues | actionable | net1h | hit1h | next probe |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:40]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.action} | "
                f"{row.survival_score:.4f} | "
                f"{row.derivatives_score:.4f} | "
                f"{row.label_gate_score:.4f} | "
                f"{row.venue_count} | "
                f"{row.actionable_venue_count} | "
                f"{row.net_directional_return_1h_proxy:.8f} | "
                f"{row.positive_directional_1h_rate:.4f} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _build_row(
    *,
    derivatives: dict[str, str],
    label_gate: dict[str, str] | None,
) -> CrowdedPositioningSurvivalRow:
    asset = derivatives.get("index_id", "")
    label = label_gate or {}
    status = _status(derivatives=derivatives, label_gate=label)
    derivatives_score = _float(derivatives.get("score"))
    label_gate_score = _float(label.get("label_gate_score"))
    venue_count = _int(label.get("venue_count"))
    actionable_venue_count = _int(label.get("actionable_venue_count"))
    net_proxy = _float(label.get("net_directional_return_1h_proxy"))
    hit_rate = _float(label.get("positive_directional_1h_rate"))
    survival_score = _survival_score(
        status=status,
        derivatives_score=derivatives_score,
        label_gate_score=label_gate_score,
        actionable_venue_count=actionable_venue_count,
        net_proxy=net_proxy,
        hit_rate=hit_rate,
    )
    evidence = (
        f"market={derivatives.get('market', '')}; "
        f"symbol={derivatives.get('symbol', '')}; "
        f"status={derivatives.get('status', '')}; "
        f"basis={derivatives.get('basis', '')}; "
        f"funding={derivatives.get('funding_rate', '')}; "
        f"oi_volume={derivatives.get('oi_volume_ratio', '')}; "
        f"label_status={label.get('label_gate_status', 'missing_label_gate')}"
    )
    return CrowdedPositioningSurvivalRow(
        asset=asset,
        status=status,
        side=derivatives.get("side", ""),
        action=label.get("action", _action_from_side(derivatives.get("side", ""))),
        survival_score=survival_score,
        derivatives_score=derivatives_score,
        label_gate_score=label_gate_score,
        venue_count=venue_count,
        actionable_venue_count=actionable_venue_count,
        funding_rate=_float(derivatives.get("funding_rate")),
        oi_volume_ratio=_float(derivatives.get("oi_volume_ratio")),
        net_directional_return_1h_proxy=net_proxy,
        positive_directional_1h_rate=hit_rate,
        evidence=evidence,
        missing_work=_missing_work(status),
        next_probe=_next_probe(status=status, asset=asset),
    )


def _best_derivatives_by_asset(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        asset = row.get("index_id", "")
        if not asset:
            continue
        current = rows.get(asset)
        if current is None or _float(row.get("score")) > _float(current.get("score")):
            rows[asset] = row
    return rows


def _best_label_gate_by_asset(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        asset = row.get("asset", "")
        if not asset:
            continue
        current = rows.get(asset)
        if current is None or _float(row.get("label_gate_score")) > _float(current.get("label_gate_score")):
            rows[asset] = row
    return rows


def _status(*, derivatives: dict[str, str], label_gate: dict[str, str]) -> str:
    if not label_gate:
        return "crowded_context_without_unwind_label"
    label_status = label_gate.get("label_gate_status", "")
    if label_status == "crowding_unwind_label_not_supported":
        return "crowding_context_not_alpha"
    if label_status == "crowding_context_only_needs_forward_labels":
        return "needs_forward_unwind_label"
    if _float(label_gate.get("net_directional_return_1h_proxy")) > 0.0 and _float(label_gate.get("positive_directional_1h_rate")) > 0.0:
        return "crowding_unwind_survival_candidate"
    if derivatives.get("status") in {"paper_oi_funding_crowding_watch", "paper_basis_funding_dislocation_watch"}:
        return "positioning_watch_needs_label"
    return "derivatives_context_only"


def _survival_score(
    *,
    status: str,
    derivatives_score: float,
    label_gate_score: float,
    actionable_venue_count: int,
    net_proxy: float,
    hit_rate: float,
) -> float:
    status_bonus = {
        "crowding_unwind_survival_candidate": 160.0,
        "positioning_watch_needs_label": 80.0,
        "needs_forward_unwind_label": 45.0,
        "crowded_context_without_unwind_label": 25.0,
        "derivatives_context_only": 10.0,
        "crowding_context_not_alpha": -60.0,
    }.get(status, 0.0)
    return (
        status_bonus
        + derivatives_score
        + label_gate_score * 0.5
        + actionable_venue_count * 12.0
        + net_proxy * 10_000.0
        + hit_rate * 40.0
    )


def _missing_work(status: str) -> str:
    if status == "crowding_unwind_survival_candidate":
        return "execution cost, funding PnL, impact, and repeated non-overlapping labels"
    if status == "crowding_context_not_alpha":
        return "current unwind labels reject the positioning thesis"
    if status == "needs_forward_unwind_label":
        return "cross-venue context exists but forward labels are not established"
    return "crowding context is visible but not connected to return-supported unwind labels"


def _next_probe(*, status: str, asset: str) -> str:
    if status == "crowding_unwind_survival_candidate":
        return f"paper-check {asset} crowding unwind with funding PnL, impact, and stop notes"
    if status == "crowding_context_not_alpha":
        return f"do not promote {asset} crowding; only revisit after fresh positive unwind labels"
    if status == "needs_forward_unwind_label":
        return f"label {asset} continuation versus unwind before treating crowding as alpha"
    return f"collect {asset} unwind labels before using crowded positioning as a signal"


def _action_from_side(side: str) -> str:
    if "watch_short" in side:
        return "short_carry_reversion_watch"
    if "watch_long" in side:
        return "long_carry_reversion_watch"
    return ""


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


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=PERP_ROOT / "current_crowded_positioning_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=PERP_ROOT / "current_crowded_positioning_survival.md")
    args = parser.parse_args()

    rows = build_crowded_positioning_survival_rows()
    write_crowded_positioning_survival_csv(rows, output_path=args.output_path)
    write_crowded_positioning_survival_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.asset, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
