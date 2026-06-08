from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://link.springer.com/article/10.1186/s40854-025-00831-7"


@dataclass(frozen=True)
class TailConnectednessRegimeRow:
    regime_id: str
    regime_role: str
    status: str
    affected_assets: str
    source_count: int
    severity_score: float
    connectedness_score: float
    evidence: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_tail_connectedness_regime_rows(root: Path = ROOT) -> tuple[TailConnectednessRegimeRow, ...]:
    stress_rows = _read_rows(root / "anomaly_stress" / "current_cross_market_stress_anomaly.csv")
    event_rows = _read_rows(root / "news_social" / "current_event_pressure_cluster.csv")
    sector_rows = _read_rows(root / "sector_rotation" / "current_category_perp_context.csv")
    rows = (
        _stress_cluster_row(stress_rows, lane="stablecoin_liquidity", regime_id="stablecoin_peg_tail_regime"),
        _stress_cluster_row(stress_rows, lane="defi_lending", regime_id="defi_lending_tail_regime"),
        _stress_cluster_row(stress_rows, lane="options_volatility", regime_id="volatility_tail_regime"),
        _event_connectedness_row(event_rows),
        _sector_connectedness_row(sector_rows),
    )
    return tuple(sorted((row for row in rows if row is not None), key=lambda row: row.connectedness_score, reverse=True))


def write_tail_connectedness_regime_csv(
    rows: tuple[TailConnectednessRegimeRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "regime_id",
                "regime_role",
                "status",
                "affected_assets",
                "source_count",
                "severity_score",
                "connectedness_score",
                "evidence",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.regime_id,
                    row.regime_role,
                    row.status,
                    row.affected_assets,
                    row.source_count,
                    f"{row.severity_score:.8f}",
                    f"{row.connectedness_score:.8f}",
                    row.evidence,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_tail_connectedness_regime_md(
    rows: tuple[TailConnectednessRegimeRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Tail Connectedness Regime\n\n")
        handle.write(
            "This groups current anomaly, event-pressure, and sector states into broad tail or "
            "connectedness regimes. It is a regime/control table, not a directional trade list.\n\n"
        )
        handle.write("| regime | role | status | assets | sources | severity | connectedness | next probe |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.regime_id} | {row.regime_role} | {row.status} | {_escape(row.affected_assets)} | "
                f"{row.source_count} | {row.severity_score:.4f} | {row.connectedness_score:.4f} | "
                f"{_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Tail-regime rows should condition downstream labels. They should not be collapsed into "
            "a long/short action until the connected assets, timestamps, execution costs, and regime "
            "persistence are measured.\n"
        )
    return output_path


def _stress_cluster_row(
    rows: tuple[dict[str, str], ...],
    *,
    lane: str,
    regime_id: str,
) -> TailConnectednessRegimeRow | None:
    lane_rows = tuple(row for row in rows if row.get("source_lane") == lane)
    if not lane_rows:
        return None
    source_count = len(lane_rows)
    severity = sum(_float(row.get("severity")) for row in lane_rows[:8])
    score = sum(_float(row.get("score")) for row in lane_rows[:8]) / min(source_count, 8)
    subjects = ", ".join(row.get("subject", "") for row in lane_rows[:6])
    return TailConnectednessRegimeRow(
        regime_id=regime_id,
        regime_role=_stress_regime_role(lane),
        status=_stress_status(lane=lane, source_count=source_count, score=score),
        affected_assets=subjects,
        source_count=source_count,
        severity_score=severity,
        connectedness_score=score + min(source_count, 10) * 4.0,
        evidence=f"top_subjects={subjects}; mean_score={score:.4f}; source_lane={lane}",
        missing_data="rolling tail dependence, tradable-route validation, regime duration, and cross-asset labels",
        next_probe=_stress_next_probe(lane),
    )


def _event_connectedness_row(rows: tuple[dict[str, str], ...]) -> TailConnectednessRegimeRow | None:
    candidates = tuple(row for row in rows if _float(row.get("score")) >= 40.0)
    if not candidates:
        return None
    side_counts = Counter(row.get("side", "") for row in candidates)
    assets = ", ".join(row.get("symbol", "") for row in candidates[:8])
    severity = sum(_float(row.get("score")) for row in candidates[:8]) / 100.0
    connectedness = severity * 12.0 + len(candidates) * 5.0 + len(side_counts) * 6.0
    status = "mixed_event_tail_context" if len(side_counts) > 1 else "same_side_event_tail_context"
    return TailConnectednessRegimeRow(
        regime_id="event_pressure_connectedness_regime",
        regime_role="event_connectedness_control",
        status=status,
        affected_assets=assets,
        source_count=len(candidates),
        severity_score=severity,
        connectedness_score=connectedness,
        evidence=f"sides={dict(side_counts)}; assets={assets}",
        missing_data="duplicate-source control, beta attribution, event timestamp quality, and cross-asset spillover labels",
        next_probe="label event-pressure assets as a connected regime before using any one event as standalone alpha",
    )


def _sector_connectedness_row(rows: tuple[dict[str, str], ...]) -> TailConnectednessRegimeRow | None:
    candidates = tuple(row for row in rows if _float(row.get("context_score")) > 0.0)
    if not candidates:
        return None
    by_category: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        by_category[row.get("category_name", "")].add(row.get("symbol", ""))
    category, assets_set = max(by_category.items(), key=lambda item: len(item[1]))
    assets = ", ".join(sorted(assets_set))
    severity = sum(abs(_float(row.get("context_score"))) for row in candidates[:12])
    connectedness = severity * 18.0 + len(assets_set) * 7.0
    return TailConnectednessRegimeRow(
        regime_id="sector_momentum_connectedness_regime",
        regime_role="sector_connectedness_control",
        status="sector_regime_before_single_asset_label",
        affected_assets=assets,
        source_count=len(candidates),
        severity_score=severity,
        connectedness_score=connectedness,
        evidence=f"largest_category={category}; category_asset_count={len(assets_set)}",
        missing_data="category constituent weights, spillover labels, funding support by asset, and regime persistence",
        next_probe=f"label {category} as a sector regime before promoting a single constituent",
    )


def _stress_regime_role(lane: str) -> str:
    return {
        "stablecoin_liquidity": "peg_tail_control",
        "defi_lending": "credit_liquidity_tail_control",
        "options_volatility": "volatility_tail_control",
    }.get(lane, "tail_control")


def _stress_status(*, lane: str, source_count: int, score: float) -> str:
    if lane == "stablecoin_liquidity" and source_count >= 3:
        return "multi_peg_tail_regime"
    if lane == "defi_lending" and source_count >= 3:
        return "lending_liquidity_tail_regime"
    if lane == "options_volatility":
        return "volatility_mispricing_tail_regime"
    if score >= 80.0:
        return "tail_regime_watch"
    return "tail_context_only"


def _stress_next_probe(lane: str) -> str:
    return {
        "stablecoin_liquidity": "treat peg anomalies as a tail regime and validate redemption/route mechanics before trading",
        "defi_lending": "condition lending/yield labels on credit-liquidity stress before any supply action",
        "options_volatility": "test whether cheap-vol candidates persist as a volatility regime after hedge costs",
    }.get(lane, "condition downstream labels on this tail regime")


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_tail_connectedness_regime.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=LANE_ROOT / "current_tail_connectedness_regime.md",
    )
    args = parser.parse_args()

    rows = build_tail_connectedness_regime_rows()
    write_tail_connectedness_regime_csv(rows, output_path=args.output_path)
    write_tail_connectedness_regime_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(row.status, row.regime_id, f"connectedness={row.connectedness_score:.4f}")


if __name__ == "__main__":
    main()
