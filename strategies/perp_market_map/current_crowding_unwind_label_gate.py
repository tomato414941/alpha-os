from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CrowdingUnwindLabelGateRow:
    asset: str
    action: str
    label_gate_status: str
    label_gate_score: float
    label_observations: int
    net_directional_return_1h_proxy: float
    positive_directional_1h_rate: float
    cross_venue_status: str
    venue_count: int
    actionable_venue_count: int
    execution_gate_action: str
    conservative_net_1h_bps: float
    paper_outcome_1h: str
    reason: str
    next_step: str


def build_crowding_unwind_label_gate_rows(
    *,
    validated_path: Path = ROOT / "current_crowding_reversion_validated_candidates.csv",
    cross_venue_path: Path = ROOT / "current_crowding_cross_venue_confirmation.csv",
    execution_path: Path = ROOT / "current_crowding_reversion_execution_check.csv",
    outcome_path: Path = ROOT / "current_crowding_reversion_paper_outcome.csv",
) -> tuple[CrowdingUnwindLabelGateRow, ...]:
    cross_venue_rows = _best_by_asset_action(_read_rows(cross_venue_path), score_column="score")
    execution_rows = _best_by_asset_action(_read_rows(execution_path), score_column="conservative_net_1h_bps")
    outcome_rows = _latest_by_asset_action(_read_rows(outcome_path), timestamp_column="entry_timestamp")
    rows = tuple(
        _build_row(
            validated=row,
            cross_venue=cross_venue_rows.get((row.get("asset", ""), row.get("action", "")), {}),
            execution=execution_rows.get((row.get("asset", ""), row.get("action", "")), {}),
            outcome=outcome_rows.get((row.get("asset", ""), row.get("action", "")), {}),
        )
        for row in _read_rows(validated_path)
    )
    return tuple(sorted(rows, key=lambda row: row.label_gate_score, reverse=True))


def write_crowding_unwind_label_gate_csv(
    rows: tuple[CrowdingUnwindLabelGateRow, ...],
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
                "label_gate_status",
                "label_gate_score",
                "label_observations",
                "net_directional_return_1h_proxy",
                "positive_directional_1h_rate",
                "cross_venue_status",
                "venue_count",
                "actionable_venue_count",
                "execution_gate_action",
                "conservative_net_1h_bps",
                "paper_outcome_1h",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.label_gate_status,
                    f"{row.label_gate_score:.8f}",
                    row.label_observations,
                    f"{row.net_directional_return_1h_proxy:.8f}",
                    f"{row.positive_directional_1h_rate:.8f}",
                    row.cross_venue_status,
                    row.venue_count,
                    row.actionable_venue_count,
                    row.execution_gate_action,
                    f"{row.conservative_net_1h_bps:.8f}",
                    row.paper_outcome_1h,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_crowding_unwind_label_gate_md(
    rows: tuple[CrowdingUnwindLabelGateRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    supported = tuple(row for row in rows if row.label_gate_status == "crowding_unwind_label_supported")
    rejected = tuple(row for row in rows if row.label_gate_status == "crowding_unwind_label_not_supported")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Unwind Label Gate\n\n")
        handle.write(
            "This gates crowded-positioning unwind ideas by forward label evidence before "
            "cross-venue derivatives context can promote them. Cross-venue OI/funding is "
            "context only; it is not a return label.\n\n"
        )
        handle.write(f"- rows: `{len(rows)}`\n")
        handle.write(f"- supported unwind labels: `{len(supported)}`\n")
        handle.write(f"- unsupported unwind labels: `{len(rejected)}`\n\n")
        handle.write(
            "| asset | action | gate | score | labels | net1h proxy | hit1h | cross venue | venues | execution | out1h | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.action} | "
                f"{row.label_gate_status} | "
                f"{row.label_gate_score:.4f} | "
                f"{row.label_observations} | "
                f"{row.net_directional_return_1h_proxy:.6f} | "
                f"{row.positive_directional_1h_rate:.4f} | "
                f"{row.cross_venue_status} | "
                f"{row.venue_count}/{row.actionable_venue_count} | "
                f"{row.execution_gate_action or '-'} | "
                f"{row.paper_outcome_1h or '-'} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`crowding_unwind_label_not_supported` means the current forward-label sample "
            "does not support treating the crowded-positioning unwind as alpha. It may "
            "still be useful as context for a different continuation, squeeze, or risk "
            "filter hypothesis.\n"
        )
    return output_path


def _build_row(
    *,
    validated: dict[str, str],
    cross_venue: dict[str, str],
    execution: dict[str, str],
    outcome: dict[str, str],
) -> CrowdingUnwindLabelGateRow:
    label_observations = _int(validated.get("label_observations"))
    net_1h = _float(validated.get("net_directional_return_1h_proxy"))
    hit_1h = _float(validated.get("positive_directional_1h_rate"))
    execution_gate = execution.get("gate_action", "")
    conservative_net_1h_bps = _float(execution.get("conservative_net_1h_bps"))
    outcome_1h = outcome.get("outcome_1h", "")
    status, reason, score = _label_gate(
        label_observations=label_observations,
        net_1h=net_1h,
        hit_1h=hit_1h,
        cross_venue_status=cross_venue.get("status", ""),
        actionable_venue_count=_int(cross_venue.get("actionable_venue_count")),
        execution_gate=execution_gate,
        conservative_net_1h_bps=conservative_net_1h_bps,
        outcome_1h=outcome_1h,
    )
    asset = validated.get("asset", "")
    return CrowdingUnwindLabelGateRow(
        asset=asset,
        action=validated.get("action", ""),
        label_gate_status=status,
        label_gate_score=score,
        label_observations=label_observations,
        net_directional_return_1h_proxy=net_1h,
        positive_directional_1h_rate=hit_1h,
        cross_venue_status=cross_venue.get("status", ""),
        venue_count=_int(cross_venue.get("venue_count")),
        actionable_venue_count=_int(cross_venue.get("actionable_venue_count")),
        execution_gate_action=execution_gate,
        conservative_net_1h_bps=conservative_net_1h_bps,
        paper_outcome_1h=outcome_1h,
        reason=reason,
        next_step=_next_step(asset=asset, status=status),
    )


def _label_gate(
    *,
    label_observations: int,
    net_1h: float,
    hit_1h: float,
    cross_venue_status: str,
    actionable_venue_count: int,
    execution_gate: str,
    conservative_net_1h_bps: float,
    outcome_1h: str,
) -> tuple[str, str, float]:
    cross_venue_bonus = min(actionable_venue_count, 3) * 1.0
    if outcome_1h == "paper_1h_win":
        return (
            "crowding_unwind_paper_outcome_supported",
            "paper outcome moved in the candidate direction after rough costs",
            85.0 + conservative_net_1h_bps / 10.0 + cross_venue_bonus,
        )
    if outcome_1h == "paper_1h_loss":
        return (
            "crowding_unwind_paper_outcome_failed",
            "paper outcome moved against the candidate after rough costs",
            20.0,
        )
    if execution_gate == "paper_execution_probe" and conservative_net_1h_bps > 0.0:
        return (
            "crowding_unwind_execution_probe_ready",
            "forward proxy and rough execution gate are both positive",
            72.0 + min(conservative_net_1h_bps / 10.0, 8.0) + cross_venue_bonus,
        )
    if net_1h > 0.0 and hit_1h >= 0.5 and label_observations >= 6:
        return (
            "crowding_unwind_label_supported",
            "repeat forward labels support the unwind before execution checks",
            68.0 + min(net_1h * 10_000.0 / 10.0, 8.0) + cross_venue_bonus,
        )
    if label_observations >= 3 and net_1h <= 0.0:
        return (
            "crowding_unwind_label_not_supported",
            "current forward labels do not support the unwind after funding and impact proxy",
            24.0 + max(net_1h * 10_000.0 / 10.0, -10.0),
        )
    if cross_venue_status:
        return (
            "crowding_context_only_needs_forward_labels",
            "cross-venue derivatives context exists but the unwind label is not established",
            38.0 + cross_venue_bonus,
        )
    return (
        "crowding_mapping_or_label_gap",
        "missing cross-venue context or forward labels",
        30.0,
    )


def _next_step(*, asset: str, status: str) -> str:
    if status in {"crowding_unwind_label_not_supported", "crowding_unwind_paper_outcome_failed"}:
        return f"do not promote {asset} unwind; only revisit if a fresh snapshot creates positive forward labels"
    if status.endswith("_supported") or status == "crowding_unwind_execution_probe_ready":
        return f"repeat {asset} unwind on fresh snapshots with depth, funding timing, and stop behavior"
    if status == "crowding_context_only_needs_forward_labels":
        return f"label {asset} continuation versus unwind separately before treating crowding as alpha"
    return f"fix {asset} mapping and collect forward labels before using crowding context"


def _best_by_asset_action(
    rows: tuple[dict[str, str], ...],
    *,
    score_column: str,
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda row: _float(row.get(score_column)), reverse=True):
        key = (row.get("asset", ""), row.get("action", ""))
        if not key[0] or not key[1] or key in output:
            continue
        output[key] = row
    return output


def _latest_by_asset_action(
    rows: tuple[dict[str, str], ...],
    *,
    timestamp_column: str,
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda row: row.get(timestamp_column, ""), reverse=True):
        key = (row.get("asset", ""), row.get("action", ""))
        if not key[0] or not key[1] or key in output:
            continue
        output[key] = row
    return output


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    return int(_float(value))


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-path", type=Path, default=ROOT / "current_crowding_reversion_validated_candidates.csv")
    parser.add_argument("--cross-venue-path", type=Path, default=ROOT / "current_crowding_cross_venue_confirmation.csv")
    parser.add_argument("--execution-path", type=Path, default=ROOT / "current_crowding_reversion_execution_check.csv")
    parser.add_argument("--outcome-path", type=Path, default=ROOT / "current_crowding_reversion_paper_outcome.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_crowding_unwind_label_gate.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_crowding_unwind_label_gate.md")
    args = parser.parse_args()

    rows = build_crowding_unwind_label_gate_rows(
        validated_path=args.validated_path,
        cross_venue_path=args.cross_venue_path,
        execution_path=args.execution_path,
        outcome_path=args.outcome_path,
    )
    write_crowding_unwind_label_gate_csv(rows, output_path=args.output_path)
    write_crowding_unwind_label_gate_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.asset, row.label_gate_status, f"{row.label_gate_score:.4f}")


if __name__ == "__main__":
    main()
