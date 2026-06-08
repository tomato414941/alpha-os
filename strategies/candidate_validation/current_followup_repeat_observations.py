from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class FollowupRepeatObservation:
    timestamp: str
    asset: str
    source: str
    source_action: str
    direction: int
    priority: float
    mark_price: float
    annualized_funding: float
    spread_bps: float
    near_depth_10bps_notional: float
    observation_status: str
    reason: str


def build_followup_repeat_observations(
    *,
    context_path: Path = ROOT / "current_followup_execution_context.csv",
    top_assets: int = 12,
) -> tuple[FollowupRepeatObservation, ...]:
    context_rows = tuple(
        row for row in _read_rows(context_path) if row.get("action") == "tradable_context_ok"
    )[:top_assets]
    source_lookup = _source_lookup()
    observed_at = datetime.now(UTC).isoformat()
    observations: list[FollowupRepeatObservation] = []
    for context_row in context_rows:
        for source in _split(context_row.get("source", "")):
            source_key = (context_row["asset"], source)
            source_details = source_lookup.get(source_key)
            observations.append(
                _build_observation(
                    context_row=context_row,
                    source=source,
                    source_details=source_details,
                    timestamp=observed_at,
                )
            )
    return tuple(observations)


def write_followup_repeat_observations(
    rows: tuple[FollowupRepeatObservation, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset",
                "source",
                "source_action",
                "direction",
                "priority",
                "mark_price",
                "annualized_funding",
                "spread_bps",
                "near_depth_10bps_notional",
                "observation_status",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.source,
                    row.source_action,
                    row.direction,
                    f"{row.priority:.4f}",
                    f"{row.mark_price:.12f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    row.observation_status,
                    row.reason,
                )
            )
    return output_path


def write_followup_repeat_observations_md(
    rows: tuple[FollowupRepeatObservation, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Repeat Observations\n\n")
        handle.write(
            "This records fresh source-specific observations from the follow-up "
            "queue. Each row is asset x source, so mixed evidence is not averaged "
            "together before labeling.\n\n"
        )
        handle.write(
            "| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.source_action} | "
                f"{row.direction} | "
                f"{row.priority:.4f} | "
                f"{row.mark_price:.8f} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.spread_bps:.4f} | "
                f"{row.near_depth_10bps_notional:.0f} | "
                f"{row.observation_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`ready_for_label` means the source had a direction and can be labeled "
            "after 15m/1h. `missing_source_direction` keeps the context visible but "
            "does not create a directional alpha label.\n"
        )
    return output_path


def _build_observation(
    *,
    context_row: dict[str, str],
    source: str,
    source_details: dict[str, str] | None,
    timestamp: str,
) -> FollowupRepeatObservation:
    direction = 0 if source_details is None else int(source_details.get("direction") or "0")
    status = "ready_for_label" if direction != 0 else "missing_source_direction"
    return FollowupRepeatObservation(
        timestamp=timestamp,
        asset=context_row["asset"],
        source=source,
        source_action="" if source_details is None else source_details.get("action", ""),
        direction=direction,
        priority=float(context_row.get("priority") or "0"),
        mark_price=float(context_row.get("mark_price") or "0"),
        annualized_funding=float(context_row.get("annualized_funding") or "0"),
        spread_bps=float(context_row.get("spread_bps") or "0"),
        near_depth_10bps_notional=float(context_row.get("near_depth_10bps_notional") or "0"),
        observation_status=status,
        reason=_reason(source=source, direction=direction),
    )


def _source_lookup() -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    _add_hl_candidate_lookup(lookup)
    _add_direction_lookup(
        lookup,
        source="okx_pressure",
        path=STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure_forward_labels.csv",
        action_field="action",
        direction_field="direction",
    )
    _add_direction_lookup(
        lookup,
        source="liquidation",
        path=STRATEGIES_ROOT / "liquidation_flow" / "current_okx_liquidation_forward_labels.csv",
        action_field="action",
        direction_field="direction",
    )
    _add_direction_lookup(
        lookup,
        source="l2_imbalance",
        path=STRATEGIES_ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv",
        action_field="source",
        direction_field="direction",
        default_action="visible_book_imbalance",
    )
    _add_direction_lookup(
        lookup,
        source="sector_rotation",
        path=STRATEGIES_ROOT / "sector_rotation" / "current_category_tradable_forward_labels.csv",
        asset_field="symbol",
        action_field="category_action",
        direction_field="direction",
    )
    _add_direction_lookup(
        lookup,
        source="sector_perp_context",
        path=STRATEGIES_ROOT / "sector_rotation" / "current_category_perp_context.csv",
        asset_field="symbol",
        action_field="category_action",
        direction_field="direction",
    )
    _add_direction_lookup(
        lookup,
        source="exchange_catalyst",
        path=STRATEGIES_ROOT / "news_social" / "current_exchange_catalyst_forward_labels.csv",
        asset_field="symbol",
        action_field="catalyst_kind",
        direction_field="direction_hint",
    )
    _add_direction_lookup(
        lookup,
        source="protocol_activity",
        path=STRATEGIES_ROOT / "protocol_activity" / "current_protocol_activity_forward_labels.csv",
        asset_field="symbol",
        action_field="action",
        direction_field="direction_hint",
    )
    _add_direction_lookup(
        lookup,
        source="on_chain_flow",
        path=STRATEGIES_ROOT / "on_chain_flow" / "current_chain_tvl_flow_forward_labels.csv",
        asset_field="token_symbol",
        action_field="action",
        direction_field="direction",
    )
    return lookup


def _add_hl_candidate_lookup(lookup: dict[tuple[str, str], dict[str, str]]) -> None:
    path = ROOT / "current_hl_signal_forward_labels.csv"
    for row in _read_rows(path):
        action = row.get("action", "")
        direction = _direction_from_action(action)
        if direction == 0:
            continue
        lookup[(row["asset"], "hl_candidate")] = {
            "action": action,
            "direction": str(direction),
        }


def _add_direction_lookup(
    lookup: dict[tuple[str, str], dict[str, str]],
    *,
    source: str,
    path: Path,
    action_field: str,
    direction_field: str,
    asset_field: str = "asset",
    default_action: str = "",
) -> None:
    for row in _read_rows(path):
        asset = row.get(asset_field, "")
        if not asset:
            continue
        lookup[(asset, source)] = {
            "action": row.get(action_field, "") or default_action,
            "direction": row.get(direction_field, "0"),
        }


def _direction_from_action(action: str) -> int:
    if action.startswith("long_"):
        return 1
    if action.startswith("short_"):
        return -1
    return 0


def _reason(*, source: str, direction: int) -> str:
    if direction == 0:
        return f"{source} has no reusable direction in current labels"
    return f"{source} direction is ready for fresh 15m/1h labeling"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _split(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-path",
        type=Path,
        default=ROOT / "current_followup_execution_context.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_repeat_observations.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_repeat_observations.md",
    )
    parser.add_argument("--top-assets", type=int, default=12)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_followup_repeat_observations(
        context_path=args.context_path,
        top_assets=args.top_assets,
    )
    write_followup_repeat_observations(rows, output_path=args.output_path)
    write_followup_repeat_observations_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.source,
            row.source_action,
            f"dir={row.direction}",
            row.observation_status,
        )


if __name__ == "__main__":
    main()
