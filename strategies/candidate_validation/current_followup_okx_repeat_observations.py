from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.candidate_validation.current_followup_repeat_observations import (
    _source_lookup,
    _split,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FollowupOkxRepeatObservation:
    timestamp: str
    asset: str
    source: str
    source_action: str
    direction: int
    priority: float
    inst_id: str
    last_price: float
    annualized_funding: float | None
    spread_bps: float
    near_depth_10bps_notional: float
    observation_status: str
    reason: str


def build_followup_okx_repeat_observations(
    *,
    context_path: Path = ROOT / "current_followup_okx_execution_context.csv",
    top_assets: int = 15,
) -> tuple[FollowupOkxRepeatObservation, ...]:
    context_rows = tuple(
        row for row in _read_rows(context_path) if row.get("action") == "okx_context_ok"
    )[:top_assets]
    source_lookup = _source_lookup()
    observed_at = datetime.now(UTC).isoformat()
    observations: list[FollowupOkxRepeatObservation] = []
    for context_row in context_rows:
        for source in _split(context_row.get("source", "")):
            source_details = source_lookup.get((context_row["asset"], source))
            observations.append(
                _build_observation(
                    context_row=context_row,
                    source=source,
                    source_details=source_details,
                    timestamp=observed_at,
                )
            )
    return tuple(observations)


def write_followup_okx_repeat_observations(
    rows: tuple[FollowupOkxRepeatObservation, ...],
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
                "inst_id",
                "last_price",
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
                    row.inst_id,
                    f"{row.last_price:.12f}",
                    "" if row.annualized_funding is None else f"{row.annualized_funding:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    row.observation_status,
                    row.reason,
                )
            )
    return output_path


def write_followup_okx_repeat_observations_md(
    rows: tuple[FollowupOkxRepeatObservation, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up OKX Repeat Observations\n\n")
        handle.write(
            "This records source-specific OKX observations from the follow-up "
            "queue. It keeps OKX-only candidates visible for later labels.\n\n"
        )
        handle.write(
            "| asset | source | source action | dir | priority | inst | last | funding ann | spread bps | depth 10bps USD | status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.source_action} | "
                f"{row.direction} | "
                f"{row.priority:.4f} | "
                f"{row.inst_id} | "
                f"{row.last_price:.8f} | "
                f"{'' if row.annualized_funding is None else f'{row.annualized_funding:.6f}'} | "
                f"{row.spread_bps:.4f} | "
                f"{row.near_depth_10bps_notional:.0f} | "
                f"{row.observation_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "These rows should be labeled on OKX candles after 15m/1h. Positive "
            "or negative outcomes should be compared by source, not only by asset.\n"
        )
    return output_path


def _build_observation(
    *,
    context_row: dict[str, str],
    source: str,
    source_details: dict[str, str] | None,
    timestamp: str,
) -> FollowupOkxRepeatObservation:
    direction = 0 if source_details is None else int(source_details.get("direction") or "0")
    status = "ready_for_label" if direction != 0 else "missing_source_direction"
    return FollowupOkxRepeatObservation(
        timestamp=timestamp,
        asset=context_row["asset"],
        source=source,
        source_action="" if source_details is None else source_details.get("action", ""),
        direction=direction,
        priority=float(context_row.get("priority") or "0"),
        inst_id=context_row["inst_id"],
        last_price=float(context_row.get("last_price") or "0"),
        annualized_funding=_float_or_none(context_row.get("annualized_funding", "")),
        spread_bps=float(context_row.get("spread_bps") or "0"),
        near_depth_10bps_notional=float(context_row.get("near_depth_10bps_notional") or "0"),
        observation_status=status,
        reason=(
            f"{source} direction is ready for OKX 15m/1h labeling"
            if direction != 0
            else f"{source} has no reusable direction in current labels"
        ),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-path",
        type=Path,
        default=ROOT / "current_followup_okx_execution_context.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_okx_repeat_observations.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_okx_repeat_observations.md",
    )
    parser.add_argument("--top-assets", type=int, default=15)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_followup_okx_repeat_observations(
        context_path=args.context_path,
        top_assets=args.top_assets,
    )
    write_followup_okx_repeat_observations(rows, output_path=args.output_path)
    write_followup_okx_repeat_observations_md(rows, output_path=args.md_output_path, top=args.top)
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
