from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FollowupRepeatHistoryRow:
    timestamp: str
    venue: str
    asset: str
    source: str
    source_action: str
    direction: int
    priority: float
    instrument: str
    price: float
    annualized_funding: float | None
    spread_bps: float
    near_depth_10bps_notional: float
    observation_status: str
    reason: str


def build_followup_repeat_history_rows(
    *,
    history_path: Path = ROOT / "followup_repeat_observation_history.csv",
    hl_observation_path: Path = ROOT / "current_followup_repeat_observations.csv",
    okx_observation_path: Path = ROOT / "current_followup_okx_repeat_observations.csv",
) -> tuple[FollowupRepeatHistoryRow, ...]:
    existing = tuple(_history_row(row) for row in _read_rows(history_path))
    new_rows = tuple(_hl_row(row) for row in _read_rows(hl_observation_path)) + tuple(
        _okx_row(row) for row in _read_rows(okx_observation_path)
    )
    rows_by_key = {_key(row): row for row in existing}
    for row in new_rows:
        rows_by_key.setdefault(_key(row), row)
    return tuple(sorted(rows_by_key.values(), key=lambda row: row.timestamp))


def write_followup_repeat_history_csv(
    rows: tuple[FollowupRepeatHistoryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "venue",
                "asset",
                "source",
                "source_action",
                "direction",
                "priority",
                "instrument",
                "price",
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
                    row.venue,
                    row.asset,
                    row.source,
                    row.source_action,
                    row.direction,
                    f"{row.priority:.4f}",
                    row.instrument,
                    f"{row.price:.12f}",
                    "" if row.annualized_funding is None else f"{row.annualized_funding:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    row.observation_status,
                    row.reason,
                )
            )
    return output_path


def write_followup_repeat_history_md(
    rows: tuple[FollowupRepeatHistoryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready_rows = tuple(row for row in rows if row.observation_status == "ready_for_label")
    by_venue: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for row in ready_rows:
        by_venue[row.venue] = by_venue.get(row.venue, 0) + 1
        by_source[row.source] = by_source.get(row.source, 0) + 1
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Repeat History\n\n")
        handle.write(
            "This preserves source-specific repeat observations across runs. "
            "Current observation files may be regenerated; this history is the "
            "sample store for repeated alpha checks.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- ready rows: `{len(ready_rows)}`\n")
        handle.write(f"- by venue: `{_format_counts(by_venue)}`\n")
        handle.write(f"- by source: `{_format_counts(by_source)}`\n\n")
        handle.write("| timestamp | venue | asset | source | action | dir | priority | status |\n")
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | --- |\n")
        for row in tuple(sorted(rows, key=lambda item: item.timestamp, reverse=True))[:50]:
            handle.write(
                "| "
                f"{row.timestamp} | "
                f"{row.venue} | "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.source_action} | "
                f"{row.direction} | "
                f"{row.priority:.4f} | "
                f"{row.observation_status} |\n"
            )
    return output_path


def _hl_row(row: dict[str, str]) -> FollowupRepeatHistoryRow:
    return FollowupRepeatHistoryRow(
        timestamp=row["timestamp"],
        venue="HL",
        asset=row["asset"],
        source=row["source"],
        source_action=row.get("source_action", ""),
        direction=int(row.get("direction") or "0"),
        priority=float(row.get("priority") or "0"),
        instrument=row["asset"],
        price=float(row.get("mark_price") or "0"),
        annualized_funding=_float_or_none(row.get("annualized_funding", "")),
        spread_bps=float(row.get("spread_bps") or "0"),
        near_depth_10bps_notional=float(row.get("near_depth_10bps_notional") or "0"),
        observation_status=row.get("observation_status", ""),
        reason=row.get("reason", ""),
    )


def _okx_row(row: dict[str, str]) -> FollowupRepeatHistoryRow:
    return FollowupRepeatHistoryRow(
        timestamp=row["timestamp"],
        venue="OKX",
        asset=row["asset"],
        source=row["source"],
        source_action=row.get("source_action", ""),
        direction=int(row.get("direction") or "0"),
        priority=float(row.get("priority") or "0"),
        instrument=row.get("inst_id", ""),
        price=float(row.get("last_price") or "0"),
        annualized_funding=_float_or_none(row.get("annualized_funding", "")),
        spread_bps=float(row.get("spread_bps") or "0"),
        near_depth_10bps_notional=float(row.get("near_depth_10bps_notional") or "0"),
        observation_status=row.get("observation_status", ""),
        reason=row.get("reason", ""),
    )


def _history_row(row: dict[str, str]) -> FollowupRepeatHistoryRow:
    return FollowupRepeatHistoryRow(
        timestamp=row["timestamp"],
        venue=row["venue"],
        asset=row["asset"],
        source=row["source"],
        source_action=row.get("source_action", ""),
        direction=int(row.get("direction") or "0"),
        priority=float(row.get("priority") or "0"),
        instrument=row.get("instrument", ""),
        price=float(row.get("price") or "0"),
        annualized_funding=_float_or_none(row.get("annualized_funding", "")),
        spread_bps=float(row.get("spread_bps") or "0"),
        near_depth_10bps_notional=float(row.get("near_depth_10bps_notional") or "0"),
        observation_status=row.get("observation_status", ""),
        reason=row.get("reason", ""),
    )


def _key(row: FollowupRepeatHistoryRow) -> tuple[str, str, str, str, str]:
    return (row.timestamp, row.venue, row.asset, row.source, row.source_action)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def _format_counts(counts: dict[str, int]) -> str:
    return "; ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "followup_repeat_observation_history.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "followup_repeat_observation_history.md",
    )
    args = parser.parse_args()

    rows = build_followup_repeat_history_rows(history_path=args.history_path)
    write_followup_repeat_history_csv(rows, output_path=args.history_path)
    write_followup_repeat_history_md(rows, output_path=args.md_output_path)
    print(
        f"rows={len(rows)}",
        f"ready={sum(row.observation_status == 'ready_for_label' for row in rows)}",
    )


if __name__ == "__main__":
    main()
