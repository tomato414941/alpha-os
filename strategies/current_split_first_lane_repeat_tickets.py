from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SplitFirstLaneRepeatTicket:
    ticket_id: str
    opened_at: str
    previous_ticket_id: str
    asset: str
    opportunity: str
    decision: str
    candidate_size_usd: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    estimated_net_after_cost_bps: str
    required_record: str
    next_step: str


def build_split_first_lane_repeat_tickets(
    *,
    queue_path: Path = ROOT / "current_split_first_lane_repeat_queue.csv",
    candidates_path: Path = ROOT / "current_cost_adjusted_alpha_candidates.csv",
    existing_tickets_path: Path | None = None,
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
) -> tuple[SplitFirstLaneRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    candidates = _candidate_context(candidates_path)
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = []
    for queue_row in _read_rows(queue_path):
        if queue_row.get("action") != "open_lane_repeat_probe":
            continue
        ticket_id = _ticket_id(queue_row)
        if ticket_id in existing:
            rows.append(existing[ticket_id])
            continue
        candidate = candidates.get(
            (
                queue_row.get("asset", ""),
                queue_row.get("lane_opportunity", ""),
                queue_row.get("cluster_decision", ""),
            ),
            {},
        )
        entry_mark, entry_source = _entry_mark(asset=queue_row.get("asset", ""), marks=marks)
        rows.append(
            SplitFirstLaneRepeatTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                previous_ticket_id=queue_row.get("queue_id", ""),
                asset=queue_row.get("asset", ""),
                opportunity=queue_row.get("lane_opportunity", ""),
                decision=queue_row.get("cluster_decision", ""),
                candidate_size_usd=candidate.get("candidate_size_usd", ""),
                checkpoints="15m,1h",
                entry_mark=entry_mark,
                entry_source=entry_source,
                estimated_net_after_cost_bps=candidate.get("estimated_net_after_cost_bps", ""),
                required_record=queue_row.get("required_record", ""),
                next_step=queue_row.get("next_step", ""),
            )
        )
    return tuple(rows)


def write_split_first_lane_repeat_tickets_csv(
    rows: tuple[SplitFirstLaneRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "opened_at",
                "previous_ticket_id",
                "asset",
                "opportunity",
                "decision",
                "candidate_size_usd",
                "checkpoints",
                "entry_mark",
                "entry_source",
                "estimated_net_after_cost_bps",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.previous_ticket_id,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.candidate_size_usd,
                    row.checkpoints,
                    row.entry_mark,
                    row.entry_source,
                    row.estimated_net_after_cost_bps,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_split_first_lane_repeat_tickets_md(
    rows: tuple[SplitFirstLaneRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Split First Lane Repeat Tickets\n\n")
        handle.write(
            "These are lane-level repeat tickets opened after mixed symbol clusters were split. "
            "They are not live trade instructions.\n\n"
        )
        handle.write("| ticket | previous | asset | decision | size USD | entry | checkpoints | net after cost | next step |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.previous_ticket_id} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{row.estimated_net_after_cost_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _candidate_context(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in _read_rows(path):
        key = (row.get("asset", ""), row.get("opportunity", ""), row.get("decision", ""))
        rows_by_key.setdefault(key, []).append(row)
    return {
        key: max(rows, key=lambda row: _float(row.get("priority_score")))
        for key, rows in rows_by_key.items()
    }


def _entry_mark(*, asset: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _ticket_id(row: dict[str, str]) -> str:
    return f"split-repeat-{row.get('asset', '').lower()}-{_slug(row.get('lane_opportunity', ''))}"


def _existing_tickets(path: Path | None) -> tuple[SplitFirstLaneRepeatTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            SplitFirstLaneRepeatTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                previous_ticket_id=row.get("previous_ticket_id", ""),
                asset=row.get("asset", ""),
                opportunity=row.get("opportunity", ""),
                decision=row.get("decision", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                checkpoints=row.get("checkpoints", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                estimated_net_after_cost_bps=row.get("estimated_net_after_cost_bps", ""),
                required_record=row.get("required_record", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


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


def _slug(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-path", type=Path, default=ROOT / "current_split_first_lane_repeat_queue.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_split_first_lane_repeat_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_split_first_lane_repeat_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_split_first_lane_repeat_tickets(
        queue_path=args.queue_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_split_first_lane_repeat_tickets_csv(rows, output_path=args.output_path)
    write_split_first_lane_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.entry_mark, row.estimated_net_after_cost_bps)


if __name__ == "__main__":
    main()
