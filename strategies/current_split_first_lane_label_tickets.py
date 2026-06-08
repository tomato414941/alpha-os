from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SplitFirstLaneLabelTicket:
    ticket_id: str
    opened_at: str
    rank: int
    opportunity: str
    probe_type: str
    status: str
    side: str
    asset: str
    venue: str
    candidate_size_usd: str
    observation_horizon: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    decision: str
    required_record: str
    next_step: str


def build_split_first_lane_label_tickets(
    *,
    queue_path: Path = ROOT / "current_split_first_lane_repeat_queue.csv",
    existing_tickets_path: Path | None = None,
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
) -> tuple[SplitFirstLaneLabelTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = []
    for rank, queue_row in enumerate(_label_rows(queue_path), start=1):
        ticket_id = _ticket_id(queue_row)
        if ticket_id in existing:
            rows.append(existing[ticket_id])
            continue
        entry_mark, entry_source = _entry_mark(asset=queue_row.get("asset", ""), marks=marks)
        rows.append(
            SplitFirstLaneLabelTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=queue_row.get("lane_opportunity", ""),
                probe_type="split_first_lane_label",
                status=queue_row.get("lane_status", ""),
                side=queue_row.get("lane_side", ""),
                asset=queue_row.get("asset", ""),
                venue="",
                candidate_size_usd="label_only",
                observation_horizon="15m,1h,4h",
                checkpoints="15m,1h,4h",
                entry_mark=entry_mark,
                entry_source=entry_source,
                decision=_decision(queue_row),
                required_record=queue_row.get("required_record", ""),
                next_step=queue_row.get("next_step", ""),
            )
        )
    return tuple(rows)


def write_split_first_lane_label_tickets_csv(
    rows: tuple[SplitFirstLaneLabelTicket, ...],
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
                "rank",
                "opportunity",
                "probe_type",
                "status",
                "side",
                "asset",
                "venue",
                "candidate_size_usd",
                "observation_horizon",
                "checkpoints",
                "entry_mark",
                "entry_source",
                "decision",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.rank,
                    row.opportunity,
                    row.probe_type,
                    row.status,
                    row.side,
                    row.asset,
                    row.venue,
                    row.candidate_size_usd,
                    row.observation_horizon,
                    row.checkpoints,
                    row.entry_mark,
                    row.entry_source,
                    row.decision,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_split_first_lane_label_tickets_md(
    rows: tuple[SplitFirstLaneLabelTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Split First Lane Label Tickets\n\n")
        handle.write(
            "These are lane-only forward-label tickets opened from split-first queue rows. "
            "They are not repeat probes and are not trade instructions.\n\n"
        )
        handle.write("| ticket | rank | asset | opportunity | side | entry | checkpoints | decision | next step |\n")
        handle.write("| --- | ---: | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.rank} | "
                f"{row.asset} | "
                f"{row.opportunity} | "
                f"{row.side} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _label_rows(path: Path) -> tuple[dict[str, str], ...]:
    return tuple(row for row in _read_rows(path) if row.get("action") == "open_lane_label")


def _entry_mark(*, asset: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _decision(row: dict[str, str]) -> str:
    side = row.get("lane_side", "").lower()
    if side == "short":
        return "paper_short"
    if side == "long":
        return "paper_long"
    return "paper_observe"


def _ticket_id(row: dict[str, str]) -> str:
    return f"split-label-{row.get('asset', '').lower()}-{_slug(row.get('lane_opportunity', ''))}"


def _existing_tickets(path: Path | None) -> tuple[SplitFirstLaneLabelTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            SplitFirstLaneLabelTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                rank=int(float(row.get("rank") or 0)),
                opportunity=row.get("opportunity", ""),
                probe_type=row.get("probe_type", ""),
                status=row.get("status", ""),
                side=row.get("side", ""),
                asset=row.get("asset", ""),
                venue=row.get("venue", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                observation_horizon=row.get("observation_horizon", ""),
                checkpoints=row.get("checkpoints", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                decision=row.get("decision", ""),
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


def _slug(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-path", type=Path, default=ROOT / "current_split_first_lane_repeat_queue.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_split_first_lane_label_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_split_first_lane_label_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_split_first_lane_label_tickets(
        queue_path=args.queue_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_split_first_lane_label_tickets_csv(rows, output_path=args.output_path)
    write_split_first_lane_label_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.entry_mark, row.decision)


if __name__ == "__main__":
    main()
