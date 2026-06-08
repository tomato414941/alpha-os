from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PortableMicrostructureHorizonTicket:
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


def build_portable_microstructure_horizon_tickets(
    *,
    candidates_path: Path = ROOT / "current_portable_microstructure_horizon_candidates.csv",
    labels_path: Path = ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv",
    snapshot_path: Path = ROOT / "market_making" / "current_microstructure_flow_snapshot.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[PortableMicrostructureHorizonTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    directions = _directions(labels_path)
    marks = _snapshot_marks(snapshot_path)
    rows = []
    for rank, candidate in enumerate(_read_rows(candidates_path), start=1):
        ticket_id = candidate.get("candidate_id", "")
        if ticket_id in existing:
            rows.append(existing[ticket_id])
            continue
        asset = candidate.get("asset", "")
        side = _side(directions.get(asset.upper(), ""))
        entry_mark, entry_source = marks.get(asset.upper(), ("", ""))
        horizon = candidate.get("candidate_horizon", "")
        rows.append(
            PortableMicrostructureHorizonTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=ticket_id,
                probe_type="portable_microstructure_horizon",
                status=candidate.get("status", ""),
                side=side,
                asset=asset,
                venue="",
                candidate_size_usd="label_only",
                observation_horizon=horizon,
                checkpoints=horizon,
                entry_mark=entry_mark,
                entry_source=entry_source,
                decision=_decision(side),
                required_record=candidate.get("required_record", ""),
                next_step=candidate.get("next_step", ""),
            )
        )
    return tuple(rows)


def write_portable_microstructure_horizon_tickets_csv(
    rows: tuple[PortableMicrostructureHorizonTicket, ...],
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


def write_portable_microstructure_horizon_tickets_md(
    rows: tuple[PortableMicrostructureHorizonTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Portable Microstructure Horizon Tickets\n\n")
        handle.write(
            "These are paper observation tickets for horizon-specific portable microstructure candidates. "
            "They are not trade instructions.\n\n"
        )
        handle.write("| ticket | rank | asset | side | status | entry | checkpoint | decision | next step |\n")
        handle.write("| --- | ---: | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.rank} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.status} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{row.decision} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _directions(path: Path) -> dict[str, str]:
    return {row.get("asset", "").upper(): row.get("direction", "") for row in _read_rows(path) if row.get("asset")}


def _snapshot_marks(path: Path) -> dict[str, tuple[str, str]]:
    return {
        row.get("asset", "").upper(): (row.get("mid_price", ""), "current_microstructure_flow_snapshot")
        for row in _read_rows(path)
        if row.get("asset")
    }


def _side(direction: str) -> str:
    if direction == "-1":
        return "short"
    if direction == "1":
        return "long"
    return ""


def _decision(side: str) -> str:
    if side == "short":
        return "paper_short"
    if side == "long":
        return "paper_long"
    return "paper_observe"


def _existing_tickets(path: Path | None) -> tuple[PortableMicrostructureHorizonTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            PortableMicrostructureHorizonTicket(
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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_horizon_candidates.csv",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "market_making" / "current_microstructure_flow_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_horizon_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_horizon_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_portable_microstructure_horizon_tickets(
        candidates_path=args.candidates_path,
        labels_path=args.labels_path,
        snapshot_path=args.snapshot_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_portable_microstructure_horizon_tickets_csv(rows, output_path=args.output_path)
    write_portable_microstructure_horizon_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.side, row.entry_mark, row.checkpoints)


if __name__ == "__main__":
    main()
