from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperTicket:
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


def build_paper_tickets(
    *,
    plan_path: Path = ROOT / "current_paper_probe_plan.csv",
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    top: int = 20,
) -> tuple[PaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = tuple(_read_rows(plan_path))[:top]
    return tuple(_build_ticket(opened_at=opened_at, row=row, marks=marks) for row in rows)


def write_paper_tickets_csv(rows: tuple[PaperTicket, ...], *, output_path: Path) -> Path:
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


def write_paper_tickets_md(rows: tuple[PaperTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Tickets\n\n")
        handle.write(
            "These are current paper-observation tickets opened from the cross-lane "
            "probe plan. They are not trade instructions and do not imply live execution.\n\n"
        )
        handle.write(
            "| ticket | rank | opportunity | side | asset | venue | size USD | entry mark | checkpoints | decision | required record |\n"
        )
        handle.write("| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.rank} | "
                f"{row.opportunity} | "
                f"{row.side} | "
                f"{row.asset} | "
                f"{row.venue} | "
                f"{row.candidate_size_usd} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{row.decision} | "
                f"{_escape(row.required_record)} |\n"
            )
        handle.write("\n## Rule\n\n")
        handle.write(
            "A ticket can only promote a candidate after the checkpoint record includes "
            "mark movement, spread or fill assumption, funding where relevant, and stop "
            "or adverse-excursion notes. Missing entry marks are allowed for non-perp "
            "or externally quoted candidates, but they must be filled before promotion.\n"
        )
    return output_path


def _build_ticket(*, opened_at: str, row: dict[str, str], marks: dict[tuple[str, str], tuple[str, str]]) -> PaperTicket:
    rank = int(float(row.get("rank") or 0))
    asset = row.get("asset", "")
    venue = row.get("venue", "")
    entry_mark, entry_source = _entry_mark(asset=asset, venue=venue, marks=marks)
    checkpoints = _checkpoints(row.get("observation_horizon", ""))
    return PaperTicket(
        ticket_id=_ticket_id(row),
        opened_at=opened_at,
        rank=rank,
        opportunity=row.get("opportunity", ""),
        probe_type=row.get("probe_type", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        asset=asset,
        venue=venue,
        candidate_size_usd=row.get("candidate_size_usd", ""),
        observation_horizon=row.get("observation_horizon", ""),
        checkpoints=checkpoints,
        entry_mark=entry_mark,
        entry_source=entry_source,
        decision=_decision(row),
        required_record=_required_record(row),
        next_step=row.get("next_step", ""),
    )


def _ticket_id(row: dict[str, str]) -> str:
    rank = int(float(row.get("rank") or 0))
    asset = (row.get("asset") or "na").lower()
    probe = (row.get("probe_type") or "probe").replace("_probe", "").replace("_", "-")
    return f"paper-{rank:02d}-{asset}-{probe}"


def _entry_mark(*, asset: str, venue: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    keys = ((venue.upper(), asset.upper()), ("HL", asset.upper()), ("", asset.upper()))
    for key in keys:
        if key in marks:
            return marks[key]
    return "", ""


def _load_marks(
    *,
    hyperliquid_snapshot_path: Path,
    hl_context_path: Path,
    okx_context_path: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    marks: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _read_rows(hyperliquid_snapshot_path):
        asset = row.get("asset", "").upper()
        mark = row.get("mark_price", "")
        if asset and mark:
            marks[("HL", asset)] = (mark, "hyperliquid_snapshot")
    for row in _read_rows(hl_context_path):
        asset = row.get("asset", "").upper()
        mark = row.get("mark_price", "")
        if asset and mark:
            marks[("HL", asset)] = (mark, "hl_execution_context")
    for row in _read_rows(okx_context_path):
        asset = row.get("asset", "").upper()
        mark = row.get("last_price", "")
        if asset and mark:
            marks[("OKX", asset)] = (mark, "okx_execution_context")
    return marks


def _checkpoints(horizon: str) -> str:
    checkpoints: list[str] = []
    for token in horizon.split("/"):
        token = token.strip()
        if token in {"15m", "1h", "4h", "12h", "24h"}:
            checkpoints.append(token)
    if not checkpoints:
        checkpoints = ["fresh"]
    return ",".join(checkpoints)


def _decision(row: dict[str, str]) -> str:
    side = row.get("side", "")
    if side.startswith("long") or "buy_yes" in side:
        return "paper_long"
    if side.startswith("short"):
        return "paper_short"
    return "paper_observe"


def _required_record(row: dict[str, str]) -> str:
    probe_type = row.get("probe_type", "")
    if probe_type == "event_probability_probe":
        return "quote, depth, fee, fill assumption, resolution risk, adverse move"
    if probe_type == "microstructure_flow_probe":
        return "mark move, spread, queue/fill assumption, funding, adverse selection"
    if probe_type in {"repeat_execution_probe", "liquidation_intensity_probe"}:
        return "mark move, spread/fill assumption, funding, stop, adverse excursion"
    return "mark move, spread/fill assumption, cost, funding where relevant, stop"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-path", type=Path, default=ROOT / "current_paper_probe_plan.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_tickets.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_paper_tickets(plan_path=args.plan_path, top=args.top)
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.asset, row.venue, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
