from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP = 3
DEFAULT_STOP_BPS = 100.0


@dataclass(frozen=True)
class SurvivingAlphaExitRegimeTicket:
    ticket_id: str
    opened_at: str
    candidate_id: str
    asset: str
    decision: str
    side: str
    entry_mark: str
    entry_source: str
    candidate_size_usd: str
    exit_horizon_minutes: int
    stop_bps: float
    required_record: str
    next_step: str


def build_surviving_alpha_exit_regime_tickets(
    *,
    candidates_path: Path = ROOT / "current_surviving_alpha_exit_regime_candidates.csv",
    existing_tickets_path: Path | None = None,
    second_tickets_path: Path = ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    top: int = DEFAULT_TOP,
) -> tuple[SurvivingAlphaExitRegimeTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    sizes = _candidate_sizes(second_tickets_path)
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    candidate_rows = tuple(
        row for row in _read_rows(candidates_path) if row.get("status") == "wide_stop_exit_candidate"
    )[:top]
    return tuple(
        _ticket_for_candidate(
            row=row,
            opened_at=opened_at,
            existing=existing,
            sizes=sizes,
            marks=marks,
        )
        for row in candidate_rows
    )


def write_surviving_alpha_exit_regime_tickets_csv(
    rows: tuple[SurvivingAlphaExitRegimeTicket, ...],
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
                "candidate_id",
                "asset",
                "decision",
                "side",
                "entry_mark",
                "entry_source",
                "candidate_size_usd",
                "exit_horizon_minutes",
                "stop_bps",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.candidate_id,
                    row.asset,
                    row.decision,
                    row.side,
                    row.entry_mark,
                    row.entry_source,
                    row.candidate_size_usd,
                    row.exit_horizon_minutes,
                    f"{row.stop_bps:.2f}",
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_surviving_alpha_exit_regime_tickets_md(
    rows: tuple[SurvivingAlphaExitRegimeTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Exit Regime Tickets\n\n")
        handle.write(
            "These are fresh paper tickets for exit regimes that survived a second-repeat "
            "review but failed the tighter path-risk check. They are not live trade instructions.\n\n"
        )
        handle.write(
            "| ticket | candidate | asset | side | entry | source | size USD | exit min | stop bps | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.candidate_id} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.entry_mark} | "
                f"{row.entry_source} | "
                f"{row.candidate_size_usd} | "
                f"{row.exit_horizon_minutes} | "
                f"{row.stop_bps:.2f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _ticket_for_candidate(
    *,
    row: dict[str, str],
    opened_at: str,
    existing: dict[str, SurvivingAlphaExitRegimeTicket],
    sizes: dict[tuple[str, str], str],
    marks: dict[tuple[str, str], tuple[str, str]],
) -> SurvivingAlphaExitRegimeTicket:
    asset = row.get("asset", "")
    decision = row.get("decision", "")
    horizon = int(float(row.get("horizon_minutes") or 0))
    ticket_id = _ticket_id(row)
    entry_mark, entry_source = _entry_mark(asset=asset, marks=marks)
    ticket = SurvivingAlphaExitRegimeTicket(
        ticket_id=ticket_id,
        opened_at=opened_at,
        candidate_id=row.get("candidate_id", ""),
        asset=asset,
        decision=decision,
        side=_side(decision),
        entry_mark=entry_mark,
        entry_source=entry_source,
        candidate_size_usd=sizes.get((asset, decision), "100.00"),
        exit_horizon_minutes=horizon,
        stop_bps=DEFAULT_STOP_BPS,
        required_record=row.get("required_record", ""),
        next_step=row.get("next_step", ""),
    )
    prior = existing.get(ticket_id)
    if prior is None:
        return ticket
    return replace(
        ticket,
        opened_at=prior.opened_at,
        entry_mark=prior.entry_mark or ticket.entry_mark,
        entry_source=prior.entry_source or ticket.entry_source,
    )


def _ticket_id(row: dict[str, str]) -> str:
    return f"exit-regime-{row.get('candidate_id', '')}-100bps-stop"


def _side(decision: str) -> str:
    if decision == "paper_short":
        return "short"
    if decision == "paper_long":
        return "long"
    return "observe"


def _entry_mark(*, asset: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", asset.upper()), ("", asset.upper()), ("OKX", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _candidate_sizes(path: Path) -> dict[tuple[str, str], str]:
    return {
        (row.get("asset", ""), row.get("decision", "")): row.get("candidate_size_usd", "")
        for row in _read_rows(path)
        if row.get("asset") and row.get("decision") and row.get("candidate_size_usd")
    }


def _existing_tickets(path: Path | None) -> tuple[SurvivingAlphaExitRegimeTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        rows.append(
            SurvivingAlphaExitRegimeTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("asset", ""),
                decision=row.get("decision", ""),
                side=row.get("side", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                exit_horizon_minutes=int(float(row.get("exit_horizon_minutes") or 0)),
                stop_bps=float(row.get("stop_bps") or 0.0),
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
        default=ROOT / "current_surviving_alpha_exit_regime_candidates.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_tickets.md",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_surviving_alpha_exit_regime_tickets(
        candidates_path=args.candidates_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top=args.top,
    )
    write_surviving_alpha_exit_regime_tickets_csv(rows, output_path=args.output_path)
    write_surviving_alpha_exit_regime_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.side, row.entry_mark, row.exit_horizon_minutes)


if __name__ == "__main__":
    main()
