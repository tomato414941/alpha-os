from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SymbolLanePaperTicket:
    ticket_id: str
    opened_at: str
    symbol: str
    lane_bias: str
    opportunity: str
    status: str
    decision: str
    candidate_size_usd: str
    checkpoints: str
    entry_mark: str
    entry_source: str
    support_state: str
    required_record: str
    next_step: str


def build_symbol_lane_paper_tickets(
    *,
    lane_review_path: Path = ROOT / "current_symbol_lane_split_review.csv",
    existing_tickets_path: Path | None = None,
    top_symbols: int = 1,
    top_lanes_per_symbol: int = 8,
) -> tuple[SymbolLanePaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing_opened_at = _existing_opened_at(existing_tickets_path)
    marks = _load_marks(
        hyperliquid_snapshot_path=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
        hl_context_path=ROOT / "candidate_validation" / "current_followup_execution_context.csv",
        okx_context_path=ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    )
    rows = _top_symbol_rows(
        rows=_read_rows(lane_review_path),
        top_symbols=top_symbols,
        top_lanes_per_symbol=top_lanes_per_symbol,
    )
    tickets = []
    for row in rows:
        ticket_id = _ticket_id(row)
        symbol = row.get("symbol", "")
        entry_mark, entry_source = _entry_mark(symbol=symbol, marks=marks)
        tickets.append(
            SymbolLanePaperTicket(
                ticket_id=ticket_id,
                opened_at=existing_opened_at.get(ticket_id, opened_at),
                symbol=symbol,
                lane_bias=row.get("lane_bias", ""),
                opportunity=row.get("opportunity", ""),
                status=row.get("status", ""),
                decision=_decision(row.get("lane_bias", "")),
                candidate_size_usd=_candidate_size(row),
                checkpoints=_checkpoints(row),
                entry_mark=entry_mark,
                entry_source=entry_source,
                support_state=row.get("support_state", ""),
                required_record=_required_record(row),
                next_step=row.get("next_step", ""),
            )
        )
    known_ticket_ids = {ticket.ticket_id for ticket in tickets}
    tickets.extend(
        ticket
        for ticket in _existing_tickets(existing_tickets_path)
        if ticket.ticket_id not in known_ticket_ids
    )
    return tuple(tickets)


def write_symbol_lane_paper_tickets_csv(
    rows: tuple[SymbolLanePaperTicket, ...],
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
                "asset",
                "symbol",
                "lane_bias",
                "opportunity",
                "status",
                "decision",
                "candidate_size_usd",
                "checkpoints",
                "entry_mark",
                "entry_source",
                "support_state",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.symbol,
                    row.symbol,
                    row.lane_bias,
                    row.opportunity,
                    row.status,
                    row.decision,
                    row.candidate_size_usd,
                    row.checkpoints,
                    row.entry_mark,
                    row.entry_source,
                    row.support_state,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_symbol_lane_paper_tickets_md(
    rows: tuple[SymbolLanePaperTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Symbol Lane Paper Tickets\n\n")
        handle.write(
            "These tickets open separate paper observations for the top symbol's lanes. "
            "They deliberately do not collapse conflicting hypotheses into one trade.\n\n"
        )
        handle.write(
            "| ticket | symbol | bias | opportunity | decision | size USD | checkpoints | entry | support | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.symbol} | "
                f"{row.lane_bias} | "
                f"{row.opportunity} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.checkpoints} | "
                f"{row.entry_mark} | "
                f"{row.support_state} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _top_symbol_rows(
    *,
    rows: tuple[dict[str, str], ...],
    top_symbols: int,
    top_lanes_per_symbol: int,
) -> tuple[dict[str, str], ...]:
    symbols: list[str] = []
    selected: list[dict[str, str]] = []
    for row in rows:
        symbol = row.get("symbol", "")
        if symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) > top_symbols:
            break
    for symbol in symbols[:top_symbols]:
        symbol_rows = [row for row in rows if row.get("symbol") == symbol]
        selected.extend(sorted(symbol_rows, key=lambda row: _float(row.get("priority_score")), reverse=True)[:top_lanes_per_symbol])
    return tuple(selected)


def _ticket_id(row: dict[str, str]) -> str:
    symbol = row.get("symbol", "").lower()
    opportunity = row.get("opportunity", "").replace("_", "-")
    return f"lane-{symbol}-{opportunity}"


def _entry_mark(*, symbol: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in (("HL", symbol.upper()), ("OKX", symbol.upper()), ("", symbol.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _decision(lane_bias: str) -> str:
    if lane_bias == "short":
        return "paper_short"
    if lane_bias == "long":
        return "paper_long"
    return "paper_observe"


def _candidate_size(row: dict[str, str]) -> str:
    evidence = row.get("evidence", "")
    size = _extract_regex(evidence, r"\bsize=([0-9]+(?:\.[0-9]+)?)")
    if size:
        return size
    usage_size = _extract_regex(evidence, r"\bdepth_usage_([0-9]+)=")
    if usage_size:
        return usage_size
    if row.get("support_state") == "paper_execution_gated":
        return "250"
    return "100"


def _checkpoints(row: dict[str, str]) -> str:
    text = " ".join((row.get("status", ""), row.get("support_state", ""), row.get("next_step", ""))).lower()
    if "4h" in text:
        return "4h,12h"
    if "1h" in text:
        return "1h"
    return "15m,1h"


def _required_record(row: dict[str, str]) -> str:
    if row.get("support_state") == "paper_execution_gated":
        return "lane mark move, spread/fill assumption, funding, stop, adverse excursion"
    return "lane mark move and independent evidence for this hypothesis"


def _extract_regex(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return "" if match is None else match.group(1)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _existing_opened_at(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        row.get("ticket_id", ""): row.get("opened_at", "")
        for row in _read_rows(path)
        if row.get("ticket_id") and row.get("opened_at")
    }


def _existing_tickets(path: Path | None) -> tuple[SymbolLanePaperTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            SymbolLanePaperTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                symbol=row.get("symbol", "") or row.get("asset", ""),
                lane_bias=row.get("lane_bias", ""),
                opportunity=row.get("opportunity", ""),
                status=row.get("status", ""),
                decision=row.get("decision", ""),
                candidate_size_usd=row.get("candidate_size_usd", ""),
                checkpoints=row.get("checkpoints", ""),
                entry_mark=row.get("entry_mark", ""),
                entry_source=row.get("entry_source", ""),
                support_state=row.get("support_state", ""),
                required_record=row.get("required_record", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-review-path", type=Path, default=ROOT / "current_symbol_lane_split_review.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_symbol_lane_paper_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_symbol_lane_paper_tickets.md")
    parser.add_argument("--top-symbols", type=int, default=1)
    parser.add_argument("--top-lanes-per-symbol", type=int, default=8)
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_symbol_lane_paper_tickets(
        lane_review_path=args.lane_review_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top_symbols=args.top_symbols,
        top_lanes_per_symbol=args.top_lanes_per_symbol,
    )
    write_symbol_lane_paper_tickets_csv(rows, output_path=args.output_path)
    write_symbol_lane_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.symbol, row.lane_bias, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
