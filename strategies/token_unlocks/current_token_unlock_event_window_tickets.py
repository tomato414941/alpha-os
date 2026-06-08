from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class TokenUnlockEventWindowTicket:
    ticket_id: str
    opened_at: str
    opportunity: str
    decision: str
    asset: str
    venue: str
    entry_mark: str
    entry_source: str
    side: str
    days_until: str
    unlock_value_usd: str
    percent_supply: str
    checkpoints: str
    required_record: str
    next_step: str


def build_token_unlock_event_window_tickets(
    *,
    actionability_path: Path = ROOT / "current_token_unlock_actionability.csv",
    hyperliquid_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[TokenUnlockEventWindowTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    marks = {row.get("asset", ""): row for row in _read_rows(hyperliquid_snapshot_path)}
    tickets: list[TokenUnlockEventWindowTicket] = []
    for row in _read_rows(actionability_path):
        if row.get("action") not in {"create_event_window_label", "label_before_short"}:
            continue
        symbol = row.get("symbol", "")
        decision = _decision(row.get("side", ""))
        ticket_id = f"unlock-event-{_slug(symbol)}-{_slug(decision)}"
        if ticket_id in existing:
            tickets.append(existing[ticket_id])
            continue
        mark = marks.get(symbol, {})
        tickets.append(
            TokenUnlockEventWindowTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                opportunity=f"token_unlock_event_window:{symbol}",
                decision=decision,
                asset=symbol,
                venue="HL",
                entry_mark=mark.get("mark_price", ""),
                entry_source="hyperliquid_snapshot" if mark else "",
                side=row.get("side", ""),
                days_until=row.get("days_until", ""),
                unlock_value_usd=row.get("unlock_value_usd", ""),
                percent_supply=row.get("percent_supply", ""),
                checkpoints="15m,1h,4h",
                required_record=(
                    "event-window forward return, funding persistence, crowding/squeeze split, "
                    "depth, and pre/post-unlock behavior"
                ),
                next_step=_next_step(symbol=symbol, side=row.get("side", "")),
            )
        )
    return tuple(tickets)


def write_token_unlock_event_window_tickets_csv(
    rows: tuple[TokenUnlockEventWindowTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(TokenUnlockEventWindowTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_token_unlock_event_window_tickets_md(
    rows: tuple[TokenUnlockEventWindowTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Event Window Tickets\n\n")
        handle.write(
            "These preserve entry marks for scheduled unlock event-window labels. "
            "They are not live orders and not a reusable strategy abstraction.\n\n"
        )
        handle.write("| ticket | asset | decision | side | in | value USD | % supply | entry | checkpoints | next step |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.asset} | {row.decision} | {row.side} | "
                f"{row.days_until} | {row.unlock_value_usd} | {row.percent_supply} | "
                f"{row.entry_mark} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _decision(side: str) -> str:
    if side == "short":
        return "paper_short"
    if side == "watch_squeeze":
        return "paper_long"
    return "paper_observe"


def _next_step(*, symbol: str, side: str) -> str:
    if side == "watch_squeeze":
        return f"label {symbol} squeeze-risk path before considering any unlock short"
    return f"label {symbol} short-pressure path before treating the unlock as tradable"


def _existing_tickets(path: Path | None) -> tuple[TokenUnlockEventWindowTicket, ...]:
    if path is None:
        return ()
    return tuple(
        TokenUnlockEventWindowTicket(
            ticket_id=row.get("ticket_id", ""),
            opened_at=row.get("opened_at", ""),
            opportunity=row.get("opportunity", ""),
            decision=row.get("decision", ""),
            asset=row.get("asset", ""),
            venue=row.get("venue", ""),
            entry_mark=row.get("entry_mark", ""),
            entry_source=row.get("entry_source", ""),
            side=row.get("side", ""),
            days_until=row.get("days_until", ""),
            unlock_value_usd=row.get("unlock_value_usd", ""),
            percent_supply=row.get("percent_supply", ""),
            checkpoints=row.get("checkpoints", ""),
            required_record=row.get("required_record", ""),
            next_step=row.get("next_step", ""),
        )
        for row in _read_rows(path)
        if row.get("ticket_id")
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionability-path", type=Path, default=ROOT / "current_token_unlock_actionability.csv")
    parser.add_argument(
        "--hyperliquid-snapshot-path",
        type=Path,
        default=STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_token_unlock_event_window_tickets.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_token_unlock_event_window_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_token_unlock_event_window_tickets(
        actionability_path=args.actionability_path,
        hyperliquid_snapshot_path=args.hyperliquid_snapshot_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_token_unlock_event_window_tickets_csv(rows, output_path=args.output_path)
    write_token_unlock_event_window_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.decision, row.asset, row.entry_mark)


if __name__ == "__main__":
    main()
