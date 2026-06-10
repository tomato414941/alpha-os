from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.crypto_pair_spread.current_crypto_pair_spread_tickets import LOCAL_ROOT, ROOT


@dataclass(frozen=True)
class PairSpreadRepeatTicket:
    ticket_id: str
    opened_at: str
    rank: int
    pair: str
    base_asset: str
    quote_asset: str
    entry_ratio: str
    base_mark: str
    quote_mark: str
    base_return_24h_bps: str
    quote_return_24h_bps: str
    dislocation_bps: str
    decision: str
    side: str
    checkpoints: str
    required_record: str
    next_step: str


def build_pair_spread_repeat_tickets(
    *,
    risk_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_fill_risk_check.csv",
    outcomes_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_outcomes.csv",
    tickets_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv",
    snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    existing_tickets_path: Path | None = None,
) -> tuple[PairSpreadRepeatTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    outcomes = {row.get("ticket_id", ""): row for row in _read_rows(outcomes_path)}
    tickets = {row.get("ticket_id", ""): row for row in _read_rows(tickets_path)}
    marks = _asset_marks(snapshot_path)
    rows: list[PairSpreadRepeatTicket] = []
    winners = (
        row
        for row in _read_rows(risk_path)
        if row.get("risk_action") == "cost_adjusted_pair_probe"
    )
    for rank, row in enumerate(winners, start=1):
        source = tickets.get(row.get("ticket_id", ""), {})
        outcome = outcomes.get(row.get("ticket_id", ""), {})
        base = source.get("base_asset", "")
        quote = source.get("quote_asset", "")
        repeat_id = f"pair-spread-repeat-{_slug(base)}-{_slug(quote)}"
        if repeat_id in existing:
            rows.append(existing[repeat_id])
            continue
        rows.append(
            PairSpreadRepeatTicket(
                ticket_id=repeat_id,
                opened_at=opened_at,
                rank=rank,
                pair=row.get("pair", ""),
                base_asset=base,
                quote_asset=quote,
                entry_ratio=outcome.get("current_mark", ""),
                base_mark=marks.get(base, {}).get("mark_price", ""),
                quote_mark=marks.get(quote, {}).get("mark_price", ""),
                base_return_24h_bps=source.get("base_return_24h_bps", ""),
                quote_return_24h_bps=source.get("quote_return_24h_bps", ""),
                dislocation_bps=source.get("dislocation_bps", ""),
                decision=row.get("decision", ""),
                side=source.get("side", ""),
                checkpoints="5m,15m",
                required_record="repeat pair-ratio move, both-leg costs, funding carry, hedge ratio, stop/adverse excursion",
                next_step=f"repeat {row.get('pair', '')} pair-spread label and compare against the first cost-adjusted win",
            )
        )
    return tuple(rows)


def write_pair_spread_repeat_tickets_csv(
    rows: tuple[PairSpreadRepeatTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(PairSpreadRepeatTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_pair_spread_repeat_tickets_md(rows: tuple[PairSpreadRepeatTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Pair Spread Repeat Tickets\n\n")
        handle.write(
            "These preserve fresh entry ratios for pair-spread probes that survived rough two-leg cost checks. "
            "They are not live spread trades.\n\n"
        )
        handle.write("| ticket | rank | pair | decision | entry ratio | checkpoints | next step |\n")
        handle.write("| --- | ---: | --- | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.rank} | {row.pair} | {row.decision} | "
                f"{row.entry_ratio} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _existing_tickets(path: Path | None) -> tuple[PairSpreadRepeatTicket, ...]:
    if path is None:
        return ()
    return tuple(
        PairSpreadRepeatTicket(
            ticket_id=row.get("ticket_id", ""),
            opened_at=row.get("opened_at", ""),
            rank=_int(row.get("rank")),
            pair=row.get("pair", ""),
            base_asset=row.get("base_asset", ""),
            quote_asset=row.get("quote_asset", ""),
            entry_ratio=row.get("entry_ratio", ""),
            base_mark=row.get("base_mark", ""),
            quote_mark=row.get("quote_mark", ""),
            base_return_24h_bps=row.get("base_return_24h_bps", ""),
            quote_return_24h_bps=row.get("quote_return_24h_bps", ""),
            dislocation_bps=row.get("dislocation_bps", ""),
            decision=row.get("decision", ""),
            side=row.get("side", ""),
            checkpoints=row.get("checkpoints", ""),
            required_record=row.get("required_record", ""),
            next_step=row.get("next_step", ""),
        )
        for row in _read_rows(path)
        if row.get("ticket_id")
    )


def _asset_marks(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("asset", ""): row for row in _read_rows(path) if row.get("asset", "")}


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0.0))
    except ValueError:
        return 0


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_fill_risk_check.csv")
    parser.add_argument("--outcomes-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_outcomes.csv")
    parser.add_argument("--tickets-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=LOCAL_ROOT / "current_crypto_pair_spread_repeat_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=LOCAL_ROOT / "current_crypto_pair_spread_repeat_tickets.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_pair_spread_repeat_tickets(
        risk_path=args.risk_path,
        outcomes_path=args.outcomes_path,
        tickets_path=args.tickets_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_pair_spread_repeat_tickets_csv(rows, output_path=args.output_path)
    write_pair_spread_repeat_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.pair, row.decision, row.entry_ratio)


if __name__ == "__main__":
    main()
