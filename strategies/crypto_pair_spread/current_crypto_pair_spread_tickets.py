from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ("BTC", "ETH", "SOL", "HYPE")


@dataclass(frozen=True)
class PairSpreadTicket:
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


def build_pair_spread_tickets(
    *,
    snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    existing_tickets_path: Path | None = None,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
) -> tuple[PairSpreadTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    marks = _asset_marks(snapshot_path)
    rows = []
    for base, quote in combinations(assets, 2):
        if base not in marks or quote not in marks:
            continue
        ticket = _ticket_for_pair(base=base, quote=quote, marks=marks, opened_at=opened_at)
        rows.append(existing.get(ticket.ticket_id, ticket))
    rows.sort(key=lambda row: abs(_float(row.dislocation_bps)), reverse=True)
    return tuple(row.__class__(**{**row.__dict__, "rank": rank}) for rank, row in enumerate(rows, start=1))


def write_pair_spread_tickets_csv(rows: tuple[PairSpreadTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(PairSpreadTicket.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_pair_spread_tickets_md(rows: tuple[PairSpreadTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Pair Spread Tickets\n\n")
        handle.write(
            "These are fresh relative-value paper labels over current BTC/ETH/SOL/HYPE marks. "
            "They are not live spread trades and do not assume borrow, margin, or hedge execution quality.\n\n"
        )
        handle.write("| ticket | pair | decision | entry ratio | base 24h | quote 24h | dislocation | checkpoints | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.pair} | {row.decision} | {row.entry_ratio} | "
                f"{row.base_return_24h_bps} | {row.quote_return_24h_bps} | {row.dislocation_bps} | "
                f"{row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _ticket_for_pair(
    *,
    base: str,
    quote: str,
    marks: dict[str, dict[str, str]],
    opened_at: str,
) -> PairSpreadTicket:
    base_mark = _float(marks[base].get("mark_price"))
    quote_mark = _float(marks[quote].get("mark_price"))
    base_return = _float(marks[base].get("return_24h")) * 10_000.0
    quote_return = _float(marks[quote].get("return_24h")) * 10_000.0
    dislocation = base_return - quote_return
    decision = "paper_short" if dislocation > 0.0 else "paper_long"
    side = "short_base_long_quote" if decision == "paper_short" else "long_base_short_quote"
    return PairSpreadTicket(
        ticket_id=f"pair-spread-{base.lower()}-{quote.lower()}-mean-reversion",
        opened_at=opened_at,
        rank=0,
        pair=f"{base}/{quote}",
        base_asset=base,
        quote_asset=quote,
        entry_ratio=f"{(base_mark / quote_mark):.12f}" if quote_mark > 0.0 else "",
        base_mark=f"{base_mark:.12f}",
        quote_mark=f"{quote_mark:.12f}",
        base_return_24h_bps=f"{base_return:.8f}",
        quote_return_24h_bps=f"{quote_return:.8f}",
        dislocation_bps=f"{dislocation:.8f}",
        decision=decision,
        side=side,
        checkpoints="5m,15m",
        required_record="pair-ratio move, both-leg spread/depth, funding carry, hedge notional, stop/adverse excursion",
        next_step=f"paper-label {base}/{quote} mean-reversion spread over 5m/15m before any execution overlay",
    )


def _asset_marks(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("asset", ""): row for row in _read_rows(path) if row.get("asset", "")}


def _existing_tickets(path: Path | None) -> tuple[PairSpreadTicket, ...]:
    if path is None or not path.exists():
        return ()
    rows = []
    for row in _read_rows(path):
        rows.append(
            PairSpreadTicket(
                ticket_id=row.get("ticket_id", ""),
                opened_at=row.get("opened_at", ""),
                rank=int(float(row.get("rank") or 0)),
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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_pair_spread_tickets(existing_tickets_path=args.output_path if args.preserve_opened_at else None)
    write_pair_spread_tickets_csv(rows, output_path=args.output_path)
    write_pair_spread_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.pair, row.decision, row.dislocation_bps)


if __name__ == "__main__":
    main()
