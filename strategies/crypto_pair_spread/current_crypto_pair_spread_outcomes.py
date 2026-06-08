from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.crypto_pair_spread.current_crypto_pair_spread_tickets import LOCAL_ROOT, ROOT


@dataclass(frozen=True)
class PairSpreadOutcome:
    ticket_id: str
    opened_at: str
    checked_at: str
    elapsed_minutes: float
    checkpoint_status: str
    opportunity: str
    decision: str
    asset: str
    venue: str
    entry_mark: str
    current_mark: str
    current_source: str
    raw_return_bps: str
    directional_return_bps: str
    outcome: str
    missing_evidence: str
    next_step: str


def build_pair_spread_outcomes(
    *,
    tickets_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv",
    snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
) -> tuple[PairSpreadOutcome, ...]:
    checked_at = datetime.now(UTC)
    marks = _asset_marks(snapshot_path)
    return tuple(_outcome_for_ticket(row=row, checked_at=checked_at, marks=marks) for row in _read_rows(tickets_path))


def write_pair_spread_outcomes_csv(rows: tuple[PairSpreadOutcome, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(PairSpreadOutcome.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.checked_at,
                    f"{row.elapsed_minutes:.2f}",
                    row.checkpoint_status,
                    row.opportunity,
                    row.decision,
                    row.asset,
                    row.venue,
                    row.entry_mark,
                    row.current_mark,
                    row.current_source,
                    row.raw_return_bps,
                    row.directional_return_bps,
                    row.outcome,
                    row.missing_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_pair_spread_outcomes_md(rows: tuple[PairSpreadOutcome, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Pair Spread Outcomes\n\n")
        handle.write(
            "This checks pair-ratio paper labels against current public marks. "
            "It is not a fill report and not a deployable pair execution overlay.\n\n"
        )
        handle.write("| ticket | status | pair | decision | entry ratio | current ratio | dir bps | outcome | next step |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.checkpoint_status} | {row.asset} | {row.decision} | "
                f"{row.entry_mark} | {row.current_mark} | {row.directional_return_bps} | "
                f"{row.outcome} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _outcome_for_ticket(
    *,
    row: dict[str, str],
    checked_at: datetime,
    marks: dict[str, dict[str, str]],
) -> PairSpreadOutcome:
    opened_at = _parse_time(row.get("opened_at", ""))
    elapsed_minutes = (checked_at - opened_at).total_seconds() / 60.0 if opened_at else 0.0
    checkpoint_status = _checkpoint_status(row.get("checkpoints", ""), elapsed_minutes)
    base = row.get("base_asset", "")
    quote = row.get("quote_asset", "")
    current_ratio = _current_ratio(base=base, quote=quote, marks=marks)
    raw_bps, dir_bps, outcome, missing = _mark_outcome(
        entry_ratio=row.get("entry_ratio", ""),
        current_ratio=current_ratio,
        decision=row.get("decision", ""),
        checkpoint_status=checkpoint_status,
    )
    pair = row.get("pair", "")
    return PairSpreadOutcome(
        ticket_id=row.get("ticket_id", ""),
        opened_at=row.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        elapsed_minutes=elapsed_minutes,
        checkpoint_status=checkpoint_status,
        opportunity=f"crypto_pair_spread:{pair}:24h_dislocation_mean_reversion",
        decision=row.get("decision", ""),
        asset=pair,
        venue="HL_PAIR",
        entry_mark=row.get("entry_ratio", ""),
        current_mark=current_ratio,
        current_source="hyperliquid_pair_ratio",
        raw_return_bps=raw_bps,
        directional_return_bps=dir_bps,
        outcome=outcome,
        missing_evidence=missing,
        next_step=_next_step(checkpoint_status=checkpoint_status, outcome=outcome),
    )


def _mark_outcome(
    *,
    entry_ratio: str,
    current_ratio: str,
    decision: str,
    checkpoint_status: str,
) -> tuple[str, str, str, str]:
    if checkpoint_status != "ready":
        return "", "", "pending_checkpoint", "wait for the first 5m/15m checkpoint"
    entry = _float(entry_ratio)
    current = _float(current_ratio)
    if entry <= 0.0 or current <= 0.0:
        return "", "", "missing_current_mark", "missing one or both pair-leg marks"
    raw_bps = (current / entry - 1.0) * 10_000.0
    directional_bps = -raw_bps if decision == "paper_short" else raw_bps
    outcome = "paper_mark_win" if directional_bps > 0.0 else "paper_mark_loss"
    missing = "both-leg spread/depth, funding carry, hedge ratio, stop/adverse excursion"
    return f"{raw_bps:.8f}", f"{directional_bps:.8f}", outcome, missing


def _next_step(*, checkpoint_status: str, outcome: str) -> str:
    if checkpoint_status != "ready":
        return "wait for pair-ratio checkpoint before scoring the label"
    if outcome == "paper_mark_win":
        return "repeat the pair label with explicit both-leg costs, funding carry, and hedge execution notes"
    if outcome == "paper_mark_loss":
        return "keep as a negative pair-spread label before trying execution overlay logic"
    return "repair missing pair marks before scoring"


def _checkpoint_status(checkpoints: str, elapsed_minutes: float) -> str:
    required = min((_checkpoint_minutes(value) for value in checkpoints.split(",") if value.strip()), default=5.0)
    return "ready" if elapsed_minutes >= required else "pending"


def _checkpoint_minutes(value: str) -> float:
    cleaned = value.strip().lower()
    if cleaned.endswith("m"):
        return _float(cleaned[:-1])
    if cleaned.endswith("h"):
        return _float(cleaned[:-1]) * 60.0
    return _float(cleaned)


def _current_ratio(*, base: str, quote: str, marks: dict[str, dict[str, str]]) -> str:
    base_mark = _float(marks.get(base, {}).get("mark_price"))
    quote_mark = _float(marks.get(quote, {}).get("mark_price"))
    if base_mark <= 0.0 or quote_mark <= 0.0:
        return ""
    return f"{(base_mark / quote_mark):.12f}"


def _asset_marks(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("asset", ""): row for row in _read_rows(path) if row.get("asset", "")}


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_outcomes.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_outcomes.md")
    args = parser.parse_args()

    rows = build_pair_spread_outcomes(tickets_path=args.tickets_path)
    write_pair_spread_outcomes_csv(rows, output_path=args.output_path)
    write_pair_spread_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.checkpoint_status, row.directional_return_bps, row.outcome)


if __name__ == "__main__":
    main()
