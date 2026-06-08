from __future__ import annotations

import argparse
import csv
import re
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
    existing_tickets_path: Path | None = None,
    hyperliquid_snapshot_path: Path = ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    intraday_live_gate_path: Path = ROOT / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv",
    top: int = 50,
) -> tuple[PaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing_opened_at = _existing_opened_at(existing_tickets_path)
    existing_tickets = {ticket.ticket_id: ticket for ticket in _existing_tickets(existing_tickets_path)}
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
        intraday_live_gate_path=intraday_live_gate_path,
    )
    rows = tuple(_read_rows(plan_path))[:top]
    return tuple(
        _ticket_for_plan_row(
            row=row,
            opened_at=opened_at,
            existing_opened_at=existing_opened_at,
            existing_tickets=existing_tickets,
            marks=marks,
        )
        for row in rows
    )


def _ticket_for_plan_row(
    *,
    row: dict[str, str],
    opened_at: str,
    existing_opened_at: dict[str, str],
    existing_tickets: dict[str, PaperTicket],
    marks: dict[tuple[str, str], tuple[str, str]],
) -> PaperTicket:
    ticket_id = _ticket_id(row)
    existing = existing_tickets.get(ticket_id)
    existing_matches = existing is not None and _ticket_matches_plan(existing, row)
    if existing_matches and row.get("probe_type") != "event_crypto_hedge_probe":
        assert existing is not None
        return existing
    return _build_ticket(
        opened_at=existing_opened_at.get(ticket_id, opened_at) if existing_matches else opened_at,
        row=row,
        marks=marks,
    )


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
    if not entry_mark:
        entry_mark, entry_source = _fallback_entry_mark(row)
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
    asset = _slug(row.get("asset") or "na")
    probe = (row.get("probe_type") or "probe").replace("_probe", "").replace("_", "-")
    return f"paper-{rank:02d}-{asset}-{probe}"


def _ticket_matches_plan(ticket: PaperTicket, row: dict[str, str]) -> bool:
    return (
        ticket.opportunity == row.get("opportunity", "")
        and ticket.probe_type == row.get("probe_type", "")
        and ticket.asset == row.get("asset", "")
    )


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
    intraday_live_gate_path: Path = ROOT / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv",
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
    for row in _read_rows(intraday_live_gate_path):
        symbol = row.get("symbol", "").upper()
        mark = row.get("mid_price", "")
        if symbol and mark:
            marks[("", symbol)] = (mark, "intraday_live_execution_gate")
    return marks


def _fallback_entry_mark(row: dict[str, str]) -> tuple[str, str]:
    if row.get("probe_type") == "event_probability_probe":
        ask = _extract_evidence_value(row.get("evidence", ""), "ask")
        if ask:
            return ask, "event_probability_plan_ask"
    return "", ""


def _extract_evidence_value(evidence: str, key: str) -> str:
    prefix = f"{key}="
    for part in evidence.split(","):
        value = part.strip()
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return ""


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
    side = row.get("side", "").lower()
    if side.startswith("watch") or side.startswith("context") or side == "none":
        return "paper_observe"
    if side.startswith("long") or side.startswith("paper_long") or "buy_yes" in side:
        return "paper_long"
    if side.startswith("short") or side.startswith("paper_short"):
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
    if probe_type == "options_volatility_probe":
        return "option quote, spread, depth, premium at risk, hedge schedule, exit bid"
    if probe_type == "token_unlock_probe":
        return "event window, mark move, funding, depth, crowding, stop"
    if probe_type == "protocol_fee_probe":
        return "fee snapshot, forward label, spread/depth, funding, valuation caveat"
    if probe_type == "wallet_entity_flow_probe":
        return "seed wallet flow, forward label, market-wide flow baseline, spread/depth, copycat risk"
    if probe_type == "event_pressure_probe":
        return "source overlap, duplicate-source check, forward label, spread/depth, funding, crowding"
    if probe_type == "execution_edge_probe":
        return "execution mode, fee tier, fill assumption, queue/adverse selection, stop, funding"
    if probe_type == "policy_expansion_probe":
        return "seed preference, fresh reward, fill/cost, stop, failure regime, comparison to seed"
    if probe_type == "event_crypto_hedge_probe":
        return "event timestamp, mark move, funding, spread/depth, beta attribution, failure regime"
    if probe_type == "stablecoin_migration_probe":
        return "flow snapshot, mapped token mark, funding, venue cost, forward label"
    if probe_type == "stablecoin_peg_probe":
        return "quote freshness, redemption route, venue depth, custody, repeated peg"
    if probe_type in {"attention_event_probe", "news_event_probe"}:
        return "timestamp, source freshness, forward label, spread/depth, crowding"
    if probe_type in {"defi_lending_probe", "defi_yield_probe"}:
        return "APY source, custody, withdrawal path, capacity, peg, exit liquidity"
    return "mark move, spread/fill assumption, cost, funding where relevant, stop"


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


def _existing_tickets(path: Path | None) -> tuple[PaperTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            PaperTicket(
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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "na"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-path", type=Path, default=ROOT / "current_paper_probe_plan.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_tickets.md")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--preserve-opened-at",
        action="store_true",
        help="Keep existing ticket opened_at values when ticket ids already exist.",
    )
    args = parser.parse_args()

    rows = build_paper_tickets(
        plan_path=args.plan_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top=args.top,
    )
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.asset, row.venue, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
