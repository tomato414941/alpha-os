from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.current_paper_tickets import _load_marks


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class StablecoinFlowProxyTicket:
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


def build_stablecoin_flow_proxy_tickets(
    *,
    candidates_path: Path = ROOT / "current_stablecoin_flow_probe_candidates.csv",
    market_context_path: Path = STRATEGIES_ROOT / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
    existing_tickets_path: Path | None = None,
    hyperliquid_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    hl_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
) -> tuple[StablecoinFlowProxyTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.ticket_id: row for row in _existing_tickets(existing_tickets_path)}
    contexts = _best_context_by_symbol(market_context_path)
    marks = _load_marks(
        hyperliquid_snapshot_path=hyperliquid_snapshot_path,
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = []
    proxy_candidates = (
        row for row in _read_rows(candidates_path) if row.get("candidate_type") == "chain_liquidity_proxy_label"
    )
    for rank, candidate in enumerate(proxy_candidates, start=1):
        ticket_id = candidate.get("candidate_id", "")
        if ticket_id in existing:
            rows.append(existing[ticket_id])
            continue
        asset = candidate.get("token_symbol", "")
        context = contexts.get(asset, {})
        venue = context.get("venue", "")
        side = _side(context.get("direction", ""))
        entry_mark, entry_source = _entry_mark(asset=asset, venue=venue, marks=marks)
        rows.append(
            StablecoinFlowProxyTicket(
                ticket_id=ticket_id,
                opened_at=opened_at,
                rank=rank,
                opportunity=candidate.get("subject", ""),
                probe_type="stablecoin_flow_proxy_label",
                status=candidate.get("status", ""),
                side=side,
                asset=asset,
                venue=venue,
                candidate_size_usd="label_only",
                observation_horizon="1h,4h",
                checkpoints="1h,4h",
                entry_mark=entry_mark,
                entry_source=entry_source,
                decision=_decision(side),
                required_record=candidate.get("required_record", ""),
                next_step=candidate.get("next_step", ""),
            )
        )
    return tuple(rows)


def write_stablecoin_flow_proxy_tickets_csv(
    rows: tuple[StablecoinFlowProxyTicket, ...],
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


def write_stablecoin_flow_proxy_tickets_md(
    rows: tuple[StablecoinFlowProxyTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Stablecoin Flow Proxy Tickets\n\n")
        handle.write(
            "These are paper labels for chain-liquidity proxy candidates. "
            "They are not direct exchange stablecoin inflow signals and not trade instructions.\n\n"
        )
        handle.write("| ticket | rank | asset | side | venue | entry | checkpoints | next step |\n")
        handle.write("| --- | ---: | --- | --- | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.rank} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.venue} | "
                f"{row.entry_mark} | "
                f"{row.checkpoints} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _best_context_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(path):
        symbol = row.get("token_symbol", "")
        if symbol:
            grouped.setdefault(symbol, []).append(row)
    return {
        symbol: max(rows, key=lambda row: _float(row.get("context_score")))
        for symbol, rows in grouped.items()
    }


def _entry_mark(*, asset: str, venue: str, marks: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    for key in ((venue.upper(), asset.upper()), ("HL", asset.upper()), ("OKX", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


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


def _existing_tickets(path: Path | None) -> tuple[StablecoinFlowProxyTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
        if not row.get("ticket_id"):
            continue
        rows.append(
            StablecoinFlowProxyTicket(
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


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", type=Path, default=ROOT / "current_stablecoin_flow_probe_candidates.csv")
    parser.add_argument(
        "--market-context-path",
        type=Path,
        default=STRATEGIES_ROOT / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_stablecoin_flow_proxy_tickets.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_stablecoin_flow_proxy_tickets.md")
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_stablecoin_flow_proxy_tickets(
        candidates_path=args.candidates_path,
        market_context_path=args.market_context_path,
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
    )
    write_stablecoin_flow_proxy_tickets_csv(rows, output_path=args.output_path)
    write_stablecoin_flow_proxy_tickets_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.asset, row.side, row.entry_mark, row.checkpoints)


if __name__ == "__main__":
    main()
