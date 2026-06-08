from __future__ import annotations

import argparse
import csv
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from strategies.current_paper_tickets import PaperTicket, _load_marks, write_paper_tickets_csv, write_paper_tickets_md


ROOT = Path(__file__).resolve().parent
DEFAULT_TOP_PER_SOURCE = 8
OKX_BASE_URL = "https://www.okx.com"


def build_broad_alpha_paper_tickets(
    *,
    existing_tickets_path: Path | None = None,
    top_per_source: int = DEFAULT_TOP_PER_SOURCE,
) -> tuple[PaperTicket, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing_tickets = _existing_tickets(existing_tickets_path)
    existing = {ticket.ticket_id: ticket for ticket in existing_tickets}
    existing_by_key = {_ticket_key(ticket): ticket for ticket in existing_tickets}
    marks = _load_marks(
        hyperliquid_snapshot_path=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
        hl_context_path=ROOT / "candidate_validation" / "current_followup_execution_context.csv",
        okx_context_path=ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
        intraday_live_gate_path=ROOT / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv",
    )
    candidates = (
        _promotion_frontier_candidates(top_per_source)
        + _cross_section_candidates(top_per_source)
        + _lane_repeat_candidates(top_per_source)
        + _seed_wallet_candidates(top_per_source)
        + _liquidation_intensity_candidates(top_per_source)
        + _btc_funding_candidates(top_per_source)
        + _intraday_derivatives_candidates(top_per_source)
        + _event_pressure_candidates(top_per_source)
        + _market_breadth_candidates(top_per_source)
        + _option_watch_candidates(top_per_source)
        + _stablecoin_proxy_candidates(top_per_source)
        + _token_unlock_candidates(top_per_source)
        + _protocol_fee_candidates(top_per_source)
        + _event_probability_candidates(top_per_source)
    )
    marks.update(_okx_marks_for_candidates(candidates))
    tickets: list[PaperTicket] = []
    seen: set[tuple[str, str]] = set()
    rank = 1
    for candidate in candidates:
        key = _dedupe_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        ticket = _ticket_for_candidate(
            rank=rank,
            candidate=candidate,
            opened_at=opened_at,
            marks=marks,
        )
        prior = existing.get(ticket.ticket_id) or existing_by_key.get(key)
        tickets.append(
            ticket
            if prior is None
            else PaperTicket(
                ticket_id=ticket.ticket_id,
                opened_at=prior.opened_at,
                rank=ticket.rank,
                opportunity=ticket.opportunity,
                probe_type=ticket.probe_type,
                status=ticket.status,
                side=ticket.side,
                asset=ticket.asset,
                venue=ticket.venue,
                candidate_size_usd=ticket.candidate_size_usd,
                observation_horizon=ticket.observation_horizon,
                checkpoints=ticket.checkpoints,
                entry_mark=prior.entry_mark or ticket.entry_mark,
                entry_source=prior.entry_source or ticket.entry_source,
                decision=ticket.decision,
                required_record=ticket.required_record,
                next_step=ticket.next_step,
            )
        )
        rank += 1
    return tuple(tickets)


def _ticket_for_candidate(
    *,
    rank: int,
    candidate: dict[str, str],
    opened_at: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> PaperTicket:
    asset = candidate["asset"]
    venue = candidate.get("venue", "")
    entry_mark, entry_source = _entry_mark(asset=asset, venue=venue, marks=marks)
    if candidate.get("entry_mark"):
        entry_mark = candidate["entry_mark"]
        entry_source = candidate.get("entry_source", "source_file")
    return PaperTicket(
        ticket_id=_ticket_id(candidate),
        opened_at=opened_at,
        rank=rank,
        opportunity=candidate["source"],
        probe_type="broad_alpha_paper_audit",
        status=candidate["status"],
        side=candidate["side"],
        asset=asset,
        venue=venue,
        candidate_size_usd=candidate.get("size_usd", "100"),
        observation_horizon=candidate.get("horizon", "15m,1h"),
        checkpoints=candidate.get("checkpoints", "15m,1h"),
        entry_mark=entry_mark,
        entry_source=entry_source,
        decision=candidate["decision"],
        required_record=candidate["required_record"],
        next_step=candidate["next_step"],
    )


def _promotion_frontier_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"promotion_frontier:{row.get('frontier_id', '')}",
            asset=row.get("asset", ""),
            decision=row.get("action", ""),
            status=row.get("status", ""),
            required_record=row.get("blocker", ""),
            next_step=row.get("next_step", ""),
            score=row.get("frontier_score", ""),
        )
        for row in _top_rows(ROOT / "current_alpha_promotion_frontier.csv", "frontier_score", top)
    )


def _cross_section_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"cost_survival_cross_section:{row.get('cluster_id', '')}",
            asset=row.get("asset", ""),
            decision=row.get("decision", ""),
            status=row.get("status", ""),
            required_record=row.get("missing_work", ""),
            next_step=row.get("next_probe", ""),
            score=row.get("survival_score", ""),
        )
        for row in _top_rows(ROOT / "current_cost_survival_cross_section.csv", "survival_score", top)
    )


def _lane_repeat_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"lane_repeat:{row.get('lane_opportunity', '')}",
            asset=row.get("asset", ""),
            decision=row.get("cluster_decision", ""),
            status=row.get("action", ""),
            required_record=row.get("required_record", ""),
            next_step=row.get("next_step", ""),
            score=row.get("priority", ""),
        )
        for row in _top_rows(ROOT / "current_split_first_lane_repeat_queue.csv", "priority", top)
    )


def _seed_wallet_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"seed_wallet_flow:{row.get('candidate_id', '')}",
            asset=row.get("execution_asset", ""),
            decision=_wallet_decision(row.get("side", "")),
            status=row.get("status", ""),
            required_record="wallet forward label, funding, spread/depth, copycat-risk, entity quality",
            next_step=row.get("next_step", ""),
            score=row.get("score", ""),
        )
        for row in _top_rows(
            ROOT / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv",
            "score",
            top,
        )
    )


def _liquidation_intensity_candidates(top: int) -> tuple[dict[str, str], ...]:
    rows = [
        row
        for row in _top_rows(
            ROOT / "liquidation_flow" / "current_okx_liquidation_intensity_paper_gate.csv",
            "conservative_net_bps",
            top * 4,
        )
        if row.get("candidate_size_usd") == "100.00"
    ][:top]
    return tuple(
        _directional_candidate(
            source=f"liquidation_intensity:{row.get('asset', '')}:{row.get('action', '')}",
            asset=row.get("asset", ""),
            decision=f"paper_{row.get('trade_direction', '')}",
            status=row.get("gate_action", ""),
            required_record=row.get("reason", ""),
            next_step=row.get("next_step", ""),
            score=row.get("conservative_net_bps", ""),
            venue="OKX",
            size_usd=row.get("candidate_size_usd", "100.00"),
        )
        for row in rows
    )


def _btc_funding_candidates(top: int) -> tuple[dict[str, str], ...]:
    rows = [
        row
        for row in _top_rows(ROOT / "institutional_flow" / "current_btc_etf_funding_paper_ticket.csv", "score", top)
        if row.get("status") == "paper_venue_candidate"
    ]
    return tuple(
        _directional_candidate(
            source=f"btc_etf_funding:{row.get('venue', '')}",
            asset="BTC",
            decision=f"paper_{row.get('side', '')}",
            status=row.get("status", ""),
            required_record="venue mark/index basis, funding timestamp, stop, fill, account fee tier",
            next_step="paper-check BTC short venue choice, stop criteria, mark/index basis, and actual account fee/fill assumptions",
            score=row.get("score", ""),
            venue="HL" if row.get("venue") == "Hyperliquid" else row.get("venue", ""),
        )
        for row in rows
    )


def _intraday_derivatives_candidates(top: int) -> tuple[dict[str, str], ...]:
    live_symbols = _intraday_live_symbols()
    rows = tuple(
        row
        for row in _top_rows(
            ROOT / "p0_parallel" / "binance_derivatives_intraday_paper_labels_2bps.csv",
            "score",
            top * 2,
        )
        if _float(row.get("score")) > 0.0
        and row.get("symbol", "") in live_symbols
    )[:top]
    return tuple(
        _directional_candidate(
            source=f"intraday_derivatives:{row.get('symbol', '')}:{row.get('feature', '')}:{row.get('action', '')}",
            asset=row.get("symbol", ""),
            decision=_intraday_derivatives_decision(row.get("action", "")),
            status=row.get("status", ""),
            required_record="fresh live feature value, live spread/depth, funding timestamp, fill delay, stop behavior",
            next_step=row.get("next_step", ""),
            score=row.get("score", ""),
            size_usd="100.00",
        )
        for row in rows
    )


def _intraday_live_symbols() -> set[str]:
    return {
        row.get("symbol", "")
        for row in _read_rows(ROOT / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv")
        if row.get("symbol") and row.get("candidate_size_usd") == "100.00"
    }


def _event_pressure_candidates(top: int) -> tuple[dict[str, str], ...]:
    attention_prices = _attention_prices()
    rows = tuple(
        row
        for row in _top_rows(ROOT / "news_social" / "current_event_pressure_cluster.csv", "score", top * 2)
        if _event_pressure_decision(row.get("side", "")) != "paper_observe"
    )[:top]
    return tuple(
        {
            "source": f"event_pressure:{row.get('symbol', '')}:{row.get('status', '')}",
            "asset": row.get("symbol", ""),
            "decision": _event_pressure_decision(row.get("side", "")),
            "side": "short" if _event_pressure_decision(row.get("side", "")) == "paper_short" else "long",
            "status": row.get("status", ""),
            "required_record": "source independence, duplicate filtering, funding, spread/depth, stop, stale-headline check",
            "next_step": row.get("next_step", ""),
            "score": row.get("score", ""),
            "horizon": "15m,1h,4h",
            "checkpoints": "15m,1h,4h",
            "size_usd": "100",
            "entry_mark": attention_prices.get(row.get("symbol", ""), ""),
            "entry_source": "attention_price_context" if row.get("symbol", "") in attention_prices else "",
        }
        for row in rows
    )


def _attention_prices() -> dict[str, str]:
    return {
        row.get("symbol", ""): row.get("current_price", "")
        for row in _read_rows(ROOT / "news_social" / "current_attention_price_context.csv")
        if row.get("symbol") and row.get("current_price")
    }


def _market_breadth_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"market_breadth:{row.get('symbol', '')}:{row.get('side', '')}",
            asset=row.get("symbol", ""),
            decision="paper_long" if row.get("side") == "long_reversal" else "paper_short",
            status=row.get("status", ""),
            required_record="fresh label, spread/depth, funding, stop, adverse excursion",
            next_step=f"paper-check {row.get('symbol', '')} market-breadth dislocation with fill and stop logs",
            score=row.get("score", ""),
        )
        for row in _top_rows(ROOT / "market_breadth" / "current_volume_price_dislocation_labels.csv", "score", top)
    )


def _option_watch_candidates(top: int) -> tuple[dict[str, str], ...]:
    rows = _top_rows(ROOT / "options_volatility" / "current_options_volatility_paper_tickets.csv", "score", top)
    return tuple(
        {
            "source": f"options_volatility:{row.get('currency', '')}:{row.get('expiry', '')}",
            "asset": f"{row.get('currency', '')}-OPTION",
            "decision": "paper_observe",
            "side": row.get("structure", "long_vol"),
            "status": row.get("status", ""),
            "required_record": "option quote refresh, sweep depth, hedge marks, exit bid, max loss, margin",
            "next_step": "track option-specific premium and hedge path before directional promotion",
            "score": row.get("score", ""),
            "horizon": "1h,4h",
            "checkpoints": "1h,4h",
            "size_usd": row.get("max_loss_usd", "100"),
        }
        for row in rows
    )


def _stablecoin_proxy_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "source": f"stablecoin_proxy:{row.get('ticket_id', '')}",
            "asset": row.get("asset", ""),
            "decision": row.get("decision", ""),
            "side": row.get("side", ""),
            "status": row.get("status", ""),
            "required_record": row.get("required_record", ""),
            "next_step": row.get("next_step", ""),
            "venue": row.get("venue", ""),
            "size_usd": row.get("candidate_size_usd", "100"),
            "horizon": row.get("observation_horizon", "1h,4h"),
            "checkpoints": row.get("checkpoints", "1h,4h"),
            "entry_mark": row.get("entry_mark", ""),
            "entry_source": row.get("entry_source", ""),
        }
        for row in _top_rows(
            ROOT / "stablecoin_liquidity" / "current_stablecoin_flow_proxy_tickets.csv",
            "rank",
            top,
            reverse=False,
        )
    )


def _token_unlock_candidates(top: int) -> tuple[dict[str, str], ...]:
    rows = tuple(
        row
        for row in _top_rows(ROOT / "token_unlocks" / "current_token_unlock_paper_tickets.csv", "score", top * 2)
        if row.get("status") == "paper_short_candidate"
    )[:top]
    return tuple(
        _directional_candidate(
            source=f"token_unlock:{row.get('symbol', '')}",
            asset=row.get("symbol", ""),
            decision="paper_short" if row.get("side") == "short" else "paper_observe",
            status=row.get("status", ""),
            required_record="event window, mark move, funding, depth, crowding, stop",
            next_step=f"paper-check {row.get('symbol', '')} token unlock short with event-window and funding logs",
            score=row.get("score", ""),
        )
        for row in rows
    )


def _protocol_fee_candidates(top: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _directional_candidate(
            source=f"protocol_fee:{row.get('protocol', '')}",
            asset=row.get("token_symbol", ""),
            decision="paper_long" if row.get("side") == "long_token" else "paper_observe",
            status="protocol_fee_paper_observation",
            required_record="4h/12h/24h return, funding, spread, depth, fee-growth persistence",
            next_step=row.get("next_step", ""),
            score=row.get("thesis_score", ""),
            size_usd=row.get("paper_notional_usd", "100"),
            venue="HL",
            horizon=row.get("observation_horizons", "4h,12h,24h"),
            checkpoints=row.get("observation_horizons", "4h,12h,24h"),
        )
        for row in _top_rows(
            ROOT / "protocol_fundamentals" / "current_protocol_fee_paper_tickets.csv",
            "thesis_score",
            top,
        )
    )


def _event_probability_candidates(top: int) -> tuple[dict[str, str], ...]:
    rows = _top_rows(
        ROOT / "prediction_markets" / "current_event_probability_paper_tickets.csv",
        "score",
        top,
    )
    return tuple(
        {
            "source": f"event_probability:{row.get('market_id', '')}:{row.get('outcome_to_buy', '')}",
            "asset": "EVENT",
            "decision": "paper_long",
            "side": f"{row.get('suggested_side', '')}: {row.get('question', '')}",
            "status": row.get("status", ""),
            "required_record": "quote refresh, depth, fee, fill assumption, source quality, event-resolution risk",
            "next_step": f"refresh quote and source quality for {row.get('question', '')}",
            "score": row.get("score", ""),
            "horizon": "1h,4h",
            "checkpoints": "1h,4h",
            "size_usd": row.get("max_loss_per_share", "0.10"),
            "entry_mark": row.get("entry_ask", ""),
            "entry_source": "event_probability_entry_ask",
        }
        for row in rows
    )


def _directional_candidate(
    *,
    source: str,
    asset: str,
    decision: str,
    status: str,
    required_record: str,
    next_step: str,
    score: str,
    venue: str = "",
    size_usd: str = "100",
    horizon: str = "15m,1h,4h",
    checkpoints: str = "15m,1h,4h",
) -> dict[str, str]:
    decision = "paper_short" if decision == "paper_short" else "paper_long"
    return {
        "source": source,
        "asset": asset,
        "decision": decision,
        "side": "short" if decision == "paper_short" else "long",
        "status": status,
        "required_record": required_record,
        "next_step": next_step,
        "score": score,
        "venue": venue,
        "size_usd": size_usd,
        "horizon": horizon,
        "checkpoints": checkpoints,
    }


def _dedupe_key(candidate: dict[str, str]) -> tuple[str, str, str]:
    asset = candidate["asset"]
    decision = candidate["decision"]
    if asset == "EVENT" or asset.endswith("-OPTION"):
        return (asset, decision, candidate["source"])
    return (asset, decision, "")


def _ticket_key(ticket: PaperTicket) -> tuple[str, str, str]:
    if ticket.asset == "EVENT" or ticket.asset.endswith("-OPTION"):
        return (ticket.asset, ticket.decision, ticket.opportunity)
    return (ticket.asset, ticket.decision, "")


def _ticket_id(candidate: dict[str, str]) -> str:
    asset = candidate["asset"]
    decision = candidate["decision"]
    if asset == "EVENT" or asset.endswith("-OPTION"):
        return f"broad-paper-{_slug(asset)}-{_slug(decision)}-{_slug(candidate['source'])}"
    return f"broad-paper-{_slug(asset)}-{_slug(decision)}"


def _wallet_decision(side: str) -> str:
    if "short" in side:
        return "paper_short"
    return "paper_long"


def _intraday_derivatives_decision(action: str) -> str:
    if action == "short_opposite":
        return "paper_short"
    return "paper_long"


def _event_pressure_decision(side: str) -> str:
    if side.startswith("short"):
        return "paper_short"
    if side.startswith("long"):
        return "paper_long"
    return "paper_observe"


def _entry_mark(
    *,
    asset: str,
    venue: str,
    marks: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    for key in ((venue.upper(), asset.upper()), ("HL", asset.upper()), ("OKX", asset.upper()), ("", asset.upper())):
        if key in marks:
            return marks[key]
    return "", ""


def _okx_marks_for_candidates(candidates: tuple[dict[str, str], ...]) -> dict[tuple[str, str], tuple[str, str]]:
    assets = sorted(
        {
            candidate["asset"].upper()
            for candidate in candidates
            if candidate.get("venue") == "OKX" and candidate.get("asset")
        }
    )
    if not assets:
        return {}
    try:
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/tickers",
            params={"instType": "SWAP"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return {}
    marks: dict[tuple[str, str], tuple[str, str]] = {}
    wanted = {f"{asset}-USDT-SWAP": asset for asset in assets}
    for item in response.json().get("data", ()):
        asset = wanted.get(str(item.get("instId", "")))
        mark = item.get("last", "")
        if asset and mark:
            marks[("OKX", asset)] = (str(mark), "okx_ticker")
    return marks


def _top_rows(path: Path, score_field: str, top: int, *, reverse: bool = True) -> tuple[dict[str, str], ...]:
    return tuple(sorted(_read_rows(path), key=lambda row: _float(row.get(score_field)), reverse=reverse)[:top])


def _existing_tickets(path: Path | None) -> tuple[PaperTicket, ...]:
    if path is None:
        return ()
    rows = []
    for row in _read_rows(path):
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "na"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_paper_tickets.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_paper_tickets.md",
    )
    parser.add_argument("--top-per-source", type=int, default=DEFAULT_TOP_PER_SOURCE)
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_broad_alpha_paper_tickets(
        existing_tickets_path=args.output_path if args.preserve_opened_at else None,
        top_per_source=args.top_per_source,
    )
    write_paper_tickets_csv(rows, output_path=args.output_path)
    write_paper_tickets_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.asset, row.decision, row.entry_mark)


if __name__ == "__main__":
    main()
