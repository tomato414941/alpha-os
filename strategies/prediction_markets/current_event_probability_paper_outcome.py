from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityPaperOutcome:
    market_id: str
    question: str
    suggested_side: str
    outcome_to_buy: str
    entry_ask: float
    current_bid: float
    current_ask: float
    current_midpoint: float
    mark_to_bid_pnl: float
    mark_to_mid_pnl: float
    estimated_payout_probability: float
    current_edge_after_ask: float
    current_edge_after_mid: float
    ask_depth_to_5c: float
    source_quality_status: str
    score: float
    status: str
    reason: str


def build_event_probability_paper_outcomes(
    *,
    paper_tickets_path: Path,
    market_tickets_path: Path,
    source_quality_path: Path,
) -> tuple[EventProbabilityPaperOutcome, ...]:
    market_rows = {
        (row.get("market_id", ""), row.get("outcome", "")): row
        for row in _read_rows(market_tickets_path)
    }
    quality_rows = {row.get("market_id", ""): row for row in _read_rows(source_quality_path)}
    outcomes: list[EventProbabilityPaperOutcome] = []
    for ticket in _read_rows(paper_tickets_path):
        if ticket.get("status") not in {"paper_event_probability_ticket", "event_probability_watch"}:
            continue
        market = market_rows.get((ticket.get("market_id", ""), ticket.get("outcome_to_buy", "")))
        if not market:
            continue
        outcomes.append(
            _build_outcome(
                ticket=ticket,
                market=market,
                source_quality=quality_rows.get(ticket.get("market_id", ""), {}),
            )
        )
    return tuple(sorted(outcomes, key=lambda row: row.score, reverse=True))


def write_event_probability_paper_outcome_csv(
    outcomes: tuple[EventProbabilityPaperOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "suggested_side",
                "outcome_to_buy",
                "entry_ask",
                "current_bid",
                "current_ask",
                "current_midpoint",
                "mark_to_bid_pnl",
                "mark_to_mid_pnl",
                "estimated_payout_probability",
                "current_edge_after_ask",
                "current_edge_after_mid",
                "ask_depth_to_5c",
                "source_quality_status",
                "score",
                "status",
                "reason",
            )
        )
        for row in outcomes:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.suggested_side,
                    row.outcome_to_buy,
                    f"{row.entry_ask:.6f}",
                    f"{row.current_bid:.6f}",
                    f"{row.current_ask:.6f}",
                    f"{row.current_midpoint:.6f}",
                    f"{row.mark_to_bid_pnl:.6f}",
                    f"{row.mark_to_mid_pnl:.6f}",
                    f"{row.estimated_payout_probability:.6f}",
                    f"{row.current_edge_after_ask:.6f}",
                    f"{row.current_edge_after_mid:.6f}",
                    f"{row.ask_depth_to_5c:.6f}",
                    row.source_quality_status,
                    f"{row.score:.8f}",
                    row.status,
                    row.reason,
                )
            )
    return output_path


def write_event_probability_paper_outcome_md(
    outcomes: tuple[EventProbabilityPaperOutcome, ...],
    *,
    output_path: Path,
    top: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Paper Outcome\n\n")
        handle.write(
            "This marks event-probability paper tickets against the current prediction-market quote. "
            "It is a paper monitor, not a live trade instruction.\n\n"
        )
        handle.write(
            "| question | side | entry ask | bid | ask | bid pnl | mid pnl | edge after ask | source quality | score | status |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in outcomes[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.suggested_side} | {row.entry_ask:.4f} | "
                f"{row.current_bid:.4f} | {row.current_ask:.4f} | {row.mark_to_bid_pnl:.4f} | "
                f"{row.mark_to_mid_pnl:.4f} | {row.current_edge_after_ask:.4f} | "
                f"{row.source_quality_status} | {row.score:.4f} | {row.status} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "A fresh run must refresh market tickets first; otherwise the mark is based on the existing "
            "snapshot. Bid/ask marks do not prove fill quality, queue priority, or adverse-selection safety.\n"
        )
    return output_path


def _build_outcome(
    *,
    ticket: dict[str, str],
    market: dict[str, str],
    source_quality: dict[str, str],
) -> EventProbabilityPaperOutcome:
    entry_ask = _float(ticket.get("entry_ask"))
    current_bid = _float(market.get("best_bid"))
    current_ask = _float(market.get("best_ask"))
    current_midpoint = _float(market.get("midpoint"))
    estimated_payout = _float(ticket.get("estimated_payout_probability"))
    mark_to_bid_pnl = current_bid - entry_ask
    mark_to_mid_pnl = current_midpoint - entry_ask
    current_edge_after_ask = estimated_payout - current_ask
    current_edge_after_mid = estimated_payout - current_midpoint
    source_quality_status = source_quality.get("status", "")
    ask_depth_to_5c = _float(market.get("ask_depth_to_5c"))
    score = _score(
        current_edge_after_ask=current_edge_after_ask,
        mark_to_mid_pnl=mark_to_mid_pnl,
        ask_depth_to_5c=ask_depth_to_5c,
        source_quality_status=source_quality_status,
    )
    status, reason = _status_reason(
        current_edge_after_ask=current_edge_after_ask,
        mark_to_bid_pnl=mark_to_bid_pnl,
        source_quality_status=source_quality_status,
    )
    return EventProbabilityPaperOutcome(
        market_id=ticket.get("market_id", ""),
        question=ticket.get("question", ""),
        suggested_side=ticket.get("suggested_side", ""),
        outcome_to_buy=ticket.get("outcome_to_buy", ""),
        entry_ask=entry_ask,
        current_bid=current_bid,
        current_ask=current_ask,
        current_midpoint=current_midpoint,
        mark_to_bid_pnl=mark_to_bid_pnl,
        mark_to_mid_pnl=mark_to_mid_pnl,
        estimated_payout_probability=estimated_payout,
        current_edge_after_ask=current_edge_after_ask,
        current_edge_after_mid=current_edge_after_mid,
        ask_depth_to_5c=ask_depth_to_5c,
        source_quality_status=source_quality_status,
        score=score,
        status=status,
        reason=reason,
    )


def _score(
    *,
    current_edge_after_ask: float,
    mark_to_mid_pnl: float,
    ask_depth_to_5c: float,
    source_quality_status: str,
) -> float:
    quality_bonus = 15.0 if source_quality_status == "source_quality_pass" else 0.0
    return (
        current_edge_after_ask * 100.0
        + mark_to_mid_pnl * 50.0
        + min(ask_depth_to_5c / 1_000.0, 15.0)
        + quality_bonus
    )


def _status_reason(
    *,
    current_edge_after_ask: float,
    mark_to_bid_pnl: float,
    source_quality_status: str,
) -> tuple[str, str]:
    if source_quality_status != "source_quality_pass":
        return "paper_outcome_source_quality_watch", "source quality has not passed the current gate"
    if current_edge_after_ask >= 0.12 and mark_to_bid_pnl >= -0.08:
        return "paper_outcome_active_watch", "paper ticket still has rough edge after current ask"
    if current_edge_after_ask >= 0.05:
        return "paper_outcome_edge_watch", "rough edge remains, but mark or source checks need attention"
    return "paper_outcome_deprioritize", "rough edge no longer survives current quote"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-tickets-path",
        type=Path,
        default=ROOT / "current_event_probability_paper_tickets.csv",
    )
    parser.add_argument(
        "--market-tickets-path",
        type=Path,
        default=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    parser.add_argument(
        "--source-quality-path",
        type=Path,
        default=ROOT / "current_event_source_quality.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_probability_paper_outcome.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_event_probability_paper_outcome.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    outcomes = build_event_probability_paper_outcomes(
        paper_tickets_path=args.paper_tickets_path,
        market_tickets_path=args.market_tickets_path,
        source_quality_path=args.source_quality_path,
    )
    write_event_probability_paper_outcome_csv(outcomes, output_path=args.output_path)
    write_event_probability_paper_outcome_md(outcomes, output_path=args.markdown_output_path, top=args.top)
    for outcome in outcomes[: args.top]:
        print(
            outcome.status,
            outcome.suggested_side,
            f"bid_pnl={outcome.mark_to_bid_pnl:.3f}",
            f"edge={outcome.current_edge_after_ask:.3f}",
            outcome.question,
        )


if __name__ == "__main__":
    main()
