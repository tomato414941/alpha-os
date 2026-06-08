from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityPaperTicket:
    market_id: str
    question: str
    suggested_side: str
    outcome_to_buy: str
    entry_ask: float
    estimated_payout_probability: float
    edge_after_ask: float
    max_loss_per_share: float
    ask_depth_to_5c: float
    visible_depth_score: float
    probability_gap: float
    confidence_score: float
    score: float
    status: str
    reason: str


def build_event_probability_paper_tickets(
    *,
    probability_gap_path: Path,
    market_tickets_path: Path,
) -> tuple[EventProbabilityPaperTicket, ...]:
    market_rows = {
        (row.get("market_id", ""), row.get("outcome", "")): row
        for row in _read_rows(market_tickets_path)
    }
    tickets: list[EventProbabilityPaperTicket] = []
    for gap in _read_rows(probability_gap_path):
        if gap.get("status") not in {"paper_probability_gap_candidate", "probability_gap_watch"}:
            continue
        outcome = "Yes" if gap.get("suggested_side") == "buy_yes" else "No"
        market = market_rows.get((gap.get("market_id", ""), outcome))
        if not market:
            continue
        tickets.append(_build_ticket(gap=gap, market=market, outcome=outcome))
    return tuple(sorted(tickets, key=lambda ticket: ticket.score, reverse=True))


def write_event_probability_paper_tickets_csv(
    tickets: tuple[EventProbabilityPaperTicket, ...],
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
                "estimated_payout_probability",
                "edge_after_ask",
                "max_loss_per_share",
                "ask_depth_to_5c",
                "visible_depth_score",
                "probability_gap",
                "confidence_score",
                "score",
                "status",
                "reason",
            )
        )
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.market_id,
                    ticket.question,
                    ticket.suggested_side,
                    ticket.outcome_to_buy,
                    f"{ticket.entry_ask:.6f}",
                    f"{ticket.estimated_payout_probability:.6f}",
                    f"{ticket.edge_after_ask:.6f}",
                    f"{ticket.max_loss_per_share:.6f}",
                    f"{ticket.ask_depth_to_5c:.6f}",
                    f"{ticket.visible_depth_score:.8f}",
                    f"{ticket.probability_gap:.6f}",
                    f"{ticket.confidence_score:.6f}",
                    f"{ticket.score:.8f}",
                    ticket.status,
                    ticket.reason,
                )
            )
    return output_path


def write_event_probability_paper_tickets_md(
    tickets: tuple[EventProbabilityPaperTicket, ...],
    *,
    output_path: Path,
    top: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Paper Tickets\n\n")
        handle.write(
            "This converts rough prediction-market probability gaps into paper tickets with entry ask, "
            "estimated payout probability, max loss, and visible near-top ask depth. "
            "It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| question | side | ask | estimated payout | edge after ask | max loss | ask depth 5c | score | status | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for ticket in tickets[:top]:
            handle.write(
                f"| {_escape(ticket.question)} | {ticket.suggested_side} | "
                f"{ticket.entry_ask:.4f} | {ticket.estimated_payout_probability:.4f} | "
                f"{ticket.edge_after_ask:.4f} | {ticket.max_loss_per_share:.4f} | "
                f"{ticket.ask_depth_to_5c:.2f} | {ticket.score:.4f} | "
                f"{ticket.status} | {_escape(ticket.reason)} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "The expected payout is based on a rough headline-derived probability proxy. "
            "Paper promotion still requires source-level verification, stale-news filtering, "
            "fill simulation, fee checks, and adverse-selection monitoring.\n"
        )
    return output_path


def _build_ticket(
    *,
    gap: dict[str, str],
    market: dict[str, str],
    outcome: str,
) -> EventProbabilityPaperTicket:
    entry_ask = _float(market.get("best_ask"))
    estimated_yes = _float(gap.get("estimated_yes_probability"))
    estimated_payout_probability = estimated_yes if outcome == "Yes" else 1.0 - estimated_yes
    edge_after_ask = estimated_payout_probability - entry_ask
    ask_depth_to_5c = _float(market.get("ask_depth_to_5c"))
    confidence = _float(gap.get("confidence_score"))
    score = _score(
        edge_after_ask=edge_after_ask,
        confidence_score=confidence,
        ask_depth_to_5c=ask_depth_to_5c,
        entry_ask=entry_ask,
    )
    status, reason = _status_reason(
        edge_after_ask=edge_after_ask,
        confidence_score=confidence,
        ask_depth_to_5c=ask_depth_to_5c,
        entry_ask=entry_ask,
    )
    return EventProbabilityPaperTicket(
        market_id=gap.get("market_id", ""),
        question=gap.get("question", ""),
        suggested_side=gap.get("suggested_side", ""),
        outcome_to_buy=outcome,
        entry_ask=entry_ask,
        estimated_payout_probability=estimated_payout_probability,
        edge_after_ask=edge_after_ask,
        max_loss_per_share=entry_ask,
        ask_depth_to_5c=ask_depth_to_5c,
        visible_depth_score=_float(market.get("visible_depth_score")),
        probability_gap=_float(gap.get("probability_gap")),
        confidence_score=confidence,
        score=score,
        status=status,
        reason=reason,
    )


def _score(
    *,
    edge_after_ask: float,
    confidence_score: float,
    ask_depth_to_5c: float,
    entry_ask: float,
) -> float:
    edge_score = max(edge_after_ask, -0.5) * 100.0
    depth_score = min(ask_depth_to_5c / 1_000.0, 20.0)
    loss_penalty = entry_ask * 8.0
    return edge_score + confidence_score + depth_score - loss_penalty


def _status_reason(
    *,
    edge_after_ask: float,
    confidence_score: float,
    ask_depth_to_5c: float,
    entry_ask: float,
) -> tuple[str, str]:
    if entry_ask <= 0.0:
        return "no_quote", "selected outcome has no usable ask quote"
    if ask_depth_to_5c < 5_000.0:
        return "too_thin", "selected outcome has too little visible near-top ask depth"
    if edge_after_ask >= 0.12 and confidence_score >= 20.0:
        return "paper_event_probability_ticket", "rough probability edge remains after crossing the current ask"
    if edge_after_ask >= 0.05:
        return "event_probability_watch", "rough probability edge exists but needs stronger confidence or depth"
    return "no_edge_after_ask", "rough probability edge does not survive current ask"


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
        "--probability-gap-path",
        type=Path,
        default=ROOT / "current_event_probability_gap.csv",
    )
    parser.add_argument(
        "--market-tickets-path",
        type=Path,
        default=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_probability_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_event_probability_paper_tickets.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    tickets = build_event_probability_paper_tickets(
        probability_gap_path=args.probability_gap_path,
        market_tickets_path=args.market_tickets_path,
    )
    write_event_probability_paper_tickets_csv(tickets, output_path=args.output_path)
    write_event_probability_paper_tickets_md(tickets, output_path=args.markdown_output_path, top=args.top)
    for ticket in tickets[: args.top]:
        print(
            ticket.status,
            ticket.suggested_side,
            f"ask={ticket.entry_ask:.3f}",
            f"edge={ticket.edge_after_ask:.3f}",
            f"depth={ticket.ask_depth_to_5c:.0f}",
            ticket.question,
        )


if __name__ == "__main__":
    main()
