from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PredictionMarketPaperTicket:
    market_id: str
    question: str
    outcome: str
    category: str
    structure: str
    best_bid: float
    best_ask: float
    spread: float
    midpoint: float
    bid_depth_to_5c: float
    ask_depth_to_5c: float
    visible_depth_score: float
    microstructure_score: float
    volume_24h: float
    one_day_price_change: float
    score: float
    status: str
    reason: str


def build_paper_tickets(
    *,
    depth_path: Path,
    microstructure_path: Path,
) -> tuple[PredictionMarketPaperTicket, ...]:
    micro_by_market = {row["market_id"]: row for row in _read_rows(microstructure_path)}
    tickets: list[PredictionMarketPaperTicket] = []
    for row in _read_rows(depth_path):
        micro = micro_by_market.get(row["market_id"], {})
        question = row["question"]
        category = _category(question)
        spread = _float(row["spread"])
        bid_depth = _float(row["bid_depth_to_5c"])
        ask_depth = _float(row["ask_depth_to_5c"])
        midpoint = (_float(row["best_bid"]) + _float(row["best_ask"])) / 2.0
        structure, status, reason = _structure_status_reason(
            question=question,
            category=category,
            spread=spread,
            bid_depth_to_5c=bid_depth,
            ask_depth_to_5c=ask_depth,
            midpoint=midpoint,
        )
        tickets.append(
            PredictionMarketPaperTicket(
                market_id=row["market_id"],
                question=question,
                outcome=row["outcome"],
                category=category,
                structure=structure,
                best_bid=_float(row["best_bid"]),
                best_ask=_float(row["best_ask"]),
                spread=spread,
                midpoint=midpoint,
                bid_depth_to_5c=bid_depth,
                ask_depth_to_5c=ask_depth,
                visible_depth_score=_float(row["visible_depth_score"]),
                microstructure_score=_float(micro.get("score", "")),
                volume_24h=_float(micro.get("volume_24h", "")),
                one_day_price_change=_float(micro.get("one_day_price_change", "")),
                score=_score(
                    category=category,
                    status=status,
                    visible_depth_score=_float(row["visible_depth_score"]),
                    microstructure_score=_float(micro.get("score", "")),
                    volume_24h=_float(micro.get("volume_24h", "")),
                    spread=spread,
                    midpoint=midpoint,
                ),
                status=status,
                reason=reason,
            )
        )
    return tuple(sorted(tickets, key=lambda ticket: ticket.score, reverse=True))


def write_tickets_csv(
    tickets: tuple[PredictionMarketPaperTicket, ...],
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
                "outcome",
                "category",
                "structure",
                "best_bid",
                "best_ask",
                "spread",
                "midpoint",
                "bid_depth_to_5c",
                "ask_depth_to_5c",
                "visible_depth_score",
                "microstructure_score",
                "volume_24h",
                "one_day_price_change",
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
                    ticket.outcome,
                    ticket.category,
                    ticket.structure,
                    f"{ticket.best_bid:.6f}",
                    f"{ticket.best_ask:.6f}",
                    f"{ticket.spread:.6f}",
                    f"{ticket.midpoint:.6f}",
                    f"{ticket.bid_depth_to_5c:.6f}",
                    f"{ticket.ask_depth_to_5c:.6f}",
                    f"{ticket.visible_depth_score:.8f}",
                    f"{ticket.microstructure_score:.8f}",
                    f"{ticket.volume_24h:.6f}",
                    f"{ticket.one_day_price_change:.6f}",
                    f"{ticket.score:.8f}",
                    ticket.status,
                    ticket.reason,
                )
            )
    return output_path


def write_tickets_md(
    tickets: tuple[PredictionMarketPaperTicket, ...],
    *,
    output_path: Path,
    top: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Prediction Market Paper Tickets\n\n")
        handle.write(
            "This converts current Polymarket microstructure and CLOB depth into research paper tickets. "
            "It is not a live trade instruction and does not estimate true event probability.\n\n"
        )
        handle.write(
            "| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for ticket in tickets[:top]:
            handle.write(
                f"| {_escape(ticket.question)} | {ticket.outcome} | {ticket.category} | "
                f"{ticket.structure} | {ticket.best_bid:.4f} | {ticket.best_ask:.4f} | "
                f"{ticket.spread:.4f} | {ticket.visible_depth_score:.4f} | "
                f"{ticket.volume_24h:.0f} | {ticket.score:.6f} | "
                f"{ticket.status} | {ticket.reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. "
            "Sports rows are treated as market-making research unless a dedicated sports model is added.\n"
        )
    return output_path


def _category(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ("bitcoin", "microstrategy", "btc")):
        return "crypto_event"
    if any(term in lowered for term in ("iran", "israel", "airspace", "strait of hormuz")):
        return "geopolitical_event"
    if any(term in lowered for term in ("vs.", "winner", "spread:", "handicap", "league", "mariners", "yankees", "rays")):
        return "sports_event"
    if any(term in lowered for term in ("election", "mayoral", "presidential")):
        return "political_event"
    return "event_market"


def _structure_status_reason(
    *,
    question: str,
    category: str,
    spread: float,
    bid_depth_to_5c: float,
    ask_depth_to_5c: float,
    midpoint: float,
) -> tuple[str, str, str]:
    min_depth = min(bid_depth_to_5c, ask_depth_to_5c)
    if min_depth < 5_000.0:
        return "none", "too_thin", "visible near-top depth is too thin"
    if midpoint <= 0.05 or midpoint >= 0.95:
        return "none", "near_certain_event", "market is too close to expiry/certainty for clean research"
    if spread > 0.05:
        return "wide_spread_making", "market_making_watch", "visible spread is wide enough for market-making research"
    if category in {"crypto_event", "geopolitical_event", "political_event"}:
        return "event_probability_model", "paper_event_model_candidate", "depth exists and the event can be tied to external information feeds"
    if category == "sports_event":
        return "maker_research", "sports_market_making_watch", "sports market has depth, but needs a dedicated model"
    return "event_probability_model", "paper_event_model_watch", "depth exists but external signal source is not identified"


def _score(
    *,
    category: str,
    status: str,
    visible_depth_score: float,
    microstructure_score: float,
    volume_24h: float,
    spread: float,
    midpoint: float,
) -> float:
    category_bonus = {
        "crypto_event": 20.0,
        "geopolitical_event": 15.0,
        "political_event": 8.0,
        "sports_event": 2.0,
    }.get(category, 0.0)
    status_bonus = {
        "paper_event_model_candidate": 20.0,
        "paper_event_model_watch": 10.0,
        "market_making_watch": 5.0,
        "sports_market_making_watch": 2.0,
    }.get(status, 0.0)
    status_penalty = {
        "near_certain_event": 120.0,
        "too_thin": 60.0,
    }.get(status, 0.0)
    probability_penalty = 10.0 if midpoint <= 0.05 or midpoint >= 0.95 else 0.0
    return (
        min(visible_depth_score, 100.0)
        + microstructure_score
        + min(volume_24h / 100_000.0, 10.0)
        + category_bonus
        + status_bonus
        - (spread * 100.0)
        - probability_penalty
        - status_penalty
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-path",
        type=Path,
        default=ROOT / "current_polymarket_clob_depth.csv",
    )
    parser.add_argument(
        "--microstructure-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_prediction_market_paper_tickets.md",
    )
    args = parser.parse_args()

    tickets = build_paper_tickets(depth_path=args.depth_path, microstructure_path=args.microstructure_path)
    write_tickets_csv(tickets, output_path=args.output_path)
    write_tickets_md(tickets, output_path=args.markdown_output_path)
    for ticket in tickets[:10]:
        print(
            ticket.category,
            ticket.status,
            ticket.outcome,
            f"spread={ticket.spread:.4f}",
            f"depth={ticket.visible_depth_score:.2f}",
            f"score={ticket.score:.4f}",
            ticket.question,
        )


if __name__ == "__main__":
    main()
