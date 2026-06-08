from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityActionabilityRow:
    market_id: str
    question: str
    suggested_side: str
    outcome_to_buy: str
    status: str
    side: str
    score: float
    entry_ask: float
    current_bid: float
    current_ask: float
    spread: float
    mark_to_bid_pnl: float
    estimated_payout_probability: float
    current_edge_after_ask: float
    ask_depth_to_5c: float
    source_quality_status: str
    refresh_status: str
    source_status: str
    reason: str
    next_step: str


def build_event_probability_actionability_rows(root: Path = ROOT) -> tuple[EventProbabilityActionabilityRow, ...]:
    refresh_by_key = {
        (row.get("market_id", ""), row.get("outcome_to_buy", "")): row
        for row in _read_rows(root / "prediction_markets" / "current_event_probability_paper_outcome_refresh.csv")
    }
    output: list[EventProbabilityActionabilityRow] = []
    for row in _read_rows(root / "prediction_markets" / "current_event_probability_paper_outcome.csv"):
        entry_ask = _float(row.get("entry_ask"))
        current_bid = _float(row.get("current_bid"))
        current_ask = _float(row.get("current_ask"))
        current_edge = _float(row.get("current_edge_after_ask"))
        mark_to_bid = _float(row.get("mark_to_bid_pnl"))
        ask_depth = _float(row.get("ask_depth_to_5c"))
        source_quality = row.get("source_quality_status", "")
        refresh = refresh_by_key.get((row.get("market_id", ""), row.get("outcome_to_buy", "")), {})
        refresh_status = refresh.get("status", "")
        spread = current_ask - current_bid
        status, side, reason = _status_side_reason(
            current_bid=current_bid,
            current_ask=current_ask,
            spread=spread,
            current_edge_after_ask=current_edge,
            mark_to_bid_pnl=mark_to_bid,
            ask_depth_to_5c=ask_depth,
            source_quality_status=source_quality,
            refresh_status=refresh_status,
        )
        output.append(
            EventProbabilityActionabilityRow(
                market_id=row.get("market_id", ""),
                question=row.get("question", ""),
                suggested_side=row.get("suggested_side", ""),
                outcome_to_buy=row.get("outcome_to_buy", ""),
                status=status,
                side=side,
                score=_score(
                    status=status,
                    current_edge_after_ask=current_edge,
                    mark_to_bid_pnl=mark_to_bid,
                    spread=spread,
                    ask_depth_to_5c=ask_depth,
                ),
                entry_ask=entry_ask,
                current_bid=current_bid,
                current_ask=current_ask,
                spread=spread,
                mark_to_bid_pnl=mark_to_bid,
                estimated_payout_probability=_float(row.get("estimated_payout_probability")),
                current_edge_after_ask=current_edge,
                ask_depth_to_5c=ask_depth,
                source_quality_status=source_quality,
                refresh_status=refresh_status,
                source_status=row.get("status", ""),
                reason=reason,
                next_step=_next_step(question=row.get("question", ""), status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_event_probability_actionability_csv(
    rows: tuple[EventProbabilityActionabilityRow, ...],
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
                "status",
                "side",
                "score",
                "entry_ask",
                "current_bid",
                "current_ask",
                "spread",
                "mark_to_bid_pnl",
                "estimated_payout_probability",
                "current_edge_after_ask",
                "ask_depth_to_5c",
                "source_quality_status",
                "refresh_status",
                "source_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.suggested_side,
                    row.outcome_to_buy,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.entry_ask:.6f}",
                    f"{row.current_bid:.6f}",
                    f"{row.current_ask:.6f}",
                    f"{row.spread:.6f}",
                    f"{row.mark_to_bid_pnl:.6f}",
                    f"{row.estimated_payout_probability:.6f}",
                    f"{row.current_edge_after_ask:.6f}",
                    f"{row.ask_depth_to_5c:.6f}",
                    row.source_quality_status,
                    row.refresh_status,
                    row.source_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_event_probability_actionability_md(
    rows: tuple[EventProbabilityActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Actionability\n\n")
        handle.write(
            "This separates rough event-probability edge from candidates that are ready for a paper "
            "fill and adverse-selection check. It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| question | side | status | score | bid | ask | spread | edge after ask | depth 5c | source | refresh | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.side} | {row.status} | {row.score:.4f} | "
                f"{row.current_bid:.4f} | {row.current_ask:.4f} | {row.spread:.4f} | "
                f"{row.current_edge_after_ask:.4f} | {row.ask_depth_to_5c:.0f} | "
                f"{row.source_quality_status} | {row.refresh_status} | {_escape(row.reason)} |\n"
            )
    return output_path


def _status_side_reason(
    *,
    current_bid: float,
    current_ask: float,
    spread: float,
    current_edge_after_ask: float,
    mark_to_bid_pnl: float,
    ask_depth_to_5c: float,
    source_quality_status: str,
    refresh_status: str,
) -> tuple[str, str, str]:
    if current_bid <= 0.0 or current_ask <= 0.0:
        return "event_probability_quote_blocked", "no_trade_until_quote", "current bid/ask is missing"
    if source_quality_status != "source_quality_pass":
        return (
            "event_probability_source_quality_blocked",
            "no_trade_until_source_quality",
            "rough probability edge is not backed by enough fresh independent sources",
        )
    if spread > 0.03:
        return (
            "event_probability_quote_mechanics_watch",
            "paper_quote_check",
            "spread is too wide for a clean paper fill check",
        )
    if current_edge_after_ask >= 0.12 and mark_to_bid_pnl >= -0.05 and ask_depth_to_5c >= 10_000.0:
        if refresh_status == "paper_outcome_survived_refresh":
            return (
                "event_probability_candidate_after_refresh_check",
                "paper_event_probability_after_refresh_check",
                "rough edge, source quality, quote, depth, and refreshed mark all pass before fill checks",
            )
        return (
            "event_probability_candidate_after_current_quote_check",
            "paper_event_probability_current_quote_check",
            "current quote and source quality pass, but the old paper entry did not prove durable refresh survival",
        )
    if current_edge_after_ask >= 0.05:
        return (
            "event_probability_edge_watch",
            "paper_edge_watch",
            "rough edge remains but quote durability, source freshness, or depth is not strong enough",
        )
    return "event_probability_deprioritize", "none", "rough edge no longer survives the current quote"


def _score(
    *,
    status: str,
    current_edge_after_ask: float,
    mark_to_bid_pnl: float,
    spread: float,
    ask_depth_to_5c: float,
) -> float:
    if status in {
        "event_probability_candidate_after_refresh_check",
        "event_probability_candidate_after_current_quote_check",
    }:
        return min(
            88.0,
            58.0
            + min(current_edge_after_ask * 50.0, 18.0)
            + min(ask_depth_to_5c / 5_000.0, 8.0)
            + min(max(mark_to_bid_pnl, -0.05) * 80.0, 4.0)
            - max(spread - 0.01, 0.0) * 120.0,
        )
    if status == "event_probability_edge_watch":
        return min(62.0, 42.0 + min(current_edge_after_ask * 40.0, 12.0) + min(ask_depth_to_5c / 20_000.0, 4.0))
    if status == "event_probability_quote_mechanics_watch":
        return min(52.0, 36.0 + min(current_edge_after_ask * 25.0, 8.0))
    if status == "event_probability_source_quality_blocked":
        return min(42.0, 28.0 + min(current_edge_after_ask * 20.0, 8.0))
    if status == "event_probability_quote_blocked":
        return 24.0
    return 20.0


def _next_step(*, question: str, status: str) -> str:
    if status == "event_probability_candidate_after_refresh_check":
        return f"paper-check {question} with explicit fee, queue, fill, resolution-risk, and adverse-selection assumptions"
    if status == "event_probability_candidate_after_current_quote_check":
        return f"restart paper ticket for {question} at current quote, then require another quote/news refresh before promotion"
    if status == "event_probability_edge_watch":
        return f"refresh news, quotes, and CLOB depth for {question}; do not promote from rough edge alone"
    if status == "event_probability_quote_mechanics_watch":
        return f"wait for tighter quote or deeper visible ask before paper-checking {question}"
    if status == "event_probability_source_quality_blocked":
        return f"collect fresher independent sources before keeping {question} in the alpha stack"
    return f"deprioritize {question} until a fresh source-backed quote edge reappears"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_event_probability_actionability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=LANE_ROOT / "current_event_probability_actionability.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_event_probability_actionability_rows()
    write_event_probability_actionability_csv(rows, output_path=args.output_path)
    write_event_probability_actionability_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.side, f"score={row.score:.2f}", row.question)


if __name__ == "__main__":
    main()
