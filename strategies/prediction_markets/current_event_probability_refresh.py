from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.prediction_markets import current_event_news_pressure as news_pressure
from strategies.prediction_markets import current_event_probability_gap as probability_gap
from strategies.prediction_markets import current_event_probability_paper_outcome as paper_outcome
from strategies.prediction_markets import current_event_probability_paper_tickets as probability_tickets
from strategies.prediction_markets import current_event_source_quality as source_quality
from strategies.prediction_markets import current_polymarket_clob_depth as clob_depth
from strategies.prediction_markets import current_polymarket_microstructure as microstructure
from strategies.prediction_markets import current_polymarket_microstructure_monitor as microstructure_monitor
from strategies.prediction_markets import current_prediction_market_paper_tickets as market_tickets


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperOutcomeRefreshRow:
    market_id: str
    question: str
    suggested_side: str
    outcome_to_buy: str
    previous_status: str
    previous_entry_ask: float
    previous_bid: float
    previous_ask: float
    current_bid: float
    current_ask: float
    current_midpoint: float
    mark_to_bid_pnl: float
    mark_to_mid_pnl: float
    estimated_payout_probability: float
    previous_edge_after_ask: float
    current_edge_after_ask: float
    edge_change: float
    ask_change: float
    ask_depth_to_5c: float
    source_quality_status: str
    score: float
    status: str
    reason: str


def run_event_probability_refresh(
    *,
    samples: int = 3,
    delay_seconds: float = 2.0,
    limit: int = 250,
    monitor_top: int = 30,
    top_markets: int = 20,
    news_top_markets: int = 10,
    max_news_records: int = 30,
) -> tuple[PaperOutcomeRefreshRow, ...]:
    previous_outcomes = _read_rows(ROOT / "current_event_probability_paper_outcome.csv")

    markets = microstructure.fetch_polymarket_markets(limit=limit)
    micro_rows = microstructure.build_polymarket_microstructure_rows(markets)
    microstructure.write_polymarket_microstructure_csv(
        micro_rows,
        output_path=ROOT / "current_polymarket_microstructure.csv",
    )
    microstructure.write_polymarket_microstructure_md(
        micro_rows,
        output_path=ROOT / "current_polymarket_microstructure.md",
    )

    monitor_samples = microstructure_monitor.run_monitor(
        samples=samples,
        delay_seconds=delay_seconds,
        limit=limit,
        top=monitor_top,
    )
    monitor_summaries = microstructure_monitor.summarize_samples(monitor_samples)
    microstructure_monitor.write_monitor_samples_csv(
        monitor_samples,
        output_path=ROOT / "current_polymarket_microstructure_monitor_samples.csv",
    )
    microstructure_monitor.write_monitor_summary_csv(
        monitor_summaries,
        output_path=ROOT / "current_polymarket_microstructure_monitor_summary.csv",
    )
    microstructure_monitor.write_monitor_summary_md(
        monitor_summaries,
        output_path=ROOT / "current_polymarket_microstructure_monitor_summary.md",
    )

    depth_rows = clob_depth.build_clob_depth_rows(
        monitor_summary_path=ROOT / "current_polymarket_microstructure_monitor_summary.csv",
        top_markets=top_markets,
    )
    clob_depth.write_clob_depth_csv(depth_rows, output_path=ROOT / "current_polymarket_clob_depth.csv")
    clob_depth.write_clob_depth_md(depth_rows, output_path=ROOT / "current_polymarket_clob_depth.md")

    market_ticket_rows = market_tickets.build_paper_tickets(
        depth_path=ROOT / "current_polymarket_clob_depth.csv",
        microstructure_path=ROOT / "current_polymarket_microstructure.csv",
    )
    market_tickets.write_tickets_csv(
        market_ticket_rows,
        output_path=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    market_tickets.write_tickets_md(
        market_ticket_rows,
        output_path=ROOT / "current_prediction_market_paper_tickets.md",
    )

    news_rows = news_pressure.build_event_news_pressure_rows(
        tickets_path=ROOT / "current_prediction_market_paper_tickets.csv",
        top_markets=news_top_markets,
        max_records=max_news_records,
    )
    news_pressure.write_event_news_pressure_csv(news_rows, output_path=ROOT / "current_event_news_pressure.csv")
    news_pressure.write_event_news_pressure_md(news_rows, output_path=ROOT / "current_event_news_pressure.md")

    gap_rows = probability_gap.build_event_probability_gap_rows(
        tickets_path=ROOT / "current_prediction_market_paper_tickets.csv",
        news_pressure_path=ROOT / "current_event_news_pressure.csv",
    )
    probability_gap.write_event_probability_gap_csv(gap_rows, output_path=ROOT / "current_event_probability_gap.csv")
    probability_gap.write_event_probability_gap_md(gap_rows, output_path=ROOT / "current_event_probability_gap.md")

    event_ticket_rows = probability_tickets.build_event_probability_paper_tickets(
        probability_gap_path=ROOT / "current_event_probability_gap.csv",
        market_tickets_path=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    probability_tickets.write_event_probability_paper_tickets_csv(
        event_ticket_rows,
        output_path=ROOT / "current_event_probability_paper_tickets.csv",
    )
    probability_tickets.write_event_probability_paper_tickets_md(
        event_ticket_rows,
        output_path=ROOT / "current_event_probability_paper_tickets.md",
    )

    source_quality_rows = source_quality.build_event_source_quality_rows(
        paper_tickets_path=ROOT / "current_event_probability_paper_tickets.csv",
        news_pressure_path=ROOT / "current_event_news_pressure.csv",
    )
    source_quality.write_event_source_quality_csv(
        source_quality_rows,
        output_path=ROOT / "current_event_source_quality.csv",
    )
    source_quality.write_event_source_quality_md(
        source_quality_rows,
        output_path=ROOT / "current_event_source_quality.md",
    )

    outcome_rows = paper_outcome.build_event_probability_paper_outcomes(
        paper_tickets_path=ROOT / "current_event_probability_paper_tickets.csv",
        market_tickets_path=ROOT / "current_prediction_market_paper_tickets.csv",
        source_quality_path=ROOT / "current_event_source_quality.csv",
    )
    paper_outcome.write_event_probability_paper_outcome_csv(
        outcome_rows,
        output_path=ROOT / "current_event_probability_paper_outcome.csv",
    )
    paper_outcome.write_event_probability_paper_outcome_md(
        outcome_rows,
        output_path=ROOT / "current_event_probability_paper_outcome.md",
    )

    refresh_rows = build_paper_outcome_refresh_rows(
        previous_outcomes=previous_outcomes,
        current_market_tickets=market_ticket_rows,
        current_source_quality=source_quality_rows,
    )
    write_paper_outcome_refresh_csv(
        refresh_rows,
        output_path=ROOT / "current_event_probability_paper_outcome_refresh.csv",
    )
    write_paper_outcome_refresh_md(
        refresh_rows,
        output_path=ROOT / "current_event_probability_paper_outcome_refresh.md",
    )
    return refresh_rows


def build_paper_outcome_refresh_rows(
    *,
    previous_outcomes: tuple[dict[str, str], ...],
    current_market_tickets: tuple[market_tickets.PredictionMarketPaperTicket, ...],
    current_source_quality: tuple[source_quality.EventSourceQualityRow, ...],
) -> tuple[PaperOutcomeRefreshRow, ...]:
    market_by_key = {
        (ticket.market_id, ticket.outcome): ticket
        for ticket in current_market_tickets
    }
    source_quality_by_market = {
        row.market_id: row.status
        for row in current_source_quality
    }
    rows: list[PaperOutcomeRefreshRow] = []
    for previous in previous_outcomes:
        if previous.get("status") not in {"paper_outcome_active_watch", "paper_outcome_edge_watch"}:
            continue
        market = market_by_key.get((previous.get("market_id", ""), previous.get("outcome_to_buy", "")))
        rows.append(_build_refresh_row(previous=previous, market=market, source_quality_by_market=source_quality_by_market))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_paper_outcome_refresh_csv(rows: tuple[PaperOutcomeRefreshRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "suggested_side",
                "outcome_to_buy",
                "previous_status",
                "previous_entry_ask",
                "previous_bid",
                "previous_ask",
                "current_bid",
                "current_ask",
                "current_midpoint",
                "mark_to_bid_pnl",
                "mark_to_mid_pnl",
                "estimated_payout_probability",
                "previous_edge_after_ask",
                "current_edge_after_ask",
                "edge_change",
                "ask_change",
                "ask_depth_to_5c",
                "source_quality_status",
                "score",
                "status",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.suggested_side,
                    row.outcome_to_buy,
                    row.previous_status,
                    f"{row.previous_entry_ask:.6f}",
                    f"{row.previous_bid:.6f}",
                    f"{row.previous_ask:.6f}",
                    f"{row.current_bid:.6f}",
                    f"{row.current_ask:.6f}",
                    f"{row.current_midpoint:.6f}",
                    f"{row.mark_to_bid_pnl:.6f}",
                    f"{row.mark_to_mid_pnl:.6f}",
                    f"{row.estimated_payout_probability:.6f}",
                    f"{row.previous_edge_after_ask:.6f}",
                    f"{row.current_edge_after_ask:.6f}",
                    f"{row.edge_change:.6f}",
                    f"{row.ask_change:.6f}",
                    f"{row.ask_depth_to_5c:.6f}",
                    row.source_quality_status,
                    f"{row.score:.8f}",
                    row.status,
                    row.reason,
                )
            )
    return output_path


def write_paper_outcome_refresh_md(rows: tuple[PaperOutcomeRefreshRow, ...], *, output_path: Path, top: int = 12) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Paper Outcome Refresh\n\n")
        handle.write(
            "This re-marks prior event-probability paper outcomes after refreshing Polymarket and news snapshots. "
            "It checks whether a prior rough edge survived fresh quotes; it is not a live trade instruction.\n\n"
        )
        handle.write(
            "| question | side | entry | current bid | current ask | bid pnl | edge now | edge change | source quality | score | status |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.suggested_side} | {row.previous_entry_ask:.4f} | "
                f"{row.current_bid:.4f} | {row.current_ask:.4f} | {row.mark_to_bid_pnl:.4f} | "
                f"{row.current_edge_after_ask:.4f} | {row.edge_change:.4f} | "
                f"{row.source_quality_status} | {row.score:.4f} | {row.status} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "A survived refresh only means the rough edge survived one public-data refresh. "
            "It still excludes actual queue priority, fills, fees, resolution risk, and adverse selection.\n"
        )
    return output_path


def _build_refresh_row(
    *,
    previous: dict[str, str],
    market: market_tickets.PredictionMarketPaperTicket | None,
    source_quality_by_market: dict[str, str],
) -> PaperOutcomeRefreshRow:
    previous_entry_ask = _float(previous.get("entry_ask"))
    previous_bid = _float(previous.get("current_bid"))
    previous_ask = _float(previous.get("current_ask"))
    estimated_payout = _float(previous.get("estimated_payout_probability"))
    previous_edge = _float(previous.get("current_edge_after_ask"))
    source_quality_status = source_quality_by_market.get(previous.get("market_id", ""), "")
    if market is None:
        return _missing_refresh_row(
            previous=previous,
            previous_entry_ask=previous_entry_ask,
            previous_bid=previous_bid,
            previous_ask=previous_ask,
            estimated_payout=estimated_payout,
            previous_edge=previous_edge,
            source_quality_status=source_quality_status,
        )
    current_midpoint = (market.best_bid + market.best_ask) / 2.0
    mark_to_bid_pnl = market.best_bid - previous_entry_ask
    mark_to_mid_pnl = current_midpoint - previous_entry_ask
    current_edge = estimated_payout - market.best_ask
    edge_change = current_edge - previous_edge
    ask_change = market.best_ask - previous_ask
    score = _refresh_score(
        current_edge_after_ask=current_edge,
        mark_to_bid_pnl=mark_to_bid_pnl,
        ask_depth_to_5c=market.ask_depth_to_5c,
        source_quality_status=source_quality_status,
    )
    status, reason = _refresh_status_reason(
        current_edge_after_ask=current_edge,
        mark_to_bid_pnl=mark_to_bid_pnl,
        source_quality_status=source_quality_status,
    )
    return PaperOutcomeRefreshRow(
        market_id=previous.get("market_id", ""),
        question=previous.get("question", ""),
        suggested_side=previous.get("suggested_side", ""),
        outcome_to_buy=previous.get("outcome_to_buy", ""),
        previous_status=previous.get("status", ""),
        previous_entry_ask=previous_entry_ask,
        previous_bid=previous_bid,
        previous_ask=previous_ask,
        current_bid=market.best_bid,
        current_ask=market.best_ask,
        current_midpoint=current_midpoint,
        mark_to_bid_pnl=mark_to_bid_pnl,
        mark_to_mid_pnl=mark_to_mid_pnl,
        estimated_payout_probability=estimated_payout,
        previous_edge_after_ask=previous_edge,
        current_edge_after_ask=current_edge,
        edge_change=edge_change,
        ask_change=ask_change,
        ask_depth_to_5c=market.ask_depth_to_5c,
        source_quality_status=source_quality_status,
        score=score,
        status=status,
        reason=reason,
    )


def _missing_refresh_row(
    *,
    previous: dict[str, str],
    previous_entry_ask: float,
    previous_bid: float,
    previous_ask: float,
    estimated_payout: float,
    previous_edge: float,
    source_quality_status: str,
) -> PaperOutcomeRefreshRow:
    return PaperOutcomeRefreshRow(
        market_id=previous.get("market_id", ""),
        question=previous.get("question", ""),
        suggested_side=previous.get("suggested_side", ""),
        outcome_to_buy=previous.get("outcome_to_buy", ""),
        previous_status=previous.get("status", ""),
        previous_entry_ask=previous_entry_ask,
        previous_bid=previous_bid,
        previous_ask=previous_ask,
        current_bid=0.0,
        current_ask=0.0,
        current_midpoint=0.0,
        mark_to_bid_pnl=0.0 - previous_entry_ask,
        mark_to_mid_pnl=0.0 - previous_entry_ask,
        estimated_payout_probability=estimated_payout,
        previous_edge_after_ask=previous_edge,
        current_edge_after_ask=0.0,
        edge_change=0.0 - previous_edge,
        ask_change=0.0 - previous_ask,
        ask_depth_to_5c=0.0,
        source_quality_status=source_quality_status,
        score=-100.0,
        status="paper_outcome_missing_current_market",
        reason="prior paper outcome no longer appears in refreshed market ticket set",
    )


def _refresh_score(
    *,
    current_edge_after_ask: float,
    mark_to_bid_pnl: float,
    ask_depth_to_5c: float,
    source_quality_status: str,
) -> float:
    quality_bonus = 15.0 if source_quality_status == "source_quality_pass" else 0.0
    return (
        current_edge_after_ask * 100.0
        + mark_to_bid_pnl * 100.0
        + min(ask_depth_to_5c / 1_000.0, 15.0)
        + quality_bonus
    )


def _refresh_status_reason(
    *,
    current_edge_after_ask: float,
    mark_to_bid_pnl: float,
    source_quality_status: str,
) -> tuple[str, str]:
    if source_quality_status != "source_quality_pass" and current_edge_after_ask >= 0.12 and mark_to_bid_pnl >= -0.08:
        return "paper_outcome_quote_survived_refresh", "prior rough edge survived refreshed quotes, but refreshed source quality is missing"
    if source_quality_status != "source_quality_pass":
        return "paper_outcome_refresh_source_watch", "refreshed source quality no longer passes"
    if mark_to_bid_pnl <= -0.15:
        return "paper_outcome_failed_refresh", "prior entry is materially underwater after refreshed bid"
    if current_edge_after_ask >= 0.12 and mark_to_bid_pnl >= -0.08:
        return "paper_outcome_survived_refresh", "prior rough edge survived refreshed public quotes"
    if current_edge_after_ask >= 0.05:
        return "paper_outcome_weak_refresh", "prior rough edge partly survived but needs another refresh"
    return "paper_outcome_failed_refresh", "prior rough edge did not survive refreshed public quotes"


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
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--monitor-top", type=int, default=30)
    parser.add_argument("--top-markets", type=int, default=20)
    parser.add_argument("--news-top-markets", type=int, default=10)
    parser.add_argument("--max-news-records", type=int, default=30)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = run_event_probability_refresh(
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        limit=args.limit,
        monitor_top=args.monitor_top,
        top_markets=args.top_markets,
        news_top_markets=args.news_top_markets,
        max_news_records=args.max_news_records,
    )
    for row in rows[: args.top]:
        print(
            row.status,
            row.suggested_side,
            f"bid_pnl={row.mark_to_bid_pnl:.3f}",
            f"edge={row.current_edge_after_ask:.3f}",
            f"edge_change={row.edge_change:.3f}",
            row.question,
        )


if __name__ == "__main__":
    main()
