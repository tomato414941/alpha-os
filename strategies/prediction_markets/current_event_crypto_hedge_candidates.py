from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RELIEF_TERMS = ("returns to normal", "peace deal", "nuclear deal", "ceasefire", "agreement")
RISK_TERMS = ("closes its airspace", "invade", "war", "missile", "leadership change")
HEDGE_ASSETS = (
    ("BTC", 1.00),
    ("ETH", 0.90),
    ("SOL", 0.75),
)


@dataclass(frozen=True)
class EventCryptoHedgeCandidate:
    candidate_id: str
    market_id: str
    question: str
    asset: str
    event_bias: str
    hedge_action: str
    status: str
    score: float
    probability_gap: float
    event_market_score: float
    news_score: float
    actionability_score: float
    current_edge_after_ask: float
    ask_depth_to_5c: float
    source_quality_status: str
    refresh_status: str
    reason: str
    next_step: str


def build_event_crypto_hedge_candidates(root: Path = ROOT) -> tuple[EventCryptoHedgeCandidate, ...]:
    news_by_market = {row.get("market_id", ""): row for row in _read_rows(root / "current_event_news_pressure.csv")}
    actionability_by_market = {
        row.get("market_id", ""): row for row in _read_rows(root / "current_event_probability_actionability.csv")
    }
    output: list[EventCryptoHedgeCandidate] = []
    for row in _read_rows(root / "current_event_probability_gap.csv"):
        if row.get("category") != "geopolitical_event":
            continue
        question = row.get("question", "")
        event_bias = _event_bias(question=question, suggested_side=row.get("suggested_side", ""))
        if event_bias == "unknown":
            continue
        news = news_by_market.get(row.get("market_id", ""), {})
        actionability = actionability_by_market.get(row.get("market_id", ""), {})
        for asset, beta_weight in HEDGE_ASSETS:
            output.append(
                _build_candidate(
                    gap=row,
                    news=news,
                    actionability=actionability,
                    asset=asset,
                    beta_weight=beta_weight,
                    event_bias=event_bias,
                )
            )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_event_crypto_hedge_candidates_csv(
    rows: tuple[EventCryptoHedgeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "market_id",
                "question",
                "asset",
                "event_bias",
                "hedge_action",
                "status",
                "score",
                "probability_gap",
                "event_market_score",
                "news_score",
                "actionability_score",
                "current_edge_after_ask",
                "ask_depth_to_5c",
                "source_quality_status",
                "refresh_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.market_id,
                    row.question,
                    row.asset,
                    row.event_bias,
                    row.hedge_action,
                    row.status,
                    f"{row.score:.8f}",
                    f"{row.probability_gap:.6f}",
                    f"{row.event_market_score:.8f}",
                    f"{row.news_score:.8f}",
                    f"{row.actionability_score:.8f}",
                    f"{row.current_edge_after_ask:.6f}",
                    f"{row.ask_depth_to_5c:.6f}",
                    row.source_quality_status,
                    row.refresh_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_event_crypto_hedge_candidates_md(
    rows: tuple[EventCryptoHedgeCandidate, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Crypto Hedge Candidates\n\n")
        handle.write(
            "This maps event-probability gaps into crypto hedge candidates. "
            "It is not a direct Polymarket trade and not a live crypto order instruction.\n\n"
        )
        handle.write(
            "| candidate | asset | action | status | score | gap | market | news | edge | depth | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.asset} | "
                f"{row.hedge_action} | "
                f"{row.status} | "
                f"{row.score:.4f} | "
                f"{row.probability_gap:.4f} | "
                f"{row.event_market_score:.2f} | "
                f"{row.news_score:.2f} | "
                f"{row.current_edge_after_ask:.4f} | "
                f"{row.ask_depth_to_5c:.0f} | "
                f"{_escape(row.reason)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "These rows use prediction-market probability gaps as event-state evidence for crypto exposure. "
            "They still need leakage-safe event timing, funding, spread/depth, and beta attribution before "
            "being treated as a trading strategy.\n"
        )
    return output_path


def _build_candidate(
    *,
    gap: dict[str, str],
    news: dict[str, str],
    actionability: dict[str, str],
    asset: str,
    beta_weight: float,
    event_bias: str,
) -> EventCryptoHedgeCandidate:
    hedge_action = "paper_long" if event_bias == "risk_relief" else "paper_short"
    status = _status(
        gap_status=gap.get("status", ""),
        news_status=news.get("status", ""),
        actionability_status=actionability.get("status", ""),
    )
    score = _score(
        status=status,
        beta_weight=beta_weight,
        probability_gap=_float(gap.get("probability_gap")),
        event_market_score=_float(gap.get("score")),
        news_score=_float(news.get("score")),
        actionability_score=_float(actionability.get("score")),
        current_edge_after_ask=_float(actionability.get("current_edge_after_ask")),
        ask_depth_to_5c=_float(actionability.get("ask_depth_to_5c")),
    )
    market_id = gap.get("market_id", "")
    return EventCryptoHedgeCandidate(
        candidate_id=f"{asset.lower()}_{market_id}_event_crypto_hedge",
        market_id=market_id,
        question=gap.get("question", ""),
        asset=asset,
        event_bias=event_bias,
        hedge_action=hedge_action,
        status=status,
        score=score,
        probability_gap=_float(gap.get("probability_gap")),
        event_market_score=_float(gap.get("score")),
        news_score=_float(news.get("score")),
        actionability_score=_float(actionability.get("score")),
        current_edge_after_ask=_float(actionability.get("current_edge_after_ask")),
        ask_depth_to_5c=_float(actionability.get("ask_depth_to_5c")),
        source_quality_status=actionability.get("source_quality_status", ""),
        refresh_status=actionability.get("refresh_status", ""),
        reason=_reason(status=status, event_bias=event_bias),
        next_step=_next_step(asset=asset, hedge_action=hedge_action, question=gap.get("question", "")),
    )


def _event_bias(*, question: str, suggested_side: str) -> str:
    if suggested_side not in {"buy_yes", "buy_no"}:
        return "unknown"
    value = question.lower()
    is_relief_event = any(term in value for term in RELIEF_TERMS)
    is_risk_event = any(term in value for term in RISK_TERMS)
    if not is_relief_event and not is_risk_event:
        return "unknown"
    if is_relief_event:
        return "risk_relief" if suggested_side == "buy_yes" else "risk_escalation"
    return "risk_escalation" if suggested_side == "buy_yes" else "risk_relief"


def _status(*, gap_status: str, news_status: str, actionability_status: str) -> str:
    if actionability_status == "event_probability_candidate_after_refresh_check":
        return "event_crypto_hedge_after_refresh_candidate"
    if actionability_status == "event_probability_candidate_after_current_quote_check":
        return "event_crypto_hedge_current_quote_candidate"
    if gap_status == "paper_probability_gap_candidate" and news_status == "external_news_active":
        return "event_crypto_hedge_news_gap_candidate"
    if gap_status in {"paper_probability_gap_candidate", "probability_gap_watch"}:
        return "event_crypto_hedge_watch"
    return "event_crypto_hedge_deprioritize"


def _score(
    *,
    status: str,
    beta_weight: float,
    probability_gap: float,
    event_market_score: float,
    news_score: float,
    actionability_score: float,
    current_edge_after_ask: float,
    ask_depth_to_5c: float,
) -> float:
    base = {
        "event_crypto_hedge_after_refresh_candidate": 54.0,
        "event_crypto_hedge_current_quote_candidate": 50.0,
        "event_crypto_hedge_news_gap_candidate": 46.0,
        "event_crypto_hedge_watch": 34.0,
    }.get(status, 18.0)
    return (
        base
        + abs(probability_gap) * 45.0
        + min(event_market_score / 8.0, 10.0)
        + min(news_score / 20.0, 8.0)
        + min(actionability_score / 12.0, 8.0)
        + min(current_edge_after_ask * 18.0, 6.0)
        + min(ask_depth_to_5c / 100_000.0, 5.0)
    ) * beta_weight


def _reason(*, status: str, event_bias: str) -> str:
    direction = "risk-relief" if event_bias == "risk_relief" else "risk-escalation"
    if status == "event_crypto_hedge_after_refresh_candidate":
        return f"{direction} event probability gap survived quote/source/refresh checks"
    if status == "event_crypto_hedge_current_quote_candidate":
        return f"{direction} event probability gap has current quote support but needs repeat refresh"
    if status == "event_crypto_hedge_news_gap_candidate":
        return f"{direction} event probability gap has active external news support"
    return f"{direction} event context needs more evidence before hedge promotion"


def _next_step(*, asset: str, hedge_action: str, question: str) -> str:
    return (
        f"paper-label {asset} {hedge_action} around event market '{question}' with event timestamp, "
        "funding, spread/depth, beta attribution, and failure regime"
    )


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
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_crypto_hedge_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_event_crypto_hedge_candidates.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    rows = build_event_crypto_hedge_candidates()
    write_event_crypto_hedge_candidates_csv(rows, output_path=args.output_path)
    write_event_crypto_hedge_candidates_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.status, row.hedge_action, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
