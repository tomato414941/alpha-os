from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityGapRow:
    market_id: str
    question: str
    category: str
    market_yes_probability: float
    estimated_yes_probability: float
    probability_gap: float
    suggested_side: str
    confidence_score: float
    score: float
    status: str
    evidence_terms: str
    reason: str


def build_event_probability_gap_rows(
    *,
    tickets_path: Path,
    news_pressure_path: Path,
) -> tuple[EventProbabilityGapRow, ...]:
    yes_tickets = {
        row["market_id"]: row
        for row in _read_rows(tickets_path)
        if row.get("status") in {"paper_event_model_candidate", "paper_event_model_watch"}
        and row.get("outcome") == "Yes"
    }
    rows: list[EventProbabilityGapRow] = []
    for news in _read_rows(news_pressure_path):
        ticket = yes_tickets.get(news.get("market_id", ""))
        if not ticket:
            continue
        rows.append(_build_gap_row(ticket=ticket, news=news))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_event_probability_gap_csv(rows: tuple[EventProbabilityGapRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "category",
                "market_yes_probability",
                "estimated_yes_probability",
                "probability_gap",
                "suggested_side",
                "confidence_score",
                "score",
                "status",
                "evidence_terms",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.category,
                    f"{row.market_yes_probability:.6f}",
                    f"{row.estimated_yes_probability:.6f}",
                    f"{row.probability_gap:.6f}",
                    row.suggested_side,
                    f"{row.confidence_score:.6f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.evidence_terms,
                    row.reason,
                )
            )
    return output_path


def write_event_probability_gap_md(
    rows: tuple[EventProbabilityGapRow, ...],
    *,
    output_path: Path,
    top: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Gap\n\n")
        handle.write(
            "This compares a rough headline-derived Yes-probability proxy against prediction-market "
            "implied odds. It is a candidate screen, not a calibrated probability model or trade instruction.\n\n"
        )
        handle.write(
            "| question | market yes | estimated yes | gap | side | confidence | score | status | evidence |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.market_yes_probability:.4f} | "
                f"{row.estimated_yes_probability:.4f} | {row.probability_gap:.4f} | "
                f"{row.suggested_side} | {row.confidence_score:.2f} | {row.score:.4f} | "
                f"{row.status} | {_escape(row.evidence_terms)} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "The estimate is intentionally simple and headline-based. It should be used to prioritize "
            "research and paper checks only. Promotion requires source verification, timing checks, "
            "calibration, execution costs, and adverse-selection analysis.\n"
        )
    return output_path


def _build_gap_row(*, ticket: dict[str, str], news: dict[str, str]) -> EventProbabilityGapRow:
    question = ticket.get("question", "")
    text = f"{question} {news.get('top_titles', '')}".lower()
    market_yes_probability = _float(ticket.get("midpoint"))
    evidence = _evidence_for_question(question, text)
    estimated_yes_probability = _estimated_yes_probability(question, evidence)
    probability_gap = estimated_yes_probability - market_yes_probability
    suggested_side = "buy_yes" if probability_gap > 0 else "buy_no"
    confidence_score = _confidence_score(news, evidence)
    score = abs(probability_gap) * 100.0 + confidence_score
    status = _status(probability_gap=probability_gap, confidence_score=confidence_score)
    return EventProbabilityGapRow(
        market_id=ticket.get("market_id", ""),
        question=question,
        category=ticket.get("category", ""),
        market_yes_probability=market_yes_probability,
        estimated_yes_probability=estimated_yes_probability,
        probability_gap=probability_gap,
        suggested_side=suggested_side,
        confidence_score=confidence_score,
        score=score,
        status=status,
        evidence_terms=", ".join(evidence),
        reason=_reason(status=status, probability_gap=probability_gap),
    )


def _evidence_for_question(question: str, text: str) -> tuple[str, ...]:
    lowered = question.lower()
    if "peace deal" in lowered and "iran" in lowered:
        return _matched_terms(
            text,
            positive=("deal", "agreement", "progress", "breakthrough", "ceasefire", "talks resume"),
            negative=("stalemate", "far from", "no deal", "impasse", "collapse", "reject", "sanctions"),
        )
    if "strait of hormuz" in lowered and "normal" in lowered:
        return _matched_terms(
            text,
            positive=("returns to normal", "reopen", "traffic resumes", "shipping resumes", "open"),
            negative=("closure", "closed", "disrupted", "reroute", "around the strait", "tanker risk"),
        )
    if "israel" in lowered and "airspace" in lowered:
        return _matched_terms(
            text,
            positive=("closes airspace", "close airspace", "airspace closed", "suspend flights", "missile fire"),
            negative=("remaining open", "remain open", "flights continuing", "open despite"),
        )
    if "keiko fujimori" in lowered:
        return _matched_terms(
            text,
            positive=("keiko", "fujimori", "edge", "lead", "win"),
            negative=("statistical tie", "sanchez", "behind", "losing"),
        )
    if "roberto" in lowered and "peruvian" in lowered:
        return _matched_terms(
            text,
            positive=("roberto", "sanchez", "lead", "win"),
            negative=("noise", "bitcoin", "fed rate", "iphone"),
        )
    return ()


def _matched_terms(text: str, *, positive: tuple[str, ...], negative: tuple[str, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for term in positive:
        if term in text:
            found.append(f"+{term}")
    for term in negative:
        if term in text:
            found.append(f"-{term}")
    return tuple(found)


def _estimated_yes_probability(question: str, evidence: tuple[str, ...]) -> float:
    lowered = question.lower()
    base = _base_probability(question)
    pos = sum(1 for item in evidence if item.startswith("+"))
    neg = sum(1 for item in evidence if item.startswith("-"))
    tilt = (pos - neg) * 0.08
    if "by june 15" in lowered:
        tilt -= 0.04
    if "by july 31" in lowered:
        tilt += 0.06
    return min(max(base + tilt, 0.02), 0.98)


def _base_probability(question: str) -> float:
    lowered = question.lower()
    if "peace deal" in lowered and "by june 15" in lowered:
        return 0.12
    if "peace deal" in lowered and "by june 30" in lowered:
        return 0.18
    if "peace deal" in lowered and "by july 31" in lowered:
        return 0.28
    if "strait of hormuz" in lowered:
        return 0.20
    if "israel" in lowered and "airspace" in lowered:
        return 0.45
    if "keiko fujimori" in lowered:
        return 0.50
    if "roberto" in lowered and "peruvian" in lowered:
        return 0.50
    return 0.50


def _confidence_score(news: dict[str, str], evidence: tuple[str, ...]) -> float:
    article_score = min(_float(news.get("article_count_24h")), 20.0) * 0.8
    source_score = min(_float(news.get("source_count_72h")), 10.0) * 1.2
    evidence_score = min(len(evidence), 6) * 2.0
    recency_score = 6.0 if _float(news.get("newest_age_hours")) <= 6.0 else 2.0
    if not evidence:
        return min(article_score + source_score + recency_score, 12.0)
    return article_score + source_score + evidence_score + recency_score


def _status(*, probability_gap: float, confidence_score: float) -> str:
    if abs(probability_gap) >= 0.15 and confidence_score >= 20.0:
        return "paper_probability_gap_candidate"
    if abs(probability_gap) >= 0.08 and confidence_score >= 12.0:
        return "probability_gap_watch"
    return "no_clear_probability_gap"


def _reason(*, status: str, probability_gap: float) -> str:
    if status == "paper_probability_gap_candidate":
        return "headline-derived estimate differs materially from market-implied Yes odds"
    if status == "probability_gap_watch":
        return "headline-derived estimate differs from market odds, but needs stronger evidence"
    return f"rough probability gap is too small or too weak for action: gap={probability_gap:.4f}"


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
        "--tickets-path",
        type=Path,
        default=ROOT / "current_prediction_market_paper_tickets.csv",
    )
    parser.add_argument(
        "--news-pressure-path",
        type=Path,
        default=ROOT / "current_event_news_pressure.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_probability_gap.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_event_probability_gap.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = build_event_probability_gap_rows(
        tickets_path=args.tickets_path,
        news_pressure_path=args.news_pressure_path,
    )
    write_event_probability_gap_csv(rows, output_path=args.output_path)
    write_event_probability_gap_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.status,
            row.suggested_side,
            f"gap={row.probability_gap:.3f}",
            f"score={row.score:.2f}",
            row.question,
        )


if __name__ == "__main__":
    main()
