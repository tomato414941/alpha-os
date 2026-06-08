from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventSourceQualityRow:
    market_id: str
    question: str
    suggested_side: str
    source_count_72h: int
    article_count_24h: int
    newest_age_hours: float
    top_title_count: int
    unique_top_title_count: int
    relevance_score: float
    quality_score: float
    status: str
    reason: str


def build_event_source_quality_rows(
    *,
    paper_tickets_path: Path,
    news_pressure_path: Path,
) -> tuple[EventSourceQualityRow, ...]:
    news_by_market = {row.get("market_id", ""): row for row in _read_rows(news_pressure_path)}
    rows: list[EventSourceQualityRow] = []
    for ticket in _read_rows(paper_tickets_path):
        if ticket.get("status") not in {"paper_event_probability_ticket", "event_probability_watch"}:
            continue
        news = news_by_market.get(ticket.get("market_id", ""))
        if not news:
            rows.append(_missing_news_row(ticket))
            continue
        rows.append(_build_row(ticket=ticket, news=news))
    return tuple(sorted(rows, key=lambda row: row.quality_score, reverse=True))


def write_event_source_quality_csv(rows: tuple[EventSourceQualityRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "suggested_side",
                "source_count_72h",
                "article_count_24h",
                "newest_age_hours",
                "top_title_count",
                "unique_top_title_count",
                "relevance_score",
                "quality_score",
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
                    row.source_count_72h,
                    row.article_count_24h,
                    f"{row.newest_age_hours:.6f}",
                    row.top_title_count,
                    row.unique_top_title_count,
                    f"{row.relevance_score:.6f}",
                    f"{row.quality_score:.8f}",
                    row.status,
                    row.reason,
                )
            )
    return output_path


def write_event_source_quality_md(
    rows: tuple[EventSourceQualityRow, ...],
    *,
    output_path: Path,
    top: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Source Quality\n\n")
        handle.write(
            "This checks whether event-probability paper tickets have enough fresh, source-diverse, "
            "non-duplicated external news context. It is not a probability model or trade instruction.\n\n"
        )
        handle.write(
            "| question | side | sources 72h | articles 24h | newest h | unique titles | relevance | score | status | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.suggested_side} | {row.source_count_72h} | "
                f"{row.article_count_24h} | {row.newest_age_hours:.2f} | "
                f"{row.unique_top_title_count}/{row.top_title_count} | {row.relevance_score:.2f} | "
                f"{row.quality_score:.4f} | {row.status} | {_escape(row.reason)} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "Passing this gate only means the external news feed is less obviously noisy. "
            "It does not validate truth, timing, calibration, fill quality, or adverse selection.\n"
        )
    return output_path


def _build_row(*, ticket: dict[str, str], news: dict[str, str]) -> EventSourceQualityRow:
    titles = _titles(news.get("top_titles", ""))
    unique_titles = {_normalize_title(title) for title in titles}
    relevance_score = _relevance_score(ticket.get("question", ""), titles)
    source_count = int(_float(news.get("source_count_72h")))
    article_count = int(_float(news.get("article_count_24h")))
    newest_age = _float(news.get("newest_age_hours"))
    quality_score = _quality_score(
        source_count_72h=source_count,
        article_count_24h=article_count,
        newest_age_hours=newest_age,
        top_title_count=len(titles),
        unique_top_title_count=len(unique_titles),
        relevance_score=relevance_score,
    )
    status, reason = _status_reason(
        source_count_72h=source_count,
        article_count_24h=article_count,
        newest_age_hours=newest_age,
        top_title_count=len(titles),
        unique_top_title_count=len(unique_titles),
        relevance_score=relevance_score,
    )
    return EventSourceQualityRow(
        market_id=ticket.get("market_id", ""),
        question=ticket.get("question", ""),
        suggested_side=ticket.get("suggested_side", ""),
        source_count_72h=source_count,
        article_count_24h=article_count,
        newest_age_hours=newest_age,
        top_title_count=len(titles),
        unique_top_title_count=len(unique_titles),
        relevance_score=relevance_score,
        quality_score=quality_score,
        status=status,
        reason=reason,
    )


def _missing_news_row(ticket: dict[str, str]) -> EventSourceQualityRow:
    return EventSourceQualityRow(
        market_id=ticket.get("market_id", ""),
        question=ticket.get("question", ""),
        suggested_side=ticket.get("suggested_side", ""),
        source_count_72h=0,
        article_count_24h=0,
        newest_age_hours=999.0,
        top_title_count=0,
        unique_top_title_count=0,
        relevance_score=0.0,
        quality_score=0.0,
        status="source_quality_fail",
        reason="missing external news pressure row",
    )


def _quality_score(
    *,
    source_count_72h: int,
    article_count_24h: int,
    newest_age_hours: float,
    top_title_count: int,
    unique_top_title_count: int,
    relevance_score: float,
) -> float:
    source_score = min(source_count_72h, 12) * 2.0
    article_score = min(article_count_24h, 20) * 0.8
    freshness_score = 12.0 if newest_age_hours <= 6.0 else 6.0 if newest_age_hours <= 24.0 else 0.0
    uniqueness_score = 0.0
    if top_title_count > 0:
        uniqueness_score = (unique_top_title_count / top_title_count) * 10.0
    return source_score + article_score + freshness_score + uniqueness_score + relevance_score


def _status_reason(
    *,
    source_count_72h: int,
    article_count_24h: int,
    newest_age_hours: float,
    top_title_count: int,
    unique_top_title_count: int,
    relevance_score: float,
) -> tuple[str, str]:
    if source_count_72h < 3:
        return "source_quality_fail", "too few independent sources"
    if article_count_24h < 2:
        return "source_quality_fail", "too little recent article flow"
    if newest_age_hours > 24.0:
        return "source_quality_fail", "newest article is stale"
    if top_title_count and unique_top_title_count <= 1:
        return "source_quality_watch", "top headlines appear duplicated"
    if relevance_score < 5.0:
        return "source_quality_watch", "top headlines have weak question relevance"
    return "source_quality_pass", "fresh multi-source news context is present and not obviously duplicated"


def _relevance_score(question: str, titles: tuple[str, ...]) -> float:
    lowered_question = question.lower()
    text = " ".join(titles).lower()
    terms = _question_terms(lowered_question)
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if term in text)
    return min((matched / len(terms)) * 20.0, 20.0)


def _question_terms(lowered_question: str) -> tuple[str, ...]:
    if "israel" in lowered_question and "airspace" in lowered_question:
        return ("israel", "airspace", "flights")
    if "keiko fujimori" in lowered_question:
        return ("keiko", "fujimori", "peru")
    if "peace deal" in lowered_question and "iran" in lowered_question:
        return ("iran", "peace", "deal")
    if "strait of hormuz" in lowered_question:
        return ("hormuz", "traffic", "shipping")
    return tuple(word for word in lowered_question.split() if len(word) > 4)[:5]


def _titles(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(" || ") if part.strip())


def _normalize_title(value: str) -> str:
    lowered = value.lower()
    return "".join(char for char in lowered if char.isalnum() or char.isspace()).strip()


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
        "--news-pressure-path",
        type=Path,
        default=ROOT / "current_event_news_pressure.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_source_quality.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_event_source_quality.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = build_event_source_quality_rows(
        paper_tickets_path=args.paper_tickets_path,
        news_pressure_path=args.news_pressure_path,
    )
    write_event_source_quality_csv(rows, output_path=args.output_path)
    write_event_source_quality_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.status,
            f"score={row.quality_score:.2f}",
            f"sources={row.source_count_72h}",
            f"recent={row.article_count_24h}",
            row.question,
        )


if __name__ == "__main__":
    main()
