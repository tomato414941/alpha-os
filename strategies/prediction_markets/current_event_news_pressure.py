from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests


ROOT = Path(__file__).resolve().parent
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


@dataclass(frozen=True)
class EventNewsPressureRow:
    market_id: str
    question: str
    category: str
    query: str
    midpoint: float
    market_score: float
    volume_24h: float
    article_count_24h: int
    article_count_72h: int
    source_count_72h: int
    newest_age_hours: float
    top_sources: str
    top_titles: str
    score: float
    status: str
    reason: str


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published_at: datetime | None


def build_event_news_pressure_rows(
    *,
    tickets_path: Path,
    now: datetime | None = None,
    top_markets: int = 8,
    max_records: int = 30,
) -> tuple[EventNewsPressureRow, ...]:
    now = now or datetime.now(UTC)
    rows: list[EventNewsPressureRow] = []
    for ticket in _unique_event_tickets(tickets_path, top_markets=top_markets):
        question = ticket["question"]
        query = _query_for_question(question)
        items = _fetch_google_news_items(query, max_records=max_records)
        rows.append(_build_row(ticket=ticket, query=query, items=items, now=now))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_event_news_pressure_csv(rows: tuple[EventNewsPressureRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "category",
                "query",
                "midpoint",
                "market_score",
                "volume_24h",
                "article_count_24h",
                "article_count_72h",
                "source_count_72h",
                "newest_age_hours",
                "top_sources",
                "top_titles",
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
                    row.category,
                    row.query,
                    f"{row.midpoint:.6f}",
                    f"{row.market_score:.8f}",
                    f"{row.volume_24h:.6f}",
                    row.article_count_24h,
                    row.article_count_72h,
                    row.source_count_72h,
                    f"{row.newest_age_hours:.6f}",
                    row.top_sources,
                    row.top_titles,
                    f"{row.score:.8f}",
                    row.status,
                    row.reason,
                )
            )
    return output_path


def write_event_news_pressure_md(
    rows: tuple[EventNewsPressureRow, ...],
    *,
    output_path: Path,
    top: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event News Pressure\n\n")
        handle.write(
            "This joins depth-positive prediction-market candidates to Google News RSS activity. "
            "It is not a probability estimate or trade instruction.\n\n"
        )
        handle.write(
            "| question | category | midpoint | market score | articles 24h | articles 72h | sources | newest h | score | status | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_escape(row.question)} | {row.category} | {row.midpoint:.4f} | "
                f"{row.market_score:.4f} | {row.article_count_24h} | {row.article_count_72h} | "
                f"{row.source_count_72h} | {row.newest_age_hours:.2f} | {row.score:.4f} | "
                f"{row.status} | {_escape(row.reason)} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "News volume is only an external information-flow proxy. It can be stale, duplicated, "
            "misleading, or already priced. A real event-market edge still needs a probability model "
            "that compares independent evidence against market-implied odds.\n"
        )
    return output_path


def _unique_event_tickets(path: Path, *, top_markets: int) -> tuple[dict[str, str], ...]:
    rows = tuple(
        row
        for row in _read_rows(path)
        if row.get("status") in {"paper_event_model_candidate", "paper_event_model_watch"}
        and row.get("category") != "sports_event"
    )
    selected: list[dict[str, str]] = []
    seen_market_ids: set[str] = set()
    for row in sorted(rows, key=lambda item: _float(item.get("score")), reverse=True):
        market_id = row.get("market_id", "")
        if not market_id or market_id in seen_market_ids:
            continue
        seen_market_ids.add(market_id)
        selected.append(row)
        if len(selected) >= top_markets:
            break
    return tuple(selected)


def _build_row(
    *,
    ticket: dict[str, str],
    query: str,
    items: tuple[NewsItem, ...],
    now: datetime,
) -> EventNewsPressureRow:
    article_count_72h = len(items)
    article_count_24h = sum(1 for item in items if _age_hours(item.published_at, now) <= 24.0)
    sources = tuple(sorted({item.source for item in items if item.source}))
    newest_age_hours = min((_age_hours(item.published_at, now) for item in items), default=999.0)
    top_titles = " || ".join(item.title for item in items[:3])
    top_sources = ", ".join(sources[:5])
    score = _score(
        article_count_24h=article_count_24h,
        article_count_72h=article_count_72h,
        source_count_72h=len(sources),
        newest_age_hours=newest_age_hours,
        market_score=_float(ticket.get("score")),
    )
    status, reason = _status_reason(
        article_count_24h=article_count_24h,
        article_count_72h=article_count_72h,
        source_count_72h=len(sources),
        newest_age_hours=newest_age_hours,
    )
    return EventNewsPressureRow(
        market_id=ticket.get("market_id", ""),
        question=ticket.get("question", ""),
        category=ticket.get("category", ""),
        query=query,
        midpoint=_float(ticket.get("midpoint")),
        market_score=_float(ticket.get("score")),
        volume_24h=_float(ticket.get("volume_24h")),
        article_count_24h=article_count_24h,
        article_count_72h=article_count_72h,
        source_count_72h=len(sources),
        newest_age_hours=newest_age_hours,
        top_sources=top_sources,
        top_titles=top_titles,
        score=score,
        status=status,
        reason=reason,
    )


def _fetch_google_news_items(query: str, *, max_records: int) -> tuple[NewsItem, ...]:
    params = {
        "q": f"{query} when:3d",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    try:
        response = requests.get(f"{GOOGLE_NEWS_RSS_URL}?{urlencode(params)}", timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return ()
    return _parse_rss_items(response.text)[:max_records]


def _parse_rss_items(xml_text: str) -> tuple[NewsItem, ...]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ()
    items: list[NewsItem] = []
    for item in root.findall("./channel/item"):
        title = _node_text(item, "title")
        source = _node_text(item, "source")
        published_at = _parse_datetime(_node_text(item, "pubDate"))
        if title:
            items.append(NewsItem(title=title, source=source, published_at=published_at))
    return tuple(items)


def _query_for_question(question: str) -> str:
    lowered = question.lower()
    if "peace deal" in lowered and "iran" in lowered:
        return '("US Iran" OR "Iran US") ("peace deal" OR negotiations OR talks)'
    if "strait of hormuz" in lowered:
        return '"Strait of Hormuz" (traffic OR shipping OR tanker OR oil)'
    if "israel" in lowered and "airspace" in lowered:
        return '"Israel" "airspace"'
    if "keiko fujimori" in lowered:
        return '"Keiko Fujimori" "Peruvian presidential election"'
    if "roberto" in lowered and "peruvian presidential" in lowered:
        return '"Roberto Sanchez Palomino" "Peruvian presidential election"'
    return " ".join(_query_terms(question))


def _query_terms(question: str) -> tuple[str, ...]:
    stop = {
        "by",
        "end",
        "of",
        "on",
        "the",
        "will",
        "win",
        "yes",
        "no",
        "june",
        "july",
        "2026",
    }
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in question.lower())
    return tuple(word for word in cleaned.split() if len(word) > 2 and word not in stop)[:8]


def _score(
    *,
    article_count_24h: int,
    article_count_72h: int,
    source_count_72h: int,
    newest_age_hours: float,
    market_score: float,
) -> float:
    recency_bonus = 15.0 if newest_age_hours <= 6.0 else 7.0 if newest_age_hours <= 24.0 else 0.0
    return (
        min(article_count_24h, 20) * 2.0
        + min(article_count_72h, 30)
        + min(source_count_72h, 10) * 4.0
        + recency_bonus
        + min(market_score, 100.0) / 5.0
    )


def _status_reason(
    *,
    article_count_24h: int,
    article_count_72h: int,
    source_count_72h: int,
    newest_age_hours: float,
) -> tuple[str, str]:
    if article_count_24h >= 3 and source_count_72h >= 2 and newest_age_hours <= 24.0:
        return "external_news_active", "recent multi-source news flow exists for this event market"
    if article_count_72h >= 2:
        return "external_news_watch", "some external news flow exists, but it is not yet strong enough"
    return "external_news_thin", "little current external news flow was found for this event market"


def _age_hours(value: datetime | None, now: datetime) -> float:
    if value is None:
        return 999.0
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return max((now - value.astimezone(UTC)).total_seconds() / 3600.0, 0.0)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _node_text(parent: ET.Element, tag: str) -> str:
    node = parent.find(tag)
    return "" if node is None or node.text is None else node.text.strip()


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
    parser.add_argument("--top-markets", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_news_pressure.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_event_news_pressure.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = build_event_news_pressure_rows(
        tickets_path=args.tickets_path,
        top_markets=args.top_markets,
        max_records=args.max_records,
    )
    write_event_news_pressure_csv(rows, output_path=args.output_path)
    write_event_news_pressure_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.status,
            f"score={row.score:.2f}",
            f"articles24={row.article_count_24h}",
            f"sources={row.source_count_72h}",
            row.question,
        )


if __name__ == "__main__":
    main()
