from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventObservation:
    symbol: str
    source: str
    side: str
    direction_hint: int
    score: float
    age_hours: float
    summary: str


@dataclass(frozen=True)
class EventPressureClusterRow:
    symbol: str
    status: str
    side: str
    score: float
    source_count: int
    event_count: int
    newest_age_hours: float
    top_sources: str
    top_events: str
    reason: str
    next_step: str


def build_event_pressure_cluster_rows(root: Path = ROOT) -> tuple[EventPressureClusterRow, ...]:
    observations = [
        *_news_event_observations(root / "current_news_event_screen.csv"),
        *_exchange_catalyst_observations(root / "current_exchange_catalyst_market_join.csv"),
        *_attention_market_observations(root / "current_attention_market_join.csv"),
        *_attention_price_observations(root / "current_attention_price_context.csv"),
    ]
    grouped: dict[str, list[EventObservation]] = {}
    for observation in observations:
        if observation.symbol:
            grouped.setdefault(observation.symbol, []).append(observation)
    rows = tuple(_build_cluster(symbol=symbol, observations=items) for symbol, items in grouped.items())
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_event_pressure_cluster_csv(rows: tuple[EventPressureClusterRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "side",
                "score",
                "source_count",
                "event_count",
                "newest_age_hours",
                "top_sources",
                "top_events",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    row.source_count,
                    row.event_count,
                    f"{row.newest_age_hours:.6f}",
                    row.top_sources,
                    row.top_events,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_event_pressure_cluster_md(
    rows: tuple[EventPressureClusterRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Pressure Cluster\n\n")
        handle.write(
            "This groups current news, exchange catalysts, and attention observations by symbol. "
            "It is a prioritization view, not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | status | side | score | sources | events | newest age h | top sources | top events |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.side} | {row.score:.4f} | "
                f"{row.source_count} | {row.event_count} | {row.newest_age_hours:.2f} | "
                f"{_escape(row.top_sources)} | {_escape(row.top_events)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A cluster is useful when independent event sources overlap on the same symbol. "
            "It still needs timestamp checks, duplicate-source controls, forward labels, venue depth, "
            "funding, and execution-cost checks.\n"
        )
    return output_path


def _news_event_observations(path: Path) -> tuple[EventObservation, ...]:
    output: list[EventObservation] = []
    for row in _read_rows(path):
        output.append(
            EventObservation(
                symbol=row.get("symbol", "").upper(),
                source=f"rss:{row.get('source', '')}",
                side=row.get("side", ""),
                direction_hint=_int(row.get("direction_hint")),
                score=_float(row.get("score")),
                age_hours=_float(row.get("age_hours")),
                summary=f"{row.get('event_kind', '')}: {row.get('title', '')}",
            )
        )
    return tuple(output)


def _exchange_catalyst_observations(path: Path) -> tuple[EventObservation, ...]:
    output: list[EventObservation] = []
    now = datetime.now(UTC)
    for row in _read_rows(path):
        output.append(
            EventObservation(
                symbol=row.get("symbol", "").upper(),
                source="exchange_catalyst",
                side=row.get("action", ""),
                direction_hint=_int(row.get("direction_hint")),
                score=_float(row.get("score")),
                age_hours=_age_hours(row.get("published_at", ""), now=now),
                summary=f"{row.get('catalyst_kind', '')}: {row.get('title', '')}",
            )
        )
    return tuple(output)


def _attention_market_observations(path: Path) -> tuple[EventObservation, ...]:
    output: list[EventObservation] = []
    for row in _read_rows(path):
        action = row.get("action", "")
        direction_hint = 1 if "funding" in action or "carry" in action else 0
        output.append(
            EventObservation(
                symbol=row.get("symbol", "").upper(),
                source="attention_market",
                side=action,
                direction_hint=direction_hint,
                score=_float(row.get("score")),
                age_hours=0.0,
                summary=(
                    f"{action}: rank={row.get('attention_rank', '')}, "
                    f"change={row.get('attention_24h_change', '')}"
                ),
            )
        )
    return tuple(output)


def _attention_price_observations(path: Path) -> tuple[EventObservation, ...]:
    output: list[EventObservation] = []
    for row in _read_rows(path):
        side = row.get("side", "")
        direction_hint = -1 if "fade" in side or "risk" in side else 1 if "long" in side else 0
        output.append(
            EventObservation(
                symbol=row.get("symbol", "").upper(),
                source="attention_price",
                side=side,
                direction_hint=direction_hint,
                score=_float(row.get("score")),
                age_hours=0.0,
                summary=f"{row.get('status', '')}: {row.get('evidence', '')}",
            )
        )
    return tuple(output)


def _build_cluster(*, symbol: str, observations: list[EventObservation]) -> EventPressureClusterRow:
    sorted_observations = sorted(observations, key=lambda item: item.score, reverse=True)
    sources = sorted({item.source for item in observations if item.source})
    positives = sum(1 for item in observations if item.direction_hint > 0)
    negatives = sum(1 for item in observations if item.direction_hint < 0)
    side = _cluster_side(positives=positives, negatives=negatives)
    status = _cluster_status(source_count=len(sources), event_count=len(observations), side=side)
    newest_age = min((item.age_hours for item in observations if item.age_hours >= 0.0), default=0.0)
    score = _cluster_score(
        max_score=sorted_observations[0].score if sorted_observations else 0.0,
        source_count=len(sources),
        event_count=len(observations),
        newest_age_hours=newest_age,
    )
    return EventPressureClusterRow(
        symbol=symbol,
        status=status,
        side=side,
        score=score,
        source_count=len(sources),
        event_count=len(observations),
        newest_age_hours=newest_age,
        top_sources=", ".join(sources[:6]),
        top_events=" || ".join(item.summary for item in sorted_observations[:4]),
        reason=_reason(status=status, side=side),
        next_step=(
            f"label {symbol} event-pressure cluster over 15m/1h/4h, then check duplicate sources, "
            "funding, depth, and execution cost"
        ),
    )


def _cluster_side(*, positives: int, negatives: int) -> str:
    if positives and negatives:
        return "mixed_event_pressure"
    if positives:
        return "long_event_pressure"
    if negatives:
        return "short_or_avoid_event_pressure"
    return "event_context"


def _cluster_status(*, source_count: int, event_count: int, side: str) -> str:
    if source_count >= 3:
        return "multi_source_event_pressure"
    if source_count >= 2:
        return "two_source_event_pressure"
    if event_count >= 2:
        return "repeated_event_context"
    return "single_event_context"


def _cluster_score(*, max_score: float, source_count: int, event_count: int, newest_age_hours: float) -> float:
    return max_score + min(source_count * 8.0, 24.0) + min(event_count * 1.5, 12.0) + max(24.0 - newest_age_hours, 0.0) / 2.0


def _reason(*, status: str, side: str) -> str:
    if status == "multi_source_event_pressure":
        return f"multiple independent event sources overlap with {side}"
    if status == "two_source_event_pressure":
        return f"two event sources overlap with {side}"
    if status == "repeated_event_context":
        return "one source reports repeated event context for the same symbol"
    return "single event observation only"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _age_hours(value: str, *, now: datetime) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max((now - parsed).total_seconds() / 3600.0, 0.0)


def _float(value: object) -> float:
    try:
        return float(value) if value not in {"", None} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(value) if value not in {"", None} else 0
    except (TypeError, ValueError):
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_pressure_cluster.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_event_pressure_cluster.md")
    args = parser.parse_args()
    rows = build_event_pressure_cluster_rows()
    write_event_pressure_cluster_csv(rows, output_path=args.output_path)
    write_event_pressure_cluster_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.symbol, row.status, row.side, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
