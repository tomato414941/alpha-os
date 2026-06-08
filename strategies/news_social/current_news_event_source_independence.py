from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent

STOPWORDS = {
    "a",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "vs",
    "with",
}


@dataclass(frozen=True)
class NewsEventSourceIndependence:
    symbol: str
    event_kind: str
    side: str
    independence_status: str
    score: float
    source_count: int
    label_count: int
    supported_count: int
    rejected_count: int
    pending_count: int
    unique_story_count: int
    dominant_story_share: float
    dominant_story_terms: str
    mean_directional_1h_bps: str
    mean_directional_4h_bps: str
    sources: str
    strongest_title: str
    reason: str
    next_step: str


def build_news_event_source_independence(
    *,
    labels_path: Path = ROOT / "current_news_event_forward_labels.csv",
) -> tuple[NewsEventSourceIndependence, ...]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in _read_rows(labels_path):
        side = row.get("side", "")
        if side == "context_label":
            continue
        key = (row.get("symbol", ""), row.get("event_kind", ""), side)
        if not all(key):
            continue
        groups.setdefault(key, []).append(row)
    rows = tuple(_build_row(key=key, rows=group_rows) for key, group_rows in groups.items())
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_news_event_source_independence_csv(
    rows: tuple[NewsEventSourceIndependence, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "event_kind",
                "side",
                "independence_status",
                "score",
                "source_count",
                "label_count",
                "supported_count",
                "rejected_count",
                "pending_count",
                "unique_story_count",
                "dominant_story_share",
                "dominant_story_terms",
                "mean_directional_1h_bps",
                "mean_directional_4h_bps",
                "sources",
                "strongest_title",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.event_kind,
                    row.side,
                    row.independence_status,
                    f"{row.score:.8f}",
                    row.source_count,
                    row.label_count,
                    row.supported_count,
                    row.rejected_count,
                    row.pending_count,
                    row.unique_story_count,
                    f"{row.dominant_story_share:.8f}",
                    row.dominant_story_terms,
                    row.mean_directional_1h_bps,
                    row.mean_directional_4h_bps,
                    row.sources,
                    row.strongest_title,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_news_event_source_independence_md(
    rows: tuple[NewsEventSourceIndependence, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current News Event Source Independence\n\n")
        handle.write(
            "This checks whether multi-source news labels are actually independent stories. "
            "Multiple outlets repeating the same story is treated as weaker evidence than unrelated sources "
            "confirming the same direction. It is a control gate, not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | kind | side | status | score | sources | labels | stories | dominant | mean 1h | mean 4h | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | "
                f"{row.event_kind} | "
                f"{row.side} | "
                f"{row.independence_status} | "
                f"{row.score:.4f} | "
                f"{row.source_count} | "
                f"{row.label_count} | "
                f"{row.unique_story_count} | "
                f"{_escape(row.dominant_story_terms)} ({row.dominant_story_share:.2f}) | "
                f"{row.mean_directional_1h_bps} | "
                f"{row.mean_directional_4h_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary_text(rows))
    return output_path


def _build_row(*, key: tuple[str, str, str], rows: list[dict[str, str]]) -> NewsEventSourceIndependence:
    symbol, event_kind, side = key
    supported = [row for row in rows if row.get("label_status") in {"direction_supported_1h_4h", "direction_supported_1h_only"}]
    rejected = [row for row in rows if row.get("label_status") == "direction_rejected_1h"]
    pending = [row for row in rows if row.get("label_status") == "pending_forward_archive"]
    sources = sorted({row.get("source", "") for row in rows if row.get("source")})
    story_counts = _story_counts(rows)
    dominant_story, dominant_count = max(story_counts.items(), key=lambda item: item[1], default=("", 0))
    dominant_share = dominant_count / len(rows) if rows else 0.0
    directional_1h = [_float(row.get("directional_1h_bps")) for row in supported if row.get("directional_1h_bps")]
    directional_4h = [_float(row.get("directional_4h_bps")) for row in supported if row.get("directional_4h_bps")]
    mean_1h = mean(directional_1h) if directional_1h else 0.0
    mean_4h = mean(directional_4h) if directional_4h else 0.0
    strongest = max(rows, key=lambda row: _float(row.get("directional_4h_bps") or row.get("directional_1h_bps")), default={})
    status, reason = _status_reason(
        source_count=len(sources),
        supported_count=len(supported),
        rejected_count=len(rejected),
        pending_count=len(pending),
        unique_story_count=len(story_counts),
        dominant_story_share=dominant_share,
        mean_1h=mean_1h,
        mean_4h=mean_4h,
    )
    return NewsEventSourceIndependence(
        symbol=symbol,
        event_kind=event_kind,
        side=side,
        independence_status=status,
        score=_score(
            source_count=len(sources),
            supported_count=len(supported),
            rejected_count=len(rejected),
            pending_count=len(pending),
            unique_story_count=len(story_counts),
            dominant_story_share=dominant_share,
            mean_1h=mean_1h,
            mean_4h=mean_4h,
        ),
        source_count=len(sources),
        label_count=len(rows),
        supported_count=len(supported),
        rejected_count=len(rejected),
        pending_count=len(pending),
        unique_story_count=len(story_counts),
        dominant_story_share=dominant_share,
        dominant_story_terms=dominant_story,
        mean_directional_1h_bps=f"{mean_1h:.8f}",
        mean_directional_4h_bps=f"{mean_4h:.8f}",
        sources=", ".join(sources),
        strongest_title=strongest.get("title", ""),
        reason=reason,
        next_step=_next_step(symbol=symbol, status=status),
    )


def _story_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        story = _story_signature(row.get("title", ""))
        counts[story] = counts.get(story, 0) + 1
    return counts


def _story_signature(title: str) -> str:
    tokens = [token for token in re.findall(r"[a-z0-9]+", title.lower()) if token not in STOPWORDS]
    if any(token in {"strategy", "saylor", "mstr", "microstrategy"} for token in tokens):
        return "strategy_saylor_btc_treasury"
    if any(token in {"zcash", "zec", "zcash"} for token in tokens) and any(
        token in {"exploit", "bug", "vulnerabilities", "vulnerability"} for token in tokens
    ):
        return "zec_security_vulnerability"
    if any(token in {"zcash", "zec", "winklevoss"} for token in tokens):
        return "zec_treasury_or_privacy"
    if any(token in {"tether", "usdt", "stablecoin"} for token in tokens):
        return "stablecoin_tether_context"
    if len(tokens) <= 3:
        return "_".join(tokens)
    return "_".join(tokens[:4])


def _status_reason(
    *,
    source_count: int,
    supported_count: int,
    rejected_count: int,
    pending_count: int,
    unique_story_count: int,
    dominant_story_share: float,
    mean_1h: float,
    mean_4h: float,
) -> tuple[str, str]:
    if supported_count == 0:
        return "reject_no_supported_label", "no supported timestamp label"
    if source_count < 2:
        return "single_source_supported_story", "supported labels still come from one source"
    if rejected_count > supported_count:
        return "mixed_or_rejected_story", "rejected labels outnumber supported labels"
    if mean_1h <= 0.0 or mean_4h <= 0.0:
        return "weak_forward_story", "mean 1h/4h directional label is not both positive"
    if pending_count:
        return "pending_archive_before_independence", "fresh labels are still pending archive"
    if unique_story_count < 2 or dominant_story_share >= 0.75:
        return "same_story_multi_source_repeat", "multiple sources mostly repeat the same story"
    return "independent_multi_source_story", "multiple sources support direction across distinct stories"


def _score(
    *,
    source_count: int,
    supported_count: int,
    rejected_count: int,
    pending_count: int,
    unique_story_count: int,
    dominant_story_share: float,
    mean_1h: float,
    mean_4h: float,
) -> float:
    return (
        min(source_count, 4) * 8.0
        + supported_count * 7.0
        + min(unique_story_count, 4) * 6.0
        - rejected_count * 7.0
        - pending_count * 4.0
        - dominant_story_share * 12.0
        + min(max(mean_1h, -100.0), 200.0) * 0.04
        + min(max(mean_4h, -200.0), 400.0) * 0.025
    )


def _next_step(*, symbol: str, status: str) -> str:
    if status == "independent_multi_source_story":
        return f"repeat {symbol} labels with execution costs, beta controls, and story-level timestamp isolation"
    if status == "same_story_multi_source_repeat":
        return f"treat {symbol} as same-story repetition; require a distinct event source before promotion"
    if status == "pending_archive_before_independence":
        return f"wait for fresh {symbol} archive labels before judging source independence"
    if status == "single_source_supported_story":
        return f"seek another independent {symbol} source before treating the story as alpha"
    return f"keep {symbol} news as context until source independence and forward labels improve"


def _summary_text(rows: tuple[NewsEventSourceIndependence, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.independence_status] = counts.get(row.independence_status, 0) + 1
    lines = [f"- {status}: {count}" for status, count in sorted(counts.items())]
    best = max(rows, key=lambda row: row.score, default=None)
    if best:
        lines.append(
            "- best source-independent candidate: "
            f"{best.symbol}/{best.event_kind}/{best.side} status={best.independence_status} "
            f"sources={best.source_count} stories={best.unique_story_count} score={best.score:.8f}"
        )
    if not lines:
        lines.append("- no news-event source independence rows yet")
    return "\n".join(lines) + "\n"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=Path, default=ROOT / "current_news_event_forward_labels.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_news_event_source_independence.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_news_event_source_independence.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_news_event_source_independence(labels_path=args.labels_path)
    write_news_event_source_independence_csv(rows, output_path=args.output_path)
    write_news_event_source_independence_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.event_kind, row.independence_status, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
