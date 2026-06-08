from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class NewsEventSurvival:
    candidate_id: str
    symbol: str
    event_kind: str
    side: str
    survival_status: str
    survival_score: float
    independence_status: str
    source_count: int
    label_count: int
    supported_count: int
    rejected_count: int
    pending_count: int
    unique_story_count: int
    dominant_story_share: float
    mean_directional_1h_bps: float
    mean_directional_4h_bps: float
    sources: str
    reason: str
    next_step: str


def build_news_event_survival_rows(
    *,
    source_independence_path: Path = ROOT / "current_news_event_source_independence.csv",
) -> tuple[NewsEventSurvival, ...]:
    rows = tuple(_build_row(row) for row in _read_rows(source_independence_path))
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_news_event_survival_csv(rows: tuple[NewsEventSurvival, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "symbol",
                "event_kind",
                "side",
                "survival_status",
                "survival_score",
                "independence_status",
                "source_count",
                "label_count",
                "supported_count",
                "rejected_count",
                "pending_count",
                "unique_story_count",
                "dominant_story_share",
                "mean_directional_1h_bps",
                "mean_directional_4h_bps",
                "sources",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.symbol,
                    row.event_kind,
                    row.side,
                    row.survival_status,
                    f"{row.survival_score:.8f}",
                    row.independence_status,
                    row.source_count,
                    row.label_count,
                    row.supported_count,
                    row.rejected_count,
                    row.pending_count,
                    row.unique_story_count,
                    f"{row.dominant_story_share:.8f}",
                    f"{row.mean_directional_1h_bps:.8f}",
                    f"{row.mean_directional_4h_bps:.8f}",
                    row.sources,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_news_event_survival_md(rows: tuple[NewsEventSurvival, ...], *, output_path: Path, top: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current News Event Survival\n\n")
        handle.write(
            "This turns source independence into a survival gate. A supported forward label from one source "
            "is context, not alpha. Multi-source repetition of the same story is also weaker than independent "
            "event confirmation.\n\n"
        )
        handle.write(
            "| candidate | status | score | sources | labels | stories | dominant | mean 1h | mean 4h | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.survival_status} | "
                f"{row.survival_score:.4f} | "
                f"{row.source_count} | "
                f"{row.label_count} | "
                f"{row.unique_story_count} | "
                f"{row.dominant_story_share:.4f} | "
                f"{row.mean_directional_1h_bps:.4f} | "
                f"{row.mean_directional_4h_bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _build_row(row: dict[str, str]) -> NewsEventSurvival:
    status = _survival_status(row)
    mean_1h = _float(row.get("mean_directional_1h_bps"))
    mean_4h = _float(row.get("mean_directional_4h_bps"))
    source_count = _int(row.get("source_count"))
    label_count = _int(row.get("label_count"))
    supported_count = _int(row.get("supported_count"))
    rejected_count = _int(row.get("rejected_count"))
    pending_count = _int(row.get("pending_count"))
    unique_story_count = _int(row.get("unique_story_count"))
    dominant_story_share = _float(row.get("dominant_story_share"))
    symbol = row.get("symbol", "")
    event_kind = row.get("event_kind", "")
    side = row.get("side", "")
    return NewsEventSurvival(
        candidate_id=f"{symbol}_{event_kind}_{side}".lower(),
        symbol=symbol,
        event_kind=event_kind,
        side=side,
        survival_status=status,
        survival_score=_survival_score(
            status=status,
            source_count=source_count,
            label_count=label_count,
            supported_count=supported_count,
            rejected_count=rejected_count,
            pending_count=pending_count,
            unique_story_count=unique_story_count,
            dominant_story_share=dominant_story_share,
            mean_1h=mean_1h,
            mean_4h=mean_4h,
        ),
        independence_status=row.get("independence_status", ""),
        source_count=source_count,
        label_count=label_count,
        supported_count=supported_count,
        rejected_count=rejected_count,
        pending_count=pending_count,
        unique_story_count=unique_story_count,
        dominant_story_share=dominant_story_share,
        mean_directional_1h_bps=mean_1h,
        mean_directional_4h_bps=mean_4h,
        sources=row.get("sources", ""),
        reason=_reason(status),
        next_step=_next_step(symbol=symbol, status=status),
    )


def _survival_status(row: dict[str, str]) -> str:
    status = row.get("independence_status", "")
    if status == "independent_multi_source_story":
        return "news_event_survival_candidate"
    if status == "same_story_multi_source_repeat":
        return "news_event_duplicate_story_blocked"
    if status == "pending_archive_before_independence":
        return "news_event_pending_archive"
    if status == "single_source_supported_story":
        return "news_event_single_source_blocked"
    if status == "weak_forward_story":
        return "news_event_weak_forward_blocked"
    return "news_event_rejected"


def _survival_score(
    *,
    status: str,
    source_count: int,
    label_count: int,
    supported_count: int,
    rejected_count: int,
    pending_count: int,
    unique_story_count: int,
    dominant_story_share: float,
    mean_1h: float,
    mean_4h: float,
) -> float:
    base = {
        "news_event_survival_candidate": 120.0,
        "news_event_pending_archive": 45.0,
        "news_event_duplicate_story_blocked": 10.0,
        "news_event_single_source_blocked": 0.0,
        "news_event_weak_forward_blocked": -30.0,
        "news_event_rejected": -80.0,
    }.get(status, 0.0)
    return (
        base
        + min(source_count, 4) * 8.0
        + min(label_count, 12) * 1.5
        + supported_count * 6.0
        + min(unique_story_count, 4) * 8.0
        + min(max(mean_1h, -100.0), 200.0) * 0.03
        + min(max(mean_4h, -200.0), 400.0) * 0.02
        - rejected_count * 8.0
        - pending_count * 5.0
        - dominant_story_share * 25.0
    )


def _reason(status: str) -> str:
    if status == "news_event_survival_candidate":
        return "distinct sources and forward labels survive the source-independence gate"
    if status == "news_event_pending_archive":
        return "fresh labels are still pending, so source independence cannot be judged yet"
    if status == "news_event_duplicate_story_blocked":
        return "multiple sources mostly repeat the same story"
    if status == "news_event_single_source_blocked":
        return "supported forward label still comes from one source only"
    if status == "news_event_weak_forward_blocked":
        return "source evidence exists but forward labels are weak"
    return "no supported independent news-event label"


def _next_step(*, symbol: str, status: str) -> str:
    if status == "news_event_survival_candidate":
        return f"repeat {symbol} labels with beta, costs, and story timestamp isolation"
    if status == "news_event_pending_archive":
        return f"wait for {symbol} forward archives, then rerun source independence"
    if status == "news_event_duplicate_story_blocked":
        return f"require a distinct {symbol} event source before promotion"
    if status == "news_event_single_source_blocked":
        return f"seek another independent {symbol} source before treating the story as alpha"
    return f"keep {symbol} news as context only"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_news_event_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_news_event_survival.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_news_event_survival_rows()
    write_news_event_survival_csv(rows, output_path=args.output_path)
    write_news_event_survival_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.survival_status, row.candidate_id, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
