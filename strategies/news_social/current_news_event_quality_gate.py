from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class NewsEventQualityGate:
    symbol: str
    event_kind: str
    side: str
    decision: str
    score: float
    source_count: int
    label_count: int
    supported_count: int
    rejected_count: int
    pending_count: int
    mean_directional_1h_bps: float
    mean_directional_4h_bps: float
    best_directional_4h_bps: float
    sources: str
    strongest_title: str
    reason: str
    next_step: str


def build_news_event_quality_gate(
    *,
    labels_path: Path = ROOT / "current_news_event_forward_labels.csv",
) -> tuple[NewsEventQualityGate, ...]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in _read_rows(labels_path):
        side = row.get("side", "")
        if side == "context_label":
            continue
        key = (row.get("symbol", ""), row.get("event_kind", ""), side)
        if not all(key):
            continue
        groups.setdefault(key, []).append(row)
    rows = tuple(_gate_for_group(key, group_rows) for key, group_rows in groups.items())
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_news_event_quality_gate_csv(
    rows: tuple[NewsEventQualityGate, ...],
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
                "decision",
                "score",
                "source_count",
                "label_count",
                "supported_count",
                "rejected_count",
                "pending_count",
                "mean_directional_1h_bps",
                "mean_directional_4h_bps",
                "best_directional_4h_bps",
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
                    row.decision,
                    f"{row.score:.8f}",
                    row.source_count,
                    row.label_count,
                    row.supported_count,
                    row.rejected_count,
                    row.pending_count,
                    f"{row.mean_directional_1h_bps:.8f}",
                    f"{row.mean_directional_4h_bps:.8f}",
                    f"{row.best_directional_4h_bps:.8f}",
                    row.sources,
                    row.strongest_title,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_news_event_quality_gate_md(
    rows: tuple[NewsEventQualityGate, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current News Event Quality Gate\n\n")
        handle.write(
            "This groups timestamped news-event labels by symbol, event kind, and side. "
            "It checks repeat support, source diversity, stale or pending labels, and rejected labels. "
            "It is a gate, not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | kind | side | decision | score | sources | labels | support/reject/pending | mean 1h | mean 4h | best 4h | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | "
                f"{row.event_kind} | "
                f"{row.side} | "
                f"{row.decision} | "
                f"{row.score:.4f} | "
                f"{row.source_count} | "
                f"{row.label_count} | "
                f"{row.supported_count}/{row.rejected_count}/{row.pending_count} | "
                f"{row.mean_directional_1h_bps:.4f} | "
                f"{row.mean_directional_4h_bps:.4f} | "
                f"{row.best_directional_4h_bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _gate_for_group(key: tuple[str, str, str], rows: list[dict[str, str]]) -> NewsEventQualityGate:
    symbol, event_kind, side = key
    supported = [row for row in rows if row.get("label_status") in {"direction_supported_1h_4h", "direction_supported_1h_only"}]
    rejected = [row for row in rows if row.get("label_status") == "direction_rejected_1h"]
    pending = [row for row in rows if row.get("label_status") == "pending_forward_archive"]
    sources = sorted({row.get("source", "") for row in rows if row.get("source")})
    directional_1h = [_float(row.get("directional_1h_bps")) for row in supported if row.get("directional_1h_bps")]
    directional_4h = [_float(row.get("directional_4h_bps")) for row in supported if row.get("directional_4h_bps")]
    best = max(rows, key=lambda row: _float(row.get("directional_4h_bps") or row.get("directional_1h_bps")))
    decision, reason = _decision(
        source_count=len(sources),
        supported_count=len(supported),
        rejected_count=len(rejected),
        pending_count=len(pending),
        mean_1h=mean(directional_1h) if directional_1h else 0.0,
        mean_4h=mean(directional_4h) if directional_4h else 0.0,
    )
    score = _score(
        source_count=len(sources),
        supported_count=len(supported),
        rejected_count=len(rejected),
        pending_count=len(pending),
        mean_1h=mean(directional_1h) if directional_1h else 0.0,
        mean_4h=mean(directional_4h) if directional_4h else 0.0,
    )
    return NewsEventQualityGate(
        symbol=symbol,
        event_kind=event_kind,
        side=side,
        decision=decision,
        score=score,
        source_count=len(sources),
        label_count=len(rows),
        supported_count=len(supported),
        rejected_count=len(rejected),
        pending_count=len(pending),
        mean_directional_1h_bps=mean(directional_1h) if directional_1h else 0.0,
        mean_directional_4h_bps=mean(directional_4h) if directional_4h else 0.0,
        best_directional_4h_bps=max(directional_4h) if directional_4h else 0.0,
        sources=", ".join(sources),
        strongest_title=best.get("title", ""),
        reason=reason,
        next_step=_next_step(symbol=symbol, decision=decision),
    )


def _decision(
    *,
    source_count: int,
    supported_count: int,
    rejected_count: int,
    pending_count: int,
    mean_1h: float,
    mean_4h: float,
) -> tuple[str, str]:
    if supported_count == 0:
        return "reject_no_supported_label", "no supported timestamp label"
    if source_count < 2:
        return "repeat_single_source_label", "supported labels come from one source only"
    if rejected_count > supported_count:
        return "mixed_or_rejected_news_label", "rejected labels outnumber supported labels"
    if mean_1h <= 0.0:
        return "reject_weak_1h_label", "mean 1h directional label is not positive"
    if mean_4h <= 0.0:
        return "watch_1h_only_news_label", "1h support exists but 4h support is weak or negative"
    if pending_count:
        return "repeat_after_pending_archive", "supported labels exist but fresh labels are still pending archive"
    return "repeat_supported_multi_source_label", "multi-source supported label survived 1h and 4h"


def _score(
    *,
    source_count: int,
    supported_count: int,
    rejected_count: int,
    pending_count: int,
    mean_1h: float,
    mean_4h: float,
) -> float:
    return (
        min(source_count, 4) * 10.0
        + supported_count * 8.0
        - rejected_count * 6.0
        - pending_count * 2.0
        + min(max(mean_1h, -100.0), 200.0) * 0.05
        + min(max(mean_4h, -200.0), 400.0) * 0.03
    )


def _next_step(*, symbol: str, decision: str) -> str:
    if decision == "repeat_supported_multi_source_label":
        return f"open a small repeated {symbol} news-event label with duplicate-source and execution-cost checks"
    if decision == "repeat_after_pending_archive":
        return f"wait for fresh {symbol} archives, then repeat the news-event quality gate"
    if decision == "repeat_single_source_label":
        return f"seek another independent {symbol} source before treating the news event as alpha"
    if decision == "watch_1h_only_news_label":
        return f"watch {symbol} 1h decay and require 4h or repeated support"
    return f"keep {symbol} news event as context only"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_news_event_quality_gate.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_news_event_quality_gate.md")
    args = parser.parse_args()

    rows = build_news_event_quality_gate(labels_path=args.labels_path)
    write_news_event_quality_gate_csv(rows, output_path=args.output_path)
    write_news_event_quality_gate_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.symbol, row.event_kind, row.side, row.decision, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
