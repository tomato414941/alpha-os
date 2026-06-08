from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.current_cross_modal_alpha_context import (
    ModalEvidence,
    ROOT,
    collect_cross_modal_evidence,
)


@dataclass(frozen=True)
class CrossModalSourceSplitRow:
    symbol: str
    source: str
    source_role: str
    context_decision: str
    source_direction: str
    aligned_direction: str
    source_score: float
    context_score: float
    priority_score: float
    paper_action: str
    evidence: str
    missing_work: str
    next_step: str


def build_cross_modal_source_split(
    *,
    context_path: Path = ROOT / "current_cross_modal_alpha_context.csv",
) -> tuple[CrossModalSourceSplitRow, ...]:
    contexts = {row.get("symbol", ""): row for row in _read_rows(context_path)}
    evidence = collect_cross_modal_evidence()
    output: list[CrossModalSourceSplitRow] = []
    for row in evidence:
        context = contexts.get(row.symbol)
        if context is None:
            continue
        output.append(_source_split_row(source=row, context=context))
    return tuple(sorted(output, key=lambda row: row.priority_score, reverse=True))


def write_cross_modal_source_split_csv(
    rows: tuple[CrossModalSourceSplitRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "source",
                "source_role",
                "context_decision",
                "source_direction",
                "aligned_direction",
                "source_score",
                "context_score",
                "priority_score",
                "paper_action",
                "evidence",
                "missing_work",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.source,
                    row.source_role,
                    row.context_decision,
                    row.source_direction,
                    row.aligned_direction,
                    f"{row.source_score:.8f}",
                    f"{row.context_score:.8f}",
                    f"{row.priority_score:.8f}",
                    row.paper_action,
                    row.evidence,
                    row.missing_work,
                    row.next_step,
                )
            )
    return output_path


def write_cross_modal_source_split_md(
    rows: tuple[CrossModalSourceSplitRow, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cross-Modal Source Split\n\n")
        handle.write(
            "This splits cross-modal alpha context back into source-level rows. "
            "It prevents event, wallet, chain, and liquidity evidence from being collapsed into one trade thesis before labels.\n\n"
        )
        handle.write(
            "| symbol | source | role | context | direction | priority | action | evidence | missing work | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | "
                f"{row.source} | "
                f"{row.source_role} | "
                f"{row.context_decision} | "
                f"{row.source_direction} | "
                f"{row.priority_score:.4f} | "
                f"{row.paper_action} | "
                f"{_escape(row.evidence)} | "
                f"{_escape(row.missing_work)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _source_split_row(*, source: ModalEvidence, context: dict[str, str]) -> CrossModalSourceSplitRow:
    aligned_sources = _split_sources(context.get("aligned_sources", ""))
    conflicting_sources = _split_sources(context.get("conflicting_sources", ""))
    source_name = source.source
    context_decision = context.get("decision", "")
    aligned_direction = context.get("aligned_direction", "")
    source_direction = "long" if source.direction > 0 else "short"
    source_role = _source_role(
        source_name=source_name,
        source_direction=source_direction,
        aligned_direction=aligned_direction,
        aligned_sources=aligned_sources,
        conflicting_sources=conflicting_sources,
    )
    context_score = _float(context.get("total_score"))
    paper_action = _paper_action(
        context_decision=context_decision,
        source_role=source_role,
        source_direction=source_direction,
        aligned_direction=aligned_direction,
    )
    priority_score = _priority_score(
        source_score=source.score,
        context_score=context_score,
        paper_action=paper_action,
        source_role=source_role,
    )
    return CrossModalSourceSplitRow(
        symbol=source.symbol,
        source=source_name,
        source_role=source_role,
        context_decision=context_decision,
        source_direction=source_direction,
        aligned_direction=aligned_direction,
        source_score=source.score,
        context_score=context_score,
        priority_score=priority_score,
        paper_action=paper_action,
        evidence=source.evidence,
        missing_work=source.missing_work,
        next_step=_next_step(symbol=source.symbol, source=source_name, paper_action=paper_action),
    )


def _source_role(
    *,
    source_name: str,
    source_direction: str,
    aligned_direction: str,
    aligned_sources: frozenset[str],
    conflicting_sources: frozenset[str],
) -> str:
    if source_name in aligned_sources and source_name in conflicting_sources:
        return "aligned" if source_direction == aligned_direction else "conflicting"
    if source_name in conflicting_sources:
        return "conflicting"
    if source_name in aligned_sources:
        return "aligned"
    return "unclassified"


def _paper_action(
    *,
    context_decision: str,
    source_role: str,
    source_direction: str,
    aligned_direction: str,
) -> str:
    if source_role == "conflicting":
        return "label_conflict_or_negative_control"
    if context_decision == "label_cross_modal_context" and source_direction == aligned_direction:
        return "label_source_component"
    if context_decision == "cross_modal_probe_after_conflict_check":
        return "label_after_conflict_check"
    if context_decision == "split_conflicting_modal_context":
        return "split_before_label"
    return "collect_more_evidence"


def _priority_score(*, source_score: float, context_score: float, paper_action: str, source_role: str) -> float:
    action_bonus = {
        "label_source_component": 35.0,
        "label_after_conflict_check": 20.0,
        "label_conflict_or_negative_control": 18.0,
        "split_before_label": 10.0,
        "collect_more_evidence": 0.0,
    }.get(paper_action, 0.0)
    role_bonus = 8.0 if source_role == "conflicting" else 0.0
    return source_score + min(context_score / 5.0, 60.0) + action_bonus + role_bonus


def _next_step(*, symbol: str, source: str, paper_action: str) -> str:
    if paper_action == "label_source_component":
        return f"paper-label {symbol} {source} alone over 15m/1h/4h before trusting the collapsed cross-modal thesis"
    if paper_action == "label_after_conflict_check":
        return f"check {symbol} {source} source timestamps and beta, then label it separately before collapsed paper action"
    if paper_action == "label_conflict_or_negative_control":
        return f"label {symbol} {source} as conflict or negative control against the aligned modal sources"
    if paper_action == "split_before_label":
        return f"split {symbol} {source} from the collapsed context and avoid merged paper action"
    return f"collect another {symbol} {source} observation before paper labeling"


def _split_sources(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


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
    parser.add_argument("--context-path", type=Path, default=ROOT / "current_cross_modal_alpha_context.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cross_modal_source_split.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cross_modal_source_split.md")
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    rows = build_cross_modal_source_split(context_path=args.context_path)
    write_cross_modal_source_split_csv(rows, output_path=args.output_path)
    write_cross_modal_source_split_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[:10]:
        print(row.symbol, row.source, row.source_role, row.paper_action, f"priority={row.priority_score:.4f}")


if __name__ == "__main__":
    main()
