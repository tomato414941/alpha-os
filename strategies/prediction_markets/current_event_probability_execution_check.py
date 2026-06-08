from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityExecutionCheck:
    market_id: str
    question: str
    suggested_side: str
    outcome_to_buy: str
    status: str
    entry_ask: str
    current_bid: str
    current_ask: str
    spread: str
    estimated_payout_probability: str
    current_edge_after_ask: str
    edge_to_max_loss: str
    ask_depth_to_5c: str
    source_quality_status: str
    refresh_status: str
    execution_action: str
    reason: str
    next_step: str


def build_event_probability_execution_checks(
    actionability_path: Path = ROOT / "current_event_probability_actionability.csv",
) -> tuple[EventProbabilityExecutionCheck, ...]:
    rows = tuple(_check_for_row(row) for row in _read_rows(actionability_path))
    rows = tuple(row for row in rows if row.execution_action != "deprioritize_event_probability_probe")
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_event_probability_execution_checks_csv(
    rows: tuple[EventProbabilityExecutionCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(EventProbabilityExecutionCheck.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_event_probability_execution_checks_md(
    rows: tuple[EventProbabilityExecutionCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Execution Check\n\n")
        handle.write(
            "This ranks pure prediction-market probability probes by quote, depth, source quality, "
            "and refresh state. It is not a live order list and not a crypto hedge signal.\n\n"
        )
        handle.write(
            "| market | side | ask | edge | edge/loss | depth 5c | refresh | action | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {_escape(row.question)} | {row.outcome_to_buy} | {row.current_ask} | "
                f"{row.current_edge_after_ask} | {row.edge_to_max_loss} | {row.ask_depth_to_5c} | "
                f"{row.refresh_status} | {row.execution_action} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _check_for_row(row: dict[str, str]) -> EventProbabilityExecutionCheck:
    current_ask = _float(row.get("current_ask"))
    current_edge = _float(row.get("current_edge_after_ask"))
    depth = _float(row.get("ask_depth_to_5c"))
    spread = _float(row.get("spread"))
    source_quality = row.get("source_quality_status", "")
    refresh = row.get("refresh_status", "")
    status = row.get("status", "")
    edge_to_loss = current_edge / current_ask if current_ask > 0.0 else 0.0
    action, reason, next_step = _execution_action(
        status=status,
        source_quality=source_quality,
        refresh=refresh,
        current_edge=current_edge,
        edge_to_loss=edge_to_loss,
        depth=depth,
        spread=spread,
        question=row.get("question", ""),
    )
    return EventProbabilityExecutionCheck(
        market_id=row.get("market_id", ""),
        question=row.get("question", ""),
        suggested_side=row.get("suggested_side", ""),
        outcome_to_buy=row.get("outcome_to_buy", ""),
        status=status,
        entry_ask=row.get("entry_ask", ""),
        current_bid=row.get("current_bid", ""),
        current_ask=row.get("current_ask", ""),
        spread=row.get("spread", ""),
        estimated_payout_probability=row.get("estimated_payout_probability", ""),
        current_edge_after_ask=row.get("current_edge_after_ask", ""),
        edge_to_max_loss=f"{edge_to_loss:.8f}",
        ask_depth_to_5c=row.get("ask_depth_to_5c", ""),
        source_quality_status=source_quality,
        refresh_status=refresh,
        execution_action=action,
        reason=reason,
        next_step=next_step,
    )


def _execution_action(
    *,
    status: str,
    source_quality: str,
    refresh: str,
    current_edge: float,
    edge_to_loss: float,
    depth: float,
    spread: float,
    question: str,
) -> tuple[str, str, str]:
    if source_quality != "source_quality_pass":
        return (
            "deprioritize_event_probability_probe",
            "source quality did not pass",
            "repair source-quality evidence before any probability paper check",
        )
    if current_edge < 0.1 or edge_to_loss < 0.75 or depth < 50_000.0 or spread > 0.02:
        return (
            "watch_event_probability_probe",
            "edge, edge-to-loss, depth, or spread is not strong enough for a clean paper check",
            f"refresh quotes and source evidence for {question}",
        )
    if refresh == "paper_outcome_survived_refresh":
        return (
            "paper_check_probability_execution",
            "edge, depth, source quality, and quote refresh survived",
            f"paper-check {question} as a pure event-probability trade with max-loss and resolution-risk notes",
        )
    if status == "event_probability_candidate_after_current_quote_check":
        return (
            "restart_probability_probe_at_current_quote",
            "current quote is strong but still needs another refresh before promotion",
            f"restart paper ticket for {question} and require quote refresh survival",
        )
    return (
        "watch_event_probability_probe",
        "probability edge exists but refresh quality is not strong enough",
        f"refresh quotes and source evidence for {question}",
    )


def _sort_key(row: EventProbabilityExecutionCheck) -> tuple[int, float, float]:
    action_rank = {
        "paper_check_probability_execution": 3,
        "restart_probability_probe_at_current_quote": 2,
        "watch_event_probability_probe": 1,
    }.get(row.execution_action, 0)
    return (action_rank, _float(row.current_edge_after_ask), _float(row.ask_depth_to_5c))


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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionability-path", type=Path, default=ROOT / "current_event_probability_actionability.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_probability_execution_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_event_probability_execution_check.md")
    args = parser.parse_args()

    rows = build_event_probability_execution_checks(actionability_path=args.actionability_path)
    write_event_probability_execution_checks_csv(rows, output_path=args.output_path)
    write_event_probability_execution_checks_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.market_id, row.execution_action, row.current_edge_after_ask, row.question)


if __name__ == "__main__":
    main()
