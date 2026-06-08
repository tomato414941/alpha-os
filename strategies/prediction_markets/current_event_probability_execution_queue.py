from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventProbabilityExecutionQueueItem:
    queue_id: str
    opened_at: str
    market_id: str
    question: str
    outcome_to_buy: str
    current_ask: str
    current_edge_after_ask: str
    edge_to_max_loss: str
    ask_depth_to_5c: str
    queue_action: str
    checkpoints: str
    required_record: str
    next_step: str


def build_event_probability_execution_queue(
    *,
    execution_check_path: Path = ROOT / "current_event_probability_execution_check.csv",
    existing_queue_path: Path | None = None,
) -> tuple[EventProbabilityExecutionQueueItem, ...]:
    opened_at = datetime.now(UTC).isoformat(timespec="seconds")
    existing = {row.get("queue_id", ""): row for row in _read_rows(existing_queue_path) if row.get("queue_id")}
    rows = tuple(
        _queue_item(row=row, opened_at=opened_at, existing=existing)
        for row in _read_rows(execution_check_path)
        if row.get("execution_action") in {
            "paper_check_probability_execution",
            "restart_probability_probe_at_current_quote",
        }
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_event_probability_execution_queue_csv(
    rows: tuple[EventProbabilityExecutionQueueItem, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(EventProbabilityExecutionQueueItem.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_event_probability_execution_queue_md(
    rows: tuple[EventProbabilityExecutionQueueItem, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Probability Execution Queue\n\n")
        handle.write(
            "This queue tracks pure prediction-market probability probes that need quote refresh, "
            "fill/queue notes, max-loss handling, and resolution-risk records. It is not a live order list.\n\n"
        )
        handle.write("| queue | market | side | ask | edge | depth 5c | action | checkpoints | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.queue_id} | {_escape(row.question)} | {row.outcome_to_buy} | "
                f"{row.current_ask} | {row.current_edge_after_ask} | {row.ask_depth_to_5c} | "
                f"{row.queue_action} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _queue_item(
    *,
    row: dict[str, str],
    opened_at: str,
    existing: dict[str, dict[str, str]],
) -> EventProbabilityExecutionQueueItem:
    market_id = row.get("market_id", "")
    queue_id = f"event-probability-execution-{market_id}-{_slug(row.get('outcome_to_buy', ''))}"
    existing_row = existing.get(queue_id, {})
    return EventProbabilityExecutionQueueItem(
        queue_id=queue_id,
        opened_at=existing_row.get("opened_at") or opened_at,
        market_id=market_id,
        question=row.get("question", ""),
        outcome_to_buy=row.get("outcome_to_buy", ""),
        current_ask=row.get("current_ask", ""),
        current_edge_after_ask=row.get("current_edge_after_ask", ""),
        edge_to_max_loss=row.get("edge_to_max_loss", ""),
        ask_depth_to_5c=row.get("ask_depth_to_5c", ""),
        queue_action=_queue_action(row.get("execution_action", "")),
        checkpoints="15m,1h,4h",
        required_record="quote refresh, fill or queue assumption, max-loss size, resolution-risk note, and source-quality update",
        next_step=row.get("next_step", ""),
    )


def _queue_action(execution_action: str) -> str:
    if execution_action == "paper_check_probability_execution":
        return "paper_check_pure_probability"
    if execution_action == "restart_probability_probe_at_current_quote":
        return "restart_quote_survival_probe"
    return "watch_probability_probe"


def _sort_key(row: EventProbabilityExecutionQueueItem) -> tuple[int, float, float]:
    action_rank = {
        "paper_check_pure_probability": 2,
        "restart_quote_survival_probe": 1,
    }.get(row.queue_action, 0)
    return (action_rank, _float(row.current_edge_after_ask), _float(row.ask_depth_to_5c))


def _read_rows(path: Path | None) -> tuple[dict[str, str], ...]:
    if path is None or not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-check-path",
        type=Path,
        default=ROOT / "current_event_probability_execution_check.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_event_probability_execution_queue.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_event_probability_execution_queue.md",
    )
    parser.add_argument("--preserve-opened-at", action="store_true")
    args = parser.parse_args()

    rows = build_event_probability_execution_queue(
        execution_check_path=args.execution_check_path,
        existing_queue_path=args.output_path if args.preserve_opened_at else None,
    )
    write_event_probability_execution_queue_csv(rows, output_path=args.output_path)
    write_event_probability_execution_queue_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.queue_id, row.queue_action, row.current_edge_after_ask)


if __name__ == "__main__":
    main()
