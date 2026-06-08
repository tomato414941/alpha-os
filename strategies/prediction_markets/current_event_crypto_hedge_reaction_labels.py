from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class EventCryptoHedgeReactionLabel:
    candidate_id: str
    ticket_id: str
    market_id: str
    question: str
    asset: str
    hedge_action: str
    event_bias: str
    candidate_status: str
    reaction_status: str
    opened_at: str
    checked_at: str
    elapsed_minutes: str
    checkpoint_status: str
    entry_mark: str
    current_mark: str
    directional_return_bps: str
    outcome: str
    probability_gap: str
    current_edge_after_ask: str
    ask_depth_to_5c: str
    next_step: str


def build_event_crypto_hedge_reaction_labels(
    *,
    candidates_path: Path = ROOT / "current_event_crypto_hedge_candidates.csv",
    outcomes_path: Path = STRATEGIES_ROOT / "current_paper_ticket_outcomes.csv",
) -> tuple[EventCryptoHedgeReactionLabel, ...]:
    candidates = {row.get("candidate_id", ""): row for row in _read_rows(candidates_path)}
    rows: list[EventCryptoHedgeReactionLabel] = []
    for outcome in _read_rows(outcomes_path):
        candidate_id = outcome.get("opportunity", "")
        candidate = candidates.get(candidate_id)
        if outcome.get("asset") not in {"BTC", "ETH", "SOL"} or candidate is None:
            continue
        rows.append(_build_label(candidate=candidate, outcome=outcome))
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_event_crypto_hedge_reaction_labels_csv(
    rows: tuple[EventCryptoHedgeReactionLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "ticket_id",
                "market_id",
                "question",
                "asset",
                "hedge_action",
                "event_bias",
                "candidate_status",
                "reaction_status",
                "opened_at",
                "checked_at",
                "elapsed_minutes",
                "checkpoint_status",
                "entry_mark",
                "current_mark",
                "directional_return_bps",
                "outcome",
                "probability_gap",
                "current_edge_after_ask",
                "ask_depth_to_5c",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.ticket_id,
                    row.market_id,
                    row.question,
                    row.asset,
                    row.hedge_action,
                    row.event_bias,
                    row.candidate_status,
                    row.reaction_status,
                    row.opened_at,
                    row.checked_at,
                    row.elapsed_minutes,
                    row.checkpoint_status,
                    row.entry_mark,
                    row.current_mark,
                    row.directional_return_bps,
                    row.outcome,
                    row.probability_gap,
                    row.current_edge_after_ask,
                    row.ask_depth_to_5c,
                    row.next_step,
                )
            )
    return output_path


def write_event_crypto_hedge_reaction_labels_md(
    rows: tuple[EventCryptoHedgeReactionLabel, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Crypto Hedge Reaction Labels\n\n")
        handle.write(
            "This joins event-crypto hedge candidates to paper-ticket mark outcomes. "
            "It labels the market reaction after the candidate is opened; it is not a live PnL report.\n\n"
        )
        handle.write(
            "| candidate | asset | action | reaction | elapsed min | entry | current | dir bps | event gap | edge | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.asset} | "
                f"{row.hedge_action}_{row.event_bias} | "
                f"{row.reaction_status} | "
                f"{row.elapsed_minutes} | "
                f"{row.entry_mark} | "
                f"{row.current_mark} | "
                f"{row.directional_return_bps} | "
                f"{row.probability_gap} | "
                f"{row.current_edge_after_ask} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary_text(rows))
    return output_path


def _build_label(*, candidate: dict[str, str], outcome: dict[str, str]) -> EventCryptoHedgeReactionLabel:
    return EventCryptoHedgeReactionLabel(
        candidate_id=candidate.get("candidate_id", ""),
        ticket_id=outcome.get("ticket_id", ""),
        market_id=candidate.get("market_id", ""),
        question=candidate.get("question", ""),
        asset=candidate.get("asset", ""),
        hedge_action=candidate.get("hedge_action", ""),
        event_bias=candidate.get("event_bias", ""),
        candidate_status=candidate.get("status", ""),
        reaction_status=_reaction_status(outcome),
        opened_at=outcome.get("opened_at", ""),
        checked_at=outcome.get("checked_at", ""),
        elapsed_minutes=outcome.get("elapsed_minutes", ""),
        checkpoint_status=outcome.get("checkpoint_status", ""),
        entry_mark=outcome.get("entry_mark", ""),
        current_mark=outcome.get("current_mark", ""),
        directional_return_bps=outcome.get("directional_return_bps", ""),
        outcome=outcome.get("outcome", ""),
        probability_gap=candidate.get("probability_gap", ""),
        current_edge_after_ask=candidate.get("current_edge_after_ask", ""),
        ask_depth_to_5c=candidate.get("ask_depth_to_5c", ""),
        next_step=_next_step(outcome),
    )


def _reaction_status(outcome: dict[str, str]) -> str:
    if outcome.get("checkpoint_status") == "pending":
        return "event_crypto_hedge_reaction_pending"
    if outcome.get("outcome") == "paper_mark_win":
        return "event_crypto_hedge_reaction_win"
    if outcome.get("outcome") == "paper_mark_loss":
        return "event_crypto_hedge_reaction_loss"
    if outcome.get("outcome") == "paper_mark_flat":
        return "event_crypto_hedge_reaction_flat"
    return "event_crypto_hedge_reaction_missing_mark"


def _next_step(outcome: dict[str, str]) -> str:
    status = _reaction_status(outcome)
    if status == "event_crypto_hedge_reaction_pending":
        return "wait for the 15m checkpoint, then refresh marks and funding"
    if status == "event_crypto_hedge_reaction_win":
        return "repeat with funding, spread/depth, beta attribution, and event timestamp controls"
    if status == "event_crypto_hedge_reaction_loss":
        return "record failure regime and check whether event odds were stale or non-causal"
    if status == "event_crypto_hedge_reaction_flat":
        return "keep only if the event probability edge persists after another refresh"
    return "fill missing current mark before judging the event hedge"


def _sort_key(row: EventCryptoHedgeReactionLabel) -> tuple[float, float]:
    status_rank = {
        "event_crypto_hedge_reaction_win": 4.0,
        "event_crypto_hedge_reaction_pending": 3.0,
        "event_crypto_hedge_reaction_flat": 2.0,
        "event_crypto_hedge_reaction_loss": 1.0,
        "event_crypto_hedge_reaction_missing_mark": 0.0,
    }.get(row.reaction_status, 0.0)
    return status_rank, _float(row.directional_return_bps)


def _summary_text(rows: tuple[EventCryptoHedgeReactionLabel, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.reaction_status] = counts.get(row.reaction_status, 0) + 1
    lines = [f"- {status}: {count}" for status, count in sorted(counts.items())]
    scored_rows = tuple(row for row in rows if row.directional_return_bps)
    best = max(
        scored_rows,
        key=lambda row: _float(row.directional_return_bps),
        default=None,
    )
    worst = min(
        scored_rows,
        key=lambda row: _float(row.directional_return_bps),
        default=None,
    )
    if best:
        lines.append(f"- best reaction: {best.candidate_id} {best.directional_return_bps}bps")
    if worst:
        lines.append(f"- worst reaction: {worst.candidate_id} {worst.directional_return_bps}bps")
    if not lines:
        lines.append("- no event crypto hedge paper outcomes yet")
    return "\n".join(lines) + "\n"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_crypto_hedge_reaction_labels.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_event_crypto_hedge_reaction_labels.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_event_crypto_hedge_reaction_labels()
    write_event_crypto_hedge_reaction_labels_csv(rows, output_path=args.output_path)
    write_event_crypto_hedge_reaction_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.reaction_status, row.directional_return_bps)


if __name__ == "__main__":
    main()
