from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class EventCryptoHedgeEventAlignment:
    candidate_id: str
    market_id: str
    question: str
    asset: str
    hedge_action: str
    event_bias: str
    asset_directional_return_bps: str
    basket_directional_return_bps: str
    residual_vs_basket_bps: str
    event_ticket_id: str
    event_mark_return_bps: str
    event_outcome: str
    same_asset_control_count: int
    same_asset_control_mean_bps: str
    same_asset_control_gap_bps: str
    alignment_status: str
    next_step: str


def build_event_crypto_hedge_event_alignment(
    *,
    beta_attribution_path: Path = ROOT / "current_event_crypto_hedge_beta_attribution.csv",
    paper_outcomes_path: Path = STRATEGIES_ROOT / "current_paper_ticket_outcomes.csv",
    paper_tickets_path: Path = STRATEGIES_ROOT / "current_paper_tickets.csv",
) -> tuple[EventCryptoHedgeEventAlignment, ...]:
    outcomes = _read_rows(paper_outcomes_path)
    event_outcomes = _event_outcomes_by_question(outcomes=outcomes, tickets=_read_rows(paper_tickets_path))
    rows = [
        _build_alignment(
            row=row,
            event_outcome=event_outcomes.get(row.get("question", ""), {}),
            control_returns=_same_asset_control_returns(row=row, outcomes=outcomes),
        )
        for row in _read_rows(beta_attribution_path)
    ]
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_event_crypto_hedge_event_alignment_csv(
    rows: tuple[EventCryptoHedgeEventAlignment, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "market_id",
                "question",
                "asset",
                "hedge_action",
                "event_bias",
                "asset_directional_return_bps",
                "basket_directional_return_bps",
                "residual_vs_basket_bps",
                "event_ticket_id",
                "event_mark_return_bps",
                "event_outcome",
                "same_asset_control_count",
                "same_asset_control_mean_bps",
                "same_asset_control_gap_bps",
                "alignment_status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.market_id,
                    row.question,
                    row.asset,
                    row.hedge_action,
                    row.event_bias,
                    row.asset_directional_return_bps,
                    row.basket_directional_return_bps,
                    row.residual_vs_basket_bps,
                    row.event_ticket_id,
                    row.event_mark_return_bps,
                    row.event_outcome,
                    row.same_asset_control_count,
                    row.same_asset_control_mean_bps,
                    row.same_asset_control_gap_bps,
                    row.alignment_status,
                    row.next_step,
                )
            )
    return output_path


def write_event_crypto_hedge_event_alignment_md(
    rows: tuple[EventCryptoHedgeEventAlignment, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Crypto Hedge Event Alignment\n\n")
        handle.write(
            "This checks whether the prediction-market event price moved with the crypto hedge return. "
            "It also compares the hedge return to same-asset non-event paper tickets. "
            "It is a rejection/control artifact, not a trade instruction.\n\n"
        )
        handle.write(
            "| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.asset} | "
                f"{row.alignment_status} | "
                f"{row.asset_directional_return_bps} | "
                f"{row.basket_directional_return_bps} | "
                f"{row.event_mark_return_bps} | "
                f"{row.same_asset_control_count} | "
                f"{row.same_asset_control_mean_bps} | "
                f"{row.same_asset_control_gap_bps} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary_text(rows))
    return output_path


def _build_alignment(
    *,
    row: dict[str, str],
    event_outcome: dict[str, str],
    control_returns: tuple[float, ...],
) -> EventCryptoHedgeEventAlignment:
    asset_return = _float(row.get("asset_directional_return_bps"))
    event_return = _float(event_outcome.get("directional_return_bps"))
    control_mean = sum(control_returns) / len(control_returns) if control_returns else 0.0
    control_gap = asset_return - control_mean if control_returns else 0.0
    status = _alignment_status(
        event_outcome=event_outcome,
        event_return=event_return,
        control_count=len(control_returns),
        control_gap=control_gap,
        asset_return=asset_return,
    )
    return EventCryptoHedgeEventAlignment(
        candidate_id=row.get("candidate_id", ""),
        market_id=row.get("market_id", ""),
        question=row.get("question", ""),
        asset=row.get("asset", ""),
        hedge_action=row.get("hedge_action", ""),
        event_bias=row.get("event_bias", ""),
        asset_directional_return_bps=row.get("asset_directional_return_bps", ""),
        basket_directional_return_bps=row.get("basket_directional_return_bps", ""),
        residual_vs_basket_bps=row.get("residual_vs_basket_bps", ""),
        event_ticket_id=event_outcome.get("ticket_id", ""),
        event_mark_return_bps=event_outcome.get("directional_return_bps", ""),
        event_outcome=event_outcome.get("outcome", ""),
        same_asset_control_count=len(control_returns),
        same_asset_control_mean_bps=_format_optional(control_returns, control_mean),
        same_asset_control_gap_bps=_format_optional(control_returns, control_gap),
        alignment_status=status,
        next_step=_next_step(status),
    )


def _event_outcomes_by_question(
    *,
    outcomes: tuple[dict[str, str], ...],
    tickets: tuple[dict[str, str], ...],
) -> dict[str, dict[str, str]]:
    questions_by_ticket = {}
    for ticket in tickets:
        if ticket.get("asset") != "EVENT":
            continue
        side = ticket.get("side", "")
        if ": " not in side:
            continue
        questions_by_ticket[ticket.get("ticket_id", "")] = side.split(": ", 1)[1]
    output = {}
    for outcome in outcomes:
        question = questions_by_ticket.get(outcome.get("ticket_id", ""))
        if question:
            output[question] = outcome
    return output


def _same_asset_control_returns(*, row: dict[str, str], outcomes: tuple[dict[str, str], ...]) -> tuple[float, ...]:
    asset = row.get("asset", "")
    values = []
    for outcome in outcomes:
        if outcome.get("asset") != asset:
            continue
        if outcome.get("checkpoint_status") != "ready":
            continue
        if not outcome.get("directional_return_bps"):
            continue
        opportunity = outcome.get("opportunity", "")
        if opportunity == row.get("candidate_id", "") or "event_crypto_hedge" in opportunity:
            continue
        values.append(_float(outcome.get("directional_return_bps")))
    return tuple(values)


def _alignment_status(
    *,
    event_outcome: dict[str, str],
    event_return: float,
    control_count: int,
    control_gap: float,
    asset_return: float,
) -> str:
    if not event_outcome:
        return "event_alignment_missing_event_ticket"
    if event_outcome.get("checkpoint_status") != "ready":
        return "event_alignment_pending_event_ticket"
    if abs(event_return) <= 1.0 and asset_return > 0.0:
        return "event_probability_flat_crypto_moved"
    if control_count and abs(control_gap) <= 1.0:
        return "same_asset_control_explains_return"
    if event_return > 0.0 and asset_return > 0.0:
        return "event_probability_and_crypto_aligned"
    if event_return < 0.0 and asset_return > 0.0:
        return "event_probability_contradicts_crypto"
    return "event_alignment_inconclusive"


def _next_step(status: str) -> str:
    if status == "event_probability_flat_crypto_moved":
        return "do not promote this event hedge; require event-market probability movement or stronger timestamp evidence"
    if status == "same_asset_control_explains_return":
        return "treat the hedge as a shared asset move until it beats same-asset non-event controls"
    if status == "event_probability_and_crypto_aligned":
        return "repeat with more event markets, costs, funding, and timestamp controls"
    if status == "event_probability_contradicts_crypto":
        return "reject or isolate the non-event crypto driver before retrying"
    return "collect a ready event-market ticket and same-asset controls before judging"


def _summary_text(rows: tuple[EventCryptoHedgeEventAlignment, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.alignment_status] = counts.get(row.alignment_status, 0) + 1
    lines = [f"- {status}: {count}" for status, count in sorted(counts.items())]
    if not lines:
        lines.append("- no event crypto hedge alignment rows yet")
    return "\n".join(lines) + "\n"


def _sort_key(row: EventCryptoHedgeEventAlignment) -> tuple[float, float]:
    status_rank = {
        "event_probability_and_crypto_aligned": 4.0,
        "same_asset_control_explains_return": 3.0,
        "event_alignment_inconclusive": 2.0,
        "event_probability_flat_crypto_moved": 1.0,
        "event_probability_contradicts_crypto": 0.0,
    }.get(row.alignment_status, 0.0)
    return status_rank, _float(row.asset_directional_return_bps)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _format_optional(source: tuple[float, ...], value: float) -> str:
    if not source:
        return ""
    return f"{value:.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_crypto_hedge_event_alignment.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_event_crypto_hedge_event_alignment.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_event_crypto_hedge_event_alignment()
    write_event_crypto_hedge_event_alignment_csv(rows, output_path=args.output_path)
    write_event_crypto_hedge_event_alignment_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.alignment_status, row.asset_directional_return_bps)


if __name__ == "__main__":
    main()
