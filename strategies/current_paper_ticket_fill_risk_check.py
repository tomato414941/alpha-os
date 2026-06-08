from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FillRiskCheck:
    ticket_id: str
    action: str
    asset: str
    opportunity: str
    decision: str
    candidate_size_usd: str
    directional_return_bps: float
    spread_bps: str
    near_depth_10bps_notional: str
    visible_depth_usage: str
    annualized_funding: str
    estimated_funding_1h_bps: str
    estimated_round_trip_cost_bps: str
    estimated_net_after_cost_bps: str
    risk_action: str
    reason: str
    next_step: str


def build_fill_risk_checks(
    *,
    action_queue_path: Path = ROOT / "current_paper_ticket_action_queue.csv",
    tickets_path: Path = ROOT / "current_paper_tickets.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    taker_fee_bps_per_fill: float = 4.0,
) -> tuple[FillRiskCheck, ...]:
    tickets = {row.get("ticket_id", ""): row for row in _read_rows(tickets_path)}
    market_context = _market_context(
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows = []
    for row in _read_rows(action_queue_path):
        if row.get("action") != "promote_to_fill_and_risk_check":
            continue
        rows.append(
            _build_check(
                action=row,
                ticket=tickets.get(row.get("ticket_id", ""), {}),
                market_context=market_context,
                taker_fee_bps_per_fill=taker_fee_bps_per_fill,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.directional_return_bps, reverse=True))


def write_fill_risk_checks_csv(rows: tuple[FillRiskCheck, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "action",
                "asset",
                "opportunity",
                "decision",
                "candidate_size_usd",
                "directional_return_bps",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage",
                "annualized_funding",
                "estimated_funding_1h_bps",
                "estimated_round_trip_cost_bps",
                "estimated_net_after_cost_bps",
                "risk_action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.action,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.candidate_size_usd,
                    f"{row.directional_return_bps:.8f}",
                    row.spread_bps,
                    row.near_depth_10bps_notional,
                    row.visible_depth_usage,
                    row.annualized_funding,
                    row.estimated_funding_1h_bps,
                    row.estimated_round_trip_cost_bps,
                    row.estimated_net_after_cost_bps,
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_fill_risk_checks_md(rows: tuple[FillRiskCheck, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Ticket Fill Risk Check\n\n")
        handle.write(
            "This checks promoted paper-ticket mark wins against rough spread, taker fee, "
            "funding, and visible-depth assumptions. It is not a live fill report.\n\n"
        )
        handle.write(
            "| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.directional_return_bps:.4f} | "
                f"{row.spread_bps} | "
                f"{row.near_depth_10bps_notional} | "
                f"{row.visible_depth_usage} | "
                f"{row.estimated_funding_1h_bps} | "
                f"{row.estimated_round_trip_cost_bps} | "
                f"{row.estimated_net_after_cost_bps} | "
                f"{row.risk_action} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _build_check(
    *,
    action: dict[str, str],
    ticket: dict[str, str],
    market_context: dict[tuple[str, str], dict[str, str]],
    taker_fee_bps_per_fill: float,
) -> FillRiskCheck:
    asset = action.get("asset", "")
    venue = ticket.get("venue", "") or "HL"
    context = market_context.get((venue, asset)) or market_context.get(("HL", asset), {})
    directional_bps = _float(action.get("directional_return_bps"))
    spread_bps = _float(context.get("spread_bps"))
    funding_1h_bps = _funding_1h_bps(
        annualized_funding=_float(context.get("annualized_funding")),
        decision=action.get("decision", ""),
    )
    round_trip_cost_bps = spread_bps + 2.0 * taker_fee_bps_per_fill
    estimated_net_bps = directional_bps - round_trip_cost_bps + funding_1h_bps
    visible_depth_usage = _visible_depth_usage(
        candidate_size=ticket.get("candidate_size_usd", ""),
        near_depth=context.get("near_depth_10bps_notional", ""),
    )
    risk_action, reason, next_step = _risk_action(
        estimated_net_bps=estimated_net_bps,
        visible_depth_usage=visible_depth_usage,
        context=context,
    )
    return FillRiskCheck(
        ticket_id=action.get("ticket_id", ""),
        action=action.get("action", ""),
        asset=asset,
        opportunity=action.get("opportunity", ""),
        decision=action.get("decision", ""),
        candidate_size_usd=ticket.get("candidate_size_usd", ""),
        directional_return_bps=directional_bps,
        spread_bps=_format_float(context.get("spread_bps")),
        near_depth_10bps_notional=_format_float(context.get("near_depth_10bps_notional")),
        visible_depth_usage="" if visible_depth_usage is None else f"{visible_depth_usage:.8f}",
        annualized_funding=_format_float(context.get("annualized_funding")),
        estimated_funding_1h_bps=f"{funding_1h_bps:.8f}",
        estimated_round_trip_cost_bps=f"{round_trip_cost_bps:.8f}",
        estimated_net_after_cost_bps=f"{estimated_net_bps:.8f}",
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _market_context(
    *,
    hl_context_path: Path,
    okx_context_path: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    contexts: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_rows(hl_context_path):
        asset = row.get("asset", "")
        if asset:
            contexts[("HL", asset)] = row
    for row in _read_rows(okx_context_path):
        asset = row.get("asset", "")
        if asset:
            contexts[("OKX", asset)] = row
    return contexts


def _funding_1h_bps(*, annualized_funding: float, decision: str) -> float:
    long_funding_bps = -annualized_funding / (365.0 * 24.0) * 10_000.0
    if decision == "paper_short":
        return -long_funding_bps
    return long_funding_bps


def _visible_depth_usage(*, candidate_size: str, near_depth: str) -> float | None:
    size = _float(candidate_size)
    depth = _float(near_depth)
    if size <= 0.0 or depth <= 0.0:
        return None
    return size / depth


def _risk_action(
    *,
    estimated_net_bps: float,
    visible_depth_usage: float | None,
    context: dict[str, str],
) -> tuple[str, str, str]:
    if not context:
        return (
            "missing_execution_context",
            "no current public execution context for the promoted paper ticket",
            "refresh execution context before repeating this ticket",
        )
    if visible_depth_usage is not None and visible_depth_usage > 0.10:
        return (
            "depth_too_thin_for_probe",
            "candidate size consumes too much visible 10bps depth",
            "reduce candidate size or skip until depth improves",
        )
    if estimated_net_bps <= 0.0:
        return (
            "cost_adjusted_edge_failed",
            "paper mark win does not survive rough spread, taker-fee, and funding haircut",
            "repeat only if the independent signal is strong enough to justify lower-cost execution",
        )
    return (
        "cost_adjusted_paper_probe",
        "paper mark win survives rough spread, taker-fee, funding, and visible-depth checks",
        "repeat the ticket and add stop/adverse-excursion notes before any promotion",
    )


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


def _format_float(value: str | None) -> str:
    parsed = _float(value)
    return "" if value in {None, ""} else f"{parsed:.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-queue-path", type=Path, default=ROOT / "current_paper_ticket_action_queue.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_ticket_fill_risk_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_ticket_fill_risk_check.md")
    args = parser.parse_args()

    rows = build_fill_risk_checks(action_queue_path=args.action_queue_path)
    write_fill_risk_checks_csv(rows, output_path=args.output_path)
    write_fill_risk_checks_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.risk_action, row.ticket_id, row.asset, row.estimated_net_after_cost_bps)


if __name__ == "__main__":
    main()
