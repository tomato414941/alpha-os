from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto_pair_spread.current_crypto_pair_spread_tickets import LOCAL_ROOT, ROOT


@dataclass(frozen=True)
class PairSpreadFillRiskCheck:
    ticket_id: str
    pair: str
    decision: str
    directional_return_bps: str
    base_spread_bps: str
    quote_spread_bps: str
    base_depth_10bps_notional: str
    quote_depth_10bps_notional: str
    base_annualized_funding: str
    quote_annualized_funding: str
    estimated_funding_1h_bps: str
    estimated_round_trip_cost_bps: str
    estimated_net_after_cost_bps: str
    risk_action: str
    reason: str
    next_step: str


def build_pair_spread_fill_risk_checks(
    *,
    outcomes_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_outcomes.csv",
    tickets_path: Path = LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv",
    hl_context_path: Path = ROOT / "candidate_validation" / "current_followup_execution_context.csv",
    taker_fee_bps_per_fill: float = 4.0,
) -> tuple[PairSpreadFillRiskCheck, ...]:
    tickets = {row.get("ticket_id", ""): row for row in _read_rows(tickets_path)}
    context = {row.get("asset", ""): row for row in _read_rows(hl_context_path)}
    rows = tuple(
        _build_check(
            outcome=row,
            ticket=tickets.get(row.get("ticket_id", ""), {}),
            context=context,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for row in _read_rows(outcomes_path)
        if row.get("checkpoint_status") == "ready" and row.get("outcome") == "paper_mark_win"
    )
    return tuple(sorted(rows, key=lambda row: _float(row.estimated_net_after_cost_bps), reverse=True))


def write_pair_spread_fill_risk_checks_csv(
    rows: tuple[PairSpreadFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(PairSpreadFillRiskCheck.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_pair_spread_fill_risk_checks_md(
    rows: tuple[PairSpreadFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Pair Spread Fill Risk Check\n\n")
        handle.write(
            "This checks pair-ratio mark wins against rough two-leg spread, taker fee, "
            "funding, and visible-depth assumptions. It is not a fill report.\n\n"
        )
        handle.write("| ticket | pair | dir bps | cost bps | funding 1h | net bps | action | reason |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.pair} | {row.directional_return_bps} | "
                f"{row.estimated_round_trip_cost_bps} | {row.estimated_funding_1h_bps} | "
                f"{row.estimated_net_after_cost_bps} | {row.risk_action} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_check(
    *,
    outcome: dict[str, str],
    ticket: dict[str, str],
    context: dict[str, dict[str, str]],
    taker_fee_bps_per_fill: float,
) -> PairSpreadFillRiskCheck:
    base = ticket.get("base_asset", "")
    quote = ticket.get("quote_asset", "")
    base_context = context.get(base, {})
    quote_context = context.get(quote, {})
    base_spread = _float(base_context.get("spread_bps"))
    quote_spread = _float(quote_context.get("spread_bps"))
    funding_1h = _pair_funding_1h_bps(
        decision=outcome.get("decision", ""),
        base_annualized=_float(base_context.get("annualized_funding")),
        quote_annualized=_float(quote_context.get("annualized_funding")),
    )
    round_trip_cost = base_spread + quote_spread + 4.0 * taker_fee_bps_per_fill
    directional = _float(outcome.get("directional_return_bps"))
    net = directional - round_trip_cost + funding_1h
    risk_action, reason, next_step = _risk_action(net_bps=net, base_context=base_context, quote_context=quote_context)
    return PairSpreadFillRiskCheck(
        ticket_id=outcome.get("ticket_id", ""),
        pair=outcome.get("asset", ""),
        decision=outcome.get("decision", ""),
        directional_return_bps=f"{directional:.8f}",
        base_spread_bps=f"{base_spread:.8f}",
        quote_spread_bps=f"{quote_spread:.8f}",
        base_depth_10bps_notional=_format_float(base_context.get("near_depth_10bps_notional")),
        quote_depth_10bps_notional=_format_float(quote_context.get("near_depth_10bps_notional")),
        base_annualized_funding=_format_float(base_context.get("annualized_funding")),
        quote_annualized_funding=_format_float(quote_context.get("annualized_funding")),
        estimated_funding_1h_bps=f"{funding_1h:.8f}",
        estimated_round_trip_cost_bps=f"{round_trip_cost:.8f}",
        estimated_net_after_cost_bps=f"{net:.8f}",
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _pair_funding_1h_bps(*, decision: str, base_annualized: float, quote_annualized: float) -> float:
    base_1h = base_annualized * 10_000.0 / (365.0 * 24.0)
    quote_1h = quote_annualized * 10_000.0 / (365.0 * 24.0)
    if decision == "paper_long":
        return base_1h - quote_1h
    if decision == "paper_short":
        return -base_1h + quote_1h
    return 0.0


def _risk_action(
    *,
    net_bps: float,
    base_context: dict[str, str],
    quote_context: dict[str, str],
) -> tuple[str, str, str]:
    if not base_context or not quote_context:
        return "missing_pair_execution_context", "one or both pair legs have no execution context", "repair pair-leg context"
    if net_bps > 5.0:
        return (
            "cost_adjusted_pair_probe",
            "pair mark win survives conservative two-leg taker cost",
            "repeat with stop/adverse-excursion and hedge execution notes",
        )
    return (
        "pair_cost_adjusted_edge_failed",
        "pair mark win does not survive conservative two-leg taker cost",
        "do not promote until maker/low-fee execution or larger edge is observed",
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
    number = _float(value)
    return "" if not value else f"{number:.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_outcomes.csv")
    parser.add_argument("--tickets-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_fill_risk_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=LOCAL_ROOT / "current_crypto_pair_spread_fill_risk_check.md")
    args = parser.parse_args()

    rows = build_pair_spread_fill_risk_checks(outcomes_path=args.outcomes_path, tickets_path=args.tickets_path)
    write_pair_spread_fill_risk_checks_csv(rows, output_path=args.output_path)
    write_pair_spread_fill_risk_checks_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.estimated_net_after_cost_bps, row.risk_action)


if __name__ == "__main__":
    main()
