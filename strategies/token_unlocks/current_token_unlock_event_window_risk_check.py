from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TokenUnlockEventWindowRiskCheckRow:
    ticket_id: str
    asset: str
    decision: str
    outcome: str
    directional_return_bps: float
    spread_bps: float
    funding_bps: float
    round_trip_cost_bps: float
    net_directional_bps: float
    annualized_funding: float
    impact_spread: float
    risk_action: str
    reason: str
    next_step: str


def build_token_unlock_event_window_risk_check_rows(
    *,
    outcomes_path: Path = ROOT / "current_token_unlock_event_window_outcomes.csv",
    actionability_path: Path = ROOT / "current_token_unlock_actionability.csv",
    taker_fee_bps_per_fill: float = 5.0,
) -> tuple[TokenUnlockEventWindowRiskCheckRow, ...]:
    actionability = {row.get("symbol", ""): row for row in _read_rows(actionability_path)}
    rows = tuple(
        _build_row(
            outcome=row,
            actionability=actionability.get(row.get("asset", ""), {}),
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for row in _read_rows(outcomes_path)
        if row.get("checkpoint_status") == "ready"
    )
    return tuple(sorted(rows, key=lambda row: row.net_directional_bps, reverse=True))


def write_token_unlock_event_window_risk_check_csv(
    rows: tuple[TokenUnlockEventWindowRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(TokenUnlockEventWindowRiskCheckRow.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.asset,
                    row.decision,
                    row.outcome,
                    f"{row.directional_return_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.funding_bps:.8f}",
                    f"{row.round_trip_cost_bps:.8f}",
                    f"{row.net_directional_bps:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_token_unlock_event_window_risk_check_md(
    rows: tuple[TokenUnlockEventWindowRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Event Window Risk Check\n\n")
        handle.write(
            "This checks matured token-unlock event-window labels against rough spread, taker fee, "
            "and funding. It is not a live order list.\n\n"
        )
        handle.write("| ticket | asset | decision | outcome | dir bps | cost bps | funding bps | net bps | action | reason |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.asset} | {row.decision} | {row.outcome} | "
                f"{row.directional_return_bps:.2f} | {row.round_trip_cost_bps:.2f} | "
                f"{row.funding_bps:.2f} | {row.net_directional_bps:.2f} | "
                f"{row.risk_action} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_row(
    *,
    outcome: dict[str, str],
    actionability: dict[str, str],
    taker_fee_bps_per_fill: float,
) -> TokenUnlockEventWindowRiskCheckRow:
    directional_bps = _float(outcome.get("directional_return_bps"))
    impact_spread = _float(actionability.get("impact_spread"))
    spread_bps = impact_spread * 10_000.0
    annualized_funding = _float(actionability.get("annualized_funding"))
    funding_bps = _funding_bps(
        annualized_funding=annualized_funding,
        decision=outcome.get("decision", ""),
        elapsed_minutes=_float(outcome.get("elapsed_minutes")),
    )
    round_trip_cost = spread_bps + 2.0 * taker_fee_bps_per_fill
    net_bps = directional_bps - round_trip_cost + funding_bps
    risk_action, reason, next_step = _risk_action(
        asset=outcome.get("asset", ""),
        outcome=outcome.get("outcome", ""),
        net_bps=net_bps,
        directional_bps=directional_bps,
    )
    return TokenUnlockEventWindowRiskCheckRow(
        ticket_id=outcome.get("ticket_id", ""),
        asset=outcome.get("asset", ""),
        decision=outcome.get("decision", ""),
        outcome=outcome.get("outcome", ""),
        directional_return_bps=directional_bps,
        spread_bps=spread_bps,
        funding_bps=funding_bps,
        round_trip_cost_bps=round_trip_cost,
        net_directional_bps=net_bps,
        annualized_funding=annualized_funding,
        impact_spread=impact_spread,
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _funding_bps(*, annualized_funding: float, decision: str, elapsed_minutes: float) -> float:
    long_funding = -annualized_funding * (elapsed_minutes / (365.0 * 24.0 * 60.0)) * 10_000.0
    if decision == "paper_short":
        return -long_funding
    return long_funding


def _risk_action(*, asset: str, outcome: str, net_bps: float, directional_bps: float) -> tuple[str, str, str]:
    if outcome != "paper_mark_win":
        return (
            "event_window_label_not_supported",
            "the first event-window label did not move in the intended direction",
            f"wait for 1h/4h labels before promoting {asset}",
        )
    if net_bps <= 0.0:
        return (
            "cost_adjusted_event_window_failed",
            "the first directional mark does not survive rough spread, taker-fee, and funding",
            f"do not promote {asset} unless 1h/4h labels produce a larger edge",
        )
    if directional_bps < 20.0:
        return (
            "thin_event_window_support",
            "direction was right, but the first mark is too small to be a strong event-window edge",
            f"keep {asset} on watch until 1h/4h labels confirm a larger move",
        )
    return (
        "cost_adjusted_event_window_probe",
        "the first event-window label survives rough trading costs",
        f"repeat {asset} event-window probe with explicit stop and adverse-excursion notes",
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


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_token_unlock_event_window_outcomes.csv")
    parser.add_argument("--actionability-path", type=Path, default=ROOT / "current_token_unlock_actionability.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_token_unlock_event_window_risk_check.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_token_unlock_event_window_risk_check.md",
    )
    args = parser.parse_args()

    rows = build_token_unlock_event_window_risk_check_rows(
        outcomes_path=args.outcomes_path,
        actionability_path=args.actionability_path,
    )
    write_token_unlock_event_window_risk_check_csv(rows, output_path=args.output_path)
    write_token_unlock_event_window_risk_check_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.risk_action, row.asset, f"net={row.net_directional_bps:.4f}")


if __name__ == "__main__":
    main()
