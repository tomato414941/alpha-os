from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class VolumePriceDislocationFillRiskCheck:
    ticket_id: str
    symbol: str
    decision: str
    notional_usd: float
    directional_return_bps: float
    spread_bps: float
    annualized_funding: float
    estimated_funding_bps: float
    visible_depth_usage: float
    visible_depth_impact_bps: float
    round_trip_cost_bps: float
    estimated_net_after_cost_bps: float
    near_depth_10bps_notional: float
    max_adverse_excursion_bps: float
    max_favorable_excursion_bps: float
    stop_50bps_survived: str
    risk_action: str
    reason: str
    next_step: str


def build_volume_price_dislocation_fill_risk_checks(
    *,
    outcomes_path: Path = ROOT / "current_volume_price_dislocation_outcomes.csv",
    execution_gate_path: Path = ROOT / "current_volume_price_dislocation_execution_gate.csv",
    notional_usd: float = 250.0,
    taker_fee_bps_per_fill: float = 4.0,
) -> tuple[VolumePriceDislocationFillRiskCheck, ...]:
    context = {row.get("symbol", ""): row for row in _read_rows(execution_gate_path)}
    rows = tuple(
        _build_check(
            outcome=row,
            context=context.get(row.get("symbol", ""), {}),
            notional_usd=notional_usd,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for row in _read_rows(outcomes_path)
        if row.get("checkpoint_status") == "ready" and row.get("outcome") == "paper_mark_win"
    )
    return tuple(sorted(rows, key=lambda row: row.estimated_net_after_cost_bps, reverse=True))


def write_volume_price_dislocation_fill_risk_checks_csv(
    rows: tuple[VolumePriceDislocationFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(VolumePriceDislocationFillRiskCheck.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.symbol,
                    row.decision,
                    f"{row.notional_usd:.2f}",
                    f"{row.directional_return_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.estimated_funding_bps:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    f"{row.visible_depth_impact_bps:.8f}",
                    f"{row.round_trip_cost_bps:.8f}",
                    f"{row.estimated_net_after_cost_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    f"{row.max_adverse_excursion_bps:.8f}",
                    f"{row.max_favorable_excursion_bps:.8f}",
                    row.stop_50bps_survived,
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_volume_price_dislocation_fill_risk_checks_md(
    rows: tuple[VolumePriceDislocationFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volume Price Dislocation Fill Risk Check\n\n")
        handle.write(
            "This checks market-breadth mark wins against rough Hyperliquid spread, "
            "taker fee, funding, visible depth, and 1m candle path risk. It is not a live fill report.\n\n"
        )
        handle.write(
            "| ticket | symbol | dir bps | cost bps | funding bps | net bps | depth 10bps | "
            "usage | MAE | MFE | stop50 | action | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.symbol} | {row.directional_return_bps:.2f} | "
                f"{row.round_trip_cost_bps:.2f} | {row.estimated_funding_bps:.2f} | "
                f"{row.estimated_net_after_cost_bps:.2f} | {row.near_depth_10bps_notional:.0f} | "
                f"{row.visible_depth_usage:.4f} | {row.max_adverse_excursion_bps:.2f} | "
                f"{row.max_favorable_excursion_bps:.2f} | {row.stop_50bps_survived} | "
                f"{row.risk_action} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_check(
    *,
    outcome: dict[str, str],
    context: dict[str, str],
    notional_usd: float,
    taker_fee_bps_per_fill: float,
) -> VolumePriceDislocationFillRiskCheck:
    symbol = outcome.get("symbol", "")
    decision = outcome.get("decision", "")
    directional_bps = _float(outcome.get("directional_return_bps"))
    spread_bps = _float(context.get("spread_bps"))
    annualized_funding = _float(context.get("annualized_funding"))
    funding_bps = _funding_bps(
        annualized_funding=annualized_funding,
        decision=decision,
        elapsed_minutes=_float(outcome.get("elapsed_minutes")),
    )
    depth = _float(context.get("near_depth_10bps_notional"))
    usage = notional_usd / depth if depth > 0.0 else 1_000_000.0
    visible_depth_impact = min(usage, 1.0) * 10.0
    round_trip_cost = spread_bps + (2.0 * taker_fee_bps_per_fill) + visible_depth_impact
    net_bps = directional_bps - round_trip_cost + funding_bps
    adverse, favorable = _path_excursions(
        symbol=symbol,
        decision=decision,
        entry_mark=_float(outcome.get("entry_mark")),
        opened_at=_parse_datetime(outcome.get("opened_at", "")),
        checked_at=_parse_datetime(outcome.get("checked_at", "")),
    )
    risk_action, reason, next_step = _risk_action(
        symbol=symbol,
        net_bps=net_bps,
        visible_depth_usage=usage,
        max_adverse_excursion_bps=adverse,
        context=context,
    )
    return VolumePriceDislocationFillRiskCheck(
        ticket_id=outcome.get("ticket_id", ""),
        symbol=symbol,
        decision=decision,
        notional_usd=notional_usd,
        directional_return_bps=directional_bps,
        spread_bps=spread_bps,
        annualized_funding=annualized_funding,
        estimated_funding_bps=funding_bps,
        visible_depth_usage=usage,
        visible_depth_impact_bps=visible_depth_impact,
        round_trip_cost_bps=round_trip_cost,
        estimated_net_after_cost_bps=net_bps,
        near_depth_10bps_notional=depth,
        max_adverse_excursion_bps=adverse,
        max_favorable_excursion_bps=favorable,
        stop_50bps_survived="yes" if adverse > -50.0 else "no",
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _funding_bps(*, annualized_funding: float, decision: str, elapsed_minutes: float) -> float:
    long_funding = -annualized_funding * (elapsed_minutes / (365.0 * 24.0 * 60.0)) * 10_000.0
    if decision == "paper_short":
        return -long_funding
    return long_funding


def _path_excursions(
    *,
    symbol: str,
    decision: str,
    entry_mark: float,
    opened_at: datetime,
    checked_at: datetime,
) -> tuple[float, float]:
    if entry_mark <= 0.0 or not symbol:
        return 0.0, 0.0
    candles = _fetch_hyperliquid_candles(symbol=symbol, start=opened_at, end=checked_at)
    if not candles:
        return 0.0, 0.0
    highs = tuple(_float(candle.get("h")) for candle in candles)
    lows = tuple(_float(candle.get("l")) for candle in candles)
    if decision == "paper_short":
        favorable = (entry_mark / min(lows) - 1.0) * 10_000.0 if lows and min(lows) > 0.0 else 0.0
        adverse = (entry_mark / max(highs) - 1.0) * 10_000.0 if highs and max(highs) > 0.0 else 0.0
    else:
        favorable = (max(highs) / entry_mark - 1.0) * 10_000.0 if highs else 0.0
        adverse = (min(lows) / entry_mark - 1.0) * 10_000.0 if lows else 0.0
    return min(adverse, 0.0), max(favorable, 0.0)


def _fetch_hyperliquid_candles(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, str], ...]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": "1m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple(response.json())


def _risk_action(
    *,
    symbol: str,
    net_bps: float,
    visible_depth_usage: float,
    max_adverse_excursion_bps: float,
    context: dict[str, str],
) -> tuple[str, str, str]:
    if not context:
        return (
            "missing_execution_context",
            "no current spread, funding, and depth context",
            f"refresh {symbol} execution context before treating this as deployable",
        )
    if visible_depth_usage > 0.10:
        return (
            "depth_too_thin_for_250_probe",
            "250 USD notional consumes more than 10% of visible 10bps depth",
            f"reduce {symbol} notional or wait for deeper book before repeating",
        )
    if net_bps <= 0.0:
        return (
            "cost_adjusted_edge_failed",
            "mark edge does not survive rough spread, taker-fee, funding, and visible-depth impact",
            f"do not promote {symbol} unless a fresh trigger survives costs",
        )
    if max_adverse_excursion_bps <= -50.0:
        return (
            "stop_risk_blocks_probe",
            "mark edge survived cost but would have breached a rough 50bps adverse-excursion stop",
            f"repeat {symbol} only with explicit stop sizing or better entry timing",
        )
    return (
        "cost_adjusted_probe_survived",
        "mark edge survives rough cost, funding, visible-depth, and 50bps adverse-excursion checks",
        f"repeat {symbol} with fresh dislocation evidence and record the same risk fields",
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_volume_price_dislocation_outcomes.csv")
    parser.add_argument(
        "--execution-gate-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_execution_gate.csv",
    )
    parser.add_argument("--notional-usd", type=float, default=250.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_fill_risk_check.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_fill_risk_check.md",
    )
    args = parser.parse_args()

    rows = build_volume_price_dislocation_fill_risk_checks(
        outcomes_path=args.outcomes_path,
        execution_gate_path=args.execution_gate_path,
        notional_usd=args.notional_usd,
    )
    write_volume_price_dislocation_fill_risk_checks_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_fill_risk_checks_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.risk_action, row.symbol, f"net={row.estimated_net_after_cost_bps:.4f}")


if __name__ == "__main__":
    main()
