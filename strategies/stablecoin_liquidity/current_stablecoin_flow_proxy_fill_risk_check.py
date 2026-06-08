from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class StablecoinFlowProxyFillRiskCheck:
    ticket_id: str
    asset: str
    decision: str
    notional_usd: float
    directional_return_bps: float
    spread_bps: float
    annualized_funding: float
    estimated_funding_bps: float
    round_trip_cost_bps: float
    estimated_net_bps: float
    near_depth_10bps_notional: float
    visible_depth_usage: float
    max_adverse_excursion_bps: float
    max_favorable_excursion_bps: float
    stop_50bps_survived: str
    risk_action: str
    reason: str
    next_step: str


def build_stablecoin_flow_proxy_fill_risk_checks(
    *,
    outcomes_path: Path = ROOT / "current_stablecoin_flow_proxy_outcomes.csv",
    okx_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    notional_usd: float = 1_000.0,
    taker_fee_bps_per_fill: float = 5.0,
) -> tuple[StablecoinFlowProxyFillRiskCheck, ...]:
    context_by_asset = {row.get("asset", ""): row for row in _read_rows(okx_context_path)}
    return tuple(
        sorted(
            (
                _build_check(
                    row=row,
                    context=context_by_asset.get(row.get("asset", ""), {}),
                    notional_usd=notional_usd,
                    taker_fee_bps_per_fill=taker_fee_bps_per_fill,
                )
                for row in _read_rows(outcomes_path)
                if row.get("checkpoint_status") == "ready"
            ),
            key=lambda row: row.estimated_net_bps,
            reverse=True,
        )
    )


def write_stablecoin_flow_proxy_fill_risk_check_csv(
    rows: tuple[StablecoinFlowProxyFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(StablecoinFlowProxyFillRiskCheck.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.asset,
                    row.decision,
                    f"{row.notional_usd:.2f}",
                    f"{row.directional_return_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.estimated_funding_bps:.8f}",
                    f"{row.round_trip_cost_bps:.8f}",
                    f"{row.estimated_net_bps:.8f}",
                    f"{row.near_depth_10bps_notional:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    f"{row.max_adverse_excursion_bps:.8f}",
                    f"{row.max_favorable_excursion_bps:.8f}",
                    row.stop_50bps_survived,
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_stablecoin_flow_proxy_fill_risk_check_md(
    rows: tuple[StablecoinFlowProxyFillRiskCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Stablecoin Flow Proxy Fill Risk Check\n\n")
        handle.write(
            "This checks stablecoin chain-liquidity proxy mark outcomes against rough OKX "
            "spread, taker fee, funding, visible depth, and adverse-excursion assumptions. "
            "It is not a live fill report.\n\n"
        )
        handle.write(
            "| ticket | asset | decision | notional | dir bps | cost bps | funding bps | net bps | "
            "depth 10bps | usage | MAE bps | MFE bps | stop 50 | action | reason |\n"
        )
        handle.write(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
        )
        for row in rows:
            handle.write(
                f"| {row.ticket_id} | {row.asset} | {row.decision} | {row.notional_usd:.0f} | "
                f"{row.directional_return_bps:.2f} | {row.round_trip_cost_bps:.2f} | "
                f"{row.estimated_funding_bps:.2f} | {row.estimated_net_bps:.2f} | "
                f"{row.near_depth_10bps_notional:.0f} | {row.visible_depth_usage:.4f} | "
                f"{row.max_adverse_excursion_bps:.2f} | {row.max_favorable_excursion_bps:.2f} | "
                f"{row.stop_50bps_survived} | {row.risk_action} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_check(
    *,
    row: dict[str, str],
    context: dict[str, str],
    notional_usd: float,
    taker_fee_bps_per_fill: float,
) -> StablecoinFlowProxyFillRiskCheck:
    decision = row.get("decision", "")
    directional_bps = _float(row.get("directional_return_bps"))
    annualized_funding = _float(context.get("annualized_funding"))
    funding_bps = _funding_bps(
        annualized_funding=annualized_funding,
        decision=decision,
        elapsed_minutes=_float(row.get("elapsed_minutes")),
    )
    spread_bps = _float(context.get("spread_bps"))
    round_trip_cost = spread_bps + (2.0 * taker_fee_bps_per_fill)
    net_bps = directional_bps - round_trip_cost + funding_bps
    depth = _float(context.get("near_depth_10bps_notional"))
    usage = notional_usd / depth if depth > 0.0 else 0.0
    adverse, favorable = _excursions(
        asset=row.get("asset", ""),
        decision=decision,
        entry_mark=_float(row.get("entry_mark")),
        opened_at=_parse_datetime(row.get("opened_at", "")),
        checked_at=_parse_datetime(row.get("checked_at", "")),
    )
    risk_action, reason, next_step = _risk_action(
        net_bps=net_bps,
        visible_depth_usage=usage,
        max_adverse_excursion_bps=adverse,
        context=context,
    )
    return StablecoinFlowProxyFillRiskCheck(
        ticket_id=row.get("ticket_id", ""),
        asset=row.get("asset", ""),
        decision=decision,
        notional_usd=notional_usd,
        directional_return_bps=directional_bps,
        spread_bps=spread_bps,
        annualized_funding=annualized_funding,
        estimated_funding_bps=funding_bps,
        round_trip_cost_bps=round_trip_cost,
        estimated_net_bps=net_bps,
        near_depth_10bps_notional=depth,
        visible_depth_usage=usage,
        max_adverse_excursion_bps=adverse,
        max_favorable_excursion_bps=favorable,
        stop_50bps_survived="yes" if adverse >= -50.0 else "no",
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _funding_bps(*, annualized_funding: float, decision: str, elapsed_minutes: float) -> float:
    long_funding = -annualized_funding * (elapsed_minutes / (365.0 * 24.0 * 60.0)) * 10_000.0
    if decision == "paper_short":
        return -long_funding
    return long_funding


def _excursions(
    *,
    asset: str,
    decision: str,
    entry_mark: float,
    opened_at: datetime,
    checked_at: datetime,
) -> tuple[float, float]:
    if entry_mark <= 0.0 or not asset:
        return 0.0, 0.0
    direction = -1 if decision == "paper_short" else 1
    candles = _fetch_okx_candles(asset=asset, start=opened_at, end=checked_at)
    path = tuple(((candle["close"] / entry_mark) - 1.0) * direction * 10_000.0 for candle in candles)
    if not path:
        return 0.0, 0.0
    return min(path), max(path)


def _fetch_okx_candles(*, asset: str, start: datetime, end: datetime) -> tuple[dict[str, float], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/candles",
        params={"instId": f"{asset}-USDT-SWAP", "bar": "1m", "limit": "300"},
        timeout=30,
    )
    response.raise_for_status()
    start_ms = (start - timedelta(minutes=2)).timestamp() * 1000.0
    end_ms = (end + timedelta(minutes=2)).timestamp() * 1000.0
    rows = tuple(
        {
            "timestamp": float(item[0]),
            "close": float(item[4]),
        }
        for item in response.json().get("data", ())
    )
    return tuple(
        sorted(
            (row for row in rows if start_ms <= row["timestamp"] <= end_ms),
            key=lambda row: row["timestamp"],
        )
    )


def _risk_action(
    *,
    net_bps: float,
    visible_depth_usage: float,
    max_adverse_excursion_bps: float,
    context: dict[str, str],
) -> tuple[str, str, str]:
    if not context:
        return (
            "missing_execution_context",
            "no current OKX spread, funding, and depth context",
            "refresh OKX execution context before treating this as deployable",
        )
    if visible_depth_usage > 0.10:
        return (
            "depth_too_thin_for_1k_probe",
            "1k notional consumes more than 10% of visible 10bps depth",
            "reduce notional or wait for deeper book before repeating",
        )
    if net_bps <= 0.0:
        return (
            "cost_adjusted_edge_failed",
            "mark edge does not survive rough spread, taker-fee, and funding haircut",
            "do not repeat unless the signal refreshes with a larger edge",
        )
    if max_adverse_excursion_bps < -50.0:
        return (
            "stop_risk_blocks_probe",
            "mark edge survived cost but would have breached a rough 50bps adverse-excursion stop",
            "repeat only with an explicit stop or better entry timing",
        )
    return (
        "cost_adjusted_probe_survived",
        "mark edge survives rough cost, funding, visible-depth, and 50bps adverse-excursion checks",
        "repeat with fresh chain-flow evidence and record the same risk fields",
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
    parser.add_argument("--outcomes-path", type=Path, default=ROOT / "current_stablecoin_flow_proxy_outcomes.csv")
    parser.add_argument(
        "--okx-context-path",
        type=Path,
        default=STRATEGIES_ROOT / "candidate_validation" / "current_followup_okx_execution_context.csv",
    )
    parser.add_argument("--notional-usd", type=float, default=1_000.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_stablecoin_flow_proxy_fill_risk_check.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_stablecoin_flow_proxy_fill_risk_check.md",
    )
    args = parser.parse_args()

    rows = build_stablecoin_flow_proxy_fill_risk_checks(
        outcomes_path=args.outcomes_path,
        okx_context_path=args.okx_context_path,
        notional_usd=args.notional_usd,
    )
    write_stablecoin_flow_proxy_fill_risk_check_csv(rows, output_path=args.output_path)
    write_stablecoin_flow_proxy_fill_risk_check_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.risk_action, row.ticket_id, row.asset, f"net={row.estimated_net_bps:.4f}")


if __name__ == "__main__":
    main()
