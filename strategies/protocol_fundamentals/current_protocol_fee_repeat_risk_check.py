from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeeRepeatRiskCheckRow:
    token_symbol: str
    protocol: str
    actionability_status: str
    paper_notional_usd: float
    mean_directional_4h_bps: float
    spread_bps: float
    funding_4h_bps: float
    round_trip_cost_bps: float
    net_mean_directional_4h_bps: float
    depth_10bps: float
    visible_depth_usage: float
    wins_4h: int
    labeled_4h: int
    risk_action: str
    reason: str
    next_step: str


def build_protocol_fee_repeat_risk_check_rows(
    *,
    actionability_path: Path = ROOT / "current_protocol_fee_actionability.csv",
    paper_notional_usd: float = 1_000.0,
    taker_fee_bps_per_fill: float = 5.0,
) -> tuple[ProtocolFeeRepeatRiskCheckRow, ...]:
    rows = tuple(
        _build_row(
            row=row,
            paper_notional_usd=paper_notional_usd,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for row in _read_rows(actionability_path)
        if row.get("action") in {"repeat_paper_probe", "refresh_execution_gate"}
    )
    return tuple(sorted(rows, key=lambda row: row.net_mean_directional_4h_bps, reverse=True))


def write_protocol_fee_repeat_risk_check_csv(
    rows: tuple[ProtocolFeeRepeatRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(ProtocolFeeRepeatRiskCheckRow.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.token_symbol,
                    row.protocol,
                    row.actionability_status,
                    f"{row.paper_notional_usd:.2f}",
                    f"{row.mean_directional_4h_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.funding_4h_bps:.8f}",
                    f"{row.round_trip_cost_bps:.8f}",
                    f"{row.net_mean_directional_4h_bps:.8f}",
                    f"{row.depth_10bps:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    row.wins_4h,
                    row.labeled_4h,
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_repeat_risk_check_md(
    rows: tuple[ProtocolFeeRepeatRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Repeat Risk Check\n\n")
        handle.write(
            "This checks protocol-fee repeat candidates against rough spread, taker fee, funding, "
            "and visible depth. It is not a live order list.\n\n"
        )
        handle.write(
            "| token | action | net 4h bps | mean 4h bps | cost bps | funding bps | depth | usage | labels | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.token_symbol}/{row.protocol} | {row.risk_action} | "
                f"{row.net_mean_directional_4h_bps:.2f} | {row.mean_directional_4h_bps:.2f} | "
                f"{row.round_trip_cost_bps:.2f} | {row.funding_4h_bps:.2f} | "
                f"{row.depth_10bps:.0f} | {row.visible_depth_usage:.4f} | "
                f"{row.wins_4h}/{row.labeled_4h} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    paper_notional_usd: float,
    taker_fee_bps_per_fill: float,
) -> ProtocolFeeRepeatRiskCheckRow:
    mean_4h_bps = _float(row.get("mean_directional_4h")) * 10_000.0
    spread_bps = _float(row.get("spread_bps"))
    round_trip_cost_bps = spread_bps + 2.0 * taker_fee_bps_per_fill
    funding_4h_bps = -_float(row.get("hl_annualized_funding")) * (4.0 / (365.0 * 24.0)) * 10_000.0
    net_bps = mean_4h_bps - round_trip_cost_bps + funding_4h_bps
    depth = _float(row.get("depth_10bps"))
    usage = paper_notional_usd / depth if depth > 0.0 else 0.0
    action, reason, next_step = _risk_action(
        token=row.get("token_symbol", ""),
        protocol=row.get("protocol", ""),
        source_action=row.get("action", ""),
        net_bps=net_bps,
        visible_depth_usage=usage,
        wins_4h=_int(row.get("wins_4h")),
        labeled_4h=_int(row.get("labeled_4h")),
    )
    return ProtocolFeeRepeatRiskCheckRow(
        token_symbol=row.get("token_symbol", ""),
        protocol=row.get("protocol", ""),
        actionability_status=row.get("status", ""),
        paper_notional_usd=paper_notional_usd,
        mean_directional_4h_bps=mean_4h_bps,
        spread_bps=spread_bps,
        funding_4h_bps=funding_4h_bps,
        round_trip_cost_bps=round_trip_cost_bps,
        net_mean_directional_4h_bps=net_bps,
        depth_10bps=depth,
        visible_depth_usage=usage,
        wins_4h=_int(row.get("wins_4h")),
        labeled_4h=_int(row.get("labeled_4h")),
        risk_action=action,
        reason=reason,
        next_step=next_step,
    )


def _risk_action(
    *,
    token: str,
    protocol: str,
    source_action: str,
    net_bps: float,
    visible_depth_usage: float,
    wins_4h: int,
    labeled_4h: int,
) -> tuple[str, str, str]:
    subject = f"{token}/{protocol}"
    if source_action != "repeat_paper_probe":
        return (
            "refresh_before_repeat",
            "label support exists but actionability has not promoted this to repeat yet",
            f"refresh execution context and require another positive label before repeating {subject}",
        )
    if visible_depth_usage > 0.10:
        return (
            "depth_too_thin_for_1k_repeat",
            "1k notional uses more than 10% of visible 10bps depth",
            f"reduce notional or wait for deeper book before repeating {subject}",
        )
    if wins_4h < 2 or labeled_4h < 2:
        return (
            "not_enough_repeat_labels",
            "repeat evidence is still too thin",
            f"wait for more 4h labels before repeating {subject}",
        )
    if net_bps <= 0.0:
        return (
            "cost_adjusted_repeat_failed",
            "mean 4h label does not survive rough cost and funding haircut",
            f"do not repeat {subject} until label strength or execution improves",
        )
    return (
        "cost_adjusted_repeat_probe",
        "repeat label survives rough spread, taker-fee, funding, and visible-depth checks",
        f"open one small repeat paper probe for {subject} and record adverse excursion",
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


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0.0))
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionability-path", type=Path, default=ROOT / "current_protocol_fee_actionability.csv")
    parser.add_argument("--paper-notional-usd", type=float, default=1_000.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_repeat_risk_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_protocol_fee_repeat_risk_check.md")
    args = parser.parse_args()

    rows = build_protocol_fee_repeat_risk_check_rows(
        actionability_path=args.actionability_path,
        paper_notional_usd=args.paper_notional_usd,
    )
    write_protocol_fee_repeat_risk_check_csv(rows, output_path=args.output_path)
    write_protocol_fee_repeat_risk_check_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.risk_action, row.token_symbol, f"net={row.net_mean_directional_4h_bps:.4f}")


if __name__ == "__main__":
    main()
