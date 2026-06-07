from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CandidateGate:
    asset: str
    action: str
    best_mode: str
    horizon: str
    fee_bps_per_fill_per_venue: Decimal
    fee_headroom_bps: Decimal
    capacity: Decimal
    both_touch_rate: Decimal
    okx_touch_rate: Decimal
    hl_touch_rate: Decimal
    reason: str


def build_candidate_gates(
    *,
    fee_ceiling_path: Path = ROOT / "okx_hl_fee_ceiling.csv",
    fee_bps_per_fill_per_venue: Decimal = Decimal("0.25"),
    min_capacity: Decimal = Decimal("50000"),
    min_both_touch_rate: Decimal = Decimal("0.2"),
    min_maker_touch_rate: Decimal = Decimal("0.2"),
) -> tuple[CandidateGate, ...]:
    rows = _read_rows(fee_ceiling_path)
    by_asset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_asset.setdefault(row["asset"], []).append(row)
    gates = tuple(
        _gate_asset(
            asset=asset,
            rows=tuple(asset_rows),
            fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
            min_capacity=min_capacity,
            min_both_touch_rate=min_both_touch_rate,
            min_maker_touch_rate=min_maker_touch_rate,
        )
        for asset, asset_rows in by_asset.items()
    )
    return tuple(
        sorted(
            gates,
            key=lambda gate: (
                _action_rank(gate.action),
                gate.horizon == "8h",
                gate.fee_headroom_bps,
                gate.capacity,
            ),
            reverse=True,
        )
    )


def write_candidate_gates_csv(
    gates: tuple[CandidateGate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "action",
                "best_mode",
                "horizon",
                "fee_bps_per_fill_per_venue",
                "fee_headroom_bps",
                "capacity",
                "both_touch_rate",
                "okx_touch_rate",
                "hl_touch_rate",
                "reason",
            )
        )
        for gate in gates:
            writer.writerow(
                (
                    gate.asset,
                    gate.action,
                    gate.best_mode,
                    gate.horizon,
                    _fmt(gate.fee_bps_per_fill_per_venue),
                    _fmt(gate.fee_headroom_bps),
                    _fmt(gate.capacity),
                    _fmt(gate.both_touch_rate),
                    _fmt(gate.okx_touch_rate),
                    _fmt(gate.hl_touch_rate),
                    gate.reason,
                )
            )
    return output_path


def write_candidate_gates_md(
    gates: tuple[CandidateGate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Promotion Gate\n\n")
        handle.write(
            "This is a research gate, not a trade instruction. It combines fee ceiling, "
            "maker-touch proxy, and capacity so maker-only false positives do not rank "
            "above executable candidates.\n\n"
        )
        handle.write(
            "| asset | action | mode | horizon | fee bps/fill/venue | headroom bps | capacity | both touch | OKX touch | HL touch | reason |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for gate in gates:
            handle.write(
                "| "
                f"{gate.asset} | "
                f"{gate.action} | "
                f"{gate.best_mode} | "
                f"{gate.horizon} | "
                f"{_fmt(gate.fee_bps_per_fill_per_venue)} | "
                f"{_fmt(gate.fee_headroom_bps)} | "
                f"{_fmt(gate.capacity)} | "
                f"{_fmt(gate.both_touch_rate)} | "
                f"{_fmt(gate.okx_touch_rate)} | "
                f"{_fmt(gate.hl_touch_rate)} | "
                f"{gate.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_*` means the current public-book proxy leaves fee headroom under "
            "the configured fee. `execution_watch` means the raw edge survives only "
            "through a mode whose maker leg did not pass the touch gate. "
            "`capacity_watch` means the edge may survive but size is too small for the "
            "configured threshold.\n"
        )
    return output_path


def _gate_asset(
    *,
    asset: str,
    rows: tuple[dict[str, str], ...],
    fee_bps_per_fill_per_venue: Decimal,
    min_capacity: Decimal,
    min_both_touch_rate: Decimal,
    min_maker_touch_rate: Decimal,
) -> CandidateGate:
    capacity = max(Decimal(row["capacity"]) for row in rows)
    feasible_rows = tuple(
        row
        for row in rows
        if _best_horizon(row, fee_bps_per_fill_per_venue) != "blocked"
        and _maker_gate_passes(
            row=row,
            min_both_touch_rate=min_both_touch_rate,
            min_maker_touch_rate=min_maker_touch_rate,
        )
    )
    best_raw_row = _best_row(rows=rows, fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue)
    if feasible_rows:
        best = _best_row(
            rows=feasible_rows,
            fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
        )
        horizon = _best_horizon(best, fee_bps_per_fill_per_venue)
        action = "paper_8h_candidate" if horizon == "8h" else "paper_24h_candidate"
        if capacity < min_capacity:
            action = "capacity_watch"
        return _candidate_gate(
            row=best,
            action=action,
            horizon=horizon,
            fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
            reason=_success_reason(row=best, capacity=capacity, min_capacity=min_capacity),
        )
    if _best_horizon(best_raw_row, fee_bps_per_fill_per_venue) != "blocked":
        return _candidate_gate(
            row=best_raw_row,
            action="execution_watch",
            horizon=_best_horizon(best_raw_row, fee_bps_per_fill_per_venue),
            fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
            reason="edge survives fees but maker-touch gate blocks the best mode",
        )
    return _candidate_gate(
        row=best_raw_row,
        action="drop_current",
        horizon="blocked",
        fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
        reason="no execution mode survives the configured fee ceiling",
    )


def _candidate_gate(
    *,
    row: dict[str, str],
    action: str,
    horizon: str,
    fee_bps_per_fill_per_venue: Decimal,
    reason: str,
) -> CandidateGate:
    both_touch, okx_touch, hl_touch = _touch_rates(row)
    return CandidateGate(
        asset=row["asset"],
        action=action,
        best_mode=row["execution_mode"],
        horizon=horizon,
        fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
        fee_headroom_bps=_headroom(
            row=row,
            horizon=horizon,
            fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
        ),
        capacity=Decimal(row["capacity"]),
        both_touch_rate=both_touch,
        okx_touch_rate=okx_touch,
        hl_touch_rate=hl_touch,
        reason=reason,
    )


def _success_reason(
    *,
    row: dict[str, str],
    capacity: Decimal,
    min_capacity: Decimal,
) -> str:
    if capacity < min_capacity:
        return "edge survives fees and execution gate, but capacity is below threshold"
    if row["execution_mode"] == "both_maker":
        return "maker-maker mode survives fees and same-window touch gate"
    if row["execution_mode"] == "both_cross":
        return "both-cross mode survives fees without maker-touch dependency"
    return "one-leg-cross mode survives fees and maker-leg touch gate"


def _best_row(
    *,
    rows: tuple[dict[str, str], ...],
    fee_bps_per_fill_per_venue: Decimal,
) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            _best_horizon(row, fee_bps_per_fill_per_venue) == "8h",
            _best_horizon(row, fee_bps_per_fill_per_venue) == "24h",
            _headroom(
                row=row,
                horizon=_best_horizon(row, fee_bps_per_fill_per_venue),
                fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
            ),
            Decimal(row["capacity"]),
        ),
    )


def _best_horizon(row: dict[str, str], fee_bps_per_fill_per_venue: Decimal) -> str:
    if Decimal(row["equal_venue_fee_8h_bps_per_fill"]) >= fee_bps_per_fill_per_venue:
        return "8h"
    if Decimal(row["equal_venue_fee_24h_bps_per_fill"]) >= fee_bps_per_fill_per_venue:
        return "24h"
    return "blocked"


def _headroom(
    *,
    row: dict[str, str],
    horizon: str,
    fee_bps_per_fill_per_venue: Decimal,
) -> Decimal:
    if horizon == "8h":
        ceiling = Decimal(row["equal_venue_fee_8h_bps_per_fill"])
    elif horizon == "24h":
        ceiling = Decimal(row["equal_venue_fee_24h_bps_per_fill"])
    else:
        ceiling = max(
            Decimal(row["equal_venue_fee_8h_bps_per_fill"]),
            Decimal(row["equal_venue_fee_24h_bps_per_fill"]),
        )
    return ceiling - fee_bps_per_fill_per_venue


def _maker_gate_passes(
    *,
    row: dict[str, str],
    min_both_touch_rate: Decimal,
    min_maker_touch_rate: Decimal,
) -> bool:
    both_touch, okx_touch, hl_touch = _touch_rates(row)
    mode = row["execution_mode"]
    if mode == "both_maker":
        return both_touch >= min_both_touch_rate
    if mode == "okx_cross_hl_maker":
        return hl_touch >= min_maker_touch_rate
    if mode == "okx_maker_hl_cross":
        return okx_touch >= min_maker_touch_rate
    if mode == "both_cross":
        return True
    raise RuntimeError(f"unknown execution mode: {mode}")


def _touch_rates(row: dict[str, str]) -> tuple[Decimal, Decimal, Decimal]:
    both_touch = _decimal_or_zero(row["both_touch_rate"])
    okx_only = _decimal_or_zero(row["okx_only_touch_rate"])
    hl_only = _decimal_or_zero(row["hl_only_touch_rate"])
    return both_touch, both_touch + okx_only, both_touch + hl_only


def _action_rank(action: str) -> int:
    return {
        "paper_8h_candidate": 5,
        "paper_24h_candidate": 4,
        "capacity_watch": 3,
        "execution_watch": 2,
        "drop_current": 1,
    }[action]


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _decimal_or_zero(value: str) -> Decimal:
    return Decimal(value) if value else Decimal("0")


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fee-ceiling-path",
        type=Path,
        default=ROOT / "okx_hl_fee_ceiling.csv",
    )
    parser.add_argument(
        "--fee-bps-per-fill-per-venue",
        type=Decimal,
        default=Decimal("0.25"),
    )
    parser.add_argument("--min-capacity", type=Decimal, default=Decimal("50000"))
    parser.add_argument("--min-both-touch-rate", type=Decimal, default=Decimal("0.2"))
    parser.add_argument("--min-maker-touch-rate", type=Decimal, default=Decimal("0.2"))
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_promotion_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_promotion_gate.md",
    )
    args = parser.parse_args()

    gates = build_candidate_gates(
        fee_ceiling_path=args.fee_ceiling_path,
        fee_bps_per_fill_per_venue=args.fee_bps_per_fill_per_venue,
        min_capacity=args.min_capacity,
        min_both_touch_rate=args.min_both_touch_rate,
        min_maker_touch_rate=args.min_maker_touch_rate,
    )
    write_candidate_gates_csv(gates, output_path=args.csv_output_path)
    write_candidate_gates_md(gates, output_path=args.md_output_path)
    for gate in gates:
        print(
            gate.asset,
            gate.action,
            gate.best_mode,
            gate.horizon,
            f"headroom_bps={_fmt(gate.fee_headroom_bps)}",
            gate.reason,
        )


if __name__ == "__main__":
    main()
