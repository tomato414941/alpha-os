from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from strategies.cross_exchange_funding.okx_hl_promotion_gate import (
    CandidateGate,
    build_candidate_gates,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FEE_TIERS = (Decimal("0.1"), Decimal("0.25"), Decimal("0.5"), Decimal("1.0"))


@dataclass(frozen=True)
class FeeTierGate:
    fee_bps_per_fill_per_venue: Decimal
    gate: CandidateGate


def build_fee_tier_gates(
    *,
    fee_ceiling_path: Path = ROOT / "okx_hl_fee_ceiling.csv",
    fee_tiers: tuple[Decimal, ...] = DEFAULT_FEE_TIERS,
    min_capacity: Decimal = Decimal("50000"),
    min_both_touch_rate: Decimal = Decimal("0.2"),
    min_maker_touch_rate: Decimal = Decimal("0.2"),
) -> tuple[FeeTierGate, ...]:
    rows: list[FeeTierGate] = []
    for fee_tier in fee_tiers:
        gates = build_candidate_gates(
            fee_ceiling_path=fee_ceiling_path,
            fee_bps_per_fill_per_venue=fee_tier,
            min_capacity=min_capacity,
            min_both_touch_rate=min_both_touch_rate,
            min_maker_touch_rate=min_maker_touch_rate,
        )
        rows.extend(FeeTierGate(fee_bps_per_fill_per_venue=fee_tier, gate=gate) for gate in gates)
    return tuple(rows)


def write_fee_tier_gates_csv(
    rows: tuple[FeeTierGate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "fee_bps_per_fill_per_venue",
                "asset",
                "action",
                "best_mode",
                "horizon",
                "fee_headroom_bps",
                "capacity",
                "both_touch_rate",
                "okx_touch_rate",
                "hl_touch_rate",
                "reason",
            )
        )
        for row in rows:
            gate = row.gate
            writer.writerow(
                (
                    _fmt(row.fee_bps_per_fill_per_venue),
                    gate.asset,
                    gate.action,
                    gate.best_mode,
                    gate.horizon,
                    _fmt(gate.fee_headroom_bps),
                    _fmt(gate.capacity),
                    _fmt(gate.both_touch_rate),
                    _fmt(gate.okx_touch_rate),
                    _fmt(gate.hl_touch_rate),
                    gate.reason,
                )
            )
    return output_path


def write_fee_tier_gates_md(
    rows: tuple[FeeTierGate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Promotion Gate Sensitivity\n\n")
        handle.write(
            "This sweeps account fee assumptions through the promotion gate. It is a "
            "research sensitivity table, not a trade instruction.\n\n"
        )
        handle.write(
            "| fee bps/fill/venue | asset | action | mode | horizon | headroom bps | capacity | both touch | OKX touch | HL touch |\n"
        )
        handle.write(
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            gate = row.gate
            handle.write(
                "| "
                f"{_fmt(row.fee_bps_per_fill_per_venue)} | "
                f"{gate.asset} | "
                f"{gate.action} | "
                f"{gate.best_mode} | "
                f"{gate.horizon} | "
                f"{_fmt(gate.fee_headroom_bps)} | "
                f"{_fmt(gate.capacity)} | "
                f"{_fmt(gate.both_touch_rate)} | "
                f"{_fmt(gate.okx_touch_rate)} | "
                f"{_fmt(gate.hl_touch_rate)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Fee sensitivity separates fee-robust candidates from candidates that only "
            "look alive under maker-only or very-low-fee assumptions. A candidate that "
            "falls from `paper_*` to `execution_watch` still has raw edge, but the "
            "current execution proxy does not support promotion.\n"
        )
    return output_path


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
        "--fee-tiers",
        nargs="+",
        type=Decimal,
        default=DEFAULT_FEE_TIERS,
    )
    parser.add_argument("--min-capacity", type=Decimal, default=Decimal("50000"))
    parser.add_argument("--min-both-touch-rate", type=Decimal, default=Decimal("0.2"))
    parser.add_argument("--min-maker-touch-rate", type=Decimal, default=Decimal("0.2"))
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_promotion_gate_sensitivity.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_promotion_gate_sensitivity.md",
    )
    args = parser.parse_args()

    rows = build_fee_tier_gates(
        fee_ceiling_path=args.fee_ceiling_path,
        fee_tiers=tuple(args.fee_tiers),
        min_capacity=args.min_capacity,
        min_both_touch_rate=args.min_both_touch_rate,
        min_maker_touch_rate=args.min_maker_touch_rate,
    )
    write_fee_tier_gates_csv(rows, output_path=args.csv_output_path)
    write_fee_tier_gates_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(
            f"fee={_fmt(row.fee_bps_per_fill_per_venue)}",
            row.gate.asset,
            row.gate.action,
            row.gate.best_mode,
            row.gate.horizon,
            f"headroom_bps={_fmt(row.gate.fee_headroom_bps)}",
        )


if __name__ == "__main__":
    main()
