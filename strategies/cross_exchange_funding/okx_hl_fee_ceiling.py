from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeeCeiling:
    asset: str
    execution_mode: str
    gross_event_8h_rate: Decimal
    gross_event_24h_rate: Decimal
    round_trip_slippage_rate: Decimal
    max_round_trip_fee_8h_rate: Decimal
    max_round_trip_fee_24h_rate: Decimal
    equal_venue_fee_8h_bps_per_fill: Decimal
    equal_venue_fee_24h_bps_per_fill: Decimal
    both_touch_rate: Decimal | None
    okx_only_touch_rate: Decimal | None
    hl_only_touch_rate: Decimal | None
    capacity: Decimal


def build_fee_ceilings(
    *,
    execution_mode_score_path: Path = ROOT / "okx_hl_execution_mode_score.csv",
) -> tuple[FeeCeiling, ...]:
    rows = _dedupe_mode_rows(_read_rows(execution_mode_score_path))
    ceilings = tuple(_fee_ceiling(row) for row in rows)
    return tuple(
        sorted(
            ceilings,
            key=lambda item: (
                item.equal_venue_fee_24h_bps_per_fill,
                item.equal_venue_fee_8h_bps_per_fill,
                item.capacity,
            ),
            reverse=True,
        )
    )


def write_fee_ceilings_csv(
    ceilings: tuple[FeeCeiling, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "execution_mode",
                "gross_event_8h_rate",
                "gross_event_24h_rate",
                "round_trip_slippage_rate",
                "max_round_trip_fee_8h_rate",
                "max_round_trip_fee_24h_rate",
                "equal_venue_fee_8h_bps_per_fill",
                "equal_venue_fee_24h_bps_per_fill",
                "both_touch_rate",
                "okx_only_touch_rate",
                "hl_only_touch_rate",
                "capacity",
            )
        )
        for ceiling in ceilings:
            writer.writerow(
                (
                    ceiling.asset,
                    ceiling.execution_mode,
                    _fmt(ceiling.gross_event_8h_rate),
                    _fmt(ceiling.gross_event_24h_rate),
                    _fmt(ceiling.round_trip_slippage_rate),
                    _fmt(ceiling.max_round_trip_fee_8h_rate),
                    _fmt(ceiling.max_round_trip_fee_24h_rate),
                    _fmt(ceiling.equal_venue_fee_8h_bps_per_fill),
                    _fmt(ceiling.equal_venue_fee_24h_bps_per_fill),
                    _fmt_optional(ceiling.both_touch_rate),
                    _fmt_optional(ceiling.okx_only_touch_rate),
                    _fmt_optional(ceiling.hl_only_touch_rate),
                    _fmt(ceiling.capacity),
                )
            )
    return output_path


def write_fee_ceilings_md(
    ceilings: tuple[FeeCeiling, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Fee Ceiling\n\n")
        handle.write(
            "This estimates the maximum equal per-fill fee bps each venue can charge "
            "before the event-window edge is erased. It uses the execution-mode "
            "slippage already measured from the public book.\n\n"
        )
        handle.write(
            "| asset | mode | max fee 8h bps/fill/venue | max fee 24h bps/fill/venue | both touch | OKX only | HL only | capacity |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for ceiling in ceilings:
            handle.write(
                "| "
                f"{ceiling.asset} | "
                f"{ceiling.execution_mode} | "
                f"{_fmt(ceiling.equal_venue_fee_8h_bps_per_fill)} | "
                f"{_fmt(ceiling.equal_venue_fee_24h_bps_per_fill)} | "
                f"{_fmt_optional(ceiling.both_touch_rate)} | "
                f"{_fmt_optional(ceiling.okx_only_touch_rate)} | "
                f"{_fmt_optional(ceiling.hl_only_touch_rate)} | "
                f"{_fmt(ceiling.capacity)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Negative fee ceilings mean the slippage-adjusted edge is already gone "
            "before fees. A ceiling below the actual account fee means the mode should "
            "not be promoted even if the raw funding spread looks positive.\n\n"
        )
        handle.write(
            "- BABY has the largest maker-only ceiling in this snapshot, but it has "
            "no maker-touch evidence here, low capacity, and loses most of the edge "
            "when one leg crosses.\n"
        )
        handle.write(
            "- ZEC has the best current one-leg-cross 24h headroom, especially "
            "`okx_cross_hl_maker`, but it is still capacity and stability constrained.\n"
        )
        handle.write(
            "- BTC has the cleanest capacity and survives all execution modes under "
            "very-low fees, but its 8h edge is extremely fee-sensitive.\n"
        )
        handle.write(
            "- JTO is weak in this event-window snapshot: even maker-only 24h has only "
            "a small fee ceiling, and one-leg-cross modes are already negative.\n"
        )
        handle.write(
            "- The next hard gate is the real account fee tier. Without that, raw "
            "funding spread is not enough to promote a mode.\n"
        )
    return output_path


def _fee_ceiling(row: dict[str, str]) -> FeeCeiling:
    gross_8h = Decimal(row["gross_event_8h_rate"])
    gross_24h = Decimal(row["gross_event_24h_rate"])
    slippage = Decimal(row["round_trip_slippage_rate"])
    max_fee_8h = gross_8h - slippage
    max_fee_24h = gross_24h - slippage
    return FeeCeiling(
        asset=row["asset"],
        execution_mode=row["execution_mode"],
        gross_event_8h_rate=gross_8h,
        gross_event_24h_rate=gross_24h,
        round_trip_slippage_rate=slippage,
        max_round_trip_fee_8h_rate=max_fee_8h,
        max_round_trip_fee_24h_rate=max_fee_24h,
        equal_venue_fee_8h_bps_per_fill=_equal_venue_fee_bps(max_fee_8h),
        equal_venue_fee_24h_bps_per_fill=_equal_venue_fee_bps(max_fee_24h),
        both_touch_rate=_optional_decimal(row["both_touch_rate"]),
        okx_only_touch_rate=_optional_decimal(row["okx_only_touch_rate"]),
        hl_only_touch_rate=_optional_decimal(row["hl_only_touch_rate"]),
        capacity=Decimal(row["capacity"]),
    )


def _equal_venue_fee_bps(max_round_trip_fee_rate: Decimal) -> Decimal:
    return max_round_trip_fee_rate * Decimal("10000") / Decimal("4")


def _dedupe_mode_rows(rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        by_key.setdefault((row["asset"], row["execution_mode"]), row)
    return tuple(by_key.values())


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _optional_decimal(value: str) -> Decimal | None:
    return Decimal(value) if value else None


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def _fmt_optional(value: Decimal | None) -> str:
    return "" if value is None else _fmt(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-mode-score-path",
        type=Path,
        default=ROOT / "okx_hl_execution_mode_score.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_fee_ceiling.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_fee_ceiling.md",
    )
    args = parser.parse_args()

    ceilings = build_fee_ceilings(
        execution_mode_score_path=args.execution_mode_score_path,
    )
    write_fee_ceilings_csv(ceilings, output_path=args.csv_output_path)
    write_fee_ceilings_md(ceilings, output_path=args.md_output_path)
    for ceiling in ceilings:
        print(
            ceiling.asset,
            ceiling.execution_mode,
            f"max8h_bps={_fmt(ceiling.equal_venue_fee_8h_bps_per_fill)}",
            f"max24h_bps={_fmt(ceiling.equal_venue_fee_24h_bps_per_fill)}",
        )


if __name__ == "__main__":
    main()
