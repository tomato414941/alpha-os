from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FEE_BPS_PER_FILL_PER_VENUE = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ExecutionCheck:
    asset: str
    fee_bps_per_fill_per_venue: float
    mean_net_24h_proxy: float
    fee_round_trip_rate: float
    combined_taker_slippage_bps: float
    fee_only_net_24h: float
    conservative_taker_net_24h: float
    okx_size_valid: bool
    hl_size_valid: bool
    book_fully_filled: bool
    action: str
    reason: str


def build_execution_checks(
    *,
    asset: str = "STABLE",
    monitor_summary_path: Path = ROOT / "current_dislocation_monitor_summary.csv",
    order_constraints_path: Path = ROOT / "okx_hl_order_constraints.csv",
    book_depth_path: Path = ROOT / "okx_hl_book_depth.csv",
    fee_bps_per_fill_per_venue_values: tuple[float, ...] = DEFAULT_FEE_BPS_PER_FILL_PER_VENUE,
) -> tuple[ExecutionCheck, ...]:
    monitor_row = _monitor_row_for_asset(monitor_summary_path, asset=asset)
    order_row = _order_constraints_row(order_constraints_path, asset=asset)
    book_rows = _book_rows(book_depth_path, asset=asset)
    mean_net_24h = float(monitor_row["mean_net_24h_proxy"])
    combined_taker_slippage_bps = _combined_taker_slippage_bps(book_rows)
    okx_size_valid = _bool(order_row["okx_size_valid"])
    hl_size_valid = _bool(order_row["hl_size_valid"])
    book_fully_filled = all(_bool(row["fully_filled"]) for row in book_rows)
    return tuple(
        _build_check(
            asset=asset,
            fee_bps_per_fill_per_venue=fee_bps,
            mean_net_24h=mean_net_24h,
            combined_taker_slippage_bps=combined_taker_slippage_bps,
            okx_size_valid=okx_size_valid,
            hl_size_valid=hl_size_valid,
            book_fully_filled=book_fully_filled,
        )
        for fee_bps in fee_bps_per_fill_per_venue_values
    )


def write_execution_checks_csv(
    checks: tuple[ExecutionCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "fee_bps_per_fill_per_venue",
                "mean_net_24h_proxy",
                "fee_round_trip_rate",
                "combined_taker_slippage_bps",
                "fee_only_net_24h",
                "conservative_taker_net_24h",
                "okx_size_valid",
                "hl_size_valid",
                "book_fully_filled",
                "action",
                "reason",
            )
        )
        for check in checks:
            writer.writerow(
                (
                    check.asset,
                    f"{check.fee_bps_per_fill_per_venue:.6f}",
                    f"{check.mean_net_24h_proxy:.10f}",
                    f"{check.fee_round_trip_rate:.10f}",
                    f"{check.combined_taker_slippage_bps:.6f}",
                    f"{check.fee_only_net_24h:.10f}",
                    f"{check.conservative_taker_net_24h:.10f}",
                    check.okx_size_valid,
                    check.hl_size_valid,
                    check.book_fully_filled,
                    check.action,
                    check.reason,
                )
            )
    return output_path


def write_execution_checks_md(
    checks: tuple[ExecutionCheck, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Dislocation Execution Check\n\n")
        handle.write(
            "This checks whether the current monitor candidate still has 24-hour "
            "headroom after fee scenarios and visible public-book taker slippage. "
            "It is not a trade instruction and does not use account-specific fee tiers.\n\n"
        )
        handle.write(
            "| asset | fee bps/fill/venue | mean net24 | fee round trip | taker slippage bps | fee-only net24 | conservative taker net24 | sizes valid | book filled | action | reason |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |\n")
        for check in checks:
            handle.write(
                "| "
                f"{check.asset} | "
                f"{check.fee_bps_per_fill_per_venue:.6f} | "
                f"{check.mean_net_24h_proxy:.6f} | "
                f"{check.fee_round_trip_rate:.6f} | "
                f"{check.combined_taker_slippage_bps:.6f} | "
                f"{check.fee_only_net_24h:.6f} | "
                f"{check.conservative_taker_net_24h:.6f} | "
                f"{check.okx_size_valid and check.hl_size_valid} | "
                f"{check.book_fully_filled} | "
                f"{check.action} | "
                f"{check.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`fee_only_monitor` means the 24-hour monitor survives the configured fee "
            "scenario but may need maker execution or lower slippage. "
            "`conservative_taker_monitor` means it also survives subtracting visible "
            "public-book taker slippage from the already friction-adjusted proxy.\n"
        )
    return output_path


def _build_check(
    *,
    asset: str,
    fee_bps_per_fill_per_venue: float,
    mean_net_24h: float,
    combined_taker_slippage_bps: float,
    okx_size_valid: bool,
    hl_size_valid: bool,
    book_fully_filled: bool,
) -> ExecutionCheck:
    fee_round_trip_rate = (fee_bps_per_fill_per_venue * 4.0) / 10_000.0
    taker_slippage_rate = combined_taker_slippage_bps / 10_000.0
    fee_only_net_24h = mean_net_24h - fee_round_trip_rate
    conservative_taker_net_24h = fee_only_net_24h - taker_slippage_rate
    action = "blocked"
    reason = "no positive 24h headroom after fees"
    if not (okx_size_valid and hl_size_valid and book_fully_filled):
        reason = "public size or visible book check failed"
    elif conservative_taker_net_24h > 0.0:
        action = "conservative_taker_monitor"
        reason = "survives fees and visible taker slippage in the conservative check"
    elif fee_only_net_24h > 0.0:
        action = "fee_only_monitor"
        reason = "survives fees, but visible taker slippage consumes the conservative edge"
    return ExecutionCheck(
        asset=asset,
        fee_bps_per_fill_per_venue=fee_bps_per_fill_per_venue,
        mean_net_24h_proxy=mean_net_24h,
        fee_round_trip_rate=fee_round_trip_rate,
        combined_taker_slippage_bps=combined_taker_slippage_bps,
        fee_only_net_24h=fee_only_net_24h,
        conservative_taker_net_24h=conservative_taker_net_24h,
        okx_size_valid=okx_size_valid,
        hl_size_valid=hl_size_valid,
        book_fully_filled=book_fully_filled,
        action=action,
        reason=reason,
    )


def _monitor_row_for_asset(path: Path, *, asset: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row["asset"] == asset and row["source"] == "okx_hl_current"
        )
    if not rows:
        raise RuntimeError(f"OKX-HL monitor row not found: {asset}")
    return max(rows, key=lambda row: float(row.get("mean_net_24h_proxy") or "0"))


def _order_constraints_row(path: Path, *, asset: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["asset"] == asset:
                return row
    raise RuntimeError(f"Order constraints row not found: {asset}")


def _book_rows(path: Path, *, asset: str) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(row for row in csv.DictReader(handle) if row["asset"] == asset)
    if not rows:
        raise RuntimeError(f"Book depth rows not found: {asset}")
    return rows


def _combined_taker_slippage_bps(rows: tuple[dict[str, str], ...]) -> float:
    return max(float(row["combined_taker_slippage_bps"]) for row in rows)


def _bool(value: str) -> bool:
    return value == "True"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="STABLE")
    parser.add_argument(
        "--fee-bps-per-fill-per-venue",
        nargs="+",
        type=float,
        default=list(DEFAULT_FEE_BPS_PER_FILL_PER_VENUE),
    )
    parser.add_argument(
        "--monitor-summary-path",
        type=Path,
        default=ROOT / "current_dislocation_monitor_summary.csv",
    )
    parser.add_argument(
        "--order-constraints-path",
        type=Path,
        default=ROOT / "okx_hl_order_constraints.csv",
    )
    parser.add_argument(
        "--book-depth-path",
        type=Path,
        default=ROOT / "okx_hl_book_depth.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_dislocation_execution_check.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_dislocation_execution_check.md",
    )
    args = parser.parse_args()

    checks = build_execution_checks(
        asset=args.asset,
        monitor_summary_path=args.monitor_summary_path,
        order_constraints_path=args.order_constraints_path,
        book_depth_path=args.book_depth_path,
        fee_bps_per_fill_per_venue_values=tuple(args.fee_bps_per_fill_per_venue),
    )
    write_execution_checks_csv(checks, output_path=args.csv_output_path)
    write_execution_checks_md(checks, output_path=args.md_output_path)
    for check in checks:
        print(
            check.asset,
            f"fee_bps={check.fee_bps_per_fill_per_venue:.6f}",
            f"fee_only={check.fee_only_net_24h:.6f}",
            f"conservative={check.conservative_taker_net_24h:.6f}",
            check.action,
        )


if __name__ == "__main__":
    main()
