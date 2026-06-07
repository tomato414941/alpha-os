from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeeScenario:
    name: str
    okx_fee_bps_per_fill: Decimal
    hl_fee_bps_per_fill: Decimal


@dataclass(frozen=True)
class FeeSensitivityRow:
    asset: str
    scenario: str
    paper_notional: Decimal
    mean_net_8h_proxy: Decimal
    mean_net_24h_proxy: Decimal
    fee_round_trip_rate: Decimal
    fee_round_trip_usdt: Decimal
    net_8h_after_fee_rate: Decimal
    net_8h_after_fee_usdt: Decimal
    net_24h_after_fee_rate: Decimal
    net_24h_after_fee_usdt: Decimal
    survives_8h: bool
    survives_24h: bool


DEFAULT_SCENARIOS = (
    FeeScenario(
        name="very_low_fee",
        okx_fee_bps_per_fill=Decimal("0.2"),
        hl_fee_bps_per_fill=Decimal("0.2"),
    ),
    FeeScenario(
        name="low_fee",
        okx_fee_bps_per_fill=Decimal("0.5"),
        hl_fee_bps_per_fill=Decimal("0.5"),
    ),
    FeeScenario(
        name="one_bps_each",
        okx_fee_bps_per_fill=Decimal("1"),
        hl_fee_bps_per_fill=Decimal("1"),
    ),
    FeeScenario(
        name="two_bps_each",
        okx_fee_bps_per_fill=Decimal("2"),
        hl_fee_bps_per_fill=Decimal("2"),
    ),
    FeeScenario(
        name="five_bps_each",
        okx_fee_bps_per_fill=Decimal("5"),
        hl_fee_bps_per_fill=Decimal("5"),
    ),
)


def build_fee_sensitivity_rows(
    *,
    summary_path: Path = ROOT / "okx_hl_funding_persistence_summary.csv",
    order_constraints_path: Path = ROOT / "okx_hl_order_constraints.csv",
    asset: str = "BTC",
    scenarios: tuple[FeeScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[FeeSensitivityRow, ...]:
    summary = _read_asset_row(summary_path, asset=asset)
    constraints = _read_first_row(order_constraints_path)
    paper_notional = Decimal(constraints["paper_notional"])
    mean_net_8h_proxy = Decimal(summary["mean_net_8h_proxy"])
    mean_net_24h_proxy = Decimal(summary["mean_net_24h_proxy"])
    return tuple(
        _scenario_row(
            asset=asset,
            scenario=scenario,
            paper_notional=paper_notional,
            mean_net_8h_proxy=mean_net_8h_proxy,
            mean_net_24h_proxy=mean_net_24h_proxy,
        )
        for scenario in scenarios
    )


def write_fee_sensitivity_rows(
    rows: tuple[FeeSensitivityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "scenario",
                "paper_notional",
                "mean_net_8h_proxy",
                "mean_net_24h_proxy",
                "fee_round_trip_rate",
                "fee_round_trip_usdt",
                "net_8h_after_fee_rate",
                "net_8h_after_fee_usdt",
                "net_24h_after_fee_rate",
                "net_24h_after_fee_usdt",
                "survives_8h",
                "survives_24h",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.scenario,
                    _fmt(row.paper_notional),
                    _fmt(row.mean_net_8h_proxy),
                    _fmt(row.mean_net_24h_proxy),
                    _fmt(row.fee_round_trip_rate),
                    _fmt(row.fee_round_trip_usdt),
                    _fmt(row.net_8h_after_fee_rate),
                    _fmt(row.net_8h_after_fee_usdt),
                    _fmt(row.net_24h_after_fee_rate),
                    _fmt(row.net_24h_after_fee_usdt),
                    row.survives_8h,
                    row.survives_24h,
                )
            )
    return output_path


def write_fee_sensitivity_md(
    rows: tuple[FeeSensitivityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Fee Sensitivity\n\n")
        handle.write("This is not a fee schedule. It is a paper sensitivity check.\n\n")
        handle.write("Assumption: one entry and one exit on each venue.\n\n")
        handle.write(
            "| scenario | round-trip fee rate | 8h after fee | 8h USDT | 24h after fee | 24h USDT | survives 8h | survives 24h |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.scenario} | "
                f"{_fmt(row.fee_round_trip_rate)} | "
                f"{_fmt(row.net_8h_after_fee_rate)} | "
                f"{_fmt(row.net_8h_after_fee_usdt)} | "
                f"{_fmt(row.net_24h_after_fee_rate)} | "
                f"{_fmt(row.net_24h_after_fee_usdt)} | "
                f"{row.survives_8h} | "
                f"{row.survives_24h} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The BTC paper ticket is fee-sensitive. If both venues require more than "
            "roughly sub-bps effective execution per fill, the 8h edge disappears. "
            "This makes maker execution, rebates, or longer holding windows central "
            "to whether the candidate is real.\n"
        )
    return output_path


def _scenario_row(
    *,
    asset: str,
    scenario: FeeScenario,
    paper_notional: Decimal,
    mean_net_8h_proxy: Decimal,
    mean_net_24h_proxy: Decimal,
) -> FeeSensitivityRow:
    fee_round_trip_rate = (
        Decimal("2")
        * (scenario.okx_fee_bps_per_fill + scenario.hl_fee_bps_per_fill)
        / Decimal("10000")
    )
    net_8h_after_fee_rate = mean_net_8h_proxy - fee_round_trip_rate
    net_24h_after_fee_rate = mean_net_24h_proxy - fee_round_trip_rate
    return FeeSensitivityRow(
        asset=asset,
        scenario=scenario.name,
        paper_notional=paper_notional,
        mean_net_8h_proxy=mean_net_8h_proxy,
        mean_net_24h_proxy=mean_net_24h_proxy,
        fee_round_trip_rate=fee_round_trip_rate,
        fee_round_trip_usdt=paper_notional * fee_round_trip_rate,
        net_8h_after_fee_rate=net_8h_after_fee_rate,
        net_8h_after_fee_usdt=paper_notional * net_8h_after_fee_rate,
        net_24h_after_fee_rate=net_24h_after_fee_rate,
        net_24h_after_fee_usdt=paper_notional * net_24h_after_fee_rate,
        survives_8h=net_8h_after_fee_rate > 0,
        survives_24h=net_24h_after_fee_rate > 0,
    )


def _read_asset_row(path: Path, *, asset: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["asset"] == asset:
                return row
    raise RuntimeError(f"asset not found in {path}: {asset}")


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")
    return rows[0]


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "okx_hl_funding_persistence_summary.csv",
    )
    parser.add_argument(
        "--order-constraints-path",
        type=Path,
        default=ROOT / "okx_hl_order_constraints.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_fee_sensitivity.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_fee_sensitivity.md",
    )
    args = parser.parse_args()

    rows = build_fee_sensitivity_rows(
        summary_path=args.summary_path,
        order_constraints_path=args.order_constraints_path,
        asset=args.asset,
    )
    write_fee_sensitivity_rows(rows, output_path=args.csv_output_path)
    write_fee_sensitivity_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(
            row.scenario,
            f"fee={_fmt(row.fee_round_trip_rate)}",
            f"8h={_fmt(row.net_8h_after_fee_rate)}",
            f"24h={_fmt(row.net_24h_after_fee_rate)}",
            f"survives8h={row.survives_8h}",
            f"survives24h={row.survives_24h}",
        )


if __name__ == "__main__":
    main()
