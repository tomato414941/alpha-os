from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class WatchRow:
    source: str
    action: str
    asset: str
    long_venue: str
    short_venue: str
    annualized_edge: float
    net_8h_proxy: float | None
    net_24h_proxy: float | None
    liquidity_proxy: float
    friction_proxy: float
    reason: str


def build_watchlist(
    *,
    funding_feasibility_path: Path = ROOT / "current_funding_feasibility.csv",
    okx_hl_path: Path = ROOT / "current_okx_hl_funding_spread.csv",
    hl_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
) -> tuple[WatchRow, ...]:
    rows = [
        *_predicted_cross_venue_rows(funding_feasibility_path),
        *_okx_hl_rows(okx_hl_path),
        *_single_venue_hl_rows(hl_snapshot_path),
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.action in {"paper_24h_monitor", "current_funding_monitor"},
                row.annualized_edge,
                row.liquidity_proxy,
            ),
            reverse=True,
        )
    )


def write_watchlist_csv(rows: tuple[WatchRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source",
                "action",
                "asset",
                "long_venue",
                "short_venue",
                "annualized_edge",
                "net_8h_proxy",
                "net_24h_proxy",
                "liquidity_proxy",
                "friction_proxy",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.source,
                    row.action,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    f"{row.annualized_edge:.8f}",
                    "" if row.net_8h_proxy is None else f"{row.net_8h_proxy:.8f}",
                    "" if row.net_24h_proxy is None else f"{row.net_24h_proxy:.8f}",
                    f"{row.liquidity_proxy:.8f}",
                    f"{row.friction_proxy:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_watchlist_md(
    rows: tuple[WatchRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Funding Dislocation Watchlist\n\n")
        handle.write(
            "This is a current-state watchlist, not a backtest and not a trade "
            "instruction. It combines current Hyperliquid funding, predicted "
            "cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.\n\n"
        )
        handle.write(
            "| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.source} | "
                f"{row.action} | "
                f"{row.asset} | "
                f"{row.long_venue} | "
                f"{row.short_venue} | "
                f"{row.annualized_edge:.6f} | "
                f"{'' if row.net_8h_proxy is None else f'{row.net_8h_proxy:.6f}'} | "
                f"{'' if row.net_24h_proxy is None else f'{row.net_24h_proxy:.6f}'} | "
                f"{row.liquidity_proxy:.2f} | "
                f"{row.friction_proxy:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_24h_monitor` means the current rough 24-hour proxy is positive, "
            "but it still needs real fee tier, fill, margin, and borrow/collateral checks. "
            "`current_funding_monitor` means the funding rate is large enough to watch, "
            "but no executable hedge has been proven.\n"
        )
    return output_path


def _predicted_cross_venue_rows(path: Path) -> tuple[WatchRow, ...]:
    if not path.exists():
        return ()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            annualized = float(row["annualized_spread"])
            day_volume = float(row.get("hl_day_notional_volume") or 0.0)
            impact = float(row.get("hl_impact_spread") or 0.0)
            if annualized < 0.5:
                continue
            rows.append(
                WatchRow(
                    source="predicted_cross_venue",
                    action=(
                        "current_funding_monitor"
                        if day_volume >= 100_000.0 and impact <= 0.005
                        else "thin_or_wide_watch"
                    ),
                    asset=row["asset"],
                    long_venue=row["long_venue"],
                    short_venue=row["short_venue"],
                    annualized_edge=annualized,
                    net_8h_proxy=None,
                    net_24h_proxy=None,
                    liquidity_proxy=day_volume,
                    friction_proxy=impact,
                    reason=row.get("notes", ""),
                )
            )
    return tuple(rows)


def _okx_hl_rows(path: Path) -> tuple[WatchRow, ...]:
    if not path.exists():
        return ()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            annualized = float(row["annualized_spread"])
            net_8h = float(row["net_8h_proxy"])
            net_24h = float(row["net_24h_proxy"])
            capacity = float(row["capacity_proxy_notional"])
            if annualized < 0.05 and net_24h <= 0.0:
                continue
            if net_8h > 0.0 and capacity >= 10_000.0:
                action = "paper_8h_monitor"
            elif net_24h > 0.0 and capacity >= 10_000.0:
                action = "paper_24h_monitor"
            else:
                action = "blocked_by_cost_or_capacity"
            rows.append(
                WatchRow(
                    source="okx_hl_current",
                    action=action,
                    asset=row["asset"],
                    long_venue=row["long_venue"],
                    short_venue=row["short_venue"],
                    annualized_edge=annualized,
                    net_8h_proxy=net_8h,
                    net_24h_proxy=net_24h,
                    liquidity_proxy=capacity,
                    friction_proxy=float(row["rough_round_trip_cost"]),
                    reason=row.get("notes", ""),
                )
            )
    return tuple(rows)


def _single_venue_hl_rows(path: Path) -> tuple[WatchRow, ...]:
    if not path.exists():
        return ()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            annualized = abs(float(row["annualized_funding"]))
            day_volume = float(row["day_notional_volume"])
            impact = float(row["impact_spread"])
            if annualized < 0.5:
                continue
            funding = float(row["funding_rate"])
            rows.append(
                WatchRow(
                    source="hl_single_venue",
                    action=(
                        "current_funding_monitor"
                        if day_volume >= 100_000.0 and impact <= 0.005
                        else "thin_or_wide_watch"
                    ),
                    asset=row["asset"],
                    long_venue="HlPerp" if funding < 0.0 else "cash_or_spot_proxy",
                    short_venue="cash_or_spot_proxy" if funding < 0.0 else "HlPerp",
                    annualized_edge=annualized,
                    net_8h_proxy=None,
                    net_24h_proxy=None,
                    liquidity_proxy=day_volume,
                    friction_proxy=impact,
                    reason=row["carry_side"],
                )
            )
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_dislocation_watchlist.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_dislocation_watchlist.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_watchlist()
    write_watchlist_csv(rows, output_path=args.csv_output_path)
    write_watchlist_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.source,
            row.action,
            row.asset,
            row.long_venue,
            row.short_venue,
            f"edge={row.annualized_edge:.4f}",
            f"net24={'' if row.net_24h_proxy is None else f'{row.net_24h_proxy:.6f}'}",
        )


if __name__ == "__main__":
    main()
