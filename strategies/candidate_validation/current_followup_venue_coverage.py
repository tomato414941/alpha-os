from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
BINANCE_FAPI_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FollowupVenueCoverageRow:
    asset: str
    priority: float
    source: str
    hyperliquid_perp: bool
    okx_usdt_swap: bool
    binance_usdt_perp: bool
    venue_count: int
    action: str
    reason: str


def build_followup_venue_coverage_rows(
    *,
    queue_path: Path = ROOT / "current_followup_queue.csv",
    top: int = 50,
) -> tuple[FollowupVenueCoverageRow, ...]:
    queue_rows = tuple(row for row in _read_rows(queue_path) if row.get("asset") != "*")[:top]
    hyperliquid_assets = _fetch_hyperliquid_assets()
    okx_assets = _fetch_okx_usdt_swap_assets()
    binance_assets = _fetch_binance_usdt_perp_assets()
    rows = tuple(
        _build_row(
            row=row,
            hyperliquid_assets=hyperliquid_assets,
            okx_assets=okx_assets,
            binance_assets=binance_assets,
        )
        for row in queue_rows
    )
    return tuple(sorted(rows, key=lambda row: (row.venue_count, row.priority), reverse=True))


def write_followup_venue_coverage_csv(
    rows: tuple[FollowupVenueCoverageRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "priority",
                "source",
                "hyperliquid_perp",
                "okx_usdt_swap",
                "binance_usdt_perp",
                "venue_count",
                "action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    f"{row.priority:.4f}",
                    row.source,
                    row.hyperliquid_perp,
                    row.okx_usdt_swap,
                    row.binance_usdt_perp,
                    row.venue_count,
                    row.action,
                    row.reason,
                )
            )
    return output_path


def write_followup_venue_coverage_md(
    rows: tuple[FollowupVenueCoverageRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Venue Coverage\n\n")
        handle.write(
            "This checks whether follow-up queue assets exist on major perp venues. "
            "It prevents Hyperliquid-only execution context from silently dropping "
            "otherwise testable candidates.\n\n"
        )
        handle.write(
            "| asset | priority | source | HL | OKX | Binance | venues | action | reason |\n"
        )
        handle.write("| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.priority:.4f} | "
                f"{row.source} | "
                f"{row.hyperliquid_perp} | "
                f"{row.okx_usdt_swap} | "
                f"{row.binance_usdt_perp} | "
                f"{row.venue_count} | "
                f"{row.action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Multi-venue coverage improves observability and execution optionality. "
            "Single-venue candidates can still be useful, but venue-specific data, "
            "fees, and depth must be checked instead of assuming Hyperliquid is the "
            "only route.\n"
        )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    hyperliquid_assets: frozenset[str],
    okx_assets: frozenset[str],
    binance_assets: frozenset[str],
) -> FollowupVenueCoverageRow:
    asset = row["asset"]
    hl = asset in hyperliquid_assets
    okx = asset in okx_assets
    binance = asset in binance_assets
    venue_count = sum((hl, okx, binance))
    action, reason = _action(
        hyperliquid_perp=hl,
        okx_usdt_swap=okx,
        binance_usdt_perp=binance,
        venue_count=venue_count,
    )
    return FollowupVenueCoverageRow(
        asset=asset,
        priority=float(row.get("priority") or "0"),
        source=row.get("source", ""),
        hyperliquid_perp=hl,
        okx_usdt_swap=okx,
        binance_usdt_perp=binance,
        venue_count=venue_count,
        action=action,
        reason=reason,
    )


def _action(
    *,
    hyperliquid_perp: bool,
    okx_usdt_swap: bool,
    binance_usdt_perp: bool,
    venue_count: int,
) -> tuple[str, str]:
    if venue_count >= 2:
        return "multi_venue_followup", "candidate can be observed or routed on multiple perp venues"
    if hyperliquid_perp:
        return "hyperliquid_only_followup", "candidate is currently visible on Hyperliquid only in this check"
    if okx_usdt_swap:
        return "okx_only_followup", "candidate is missing from Hyperliquid but exists on OKX USDT swap"
    if binance_usdt_perp:
        return "binance_only_followup", "candidate is missing from Hyperliquid but exists on Binance USD perp"
    return "venue_gap", "candidate is not found on the checked perp venues"


def _fetch_hyperliquid_assets() -> frozenset[str]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=30,
    )
    response.raise_for_status()
    meta, _contexts = response.json()
    return frozenset(str(row["name"]).upper() for row in meta["universe"])


def _fetch_okx_usdt_swap_assets() -> frozenset[str]:
    response = requests.get(
        OKX_INSTRUMENTS_URL,
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    return frozenset(
        str(row.get("uly") or "").split("-")[0].upper()
        for row in rows
        if str(row.get("instId") or "").endswith("-USDT-SWAP")
    )


def _fetch_binance_usdt_perp_assets() -> frozenset[str]:
    try:
        response = requests.get(BINANCE_FAPI_EXCHANGE_INFO_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return frozenset()
    rows = response.json().get("symbols", ())
    return frozenset(
        str(row.get("baseAsset") or "").upper()
        for row in rows
        if row.get("contractType") == "PERPETUAL"
        and row.get("quoteAsset") == "USDT"
        and row.get("status") == "TRADING"
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=ROOT / "current_followup_queue.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_venue_coverage.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_venue_coverage.md",
    )
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_followup_venue_coverage_rows(queue_path=args.queue_path, top=args.top)
    write_followup_venue_coverage_csv(rows, output_path=args.output_path)
    write_followup_venue_coverage_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"venues={row.venue_count}",
            f"HL={row.hyperliquid_perp}",
            f"OKX={row.okx_usdt_swap}",
            f"BINANCE={row.binance_usdt_perp}",
        )


if __name__ == "__main__":
    main()
