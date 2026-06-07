from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChainTvlVenueCoverageRow:
    timestamp: str
    chain: str
    token_symbol: str
    action: str
    week_change_pct: float
    day_change_pct: float
    hyperliquid_perp: bool
    okx_swap: bool
    venue_count: int
    followup: str


def build_chain_tvl_venue_coverage_rows(
    *,
    flow_path: Path = ROOT / "current_chain_tvl_flow.csv",
) -> tuple[ChainTvlVenueCoverageRow, ...]:
    rows = tuple(row for row in _read_rows(flow_path) if row.get("token_symbol"))
    hyperliquid_assets = _fetch_hyperliquid_assets()
    okx_instruments = _fetch_okx_swap_instruments()
    coverage_rows = tuple(
        _coverage_row(
            row=row,
            hyperliquid_assets=hyperliquid_assets,
            okx_instruments=okx_instruments,
        )
        for row in rows
    )
    return tuple(sorted(coverage_rows, key=_sort_key, reverse=True))


def write_chain_tvl_venue_coverage_csv(
    rows: tuple[ChainTvlVenueCoverageRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "chain",
                "token_symbol",
                "action",
                "week_change_pct",
                "day_change_pct",
                "hyperliquid_perp",
                "okx_swap",
                "venue_count",
                "followup",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.chain,
                    row.token_symbol,
                    row.action,
                    f"{row.week_change_pct:.8f}",
                    f"{row.day_change_pct:.8f}",
                    row.hyperliquid_perp,
                    row.okx_swap,
                    row.venue_count,
                    row.followup,
                )
            )
    return output_path


def write_chain_tvl_venue_coverage_md(
    rows: tuple[ChainTvlVenueCoverageRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain TVL Flow Venue Coverage\n\n")
        handle.write(
            "This checks whether chain TVL flow candidates have public perp venues. "
            "It does not validate fees, fills, or whether TVL accounting is stale.\n\n"
        )
        handle.write(
            "| chain | token | action | week % | day % | HL | OKX | venues | followup |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.chain} | "
                f"{row.token_symbol} | "
                f"{row.action} | "
                f"{row.week_change_pct:.4f} | "
                f"{row.day_change_pct:.4f} | "
                f"{row.hyperliquid_perp} | "
                f"{row.okx_swap} | "
                f"{row.venue_count} | "
                f"{row.followup} |\n"
            )
    return output_path


def _coverage_row(
    *,
    row: dict[str, str],
    hyperliquid_assets: set[str],
    okx_instruments: set[str],
) -> ChainTvlVenueCoverageRow:
    token_symbol = row["token_symbol"]
    hl = token_symbol in hyperliquid_assets
    okx = f"{token_symbol}-USDT-SWAP" in okx_instruments
    venue_count = int(hl) + int(okx)
    return ChainTvlVenueCoverageRow(
        timestamp=row["timestamp"],
        chain=row["chain"],
        token_symbol=token_symbol,
        action=row["action"],
        week_change_pct=float(row.get("week_change_pct") or "0"),
        day_change_pct=float(row.get("day_change_pct") or "0"),
        hyperliquid_perp=hl,
        okx_swap=okx,
        venue_count=venue_count,
        followup=_followup(action=row["action"], token_symbol=token_symbol, venue_count=venue_count),
    )


def _fetch_hyperliquid_assets() -> set[str]:
    response = requests.post(HYPERLIQUID_INFO_URL, json={"type": "meta"}, timeout=30)
    response.raise_for_status()
    return {str(item["name"]) for item in response.json().get("universe", ())}


def _fetch_okx_swap_instruments() -> set[str]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    return {
        str(item.get("instId"))
        for item in response.json().get("data", ())
        if str(item.get("instId", "")).endswith("-USDT-SWAP")
    }


def _followup(*, action: str, token_symbol: str, venue_count: int) -> str:
    if venue_count == 0:
        return "keep as context until a perp venue exists"
    if action == "chain_inflow_momentum_watch":
        return f"label {token_symbol} long continuation on covered venues"
    if action == "chain_outflow_stress_watch":
        return f"label {token_symbol} short stress on covered venues"
    if action == "chain_flow_reversal_watch":
        return f"label {token_symbol} rebound continuation on covered venues"
    return f"keep {token_symbol} as context"


def _sort_key(row: ChainTvlVenueCoverageRow) -> tuple[int, int, float]:
    action_priority = {
        "chain_inflow_momentum_watch": 3,
        "chain_outflow_stress_watch": 2,
        "chain_flow_reversal_watch": 1,
        "chain_flow_context": 0,
    }
    return (
        row.venue_count,
        action_priority.get(row.action, 0),
        abs(row.week_change_pct),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_venue_coverage.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_venue_coverage.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_chain_tvl_venue_coverage_rows(flow_path=args.flow_path)
    write_chain_tvl_venue_coverage_csv(rows, output_path=args.output_path)
    write_chain_tvl_venue_coverage_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.chain,
            row.token_symbol,
            row.action,
            f"venues={row.venue_count}",
            f"week={row.week_change_pct:.4f}",
        )


if __name__ == "__main__":
    main()
