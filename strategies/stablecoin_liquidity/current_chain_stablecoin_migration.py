from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from strategies.stablecoin_liquidity.current_supply_snapshot import fetch_stablecoins


ROOT = Path(__file__).resolve().parent
CHAIN_TOKEN_SYMBOLS = {
    "Arbitrum": "ARB",
    "Avalanche": "AVAX",
    "Base": "",
    "Berachain": "BERA",
    "BNB": "BNB",
    "BSC": "BNB",
    "Ethereum": "ETH",
    "Hyperliquid": "HYPE",
    "Hyperliquid L1": "HYPE",
    "Mantle": "MNT",
    "Optimism": "OP",
    "Polygon": "POL",
    "Solana": "SOL",
    "Sui": "SUI",
    "Tron": "TRX",
}
MAJOR_STABLES = {"USDT", "USDC", "DAI", "USDS", "USDE", "FDUSD", "USD1", "PYUSD", "FRAX", "USDD"}


@dataclass(frozen=True)
class ChainStablecoinMigrationRow:
    timestamp: str
    chain: str
    token_symbol: str
    current_supply_usd: float
    day_change_usd: float
    week_change_usd: float
    month_change_usd: float
    day_change_pct: float
    week_change_pct: float
    month_change_pct: float
    top_asset: str
    top_asset_week_change_usd: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def build_chain_stablecoin_migration_rows(
    payload: dict[str, object],
    *,
    timestamp: str | None = None,
    min_supply_usd: float = 20_000_000.0,
) -> tuple[ChainStablecoinMigrationRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    chain_totals: dict[str, dict[str, float | str]] = {}
    for asset in payload.get("peggedAssets") or ():
        symbol = str(asset.get("symbol") or "")
        if symbol.upper() not in MAJOR_STABLES:
            continue
        chain_rows = asset.get("chainCirculating")
        if not isinstance(chain_rows, dict):
            continue
        for chain, values in chain_rows.items():
            if not isinstance(values, dict):
                continue
            current = _pegged_usd(values.get("current"))
            prev_day = _pegged_usd(values.get("circulatingPrevDay"))
            prev_week = _pegged_usd(values.get("circulatingPrevWeek"))
            prev_month = _pegged_usd(values.get("circulatingPrevMonth"))
            if current <= 0.0:
                continue
            row = chain_totals.setdefault(
                str(chain),
                {
                    "current": 0.0,
                    "day": 0.0,
                    "week": 0.0,
                    "month": 0.0,
                    "top_asset": "",
                    "top_asset_week": 0.0,
                },
            )
            day_change = current - prev_day
            week_change = current - prev_week
            month_change = current - prev_month
            row["current"] = float(row["current"]) + current
            row["day"] = float(row["day"]) + day_change
            row["week"] = float(row["week"]) + week_change
            row["month"] = float(row["month"]) + month_change
            if abs(week_change) > abs(float(row["top_asset_week"])):
                row["top_asset"] = symbol
                row["top_asset_week"] = week_change
    rows = tuple(
        _build_row(timestamp=observed_at, chain=chain, values=values)
        for chain, values in chain_totals.items()
        if float(values["current"]) >= min_supply_usd
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_chain_stablecoin_migration_csv(
    rows: tuple[ChainStablecoinMigrationRow, ...],
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
                "current_supply_usd",
                "day_change_usd",
                "week_change_usd",
                "month_change_usd",
                "day_change_pct",
                "week_change_pct",
                "month_change_pct",
                "top_asset",
                "top_asset_week_change_usd",
                "score",
                "status",
                "side",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.chain,
                    row.token_symbol,
                    f"{row.current_supply_usd:.2f}",
                    f"{row.day_change_usd:.2f}",
                    f"{row.week_change_usd:.2f}",
                    f"{row.month_change_usd:.2f}",
                    f"{row.day_change_pct:.8f}",
                    f"{row.week_change_pct:.8f}",
                    f"{row.month_change_pct:.8f}",
                    row.top_asset,
                    f"{row.top_asset_week_change_usd:.2f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_chain_stablecoin_migration_md(
    rows: tuple[ChainStablecoinMigrationRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain Stablecoin Migration\n\n")
        handle.write(
            "This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. "
            "It is a capital-flow proxy, not a bridge-fill or trade instruction.\n\n"
        )
        handle.write(
            "| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.chain} | {row.token_symbol or '-'} | {row.status} | "
                f"{row.current_supply_usd:.0f} | {row.day_change_usd:.0f} | "
                f"{row.week_change_usd:.0f} | {row.month_change_usd:.0f} | "
                f"{row.week_change_pct:.4f} | {row.top_asset} | {row.score:.4f} | {row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, "
            "bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, "
            "bridge route checks, and forward labels.\n"
        )
    return output_path


def _build_row(*, timestamp: str, chain: str, values: dict[str, float | str]) -> ChainStablecoinMigrationRow:
    current = float(values["current"])
    day_change = float(values["day"])
    week_change = float(values["week"])
    month_change = float(values["month"])
    day_pct = _pct_change(day_change, current)
    week_pct = _pct_change(week_change, current)
    month_pct = _pct_change(month_change, current)
    status, side, reason = _status_side_reason(
        current_supply_usd=current,
        day_change_usd=day_change,
        week_change_usd=week_change,
        week_change_pct=week_pct,
    )
    token_symbol = CHAIN_TOKEN_SYMBOLS.get(chain, "")
    if not token_symbol and side in {"long_chain_beta_or_activity", "short_chain_beta_or_reduce_exposure"}:
        side = "chain_activity_or_related_assets"
    return ChainStablecoinMigrationRow(
        timestamp=timestamp,
        chain=chain,
        token_symbol=token_symbol,
        current_supply_usd=current,
        day_change_usd=day_change,
        week_change_usd=week_change,
        month_change_usd=month_change,
        day_change_pct=day_pct,
        week_change_pct=week_pct,
        month_change_pct=month_pct,
        top_asset=str(values["top_asset"]),
        top_asset_week_change_usd=float(values["top_asset_week"]),
        score=_score(current_supply_usd=current, week_change_usd=week_change, week_change_pct=week_pct, status=status),
        status=status,
        side=side,
        reason=reason,
        next_step=_next_step(chain=chain, token_symbol=token_symbol, status=status),
    )


def _status_side_reason(
    *,
    current_supply_usd: float,
    day_change_usd: float,
    week_change_usd: float,
    week_change_pct: float,
) -> tuple[str, str, str]:
    if week_change_usd >= 100_000_000.0 and week_change_pct >= 0.03 and day_change_usd > 0.0:
        return "paper_chain_stablecoin_inflow_watch", "long_chain_beta_or_activity", "large stablecoin inflow suggests deployable capital is arriving"
    if week_change_usd <= -100_000_000.0 and week_change_pct <= -0.03 and day_change_usd < 0.0:
        return "paper_chain_stablecoin_outflow_watch", "short_chain_beta_or_reduce_exposure", "large stablecoin outflow suggests capital is leaving"
    if abs(week_change_usd) >= 100_000_000.0:
        return "chain_stablecoin_flow_reversal_watch", "collect_label", "large weekly stablecoin flow has mixed daily confirmation"
    if current_supply_usd >= 1_000_000_000.0 and abs(week_change_pct) >= 0.01:
        return "chain_stablecoin_context_watch", "collect_label", "large chain has a material stablecoin distribution change"
    return "chain_stablecoin_context", "none", "chain stablecoin distribution context"


def _score(*, current_supply_usd: float, week_change_usd: float, week_change_pct: float, status: str) -> float:
    status_bonus = {
        "paper_chain_stablecoin_inflow_watch": 40.0,
        "paper_chain_stablecoin_outflow_watch": 38.0,
        "chain_stablecoin_flow_reversal_watch": 25.0,
        "chain_stablecoin_context_watch": 15.0,
    }.get(status, 0.0)
    flow_score = min(abs(week_change_usd) / 20_000_000.0, 25.0)
    pct_score = min(abs(week_change_pct) * 200.0, 20.0)
    size_score = min(current_supply_usd / 1_000_000_000.0, 10.0)
    return status_bonus + flow_score + pct_score + size_score


def _next_step(*, chain: str, token_symbol: str, status: str) -> str:
    if status == "paper_chain_stablecoin_inflow_watch":
        if token_symbol:
            return f"label {token_symbol} returns and activity after {chain} stablecoin inflow; check bridge route and venue coverage"
        return f"label {chain} activity and related assets after stablecoin inflow; check bridge route and venue coverage"
    if status == "paper_chain_stablecoin_outflow_watch":
        if token_symbol:
            return f"label {token_symbol} downside or rotation after {chain} stablecoin outflow; check bridge and exchange liquidity"
        return f"label {chain} activity and related assets after stablecoin outflow; check bridge and exchange liquidity"
    return f"collect repeat {chain} stablecoin migration snapshots and join to chain token, TVL, DEX, and funding context"


def _pegged_usd(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    return float(value.get("peggedUSD") or 0.0)


def _pct_change(change: float, current: float) -> float:
    previous = current - change
    if previous <= 0.0:
        return 0.0
    return change / previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_chain_stablecoin_migration.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_chain_stablecoin_migration.md")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_chain_stablecoin_migration_rows(fetch_stablecoins())
    write_chain_stablecoin_migration_csv(rows, output_path=args.output_path)
    write_chain_stablecoin_migration_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.chain, row.token_symbol or "-", f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
