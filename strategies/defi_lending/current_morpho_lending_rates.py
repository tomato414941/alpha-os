from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"
MORPHO_MARKETS_QUERY = """
query {
  markets(
    first: 100,
    orderBy: SupplyAssetsUsd,
    orderDirection: Desc,
    where: { chainId_in: [1, 8453] }
  ) {
    items {
      marketId
      chain { id network }
      loanAsset { symbol }
      collateralAsset { symbol }
      lltv
      state {
        supplyAssetsUsd
        borrowAssetsUsd
        liquidityAssetsUsd
        utilization
        supplyApy
        borrowApy
        avgSupplyApy
        avgBorrowApy
        avgNetSupplyApy
        avgNetBorrowApy
      }
    }
  }
}
"""
STABLE_LOAN_SYMBOLS = {"USDC", "USDT", "DAI", "USDS", "USDE", "SUSDE", "USR", "USDA", "USD0", "GHO"}


@dataclass(frozen=True)
class MorphoLendingRateRow:
    market_id: str
    chain_id: int
    chain: str
    loan_asset: str
    collateral_asset: str
    lltv: float
    supply_usd: float
    borrow_usd: float
    liquidity_usd: float
    utilization: float
    supply_apy: float
    borrow_apy: float
    avg_supply_apy: float
    avg_borrow_apy: float
    avg_net_supply_apy: float
    avg_net_borrow_apy: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def fetch_morpho_markets(url: str = MORPHO_GRAPHQL_URL) -> tuple[dict[str, object], ...]:
    response = requests.post(url, json={"query": MORPHO_MARKETS_QUERY}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    markets = payload.get("data", {}).get("markets", {}).get("items", ())
    return tuple(markets)


def build_lending_rate_rows(markets: tuple[dict[str, object], ...]) -> tuple[MorphoLendingRateRow, ...]:
    rows = tuple(_build_row(market) for market in markets)
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_lending_rate_csv(rows: tuple[MorphoLendingRateRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "chain_id",
                "chain",
                "loan_asset",
                "collateral_asset",
                "lltv",
                "supply_usd",
                "borrow_usd",
                "liquidity_usd",
                "utilization",
                "supply_apy",
                "borrow_apy",
                "avg_supply_apy",
                "avg_borrow_apy",
                "avg_net_supply_apy",
                "avg_net_borrow_apy",
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
                    row.market_id,
                    row.chain_id,
                    row.chain,
                    row.loan_asset,
                    row.collateral_asset,
                    f"{row.lltv:.8f}",
                    f"{row.supply_usd:.2f}",
                    f"{row.borrow_usd:.2f}",
                    f"{row.liquidity_usd:.2f}",
                    f"{row.utilization:.8f}",
                    f"{row.supply_apy:.8f}",
                    f"{row.borrow_apy:.8f}",
                    f"{row.avg_supply_apy:.8f}",
                    f"{row.avg_borrow_apy:.8f}",
                    f"{row.avg_net_supply_apy:.8f}",
                    f"{row.avg_net_borrow_apy:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_lending_rate_md(
    rows: tuple[MorphoLendingRateRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Morpho Lending Rates\n\n")
        handle.write(
            "This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. "
            "It is a lending-rate pressure screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.chain} | {row.loan_asset} | {row.collateral_asset} | {row.status} | "
                f"{row.supply_usd:.0f} | {row.borrow_usd:.0f} | {row.liquidity_usd:.0f} | "
                f"{row.utilization:.4f} | {row.avg_net_supply_apy:.4f} | "
                f"{row.avg_net_borrow_apy:.4f} | {row.score:.4f} | {row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High utilization and high borrow APY can indicate leverage demand or liquidity stress. "
            "A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, "
            "withdrawal, gas, and smart-contract risk checks.\n"
        )
    return output_path


def _build_row(market: dict[str, object]) -> MorphoLendingRateRow:
    state = market.get("state") if isinstance(market.get("state"), dict) else {}
    chain = market.get("chain") if isinstance(market.get("chain"), dict) else {}
    loan_asset = market.get("loanAsset") if isinstance(market.get("loanAsset"), dict) else {}
    collateral_asset = market.get("collateralAsset") if isinstance(market.get("collateralAsset"), dict) else {}
    loan_symbol = str(loan_asset.get("symbol") or "")
    collateral_symbol = str(collateral_asset.get("symbol") or "")
    supply_usd = _float(state.get("supplyAssetsUsd"))
    borrow_usd = _float(state.get("borrowAssetsUsd"))
    liquidity_usd = _float(state.get("liquidityAssetsUsd"))
    utilization = _float(state.get("utilization"))
    avg_supply_apy = _float(state.get("avgSupplyApy"))
    avg_borrow_apy = _float(state.get("avgBorrowApy"))
    avg_net_supply_apy = _float(state.get("avgNetSupplyApy"))
    avg_net_borrow_apy = _float(state.get("avgNetBorrowApy"))
    status, side, reason = _status_side_reason(
        loan_asset=loan_symbol,
        supply_usd=supply_usd,
        liquidity_usd=liquidity_usd,
        utilization=utilization,
        avg_net_supply_apy=avg_net_supply_apy,
        avg_net_borrow_apy=avg_net_borrow_apy,
    )
    return MorphoLendingRateRow(
        market_id=str(market.get("marketId") or ""),
        chain_id=int(chain.get("id") or 0),
        chain=str(chain.get("network") or ""),
        loan_asset=loan_symbol,
        collateral_asset=collateral_symbol,
        lltv=_lltv(market.get("lltv")),
        supply_usd=supply_usd,
        borrow_usd=borrow_usd,
        liquidity_usd=liquidity_usd,
        utilization=utilization,
        supply_apy=_float(state.get("supplyApy")),
        borrow_apy=_float(state.get("borrowApy")),
        avg_supply_apy=avg_supply_apy,
        avg_borrow_apy=avg_borrow_apy,
        avg_net_supply_apy=avg_net_supply_apy,
        avg_net_borrow_apy=avg_net_borrow_apy,
        score=_score(
            loan_asset=loan_symbol,
            supply_usd=supply_usd,
            liquidity_usd=liquidity_usd,
            utilization=utilization,
            avg_net_supply_apy=avg_net_supply_apy,
            avg_net_borrow_apy=avg_net_borrow_apy,
            status=status,
        ),
        status=status,
        side=side,
        reason=reason,
        next_step=(
            f"check Morpho {loan_symbol}/{collateral_symbol} rate persistence, collateral drawdown, "
            "oracle, liquidation, withdrawal liquidity, gas, and smart-contract risk"
        ),
    )


def _status_side_reason(
    *,
    loan_asset: str,
    supply_usd: float,
    liquidity_usd: float,
    utilization: float,
    avg_net_supply_apy: float,
    avg_net_borrow_apy: float,
) -> tuple[str, str, str]:
    stable_loan = loan_asset.upper() in STABLE_LOAN_SYMBOLS
    if supply_usd >= 20_000_000.0 and utilization >= 0.98 and liquidity_usd <= 1_000_000.0:
        return "paper_borrow_liquidity_stress_watch", "watch_borrow_squeeze_or_avoid_supply", "market is highly utilized with little remaining liquidity"
    if stable_loan and supply_usd >= 20_000_000.0 and liquidity_usd >= 2_000_000.0 and avg_net_supply_apy >= 0.06:
        return "paper_stable_lending_yield_watch", "lend_after_risk_check", "stable loan asset offers material lending APY with remaining liquidity"
    if supply_usd >= 20_000_000.0 and avg_net_borrow_apy >= 0.10:
        return "borrow_demand_context_watch", "watch_borrow_pressure", "borrow APY is elevated versus typical stable carry"
    return "lending_context_watch", "none", "lending market context exists but is not yet actionable"


def _score(
    *,
    loan_asset: str,
    supply_usd: float,
    liquidity_usd: float,
    utilization: float,
    avg_net_supply_apy: float,
    avg_net_borrow_apy: float,
    status: str,
) -> float:
    stable_bonus = 8.0 if loan_asset.upper() in STABLE_LOAN_SYMBOLS else 0.0
    size_score = min(supply_usd / 100_000_000.0, 12.0)
    liquidity_score = min(liquidity_usd / 25_000_000.0, 5.0)
    utilization_score = utilization * 20.0
    supply_rate_score = min(avg_net_supply_apy * 100.0, 25.0)
    borrow_rate_score = min(avg_net_borrow_apy * 50.0, 20.0)
    status_bonus = {
        "paper_borrow_liquidity_stress_watch": 18.0,
        "paper_stable_lending_yield_watch": 16.0,
        "borrow_demand_context_watch": 8.0,
    }.get(status, 0.0)
    return stable_bonus + size_score + liquidity_score + utilization_score + supply_rate_score + borrow_rate_score + status_bonus


def _float(value: object) -> float:
    return float(value or 0.0)


def _lltv(value: object) -> float:
    raw = _float(value)
    return raw / 1_000_000_000_000_000_000.0 if raw > 1.0 else raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_morpho_lending_rates.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_morpho_lending_rates.md")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_lending_rate_rows(fetch_morpho_markets())
    write_lending_rate_csv(rows, output_path=args.output_path)
    write_lending_rate_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.chain, row.loan_asset, row.collateral_asset, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
