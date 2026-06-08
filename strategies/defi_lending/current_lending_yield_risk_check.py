from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLUE_CHIP_COLLATERAL = {"WETH", "ETH", "wstETH", "stETH", "WBTC"}
STABLE_OR_RWA_COLLATERAL = {"USDC", "USDT", "DAI", "USDS", "sUSDS", "USDE", "sUSDE", "USR", "USDY", "BUIDL"}


@dataclass(frozen=True)
class LendingYieldRiskCheckRow:
    chain: str
    loan_asset: str
    collateral_asset: str
    market_id: str
    paper_notional_usd: float
    supply_usd: float
    liquidity_usd: float
    utilization: float
    lltv: float
    supply_apy: float
    avg_net_supply_apy: float
    supply_apy_spike_ratio: float
    capacity_usage: float
    collateral_category: str
    risk_score: float
    risk_action: str
    reason: str
    next_step: str


def build_lending_yield_risk_check_rows(
    *,
    actionability_path: Path = ROOT / "current_lending_stress_actionability.csv",
    morpho_path: Path = ROOT / "current_morpho_lending_rates.csv",
    paper_notional_usd: float = 10_000.0,
) -> tuple[LendingYieldRiskCheckRow, ...]:
    markets = {
        (row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")): row
        for row in _read_rows(morpho_path)
    }
    rows = tuple(
        _build_row(
            row=row,
            market=markets.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {}),
            paper_notional_usd=paper_notional_usd,
        )
        for row in _read_rows(actionability_path)
        if row.get("side") == "paper_lend_after_risk_check"
    )
    return tuple(sorted(rows, key=lambda row: row.risk_score, reverse=True))


def write_lending_yield_risk_check_csv(
    rows: tuple[LendingYieldRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(LendingYieldRiskCheckRow.__dataclass_fields__))
        for row in rows:
            writer.writerow(
                (
                    row.chain,
                    row.loan_asset,
                    row.collateral_asset,
                    row.market_id,
                    f"{row.paper_notional_usd:.2f}",
                    f"{row.supply_usd:.2f}",
                    f"{row.liquidity_usd:.2f}",
                    f"{row.utilization:.8f}",
                    f"{row.lltv:.8f}",
                    f"{row.supply_apy:.8f}",
                    f"{row.avg_net_supply_apy:.8f}",
                    f"{row.supply_apy_spike_ratio:.8f}",
                    f"{row.capacity_usage:.8f}",
                    row.collateral_category,
                    f"{row.risk_score:.8f}",
                    row.risk_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_lending_yield_risk_check_md(
    rows: tuple[LendingYieldRiskCheckRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Lending Yield Risk Check\n\n")
        handle.write(
            "This checks Morpho lending yield candidates against capacity, utilization, LLTV, "
            "collateral familiarity, and APY spike risk. It is a paper-risk gate, not a deposit instruction.\n\n"
        )
        handle.write(
            "| market | action | score | notional | liquidity | usage | util | LLTV | APY | avg APY | spike | collateral | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.chain} {row.loan_asset}/{row.collateral_asset} | {row.risk_action} | "
                f"{row.risk_score:.2f} | {row.paper_notional_usd:.0f} | {row.liquidity_usd:.0f} | "
                f"{row.capacity_usage:.4f} | {row.utilization:.4f} | {row.lltv:.2f} | "
                f"{row.supply_apy:.4f} | {row.avg_net_supply_apy:.4f} | {row.supply_apy_spike_ratio:.2f} | "
                f"{row.collateral_category} | {_escape(row.reason)} |\n"
            )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    market: dict[str, str],
    paper_notional_usd: float,
) -> LendingYieldRiskCheckRow:
    liquidity = _float(row.get("liquidity_usd"))
    utilization = _float(row.get("utilization"))
    avg_net_supply_apy = _float(row.get("avg_net_supply_apy"))
    supply_apy = _float(market.get("supply_apy")) or avg_net_supply_apy
    spike_ratio = supply_apy / avg_net_supply_apy if avg_net_supply_apy > 0.0 else 0.0
    capacity_usage = paper_notional_usd / liquidity if liquidity > 0.0 else 0.0
    collateral = row.get("collateral_asset", "")
    collateral_category = _collateral_category(collateral)
    risk_action, reason, next_step = _risk_action(
        chain=row.get("chain", ""),
        loan=row.get("loan_asset", ""),
        collateral=collateral,
        capacity_usage=capacity_usage,
        utilization=utilization,
        lltv=_float(market.get("lltv")),
        spike_ratio=spike_ratio,
        collateral_category=collateral_category,
        market=market,
    )
    return LendingYieldRiskCheckRow(
        chain=row.get("chain", ""),
        loan_asset=row.get("loan_asset", ""),
        collateral_asset=collateral,
        market_id=market.get("market_id", ""),
        paper_notional_usd=paper_notional_usd,
        supply_usd=_float(row.get("supply_usd")),
        liquidity_usd=liquidity,
        utilization=utilization,
        lltv=_float(market.get("lltv")),
        supply_apy=supply_apy,
        avg_net_supply_apy=avg_net_supply_apy,
        supply_apy_spike_ratio=spike_ratio,
        capacity_usage=capacity_usage,
        collateral_category=collateral_category,
        risk_score=_risk_score(
            liquidity_usd=liquidity,
            utilization=utilization,
            lltv=_float(market.get("lltv")),
            avg_net_supply_apy=avg_net_supply_apy,
            spike_ratio=spike_ratio,
            collateral_category=collateral_category,
        ),
        risk_action=risk_action,
        reason=reason,
        next_step=next_step,
    )


def _risk_action(
    *,
    chain: str,
    loan: str,
    collateral: str,
    capacity_usage: float,
    utilization: float,
    lltv: float,
    spike_ratio: float,
    collateral_category: str,
    market: dict[str, str],
) -> tuple[str, str, str]:
    market_name = f"{chain} {loan}/{collateral}"
    if not market:
        return "missing_morpho_market_context", "no matching Morpho market row", f"refresh Morpho data before checking {market_name}"
    if capacity_usage > 0.01:
        return "capacity_too_small_for_10k_probe", "10k paper notional uses more than 1% of remaining liquidity", f"lower notional or skip {market_name}"
    if collateral_category == "opaque_collateral":
        return "collateral_review_required", "collateral is not a familiar blue-chip, stable, or RWA symbol", f"review collateral mechanics before any {market_name} paper lend"
    if utilization >= 0.95:
        return "exit_liquidity_watch", "high utilization means exit and withdrawal timing dominate the headline APY", f"paper-check exit path before any {market_name} deposit simulation"
    if lltv >= 0.90:
        return "high_lltv_liquidation_watch", "LLTV is high enough that collateral drawdown and liquidation mechanics dominate", f"review oracle and liquidation path for {market_name}"
    if spike_ratio >= 1.30:
        return "rate_spike_watch", "current supply APY is materially above average net supply APY", f"wait for rate persistence before any {market_name} paper lend"
    return "paper_lending_risk_check_survived", "candidate survives the first capacity, utilization, LLTV, collateral, and APY spike checks", f"record a small paper lending simulation for {market_name}"


def _risk_score(
    *,
    liquidity_usd: float,
    utilization: float,
    lltv: float,
    avg_net_supply_apy: float,
    spike_ratio: float,
    collateral_category: str,
) -> float:
    score = 50.0
    score += min(avg_net_supply_apy * 200.0, 25.0)
    score += min(liquidity_usd / 2_000_000.0, 8.0)
    if utilization >= 0.95:
        score -= 15.0
    elif utilization >= 0.90:
        score -= 7.0
    if lltv >= 0.90:
        score -= 15.0
    elif lltv >= 0.85:
        score -= 6.0
    if spike_ratio >= 1.30:
        score -= 10.0
    if collateral_category == "opaque_collateral":
        score -= 20.0
    elif collateral_category == "stable_or_rwa_collateral":
        score -= 6.0
    return max(0.0, min(100.0, score))


def _collateral_category(collateral: str) -> str:
    if collateral in BLUE_CHIP_COLLATERAL:
        return "blue_chip_collateral"
    if collateral in STABLE_OR_RWA_COLLATERAL:
        return "stable_or_rwa_collateral"
    return "opaque_collateral"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionability-path", type=Path, default=ROOT / "current_lending_stress_actionability.csv")
    parser.add_argument("--morpho-path", type=Path, default=ROOT / "current_morpho_lending_rates.csv")
    parser.add_argument("--paper-notional-usd", type=float, default=10_000.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_lending_yield_risk_check.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_lending_yield_risk_check.md")
    args = parser.parse_args()

    rows = build_lending_yield_risk_check_rows(
        actionability_path=args.actionability_path,
        morpho_path=args.morpho_path,
        paper_notional_usd=args.paper_notional_usd,
    )
    write_lending_yield_risk_check_csv(rows, output_path=args.output_path)
    write_lending_yield_risk_check_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.risk_action, f"{row.chain} {row.loan_asset}/{row.collateral_asset}", f"score={row.risk_score:.4f}")


if __name__ == "__main__":
    main()
