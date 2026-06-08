from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LendingStressActionabilityRow:
    chain: str
    loan_asset: str
    collateral_asset: str
    status: str
    side: str
    score: float
    supply_usd: float
    borrow_usd: float
    liquidity_usd: float
    utilization: float
    avg_net_supply_apy: float
    avg_net_borrow_apy: float
    source_status: str
    reason: str
    next_step: str


def build_lending_stress_actionability_rows(root: Path = ROOT) -> tuple[LendingStressActionabilityRow, ...]:
    output: list[LendingStressActionabilityRow] = []
    rows = tuple(
        row
        for row in _read_rows(root / "defi_lending" / "current_morpho_lending_rates.csv")
        if row.get("status")
        in {
            "paper_borrow_liquidity_stress_watch",
            "paper_stable_lending_yield_watch",
            "borrow_demand_context_watch",
        }
    )
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        chain = row.get("chain", "")
        loan = row.get("loan_asset", "")
        collateral = row.get("collateral_asset", "")
        key = (chain, loan, collateral)
        if key in seen:
            continue
        seen.add(key)
        supply_usd = _float(row.get("supply_usd"))
        borrow_usd = _float(row.get("borrow_usd"))
        liquidity_usd = _float(row.get("liquidity_usd"))
        utilization = _float(row.get("utilization"))
        avg_net_supply_apy = _float(row.get("avg_net_supply_apy"))
        avg_net_borrow_apy = _float(row.get("avg_net_borrow_apy"))
        status, side, reason = _status_side_reason(
            supply_usd=supply_usd,
            liquidity_usd=liquidity_usd,
            utilization=utilization,
            avg_net_supply_apy=avg_net_supply_apy,
            avg_net_borrow_apy=avg_net_borrow_apy,
        )
        output.append(
            LendingStressActionabilityRow(
                chain=chain,
                loan_asset=loan,
                collateral_asset=collateral,
                status=status,
                side=side,
                score=_score(
                    status=status,
                    supply_usd=supply_usd,
                    liquidity_usd=liquidity_usd,
                    utilization=utilization,
                    avg_net_supply_apy=avg_net_supply_apy,
                    avg_net_borrow_apy=avg_net_borrow_apy,
                ),
                supply_usd=supply_usd,
                borrow_usd=borrow_usd,
                liquidity_usd=liquidity_usd,
                utilization=utilization,
                avg_net_supply_apy=avg_net_supply_apy,
                avg_net_borrow_apy=avg_net_borrow_apy,
                source_status=row.get("status", ""),
                reason=reason,
                next_step=_next_step(chain=chain, loan=loan, collateral=collateral, status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_lending_stress_actionability_csv(
    rows: tuple[LendingStressActionabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "chain",
                "loan_asset",
                "collateral_asset",
                "status",
                "side",
                "score",
                "supply_usd",
                "borrow_usd",
                "liquidity_usd",
                "utilization",
                "avg_net_supply_apy",
                "avg_net_borrow_apy",
                "source_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.chain,
                    row.loan_asset,
                    row.collateral_asset,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.supply_usd:.2f}",
                    f"{row.borrow_usd:.2f}",
                    f"{row.liquidity_usd:.2f}",
                    f"{row.utilization:.8f}",
                    f"{row.avg_net_supply_apy:.8f}",
                    f"{row.avg_net_borrow_apy:.8f}",
                    row.source_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_lending_stress_actionability_md(
    rows: tuple[LendingStressActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Lending Stress Actionability\n\n")
        handle.write(
            "This separates Morpho lending stress from a currently actionable lending candidate. "
            "A fully utilized market with no remaining liquidity is treated as a mechanics/risk "
            "state, not as deployable yield.\n\n"
        )
        handle.write(
            "| chain | loan/collateral | status | side | score | supply USD | liquidity USD | "
            "util | avg supply APY | avg borrow APY | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.chain} | {row.loan_asset}/{row.collateral_asset} | {row.status} | {row.side} | "
                f"{row.score:.4f} | {row.supply_usd:.0f} | {row.liquidity_usd:.0f} | "
                f"{row.utilization:.4f} | {row.avg_net_supply_apy:.4f} | "
                f"{row.avg_net_borrow_apy:.4f} | {_escape(row.reason)} |\n"
            )
    return output_path


def _status_side_reason(
    *,
    supply_usd: float,
    liquidity_usd: float,
    utilization: float,
    avg_net_supply_apy: float,
    avg_net_borrow_apy: float,
) -> tuple[str, str, str]:
    if liquidity_usd <= 10_000.0 and utilization >= 0.98:
        return (
            "lending_stress_no_liquidity_risk",
            "no_new_lending_until_exit_path",
            "fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge",
        )
    if liquidity_usd >= 2_000_000.0 and avg_net_supply_apy >= 0.08 and utilization < 0.98:
        return (
            "lending_rate_candidate_after_risk_check",
            "paper_lend_after_risk_check",
            "market has visible remaining liquidity and material supply APY before protocol and withdrawal checks",
        )
    if supply_usd >= 20_000_000.0 and avg_net_borrow_apy >= 0.10:
        return (
            "lending_stress_mechanics_watch",
            "paper_mechanics_check",
            "borrow pressure is visible, but collateral, oracle, liquidation, and withdrawal mechanics dominate first",
        )
    return "lending_stress_deprioritize", "none", "lending state is not actionable after the basic liquidity screen"


def _score(
    *,
    status: str,
    supply_usd: float,
    liquidity_usd: float,
    utilization: float,
    avg_net_supply_apy: float,
    avg_net_borrow_apy: float,
) -> float:
    if status == "lending_rate_candidate_after_risk_check":
        return min(92.0, 65.0 + min(avg_net_supply_apy * 120.0, 14.0) + min(liquidity_usd / 2_000_000.0, 8.0))
    if status == "lending_stress_mechanics_watch":
        return min(72.0, 48.0 + min(avg_net_borrow_apy * 30.0, 14.0) + min(supply_usd / 500_000_000.0, 10.0))
    if status == "lending_stress_no_liquidity_risk":
        return min(48.0, 32.0 + min(supply_usd / 1_000_000_000.0, 8.0) + min(utilization * 8.0, 8.0))
    return 20.0


def _next_step(*, chain: str, loan: str, collateral: str, status: str) -> str:
    market = f"{chain} {loan}/{collateral}"
    if status == "lending_rate_candidate_after_risk_check":
        return f"paper-check {market} withdrawal path, rate persistence, gas, oracle, liquidation, and smart-contract risk"
    if status == "lending_stress_mechanics_watch":
        return f"check {market} collateral, oracle, liquidation, withdrawal, and rate persistence before any alpha label"
    if status == "lending_stress_no_liquidity_risk":
        return f"do not promote {market}; first prove remaining capacity, exit liquidity, and rate persistence"
    return f"deprioritize {market} until visible liquidity and a testable lending edge appear"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_lending_stress_actionability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=LANE_ROOT / "current_lending_stress_actionability.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_lending_stress_actionability_rows()
    write_lending_stress_actionability_csv(rows, output_path=args.output_path)
    write_lending_stress_actionability_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.chain, f"{row.loan_asset}/{row.collateral_asset}", f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
