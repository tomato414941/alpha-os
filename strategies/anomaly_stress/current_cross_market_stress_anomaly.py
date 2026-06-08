from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CrossMarketStressAnomaly:
    source_lane: str
    subject: str
    status: str
    side: str
    score: float
    severity: float
    evidence: str
    failure_mode: str
    next_step: str


def build_cross_market_stress_anomalies(root: Path = ROOT) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    rows.extend(_stablecoin_peg_rows(root))
    rows.extend(_defi_lending_rows(root))
    rows.extend(_defi_yield_rows(root))
    rows.extend(_options_volatility_rows(root))
    rows.extend(_prediction_market_rows(root))
    rows.extend(_cross_exchange_funding_rows(root))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_cross_market_stress_anomalies_csv(
    rows: tuple[CrossMarketStressAnomaly, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source_lane",
                "subject",
                "status",
                "side",
                "score",
                "severity",
                "evidence",
                "failure_mode",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.source_lane,
                    row.subject,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.severity:.8f}",
                    row.evidence,
                    row.failure_mode,
                    row.next_step,
                )
            )
    return output_path


def write_cross_market_stress_anomalies_md(
    rows: tuple[CrossMarketStressAnomaly, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cross-Market Stress Anomaly\n\n")
        handle.write(
            "This joins current anomaly-like states across peg, lending, yield, volatility, "
            "prediction-market, and execution-spread lanes. It is a broad candidate screen, "
            "not a trade instruction.\n\n"
        )
        handle.write(
            "| source | subject | status | side | score | severity | evidence | failure mode | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.source_lane} | {_escape(row.subject)} | {row.status} | {row.side} | "
                f"{row.score:.4f} | {row.severity:.4f} | {_escape(row.evidence)} | "
                f"{_escape(row.failure_mode)} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A high anomaly score means the current state is unusual enough to deserve a "
            "specific falsification test. It does not imply the anomaly is executable, "
            "fairly priced, or safe to trade.\n"
        )
    return output_path


def _stablecoin_peg_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "stablecoin_liquidity" / "current_peg_stress_screen.csv")[:8]:
        status = row.get("status", "")
        if status not in {
            "paper_depeg_repeg_watch",
            "paper_premium_mean_reversion_watch",
            "peg_supply_stress_watch",
        }:
            continue
        peg_deviation = _float(row.get("peg_deviation"))
        score = min(abs(peg_deviation) * 1_000.0 + _float(row.get("supply_stress_score")) * 5.0, 140.0)
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="stablecoin_liquidity",
                subject=row.get("symbol", ""),
                status="cross_market_peg_stress_anomaly",
                side=row.get("side", ""),
                score=score,
                severity=abs(peg_deviation),
                evidence=(
                    f"price={row.get('price', '')}, peg_deviation={row.get('peg_deviation', '')}, "
                    f"week_change={row.get('week_change_usd', '')}, mechanism={row.get('peg_mechanism', '')}"
                ),
                failure_mode="peg quote can be stale, unredeemable, insolvent, or impossible to trade at quoted size",
                next_step=row.get(
                    "next_step",
                    "check redemption route, venue depth, issuer risk, and repeated peg snapshots",
                ),
            )
        )
    return tuple(rows)


def _defi_lending_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "defi_lending" / "current_morpho_lending_rates.csv")[:6]:
        if row.get("status") != "paper_borrow_liquidity_stress_watch":
            continue
        utilization = _float(row.get("utilization"))
        liquidity = _float(row.get("liquidity_usd"))
        score = min(_float(row.get("score")), 120.0)
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="defi_lending",
                subject=f"{row.get('chain', '')} {row.get('loan_asset', '')}/{row.get('collateral_asset', '')}",
                status="cross_market_lending_stress_anomaly",
                side=row.get("side", ""),
                score=score,
                severity=utilization,
                evidence=(
                    f"utilization={row.get('utilization', '')}, liquidity={liquidity:.2f}, "
                    f"avg_borrow_apy={row.get('avg_borrow_apy', '')}, supply={row.get('supply_usd', '')}"
                ),
                failure_mode="100% utilization can be oracle, collateral, protocol, or withdrawal risk rather than clean alpha",
                next_step=row.get(
                    "next_step",
                    "check rate persistence, collateral drawdown, oracle, liquidation, withdrawal, and gas",
                ),
            )
        )
    return tuple(rows)


def _defi_yield_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "defi_yield" / "current_yield_peg_risk_join.csv")[:8]:
        status = row.get("status", "")
        if status not in {
            "paper_yield_premium_conflict_watch",
            "paper_yield_depeg_conflict_watch",
            "yield_supply_stress_watch",
            "paper_yield_without_peg_stress_watch",
        }:
            continue
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="defi_yield",
                subject=f"{row.get('chain', '')} {row.get('project', '')} {row.get('symbol', '')}",
                status="cross_market_yield_peg_anomaly",
                side=row.get("side", ""),
                score=min(_float(row.get("score")) + abs(_float(row.get("peg_deviation"))) * 200.0, 95.0),
                severity=abs(_float(row.get("peg_deviation"))) + _float(row.get("apy")) / 100.0,
                evidence=(
                    f"apy={row.get('apy', '')}, tvl={row.get('tvl_usd', '')}, "
                    f"peg={row.get('peg_symbol', '')}, peg_deviation={row.get('peg_deviation', '')}"
                ),
                failure_mode="headline APY can be compensation for peg, redemption, custody, or exit-liquidity risk",
                next_step=row.get(
                    "next_step",
                    "check APY persistence, redemption route, custody, exit liquidity, and gas",
                ),
            )
        )
    return tuple(rows)


def _options_volatility_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "options_volatility" / "current_options_volatility_paper_tickets.csv")[:6]:
        if row.get("status") != "paper_long_vol_quote_candidate":
            continue
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="options_volatility",
                subject=f"{row.get('currency', '')} {row.get('expiry', '')} {row.get('structure', '')}",
                status="cross_market_volatility_mispricing_watch",
                side=row.get("structure", ""),
                score=min(_float(row.get("score")) + max(-_float(row.get("iv_premium_24h")), 0.0) * 0.2, 110.0),
                severity=max(-_float(row.get("iv_premium_24h")), 0.0) / 100.0,
                evidence=(
                    f"atm_iv={row.get('atm_iv', '')}, realized_vol_24h={row.get('realized_vol_24h', '')}, "
                    f"iv_premium_24h={row.get('iv_premium_24h', '')}, max_loss_pct={row.get('max_loss_pct', '')}"
                ),
                failure_mode="cheap IV can be stale, unhedgeable, too wide, or explained by realized-vol mean reversion",
                next_step="paper-check multi-level depth, delta hedge cost, margin, and realized-vol persistence",
            )
        )
    return tuple(rows)


def _prediction_market_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "prediction_markets" / "current_event_probability_paper_outcome.csv")[:5]:
        if row.get("status") not in {
            "paper_outcome_active_watch",
            "paper_outcome_edge_watch",
            "paper_outcome_source_quality_watch",
        }:
            continue
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="prediction_markets",
                subject=row.get("question", ""),
                status="cross_market_event_probability_anomaly",
                side=row.get("suggested_side", ""),
                score=min(_float(row.get("score")) + max(_float(row.get("current_edge_after_ask")), 0.0) * 40.0, 105.0),
                severity=max(_float(row.get("current_edge_after_ask")), 0.0),
                evidence=(
                    f"entry_ask={row.get('entry_ask', '')}, bid={row.get('current_bid', '')}, "
                    f"ask={row.get('current_ask', '')}, edge={row.get('current_edge_after_ask', '')}, "
                    f"source_quality={row.get('source_quality_status', '')}"
                ),
                failure_mode="probability edge can be rough, stale, crowded, unfillable, or adversely selected",
                next_step="refresh market/news snapshots and require the edge to survive quote movement",
            )
        )
    return tuple(rows)


def _cross_exchange_funding_rows(root: Path) -> tuple[CrossMarketStressAnomaly, ...]:
    rows: list[CrossMarketStressAnomaly] = []
    for row in _read_rows(root / "cross_exchange_funding" / "current_dislocation_execution_check.csv")[:8]:
        if row.get("action") not in {"conservative_taker_monitor", "fee_only_monitor"}:
            continue
        conservative_net = _float(row.get("conservative_taker_net_24h"))
        rows.append(
            CrossMarketStressAnomaly(
                source_lane="cross_exchange_funding",
                subject=row.get("asset", ""),
                status="cross_market_execution_spread_anomaly",
                side=row.get("action", ""),
                score=max(min(conservative_net * 100_000.0, 90.0), 0.0),
                severity=max(conservative_net, 0.0),
                evidence=(
                    f"mean_net_24h={row.get('mean_net_24h_proxy', '')}, "
                    f"conservative_net_24h={row.get('conservative_taker_net_24h', '')}, "
                    f"slippage_bps={row.get('combined_taker_slippage_bps', '')}"
                ),
                failure_mode="apparent spread can vanish after fees, slippage, balances, margin, or transfer constraints",
                next_step="repeat monitor with actual account fees, venue constraints, and paper hedge tickets",
            )
        )
    return tuple(rows)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=LANE_ROOT / "current_cross_market_stress_anomaly.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=LANE_ROOT / "current_cross_market_stress_anomaly.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_cross_market_stress_anomalies()
    write_cross_market_stress_anomalies_csv(rows, output_path=args.csv_output_path)
    write_cross_market_stress_anomalies_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.subject, f"score={row.score:.4f}", row.side)


if __name__ == "__main__":
    main()
