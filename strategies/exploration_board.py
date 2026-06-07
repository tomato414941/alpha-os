from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExplorationRow:
    lane: str
    status: str
    strongest_current_signal: str
    main_gap: str
    next_step: str


def build_exploration_rows(root: Path = ROOT) -> tuple[ExplorationRow, ...]:
    return (
        _crypto_market_structure_row(root),
        _cross_exchange_funding_row(root),
        _perp_market_map_row(root),
        _event_flow_row(root),
        _defi_yield_row(root),
        _market_making_row(root),
        _news_social_row(root),
        _stablecoin_liquidity_row(root),
        ExplorationRow(
            lane="on_chain_flow",
            status="partial_proxy",
            strongest_current_signal="stablecoin supply proxy exists",
            main_gap="wallet, bridge, and exchange inflow/outflow data not connected",
            next_step="add direct flow source instead of only stablecoin supply proxy",
        ),
    )


def write_exploration_board(
    rows: tuple[ExplorationRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Strategy Exploration Board\n\n")
        handle.write("This board tracks broad profit-source exploration. It is not a ranking of deployable strategies.\n\n")
        handle.write("| lane | status | strongest current signal | main gap | next step |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.lane} | {row.status} | {row.strongest_current_signal} | "
                f"{row.main_gap} | {row.next_step} |\n"
            )
    return output_path


def _crypto_market_structure_row(root: Path) -> ExplorationRow:
    gate_path = root / "crypto_market_structure" / "spot_perp_carry_execution_gate.csv"
    symbol_audit_path = root / "crypto_market_structure" / "spot_perp_carry_symbol_audit.csv"
    period_audit_path = root / "crypto_market_structure" / "spot_perp_carry_period_audit.csv"
    best = _best_numeric_row(gate_path, key="headroom_bps")
    best_symbol = _best_numeric_row(symbol_audit_path, key="gross_contribution")
    best_2024 = _best_period_row(period_audit_path, period="2024")
    best_current = _best_period_row(period_audit_path, period="2026_to_date")
    signal = "spot/perp carry screen exists"
    if best:
        signal = (
            f"{best.get('candidate', 'spot_perp_carry')}: "
            f"{best.get('scenario', 'scenario')} headroom="
            f"{best.get('headroom_bps', '')}bps, "
            f"default_sharpe={best.get('default_cost_sharpe', '')}"
        )
    if best_symbol:
        signal = (
            f"{signal}; top_symbol={best_symbol.get('symbol', '')} "
            f"gross={best_symbol.get('gross_contribution', '')}"
        )
    status = "execution_gate_candidate"
    main_gap = "actual account fees, borrow/margin, and book-depth feasibility remain shallow"
    next_step = "validate WIF/INJ/FET/APT venue fees, margin, and book depth before paper trading"
    if best_2024 and best_current and float(best_current.get("total_return") or "0") <= 0.0:
        signal = (
            f"2024 {best_2024.get('candidate', '')} sharpe={best_2024.get('sharpe', '')}; "
            f"2026_to_date best_total={best_current.get('total_return', '')}"
        )
        status = "historical_dislocation"
        main_gap = "spot/perp carry did not persist after 2024 under the current rule"
        next_step = "search current funding dislocations or regime filters before paper trading"
    return ExplorationRow(
        lane="crypto_market_structure",
        status=status,
        strongest_current_signal=signal,
        main_gap=main_gap,
        next_step=next_step,
    )


def _cross_exchange_funding_row(root: Path) -> ExplorationRow:
    execution_check_path = root / "cross_exchange_funding" / "current_dislocation_execution_check.csv"
    best_execution_check = _best_execution_check_row(execution_check_path)
    if best_execution_check:
        return ExplorationRow(
            lane="cross_exchange_funding",
            status="execution_assumption_gate",
            strongest_current_signal=(
                f"{best_execution_check.get('asset', '')}: "
                f"{best_execution_check.get('action', '')}, "
                f"fee={best_execution_check.get('fee_bps_per_fill_per_venue', '')}bps/fill/venue, "
                f"conservative_net24={best_execution_check.get('conservative_taker_net_24h', '')}"
            ),
            main_gap="real account fees, fills, margin, collateral movement, and liquidation buffer are unvalidated",
            next_step="run longer STABLE monitoring and validate real fee/fill/margin assumptions before paper trading",
        )
    monitor_path = root / "cross_exchange_funding" / "current_dislocation_monitor_summary.csv"
    best_monitor = _best_monitor_row(monitor_path)
    if best_monitor:
        return ExplorationRow(
            lane="cross_exchange_funding",
            status="short_window_monitor",
            strongest_current_signal=(
                f"{best_monitor.get('asset', '')}: {best_monitor.get('action', '')} "
                f"{best_monitor.get('long_venue', '')}->{best_monitor.get('short_venue', '')}, "
                f"obs={best_monitor.get('observations', '')}, "
                f"mean_net24={best_monitor.get('mean_net_24h_proxy', '')}"
            ),
            main_gap="short-window persistence exists, but real fees, fills, and margin are unvalidated",
            next_step="validate STABLE fee/fill/margin assumptions and run longer scheduled monitoring",
        )
    watchlist_path = root / "cross_exchange_funding" / "current_dislocation_watchlist.csv"
    best_watch = _best_watchlist_row(watchlist_path)
    if best_watch:
        return ExplorationRow(
            lane="cross_exchange_funding",
            status="current_dislocation_monitor",
            strongest_current_signal=(
                f"{best_watch.get('asset', '')}: {best_watch.get('action', '')} "
                f"{best_watch.get('long_venue', '')}->{best_watch.get('short_venue', '')}, "
                f"edge={best_watch.get('annualized_edge', '')}, "
                f"net24={best_watch.get('net_24h_proxy', '')}"
            ),
            main_gap="current dislocation has not been persistence-tested with real fees and fills",
            next_step="monitor STABLE/SAGA/kNEIRO/SNX/AIXBT repeatedly before paper trading",
        )
    sensitivity_path = root / "cross_exchange_funding" / "okx_hl_promotion_gate_sensitivity.csv"
    best = _best_promotion_gate_row(sensitivity_path)
    signal = "current funding spread screen exists"
    if best:
        signal = (
            f"{best.get('asset', '')}: {best.get('action', '')} "
            f"{best.get('best_mode', '')} {best.get('horizon', '')}, "
            f"fee={best.get('fee_bps_per_fill_per_venue', '')}bps, "
            f"headroom={best.get('fee_headroom_bps', '')}bps"
        )
    return ExplorationRow(
        lane="cross_exchange_funding",
        status="paper_gate_candidate",
        strongest_current_signal=signal,
        main_gap="actual account fees, longer event monitoring, and real maker-fill evidence are still missing",
        next_step="validate actual OKX/Hyperliquid fee tier, then paper-test ZEC/BTC execution gates",
    )


def _perp_market_map_row(root: Path) -> ExplorationRow:
    path = root / "perp_market_map" / "current_hyperliquid_snapshot.csv"
    best = _best_numeric_row(path, key="attention_score")
    signal = "not run yet"
    if best:
        signal = (
            f"{best.get('asset', '')}: ann_funding={best.get('annualized_funding', '')}, "
            f"volume={best.get('day_notional_volume', '')}"
        )
    return ExplorationRow(
        lane="perp_market_map",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="no history yet, so no persistence or PnL evidence",
        next_step="collect snapshots over time and test carry/crowding persistence",
    )


def _event_flow_row(root: Path) -> ExplorationRow:
    path = root / "event_flow" / "flow_imbalance_screen.csv"
    top = _row_by_value(path, field="bucket", value="top_20")
    signal = "5m aggTrades path exists"
    if top:
        signal = (
            f"top_20 imbalance mean_next_return={top.get('mean_next_return', '')}, "
            f"hit_rate={top.get('hit_rate', '')}"
        )
    return ExplorationRow(
        lane="event_flow",
        status="implemented_probe",
        strongest_current_signal=signal,
        main_gap="tiny sample and naive label; no order book or liquidation context",
        next_step="extend sample window and add liquidation/funding-time labels",
    )


def _defi_yield_row(root: Path) -> ExplorationRow:
    path = root / "defi_yield" / "current_yield_screen.csv"
    best = _best_numeric_row(path, key="score")
    signal = "current stable-yield screen exists"
    if best:
        signal = (
            f"{best.get('chain', '')}/{best.get('project', '')} "
            f"{best.get('symbol', '')}: apy={best.get('apy', '')}, tvl={best.get('tvl_usd', '')}"
        )
    return ExplorationRow(
        lane="defi_yield",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="risk, custody, exit liquidity, and APY decay not modeled",
        next_step="separate real yield from incentive yield and add operational risk checklist",
    )


def _market_making_row(root: Path) -> ExplorationRow:
    path = root / "market_making" / "current_l2_snapshot.csv"
    best = _best_abs_numeric_row(path, key="imbalance_10_bps")
    signal = "Hyperliquid L2 snapshot exists"
    if best:
        signal = (
            f"{best.get('asset', '')}: spread_bps={best.get('spread_bps', '')}, "
            f"imbalance10={best.get('imbalance_10_bps', '')}"
        )
    return ExplorationRow(
        lane="market_making",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="no queue position, fill probability, adverse selection, or fee model",
        next_step="collect repeated L2 snapshots and estimate fill/adverse-selection risk",
    )


def _news_social_row(root: Path) -> ExplorationRow:
    path = root / "news_social" / "current_attention_snapshot.csv"
    fear = _row_by_value(path, field="source", value="alternative_me_fear_greed")
    trend = _row_by_value(path, field="source", value="coingecko_trending")
    signal = "attention snapshot exists"
    if fear and trend:
        signal = (
            f"fear_greed={fear.get('score', '')} {fear.get('label', '')}; "
            f"top_trending={trend.get('symbol', '')}"
        )
    return ExplorationRow(
        lane="news_social",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="attention data is not yet joined to leakage-safe return labels",
        next_step="build event-to-return labels and add richer news/social sources",
    )


def _stablecoin_liquidity_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_supply_snapshot.csv"
    best = _best_abs_numeric_row(path, key="week_change_usd")
    signal = "stablecoin supply snapshot exists"
    if best:
        signal = (
            f"{best.get('symbol', '')}: week_change_usd="
            f"{best.get('week_change_usd', '')}"
        )
    return ExplorationRow(
        lane="stablecoin_liquidity",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="supply changes are not yet joined to returns, funding, or regimes",
        next_step="test stablecoin supply change as market liquidity context",
    )


def _best_numeric_row(path: Path, *, key: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(key) or "-inf"))


def _best_abs_numeric_row(path: Path, *, key: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: abs(float(row.get(key) or "0")))


def _row_by_value(path: Path, *, field: str, value: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get(field) == value:
                return row
    return None


def _best_period_row(path: Path, *, period: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(row for row in csv.DictReader(handle) if row.get("period") == period)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("sharpe") or "-inf"))


def _best_watchlist_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    priority = {
        "paper_8h_monitor": 4,
        "paper_24h_monitor": 3,
        "current_funding_monitor": 2,
        "thin_or_wide_watch": 1,
        "blocked_by_cost_or_capacity": 0,
    }
    return max(
        rows,
        key=lambda row: (
            priority.get(row.get("action", ""), -1),
            float(row.get("net_24h_proxy") or "0"),
            float(row.get("annualized_edge") or "0"),
            float(row.get("liquidity_proxy") or "0"),
        ),
    )


def _best_monitor_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    priority = {
        "paper_8h_monitor": 4,
        "paper_24h_monitor": 3,
        "current_funding_monitor": 2,
    }
    return max(
        rows,
        key=lambda row: (
            priority.get(row.get("action", ""), -1),
            int(row.get("observations") or "0"),
            float(row.get("positive_net_24h_rate") or "0"),
            float(row.get("mean_net_24h_proxy") or "0"),
            float(row.get("mean_annualized_edge") or "0"),
        ),
    )


def _best_execution_check_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    priority = {
        "conservative_taker_monitor": 2,
        "fee_only_monitor": 1,
        "blocked": 0,
    }
    actionable_rows = tuple(row for row in rows if priority.get(row.get("action", ""), 0) > 0)
    if not actionable_rows:
        return None
    return max(
        actionable_rows,
        key=lambda row: (
            priority.get(row.get("action", ""), 0),
            float(row.get("fee_bps_per_fill_per_venue") or "0"),
            float(row.get("conservative_taker_net_24h") or "-inf"),
            float(row.get("fee_only_net_24h") or "0"),
        ),
    )


def _best_promotion_gate_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    paper_rows = tuple(row for row in rows if row.get("action", "").startswith("paper_"))
    if not paper_rows:
        return None
    return max(
        paper_rows,
        key=lambda row: (
            float(row.get("fee_bps_per_fill_per_venue") or "0"),
            row.get("horizon") == "8h",
            float(row.get("fee_headroom_bps") or "0"),
            float(row.get("capacity") or "0"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "exploration_board.md",
    )
    args = parser.parse_args()
    path = write_exploration_board(build_exploration_rows(), output_path=args.output_path)
    print(path)


if __name__ == "__main__":
    main()
