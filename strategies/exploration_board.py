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
    best = _best_numeric_row(gate_path, key="headroom_bps")
    signal = "spot/perp carry screen exists"
    if best:
        signal = (
            f"{best.get('candidate', 'spot_perp_carry')}: "
            f"{best.get('scenario', 'scenario')} headroom="
            f"{best.get('headroom_bps', '')}bps, "
            f"default_sharpe={best.get('default_cost_sharpe', '')}"
        )
    return ExplorationRow(
        lane="crypto_market_structure",
        status="execution_gate_candidate",
        strongest_current_signal=signal,
        main_gap="actual account fees, borrow/margin, and book-depth feasibility remain shallow",
        next_step="validate venue-specific fees, margin, and symbol availability before paper trading",
    )


def _cross_exchange_funding_row(root: Path) -> ExplorationRow:
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
