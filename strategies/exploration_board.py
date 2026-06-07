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
        _alpha_stack_row(root),
        _crypto_market_structure_row(root),
        _basis_term_structure_row(root),
        _cross_exchange_funding_row(root),
        _perp_market_map_row(root),
        _derivatives_positioning_row(root),
        _macro_regime_row(root),
        _crypto_equity_proxy_row(root),
        _speculative_beta_row(root),
        _event_flow_row(root),
        _liquidation_flow_row(root),
        _defi_yield_row(root),
        _dex_pool_flow_row(root),
        _market_making_row(root),
        _options_volatility_row(root),
        _sector_rotation_row(root),
        _exchange_catalyst_row(root),
        _token_unlocks_row(root),
        _news_social_row(root),
        _prediction_markets_row(root),
        _protocol_activity_row(root),
        _institutional_flow_row(root),
        _candidate_validation_row(root),
        _stablecoin_liquidity_row(root),
        _on_chain_flow_row(root),
        _protocol_fundamentals_row(root),
    )


def _alpha_stack_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_stack.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="alpha_stack",
            status=best.get("status", "candidate_generation"),
            strongest_current_signal=(
                f"{best.get('opportunity', '')}: "
                f"{best.get('side', '')}, "
                f"priority={best.get('priority_score', '')}, "
                f"sources={best.get('sources', '')}"
            ),
            main_gap=best.get("conflict", "cross-lane stack still needs validation"),
            next_step=best.get("next_step", "validate top cross-lane paper candidate"),
        )
    return ExplorationRow(
        lane="alpha_stack",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="current paper candidates are not joined across lanes",
        next_step="run current alpha stack to identify cross-lane candidate priorities",
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


def _basis_term_structure_row(root: Path) -> ExplorationRow:
    path = root / "basis_term_structure" / "current_deribit_futures_basis.csv"
    if not path.exists():
        rows: tuple[dict[str, str], ...] = ()
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = tuple(
                row
                for row in csv.DictReader(handle)
                if row.get("status")
                in {"paper_short_basis_watch", "paper_long_basis_watch", "basis_term_structure_watch"}
            )
    best = max(rows, key=lambda row: float(row.get("score") or "0")) if rows else None
    if best:
        return ExplorationRow(
            lane="basis_term_structure",
            status=best.get("status", "basis_screen"),
            strongest_current_signal=(
                f"{best.get('instrument_name', '')}: "
                f"ann_basis={best.get('annualized_basis', '')}, "
                f"dte={best.get('days_to_expiry', '')}, "
                f"volume={best.get('volume_usd', '')}, "
                f"spread={best.get('bid_ask_spread_pct', '')}"
            ),
            main_gap="dated-futures basis still lacks hedge route, funding, margin, fees, and order-book depth checks",
            next_step=best.get(
                "next_step",
                "check hedge route, funding, margin, fees, and depth before paper action",
            ),
        )
    return ExplorationRow(
        lane="basis_term_structure",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="dated futures basis has not been screened",
        next_step="run Deribit futures basis screen for BTC/ETH term structure",
    )


def _cross_exchange_funding_row(root: Path) -> ExplorationRow:
    best_execution_check = _best_execution_check_row(
        root / "cross_exchange_funding" / "stable_12_sample_execution_check.csv"
    ) or _best_execution_check_row(
        root / "cross_exchange_funding" / "current_dislocation_execution_check.csv"
    )
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
    okx_signal = _okx_perp_pressure_signal(
        root / "perp_market_map" / "current_okx_perp_pressure.csv",
        root / "perp_market_map" / "current_okx_perp_pressure_forward_labels.csv",
    )
    crowding_monitor_path = (
        root / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv"
    )
    best_crowding_monitor = _best_crowding_monitor_row(crowding_monitor_path)
    if best_crowding_monitor:
        return ExplorationRow(
            lane="perp_market_map",
            status="short_window_carry_reversion_monitor",
            strongest_current_signal=(
                f"{best_crowding_monitor.get('asset', '')}: "
                f"{best_crowding_monitor.get('action', '')}, "
                f"obs={best_crowding_monitor.get('observations', '')}, "
                f"mean_score={best_crowding_monitor.get('mean_score', '')}, "
                f"mean_funding={best_crowding_monitor.get('mean_annualized_funding', '')}"
                f"{okx_signal}"
            ),
            main_gap="persistent crowding and OKX pressure proxies are not yet joined to future returns or execution costs",
            next_step="label top HL/OKX pressure rows for subsequent returns, funding decay, and liquidation risk",
        )
    crowding_path = root / "perp_market_map" / "current_crowding_reversion_screen.csv"
    best_crowding = _best_numeric_row(crowding_path, key="carry_reversion_score")
    if best_crowding:
        return ExplorationRow(
            lane="perp_market_map",
            status="current_carry_reversion_screen",
            strongest_current_signal=(
                f"{best_crowding.get('asset', '')}: {best_crowding.get('action', '')}, "
                f"funding={best_crowding.get('annualized_funding', '')}, "
                f"mark_oracle={best_crowding.get('mark_oracle_diff', '')}, "
                f"score={best_crowding.get('carry_reversion_score', '')}"
                f"{okx_signal}"
            ),
            main_gap="current crowding and OKX pressure proxies are not yet joined to future returns or execution costs",
            next_step="monitor top HL/OKX pressure rows and label subsequent returns, funding decay, and liquidation risk",
        )
    path = root / "perp_market_map" / "current_hyperliquid_snapshot.csv"
    best = _best_numeric_row(path, key="attention_score")
    signal = "not run yet"
    if best:
        signal = (
            f"{best.get('asset', '')}: ann_funding={best.get('annualized_funding', '')}, "
            f"volume={best.get('day_notional_volume', '')}"
            f"{okx_signal}"
        )
    return ExplorationRow(
        lane="perp_market_map",
        status="current_snapshot",
        strongest_current_signal=signal,
        main_gap="no history yet, so no persistence or PnL evidence",
        next_step="collect snapshots over time and test carry/crowding persistence",
    )


def _derivatives_positioning_row(root: Path) -> ExplorationRow:
    current_path = root / "derivatives_positioning" / "current_coingecko_derivatives_positioning.csv"
    current_rows = tuple(
        row
        for row in _csv_rows(current_path)
        if row.get("status")
        in {
            "paper_oi_funding_crowding_watch",
            "paper_basis_funding_dislocation_watch",
            "paper_derivatives_momentum_risk_watch",
        }
    )
    best_current = max(current_rows, key=lambda row: float(row.get("score") or "0")) if current_rows else None
    if best_current:
        return ExplorationRow(
            lane="derivatives_positioning",
            status=best_current.get("status", "current_positioning_screen"),
            strongest_current_signal=(
                f"{best_current.get('market', '')} {best_current.get('symbol', '')}: "
                f"oi={best_current.get('open_interest', '')}, "
                f"vol24={best_current.get('volume_24h', '')}, "
                f"funding={best_current.get('funding_rate', '')}, "
                f"basis={best_current.get('basis', '')}, "
                f"score={best_current.get('score', '')}"
            ),
            main_gap="current derivatives positioning still lacks venue-specific depth, funding timing, fees, margin, and forward labels",
            next_step=best_current.get(
                "next_step",
                "label forward returns, funding PnL, depth, fees, and margin constraints",
            ),
        )
    path = root / "p0_parallel" / "binance_derivatives_signal_summary.csv"
    best_corr = _best_abs_numeric_row(path, key="correlation_to_next_return")
    best_hit = _best_numeric_row(path, key="high_bucket_hit_rate")
    if best_corr:
        hit_note = ""
        if best_hit:
            hit_note = (
                f"; high_hit {best_hit.get('feature', '')}="
                f"{best_hit.get('high_bucket_hit_rate', '')}"
            )
        return ExplorationRow(
            lane="derivatives_positioning",
            status="broad_history_screen",
            strongest_current_signal=(
                f"{best_corr.get('feature', '')}: "
                f"obs={best_corr.get('observations', '')}, "
                f"corr={best_corr.get('correlation_to_next_return', '')}, "
                f"high_mean={best_corr.get('high_bucket_mean_next_return', '')}"
                f"{hit_note}"
            ),
            main_gap="daily Binance positioning labels exclude regime splits, execution costs, and venue-specific carry PnL",
            next_step="split OI/funding/premium/long-short effects by regime and test cost-aware carry plus reversal labels",
        )
    return ExplorationRow(
        lane="derivatives_positioning",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="OI, funding, premium, and long-short history are not summarized",
        next_step="run Binance derivatives history over a broad symbol and date panel",
    )


def _macro_regime_row(root: Path) -> ExplorationRow:
    ticket_path = root / "macro_regime" / "current_macro_crypto_paper_tickets.csv"
    best_ticket = _best_macro_regime_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="macro_regime",
            status=best_ticket.get("status", "watch"),
            strongest_current_signal=(
                f"{best_ticket.get('name', '')}: "
                f"{best_ticket.get('side', '')}, "
                f"score={best_ticket.get('score', '')}, "
                f"{best_ticket.get('reason', '')}"
            ),
            main_gap="macro regime screen is current-context only and lacks repeated forward labels, costs, and crypto venue mapping",
            next_step="repeat macro/crypto regime labels and join them to funding, liquidation, and BTC ETF flow candidates",
        )
    snapshot_path = root / "macro_regime" / "current_macro_crypto_context.csv"
    best_snapshot = _best_abs_numeric_row(snapshot_path, key="risk_score")
    signal = "not run yet"
    if best_snapshot:
        signal = (
            f"{best_snapshot.get('symbol', '')}: "
            f"group={best_snapshot.get('group', '')}, "
            f"risk_score={best_snapshot.get('risk_score', '')}, "
            f"ret5d={best_snapshot.get('return_5d', '')}"
        )
    return ExplorationRow(
        lane="macro_regime",
        status="current_context",
        strongest_current_signal=signal,
        main_gap="macro/crypto context has not been turned into repeated labels",
        next_step="build forward labels for risk-on catch-up, risk-off lagged short, and ETH/BTC beta rotation",
    )


def _crypto_equity_proxy_row(root: Path) -> ExplorationRow:
    ticket_path = root / "crypto_equity_proxy" / "current_crypto_equity_proxy_paper_tickets.csv"
    best_ticket = _best_crypto_equity_proxy_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="crypto_equity_proxy",
            status=best_ticket.get("status", "watch"),
            strongest_current_signal=(
                f"{best_ticket.get('name', '')}: "
                f"{best_ticket.get('side', '')}, "
                f"score={best_ticket.get('score', '')}, "
                f"{best_ticket.get('reason', '')}"
            ),
            main_gap="crypto equity proxy signal is current-context only and lacks repeated forward labels, borrow/funding costs, and execution mapping",
            next_step="label MSTR/BTC, COIN/HOOD beta, and miner stress against BTC/ETH returns and funding context",
        )
    context_path = root / "crypto_equity_proxy" / "current_crypto_equity_proxy_context.csv"
    best_context = _best_abs_numeric_row(context_path, key="vs_btc_5d")
    signal = "not run yet"
    if best_context:
        signal = (
            f"{best_context.get('symbol', '')}: "
            f"group={best_context.get('group', '')}, "
            f"vs_btc_5d={best_context.get('vs_btc_5d', '')}, "
            f"ret5d={best_context.get('return_5d', '')}"
        )
    return ExplorationRow(
        lane="crypto_equity_proxy",
        status="current_context",
        strongest_current_signal=signal,
        main_gap="crypto-linked equity proxies have not been converted into repeated labels",
        next_step="build paper tickets for proxy lead/lag, MSTR/BTC dislocation, and miner stress",
    )


def _speculative_beta_row(root: Path) -> ExplorationRow:
    ticket_path = root / "speculative_beta" / "current_speculative_beta_paper_tickets.csv"
    best_ticket = _best_speculative_beta_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="speculative_beta",
            status=best_ticket.get("status", "watch"),
            strongest_current_signal=(
                f"{best_ticket.get('name', '')}: "
                f"{best_ticket.get('side', '')}, "
                f"score={best_ticket.get('score', '')}, "
                f"{best_ticket.get('reason', '')}"
            ),
            main_gap="speculative beta screen is current-context only and lacks repeated labels, crypto venue costs, and regime splits",
            next_step="label VIX/high-beta shocks, AI/BTC divergence, and semiconductor/BTC divergence against BTC/ETH returns and funding",
        )
    context_path = root / "speculative_beta" / "current_speculative_beta_context.csv"
    best_context = _best_abs_numeric_row(context_path, key="risk_score_5d")
    signal = "not run yet"
    if best_context:
        signal = (
            f"{best_context.get('symbol', '')}: "
            f"group={best_context.get('group', '')}, "
            f"risk_score_5d={best_context.get('risk_score_5d', '')}, "
            f"vs_btc_5d={best_context.get('vs_btc_5d', '')}"
        )
    return ExplorationRow(
        lane="speculative_beta",
        status="current_context",
        strongest_current_signal=signal,
        main_gap="speculative equity beta has not been converted into repeated crypto labels",
        next_step="build paper tickets for high-beta lead/lag, AI divergence, semiconductor divergence, and VIX air pockets",
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


def _liquidation_flow_row(root: Path) -> ExplorationRow:
    monitor_path = root / "liquidation_flow" / "current_okx_liquidation_monitor_summary.csv"
    actionability_path = root / "liquidation_flow" / "current_okx_liquidation_actionability_review.csv"
    paper_outcome_path = root / "liquidation_flow" / "current_okx_liquidation_paper_outcome.csv"
    best_outcome = _best_paper_outcome_row(paper_outcome_path)
    if best_outcome:
        next_step = (
            "wait for 1h outcome and repeat JTO/ONDO/LTC gate on fresh liquidation events"
        )
        if best_outcome.get("outcome_1h") != "pending_1h":
            next_step = "repeat ONDO/LTC/JTO on fresh liquidation events with live depth and fee checks"
        return ExplorationRow(
            lane="liquidation_flow",
            status="current_okx_paper_outcome",
            strongest_current_signal=(
                f"{best_outcome.get('asset', '')}: {best_outcome.get('action', '')}, "
                f"dir={best_outcome.get('paper_direction', '')}, "
                f"size={best_outcome.get('candidate_size_usd', '')}, "
                f"net15={best_outcome.get('net_15m_bps', '')}bps, "
                f"out15={best_outcome.get('outcome_15m', '')}, "
                f"out1h={best_outcome.get('outcome_1h', '')}"
            ),
            main_gap="paper outcome is retrospective and has no live fill or repeated fresh-event sample yet",
            next_step=next_step,
        )
    paper_gate_path = root / "liquidation_flow" / "current_okx_liquidation_paper_gate.csv"
    best_paper = _best_paper_gate_row(paper_gate_path)
    if best_paper:
        return ExplorationRow(
            lane="liquidation_flow",
            status="current_okx_paper_gate",
            strongest_current_signal=(
                f"{best_paper.get('asset', '')}: {best_paper.get('action', '')}, "
                f"size={best_paper.get('candidate_size_usd', '')}, "
                f"net_bps={best_paper.get('conservative_net_bps', '')}, "
                f"depth_usage={best_paper.get('visible_depth_usage', '')}, "
                f"gate={best_paper.get('gate_action', '')}"
            ),
            main_gap="paper gate still uses assumed fees, public visible depth, and short-window labels only",
            next_step="repeat JTO/ONDO/LTC gate over fresh events and compare with actual paper fills",
        )
    best_actionable = _best_numeric_row(actionability_path, key="actionability_score")
    if best_actionable:
        return ExplorationRow(
            lane="liquidation_flow",
            status="current_okx_actionability_review",
            strongest_current_signal=(
                f"{best_actionable.get('asset', '')}: {best_actionable.get('action', '')}, "
                f"score={best_actionable.get('actionability_score', '')}, "
                f"cont15={best_actionable.get('continuation_return_15m', '')}, "
                f"near_depth5={best_actionable.get('near_touch_depth_5bps', '')}, "
                f"note={best_actionable.get('note', '')}"
            ),
            main_gap="actionability is still based on visible depth and short monitor labels only",
            next_step="repeat JTO/ONDO/LTC labels and add real fee/slippage assumptions before paper sizing",
        )
    best_monitor = _best_numeric_row(monitor_path, key="mean_cascade_score")
    if best_monitor:
        depth_path = root / "liquidation_flow" / "current_okx_liquidation_depth_check.csv"
        best_depth = _best_numeric_row(depth_path, key="depth_score")
        depth_note = ""
        if best_depth:
            depth_note = (
                f"; depth {best_depth.get('asset', '')}: "
                f"spread={best_depth.get('spread_bps', '')}, "
                f"bid5={best_depth.get('bid_depth_5bps', '')}, "
                f"ask5={best_depth.get('ask_depth_5bps', '')}"
            )
        return ExplorationRow(
            lane="liquidation_flow",
            status="current_okx_event_flow_monitor",
            strongest_current_signal=(
                f"{best_monitor.get('asset', '')}: {best_monitor.get('action', '')}, "
                f"obs={best_monitor.get('observations', '')}, "
                f"mean_score={best_monitor.get('mean_cascade_score', '')}, "
                f"mean_liq={best_monitor.get('mean_total_liquidation_notional', '')}, "
                f"mean_imbalance={best_monitor.get('mean_forced_buy_sell_imbalance', '')}"
                f"{depth_note}"
            ),
            main_gap="monitor persistence exists, but strongest signals have thin visible near-touch depth",
            next_step="label BEAT/WLD monitor timestamps and test smaller paper sizes or alternate venues",
        )
    path = root / "liquidation_flow" / "current_okx_liquidation_flow.csv"
    label_path = root / "liquidation_flow" / "current_okx_liquidation_forward_labels.csv"
    best = _best_numeric_row(path, key="cascade_score")
    signal = "not run yet"
    if best:
        label = _label_row_for_asset(label_path, asset=best.get("asset", ""))
        label_note = ""
        if label and label.get("continuation_return_15m", "") != "":
            label_note = f", cont15={label.get('continuation_return_15m', '')}"
        signal = (
            f"{best.get('asset', '')}: {best.get('action', '')}, "
            f"obs={best.get('observations', '')}, "
            f"total_liq={best.get('total_liquidation_notional', '')}, "
            f"imbalance={best.get('forced_buy_sell_imbalance', '')}"
            f"{label_note}"
        )
    return ExplorationRow(
        lane="liquidation_flow",
        status="current_okx_event_flow_labeled",
        strongest_current_signal=signal,
        main_gap="15m continuation labels exist, but 1h/regime/execution labels are still missing",
        next_step="repeat liquidation snapshots and separate continuation candidates from reversal candidates",
    )


def _defi_yield_row(root: Path) -> ExplorationRow:
    quality_path = root / "defi_yield" / "current_yield_quality_screen.csv"
    best_quality = _best_numeric_row(quality_path, key="score")
    if best_quality:
        return ExplorationRow(
            lane="defi_yield",
            status=best_quality.get("status", "yield_quality_screen"),
            strongest_current_signal=(
                f"{best_quality.get('chain', '')}/{best_quality.get('project', '')} "
                f"{best_quality.get('symbol', '')}: apy={best_quality.get('apy', '')}, "
                f"base={best_quality.get('apy_base', '')}, "
                f"reward_share={best_quality.get('reward_share', '')}, "
                f"tvl={best_quality.get('tvl_usd', '')}"
            ),
            main_gap="yield candidates still need custody, smart-contract, issuer, APY-decay, and exit-liquidity checks",
            next_step=best_quality.get(
                "next_step",
                "check custody, APY source, capacity, and exit liquidity before paper allocation",
            ),
        )
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


def _dex_pool_flow_row(root: Path) -> ExplorationRow:
    path = root / "dex_pool_flow" / "current_geckoterminal_pool_flow.csv"
    rows = tuple(
        row
        for row in _csv_rows(path)
        if row.get("status")
        in {
            "paper_dex_pool_momentum_watch",
            "paper_dex_reversal_risk_watch",
            "dex_liquidity_stress_watch",
        }
    )
    best = max(rows, key=lambda row: float(row.get("score") or "0")) if rows else None
    if best:
        return ExplorationRow(
            lane="dex_pool_flow",
            status=best.get("status", "dex_pool_flow_screen"),
            strongest_current_signal=(
                f"{best.get('network', '')}/{best.get('dex', '')} {best.get('name', '')}: "
                f"vol1h={best.get('volume_h1_usd', '')}, "
                f"reserve={best.get('reserve_usd', '')}, "
                f"vol_reserve={best.get('volume_reserve_ratio_h1', '')}, "
                f"chg1h={best.get('price_change_h1', '')}, "
                f"chg24h={best.get('price_change_h24', '')}"
            ),
            main_gap="DEX pool candidates need route simulation, slippage, gas, MEV, token-transfer, and repeated-flow checks",
            next_step=best.get(
                "next_step",
                "check route depth, slippage, gas, MEV, token restrictions, and repeated pool flow",
            ),
        )
    return ExplorationRow(
        lane="dex_pool_flow",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="DEX pool flow has not been screened",
        next_step="run GeckoTerminal pool-flow screen",
    )


def _market_making_row(root: Path) -> ExplorationRow:
    paper_gate_path = root / "market_making" / "current_l2_imbalance_paper_gate.csv"
    best_gate = _best_l2_imbalance_paper_gate_row(paper_gate_path)
    if best_gate:
        net_1h_note = ""
        if best_gate.get("net_1h_bps", ""):
            net_1h_note = f"net1h={best_gate.get('net_1h_bps', '')}bps, "
        return ExplorationRow(
            lane="market_making",
            status="l2_imbalance_paper_gate",
            strongest_current_signal=(
                f"{best_gate.get('asset', '')}: "
                f"size={best_gate.get('candidate_size_usd', '')}, "
                f"net15={best_gate.get('net_15m_bps', '')}bps, "
                f"{net_1h_note}"
                f"depth_usage={best_gate.get('visible_depth_usage', '')}"
            ),
            main_gap="paper gate is directional and still excludes maker queue, fill probability, rebates, and repeated adverse-selection samples",
            next_step="repeat JTO/XLM/NEAR/XPL L2 gates on fresh snapshots and then design a maker-fill observation log",
        )
    label_path = root / "market_making" / "current_l2_imbalance_forward_labels.csv"
    best_label = _best_l2_imbalance_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="market_making",
            status="l2_imbalance_forward_label",
            strongest_current_signal=(
                f"{best_label.get('asset', '')}: "
                f"imbalance10={best_label.get('imbalance_10_bps', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"dir1h={best_label.get('directional_return_1h', '')}, "
                f"spread={best_label.get('spread_bps', '')}"
            ),
            main_gap="one L2 snapshot label excludes queue position, fill probability, fees, and repeated adverse-selection samples",
            next_step="collect repeated L2 imbalance labels and estimate maker fill/adverse-selection risk",
        )
    monitor_path = root / "market_making" / "current_l2_imbalance_monitor_summary.csv"
    best_monitor = _best_l2_imbalance_monitor_row(monitor_path)
    if best_monitor:
        return ExplorationRow(
            lane="market_making",
            status="l2_imbalance_monitor",
            strongest_current_signal=(
                f"{best_monitor.get('asset', '')}: "
                f"dir={best_monitor.get('dominant_direction', '')}, "
                f"persist={best_monitor.get('direction_persistence_rate', '')}, "
                f"mean_abs={best_monitor.get('mean_abs_imbalance_10_bps', '')}, "
                f"near_depth={best_monitor.get('mean_near_depth_10bps_notional', '')}"
            ),
            main_gap="persistent L2 imbalance candidates are still unlabeled against 15m/1h returns",
            next_step="rerun forward labels after 15m/1h and gate BTC/SOL/ONDO/XPL with costs",
        )
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
        status="current_broad_l2_snapshot",
        strongest_current_signal=signal,
        main_gap="broad L2 imbalance snapshot is unlabeled until 15m/1h outcomes mature",
        next_step="rerun L2 imbalance forward labels after 15m/1h and then gate WLD/ZEC/HYPE/SOL/BTC",
    )


def _options_volatility_row(root: Path) -> ExplorationRow:
    ticket_path = root / "options_volatility" / "current_options_volatility_paper_tickets.csv"
    best_ticket = _best_options_volatility_paper_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="options_volatility",
            status=best_ticket.get("status", "paper_options_watch"),
            strongest_current_signal=(
                f"{best_ticket.get('currency', '')} {best_ticket.get('expiry', '')}: "
                f"{best_ticket.get('structure', '')}, "
                f"prem24={best_ticket.get('iv_premium_24h', '')}, "
                f"skew={best_ticket.get('skew_iv', '')}, "
                f"volume={best_ticket.get('volume_usd', '')}"
            ),
            main_gap="options paper ticket still lacks spread quotes, delta hedge PnL, margin, and realized-vol forecast",
            next_step="paper-check BTC short put spread quotes, delta hedge cost, tail loss, and expiry handling before any live action",
        )
    label_path = root / "options_volatility" / "current_deribit_options_realized_vol_labels.csv"
    best_label = _best_numeric_row(label_path, key="score")
    if best_label:
        return ExplorationRow(
            lane="options_volatility",
            status="current_iv_realized_context",
            strongest_current_signal=(
                f"{best_label.get('currency', '')} {best_label.get('expiry', '')}: "
                f"{best_label.get('action', '')}, "
                f"iv={best_label.get('atm_iv', '')}, "
                f"rv24={best_label.get('realized_vol_24h', '')}, "
                f"prem24={best_label.get('iv_premium_24h', '')}, "
                f"skew={best_label.get('skew_iv', '')}"
            ),
            main_gap="IV-vs-realized label lacks realized-vol forecast, option execution costs, margin, hedge PnL, and tail-risk controls",
            next_step="repeat BTC/ETH IV premium labels and add hedge-cost plus realized-vol forecast checks",
        )
    path = root / "options_volatility" / "current_deribit_options_surface.csv"
    best = _best_numeric_row(path, key="score")
    signal = "Deribit BTC/ETH option surface probe exists"
    if best:
        signal = (
            f"{best.get('currency', '')} {best.get('expiry', '')}: "
            f"{best.get('action', '')}, "
            f"atm_iv={best.get('atm_iv', '')}, "
            f"skew={best.get('skew_iv', '')}, "
            f"term={best.get('term_iv_spread_to_next', '')}, "
            f"score={best.get('score', '')}"
        )
    return ExplorationRow(
        lane="options_volatility",
        status="current_deribit_surface",
        strongest_current_signal=signal,
        main_gap="surface snapshot lacks realized-vol baseline, option execution costs, margin, and hedge rules",
        next_step="join IV/skew/term candidates to realized volatility and hedge-cost labels",
    )


def _best_options_volatility_paper_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(
        row
        for row in rows
        if row.get("status") in {
            "paper_short_put_spread_candidate",
            "paper_calendar_spread_watch",
        }
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            1.0 if row.get("status") == "paper_short_put_spread_candidate" else 0.5,
            float(row.get("score") or "0"),
        ),
    )


def _sector_rotation_row(root: Path) -> ExplorationRow:
    context_path = root / "sector_rotation" / "current_category_perp_context.csv"
    best_context = _best_category_perp_context_row(context_path)
    if best_context:
        return ExplorationRow(
            lane="sector_rotation",
            status="category_perp_context",
            strongest_current_signal=(
                f"{best_context.get('category_name', '')}/{best_context.get('symbol', '')}: "
                f"action={best_context.get('action', '')}, "
                f"funding_support={best_context.get('best_funding_support', '')}, "
                f"score={best_context.get('context_score', '')}"
            ),
            main_gap="category-perp context has mostly failed short labels; remaining pending rows are not promotion evidence",
            next_step="deprioritize sector-perp continuation until repeated labels beat costs and funding support",
        )
    label_path = root / "sector_rotation" / "current_category_tradable_forward_labels.csv"
    best_label = _best_sector_tradable_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="sector_rotation",
            status="category_tradable_forward_label",
            strongest_current_signal=(
                f"{best_label.get('category_name', '')}/{best_label.get('symbol', '')}: "
                f"{best_label.get('category_action', '')}, "
                f"change24={best_label.get('category_change_24h', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"dir1h={best_label.get('directional_return_1h', '')}"
            ),
            main_gap="single category-to-symbol label; no repeated evidence, constituent weighting, liquidity, or costs",
            next_step="repeat top category-to-symbol labels and test continuation vs reversal by category family",
        )
    path = root / "sector_rotation" / "current_coingecko_category_rotation.csv"
    best = _best_numeric_row(path, key="score")
    signal = "CoinGecko category rotation probe exists"
    if best:
        signal = (
            f"{best.get('name', '')}: {best.get('action', '')}, "
            f"change24={best.get('market_cap_change_24h', '')}, "
            f"top={best.get('top_3_coins_id', '')}, "
            f"score={best.get('score', '')}"
        )
    return ExplorationRow(
        lane="sector_rotation",
        status="current_category_rotation",
        strongest_current_signal=signal,
        main_gap="category move is not mapped to tradable constituents, forward labels, liquidity, or costs",
        next_step="map top rotating categories to tradable coins and label category continuation/reversal",
    )


def _news_social_row(root: Path) -> ExplorationRow:
    label_path = root / "news_social" / "current_attention_forward_labels.csv"
    best_label = _best_attention_forward_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="news_social",
            status="attention_forward_label",
            strongest_current_signal=(
                f"{best_label.get('symbol', '')}: {best_label.get('action', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"dir1h={best_label.get('directional_return_1h', '')}, "
                f"score={best_label.get('score', '')}"
            ),
            main_gap="attention label is one event and excludes fees, funding PnL, causality, and neutral baselines",
            next_step="collect repeated attention/perp-overlap events and test 15m vs 1h decay",
        )
    join_path = root / "news_social" / "current_attention_market_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        signal = (
            f"{best_join.get('symbol', '')}: {best_join.get('action', '')}, "
            f"rank={best_join.get('attention_rank', '')}, "
            f"change={best_join.get('attention_24h_change', '')}, "
            f"carry_score={best_join.get('carry_reversion_score', '')}"
        )
        return ExplorationRow(
            lane="news_social",
            status="attention_market_join",
            strongest_current_signal=signal,
            main_gap="attention/perp overlap is not yet labeled against future returns",
            next_step="label AERO and other overlap candidates for subsequent return and funding decay",
        )
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


def _exchange_catalyst_row(root: Path) -> ExplorationRow:
    label_path = root / "news_social" / "current_exchange_catalyst_forward_labels.csv"
    best_label = _best_exchange_catalyst_forward_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="exchange_catalysts",
            status="exchange_catalyst_forward_label",
            strongest_current_signal=(
                f"{best_label.get('symbol', '')}: {best_label.get('catalyst_kind', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"dir1h={best_label.get('directional_return_1h', '')}, "
                f"score={best_label.get('score', '')}"
            ),
            main_gap="exchange catalyst label is event-reactive and excludes costs, funding PnL, and latency",
            next_step="repeat Binance listing/removal labels and join them to venue depth plus funding decay",
        )
    join_path = root / "news_social" / "current_exchange_catalyst_market_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        return ExplorationRow(
            lane="exchange_catalysts",
            status="exchange_catalyst_market_join",
            strongest_current_signal=(
                f"{best_join.get('symbol', '')}: {best_join.get('catalyst_kind', '')}, "
                f"action={best_join.get('action', '')}, "
                f"score={best_join.get('score', '')}"
            ),
            main_gap="exchange catalyst is not yet labeled against future returns",
            next_step="label announcement reactions over 15m and 1h before promotion",
        )
    snapshot_path = root / "news_social" / "current_exchange_catalyst_snapshot.csv"
    best_snapshot = _best_numeric_row(snapshot_path, key="score")
    signal = "exchange catalyst snapshot is missing"
    if best_snapshot:
        signal = (
            f"{best_snapshot.get('symbol', '')}: "
            f"{best_snapshot.get('catalyst_kind', '')}, "
            f"score={best_snapshot.get('score', '')}"
        )
    return ExplorationRow(
        lane="exchange_catalysts",
        status="exchange_catalyst_snapshot",
        strongest_current_signal=signal,
        main_gap="exchange announcements are not joined to tradable venues or labels",
        next_step="join exchange catalysts to perp venue state and label event reactions",
    )


def _token_unlocks_row(root: Path) -> ExplorationRow:
    ticket_path = root / "token_unlocks" / "current_token_unlock_paper_tickets.csv"
    best_ticket = _best_token_unlock_paper_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="token_unlocks",
            status=best_ticket.get("status", "paper_short_candidate"),
            strongest_current_signal=(
                f"{best_ticket.get('symbol', '')}: side={best_ticket.get('side', '')}, "
                f"value={best_ticket.get('unlock_value_usd', '')}, "
                f"supply={best_ticket.get('percent_supply', '')}, "
                f"funding={best_ticket.get('annualized_funding', '')}, "
                f"impact={best_ticket.get('impact_spread', '')}"
            ),
            main_gap="unlock paper tickets still lack event-window labels, depth decay, stop logic, and funding persistence",
            next_step="paper-track HYPE/ZRO/KAITO/EIGEN unlock windows and ME crowded-short squeeze risk against returns, funding, and depth",
        )
    join_path = root / "token_unlocks" / "current_token_unlock_market_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        return ExplorationRow(
            lane="token_unlocks",
            status="supply_event_market_join",
            strongest_current_signal=(
                f"{best_join.get('symbol', '')}: {best_join.get('action', '')}, "
                f"value={best_join.get('unlock_value_usd', '')}, "
                f"supply={best_join.get('percent_supply', '')}, "
                f"funding={best_join.get('annualized_funding', '')}, "
                f"score={best_join.get('score', '')}"
            ),
            main_gap="unlock events are joined to tradable venues but not labeled around the unlock window",
            next_step="label ME/ZRO/KAITO/HYPE unlock windows against returns, funding PnL, and venue depth",
        )
    path = root / "token_unlocks" / "current_token_unlock_snapshot.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="token_unlocks",
            status="supply_event_snapshot",
            strongest_current_signal=(
                f"{best.get('symbol', '')}: {best.get('action', '')}, "
                f"value={best.get('unlock_value_usd', '')}, "
                f"supply={best.get('percent_supply', '')}, "
                f"days={best.get('days_until', '')}"
            ),
            main_gap="unlock events are not joined to tradable venues or forward labels",
            next_step="join current unlocks to perp venue state and label event windows",
        )
    return ExplorationRow(
        lane="token_unlocks",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="scheduled supply events are not connected",
        next_step="fetch token unlock calendar and join tradable tokens to venue state",
    )


def _best_token_unlock_paper_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(
        row
        for row in rows
        if row.get("status") in {"paper_short_candidate", "crowded_short_risk"}
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            1.0 if row.get("status") == "paper_short_candidate" else 0.5,
            float(row.get("score") or "0"),
        ),
    )


def _prediction_markets_row(root: Path) -> ExplorationRow:
    ticket_path = root / "prediction_markets" / "current_prediction_market_paper_tickets.csv"
    best_ticket = _best_prediction_market_paper_ticket(ticket_path)
    if best_ticket:
        return ExplorationRow(
            lane="prediction_markets",
            status=best_ticket.get("status", "paper_event_model_candidate"),
            strongest_current_signal=(
                f"{best_ticket.get('question', '')} {best_ticket.get('outcome', '')}: "
                f"{best_ticket.get('structure', '')}, "
                f"spread={best_ticket.get('spread', '')}, "
                f"depth={best_ticket.get('visible_depth_score', '')}, "
                f"vol24={best_ticket.get('volume_24h', '')}"
            ),
            main_gap="prediction-market paper ticket still lacks true-probability model, news feed, latency, and adverse-selection checks",
            next_step="build event-model checks for MicroStrategy BTC purchase and Israel/Iran airspace markets before any live action",
        )
    depth_path = root / "prediction_markets" / "current_polymarket_clob_depth.csv"
    best_depth = _best_numeric_row(depth_path, key="visible_depth_score")
    if best_depth:
        return ExplorationRow(
            lane="prediction_markets",
            status="clob_depth_check",
            strongest_current_signal=(
                f"{best_depth.get('question', '')} {best_depth.get('outcome', '')}: "
                f"spread={best_depth.get('spread', '')}, "
                f"bid_depth_5c={best_depth.get('bid_depth_to_5c', '')}, "
                f"ask_depth_5c={best_depth.get('ask_depth_to_5c', '')}"
            ),
            main_gap="event probability, news flow, and adverse selection are not modeled",
            next_step="build external event model for depth-positive markets before paper trading",
        )
    monitor_path = (
        root / "prediction_markets" / "current_polymarket_microstructure_monitor_summary.csv"
    )
    best_monitor = _best_polymarket_monitor_row(monitor_path)
    if best_monitor:
        signal = (
            f"{best_monitor.get('action', '')}: {best_monitor.get('question', '')}, "
            f"obs={best_monitor.get('observations', '')}, "
            f"score={best_monitor.get('mean_score', '')}, "
            f"spread={best_monitor.get('mean_spread', '')}"
        )
        return ExplorationRow(
            lane="prediction_markets",
            status="short_window_microstructure_monitor",
            strongest_current_signal=signal,
            main_gap="event probability and adverse selection are not modeled",
            next_step="join top markets to external event models and CLOB depth checks",
        )
    path = root / "prediction_markets" / "current_polymarket_microstructure.csv"
    best = _best_numeric_row(path, key="score")
    signal = "not run yet"
    if best:
        signal = (
            f"{best.get('action', '')}: {best.get('question', '')}, "
            f"spread={best.get('spread', '')}, "
            f"change={best.get('one_day_price_change', '')}, "
            f"vol24={best.get('volume_24h', '')}"
        )
    return ExplorationRow(
        lane="prediction_markets",
        status="current_microstructure_screen",
        strongest_current_signal=signal,
        main_gap="event probability is not modeled; this only ranks active public market structure",
        next_step="join top markets to external event models, order books, and adverse-selection checks",
    )


def _best_prediction_market_paper_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(
        row
        for row in rows
        if row.get("status") in {
            "paper_event_model_candidate",
            "paper_event_model_watch",
            "market_making_watch",
            "sports_market_making_watch",
        }
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            1.0 if row.get("status") == "paper_event_model_candidate" else 0.5,
            float(row.get("score") or "0"),
        ),
    )


def _best_macro_regime_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            1.0 if row.get("status", "").endswith("_candidate") else 0.0,
            abs(float(row.get("score") or "0")),
        ),
    )


def _best_crypto_equity_proxy_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            1.0
            if row.get("status")
            in {
                "paper_long_candidate",
                "paper_short_candidate",
                "paper_relative_value_watch",
                "paper_risk_context",
            }
            else 0.0,
            abs(float(row.get("score") or "0")),
        ),
    )


def _best_speculative_beta_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            1.0 if row.get("status") in {"paper_long_candidate", "paper_short_candidate"} else 0.0,
            abs(float(row.get("score") or "0")),
        ),
    )


def _protocol_activity_row(root: Path) -> ExplorationRow:
    label_path = root / "protocol_activity" / "current_protocol_activity_forward_labels.csv"
    best_label = _best_protocol_activity_forward_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="protocol_activity",
            status="protocol_activity_forward_label",
            strongest_current_signal=(
                f"{best_label.get('symbol', '')}: {best_label.get('action', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"dir1h={best_label.get('directional_return_1h', '')}, "
                f"score={best_label.get('score', '')}"
            ),
            main_gap="developer/community activity is slow context and the current short labels are weak",
            next_step="keep protocol activity as context and test longer horizons plus funding/event overlaps",
        )
    join_path = root / "protocol_activity" / "current_protocol_activity_market_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        return ExplorationRow(
            lane="protocol_activity",
            status="protocol_activity_market_join",
            strongest_current_signal=(
                f"{best_join.get('symbol', '')}: {best_join.get('action', '')}, "
                f"commits4w={best_join.get('commit_count_4_weeks', '')}, "
                f"funding={best_join.get('annualized_funding', '')}, "
                f"score={best_join.get('score', '')}"
            ),
            main_gap="protocol activity is not yet labeled against future returns",
            next_step="label protocol-activity/perp overlaps and compare short vs longer horizons",
        )
    path = root / "protocol_activity" / "current_coingecko_protocol_activity.csv"
    best = _best_numeric_row(path, key="score")
    signal = "protocol activity snapshot is missing"
    if best:
        signal = (
            f"{best.get('symbol', '')}: {best.get('action', '')}, "
            f"commits4w={best.get('commit_count_4_weeks', '')}, "
            f"score={best.get('score', '')}"
        )
    return ExplorationRow(
        lane="protocol_activity",
        status="protocol_activity_snapshot",
        strongest_current_signal=signal,
        main_gap="protocol activity is not joined to tradable venues or labels",
        next_step="join protocol activity to perp state and label forward returns",
    )


def _institutional_flow_row(root: Path) -> ExplorationRow:
    paper_ticket_path = root / "institutional_flow" / "current_btc_etf_funding_paper_ticket.csv"
    best_paper_ticket = _current_btc_etf_funding_paper_ticket(paper_ticket_path)
    current_candidate_path = root / "institutional_flow" / "current_btc_etf_funding_candidate.csv"
    current_candidate = _current_btc_etf_funding_candidate(current_candidate_path)
    if current_candidate:
        ticket_note = ""
        if best_paper_ticket:
            ticket_note = (
                f", top_venue={best_paper_ticket.get('venue', '')}"
                f"/{best_paper_ticket.get('instrument', '')}"
            )
        return ExplorationRow(
            lane="institutional_flow",
            status=current_candidate.get("status", "active_paper_watch"),
            strongest_current_signal=(
                f"{current_candidate.get('asset', '')} {current_candidate.get('side', '')}: "
                f"5d_flow={current_candidate.get('rolling_5d_flow_btc', '')}, "
                f"funding={current_candidate.get('annualized_funding', '')}, "
                f"hist_total={current_candidate.get('historical_total_return', '')}, "
                f"hist_hit={current_candidate.get('historical_hit_rate_5d', '')}"
                f"{ticket_note}"
            ),
            main_gap="current watch survived coarse 1h entry and adverse-excursion stress, but still lacks stop/fill and venue-specific mark/index checks",
            next_step="paper-check BTC short venue choice, stop criteria, mark/index basis, and actual account fee/fill assumptions before any live action",
        )
    paper_rule_path = root / "institutional_flow" / "btc_etf_flow_funding_candidate_summary.csv"
    best_paper_rule = _btc_etf_flow_funding_candidate_summary(paper_rule_path)
    if best_paper_rule:
        return ExplorationRow(
            lane="institutional_flow",
            status=best_paper_rule.get("action", "paper_rule_candidate"),
            strongest_current_signal=(
                f"{best_paper_rule.get('rule_key', '')}: "
                f"trades={best_paper_rule.get('trades', '')}, "
                f"total={best_paper_rule.get('total_return', '')}, "
                f"hit={best_paper_rule.get('hit_rate_5d', '')}, "
                f"mdd={best_paper_rule.get('max_drawdown', '')}"
            ),
            main_gap="paper rule is non-overlapping but still daily, fee-assumption based, and missing intraday execution/liquidation checks",
            next_step="test the BTC ETF-flow/funding rule with intraday entries, account fees, liquidation buffer, and BTC regime filters",
        )
    funding_regime_path = root / "institutional_flow" / "btc_etf_flow_funding_regime_summary.csv"
    best_funding_regime = _best_btc_etf_flow_funding_regime_row(funding_regime_path)
    if best_funding_regime:
        return ExplorationRow(
            lane="institutional_flow",
            status="btc_etf_flow_funding_regime_candidate",
            strongest_current_signal=(
                f"{best_funding_regime.get('group_key', '')}: "
                f"obs={best_funding_regime.get('observations', '')}, "
                f"dir5+funding={best_funding_regime.get('mean_directional_5d_with_funding', '')}, "
                f"start_funding_support={best_funding_regime.get('mean_start_funding_support', '')}, "
                f"hit={best_funding_regime.get('hit_rate_5d_with_funding', '')}"
            ),
            main_gap="ETF flow plus funding still excludes intraday timing, drawdown filters, liquidity, and OOS market-regime splits",
            next_step="convert the best ETF-flow/funding regime into a paper BTC perp rule with drawdown and execution-cost checks",
        )
    regime_path = root / "institutional_flow" / "btc_etf_flow_regime_summary.csv"
    best_regime = _best_btc_etf_flow_regime_row(regime_path)
    if best_regime:
        return ExplorationRow(
            lane="institutional_flow",
            status="btc_etf_flow_regime_candidate",
            strongest_current_signal=(
                f"{best_regime.get('group_key', '')}: "
                f"obs={best_regime.get('observations', '')}, "
                f"mean_dir5={best_regime.get('mean_directional_5d', '')}, "
                f"hit5={best_regime.get('hit_rate_5d', '')}"
            ),
            main_gap="ETF flow regime labels exclude funding PnL, intraday timing, and market-regime OOS splits",
            next_step="test large 5d ETF outflow and ETF distribution regimes against perp funding alignment and drawdown filters",
        )
    label_path = root / "institutional_flow" / "btc_etf_flow_forward_labels.csv"
    label_summary = _btc_etf_flow_label_summary(label_path)
    if label_summary:
        return ExplorationRow(
            lane="institutional_flow",
            status="btc_etf_flow_forward_label",
            strongest_current_signal=(
                f"obs={label_summary['observations']:.0f}, "
                f"mean_dir5={label_summary['mean_directional_5d']:.8f}, "
                f"hit5={label_summary['hit_rate_5d']:.4f}, "
                f"latest={label_summary['latest_action']}"
            ),
            main_gap="BTC ETF flow is a coarse daily regime label and excludes funding PnL, intraday timing, and regime splits",
            next_step="split ETF flow labels by BTC regime, perp funding alignment, and large-flow thresholds",
        )
    join_path = root / "institutional_flow" / "current_btc_etf_market_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        return ExplorationRow(
            lane="institutional_flow",
            status="btc_etf_market_context",
            strongest_current_signal=(
                f"{best_join.get('asset', '')}: {best_join.get('action', '')}, "
                f"latest={best_join.get('latest_flow_btc', '')} BTC, "
                f"5d={best_join.get('rolling_5d_flow_btc', '')} BTC, "
                f"funding={best_join.get('annualized_funding', '')}, "
                f"score={best_join.get('score', '')}"
            ),
            main_gap="ETF flow context is joined to BTC perp state but not labeled by regime or execution costs",
            next_step="label ETF inflow/outflow plus funding alignment against BTC forward returns and drawdown",
        )
    path = root / "institutional_flow" / "current_btc_etf_flow_snapshot.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="institutional_flow",
            status="btc_etf_flow_snapshot",
            strongest_current_signal=(
                f"{best.get('action', '')}: latest={best.get('latest_flow_btc', '')} BTC, "
                f"5d={best.get('rolling_5d_flow_btc', '')} BTC, "
                f"10d={best.get('rolling_10d_flow_btc', '')} BTC"
            ),
            main_gap="ETF flow context is not joined to perp state or forward labels",
            next_step="join BTC ETF flow to BTC perp funding/OI and label forward returns",
        )
    return ExplorationRow(
        lane="institutional_flow",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="ETF and institutional flow context is not connected",
        next_step="fetch BTC ETF flow history and join it to BTC market state",
    )


def _btc_etf_flow_label_summary(path: Path) -> dict[str, float | str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    labeled = tuple(row for row in rows if row.get("directional_return_5d"))
    if not labeled:
        return None
    dir_5d = tuple(float(row["directional_return_5d"]) for row in labeled)
    latest = rows[-1]
    return {
        "observations": float(len(labeled)),
        "mean_directional_5d": sum(dir_5d) / len(dir_5d),
        "hit_rate_5d": sum(1.0 for value in dir_5d if value > 0.0) / len(dir_5d),
        "latest_action": latest.get("action", ""),
    }


def _best_btc_etf_flow_regime_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(row for row in rows if row.get("action") == "regime_candidate")
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("mean_directional_5d") or "0"),
            float(row.get("hit_rate_5d") or "0"),
        ),
    )


def _best_btc_etf_flow_funding_regime_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(row for row in rows if row.get("action") == "funding_regime_candidate")
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("mean_directional_5d_with_funding") or "0"),
            float(row.get("hit_rate_5d_with_funding") or "0"),
        ),
    )


def _btc_etf_flow_funding_candidate_summary(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[0]
    if row.get("action") not in {"paper_rule_candidate", "paper_rule_watch"}:
        return None
    return row


def _current_btc_etf_funding_candidate(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[0]
    if row.get("status") != "active_paper_watch":
        return None
    return row


def _current_btc_etf_funding_paper_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    candidates = tuple(row for row in rows if row.get("status") == "paper_venue_candidate")
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("score") or "0"))


def _candidate_validation_row(root: Path) -> ExplorationRow:
    repeat_summary_path = (
        root / "candidate_validation" / "current_followup_repeat_history_summary.csv"
    )
    best_repeat_summary = _best_followup_repeat_summary_row(repeat_summary_path)
    if best_repeat_summary:
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_repeat_summary",
            strongest_current_signal=(
                f"{best_repeat_summary.get('group_key', '')}: "
                f"hit15={best_repeat_summary.get('hit_rate_15m', '')}, "
                f"mean15={best_repeat_summary.get('mean_dir15', '')}, "
                f"pending={best_repeat_summary.get('pending_rows', '')}, "
                f"action={best_repeat_summary.get('action', '')}"
            ),
            main_gap="repeat summary is still short-horizon and excludes costs, funding PnL, slippage, and neutral baselines",
            next_step="collect another liquidation-specific repeat batch for JTO/LTC and add rough costs, funding PnL, and slippage",
        )
    repeat_history_label_path = (
        root / "candidate_validation" / "current_followup_repeat_history_labels.csv"
    )
    best_repeat_history_label = _best_followup_repeat_label_row(repeat_history_label_path)
    if best_repeat_history_label:
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_repeat_history_label",
            strongest_current_signal=(
                f"{best_repeat_history_label.get('venue', '')}/"
                f"{best_repeat_history_label.get('asset', '')}/"
                f"{best_repeat_history_label.get('source', '')}: "
                f"{best_repeat_history_label.get('source_action', '')}, "
                f"dir15={best_repeat_history_label.get('directional_return_15m', '')}, "
                f"status={best_repeat_history_label.get('label_status', '')}"
            ),
            main_gap="history label is still one repeat batch and excludes costs, funding PnL, slippage, and neutral baseline",
            next_step="collect more source-specific repeat batches and compare 15m/1h by source and venue",
        )
    repeat_label_path = root / "candidate_validation" / "current_followup_repeat_forward_labels.csv"
    best_repeat_label = _best_followup_repeat_label_row(repeat_label_path)
    if best_repeat_label:
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_repeat_label",
            strongest_current_signal=(
                f"{best_repeat_label.get('asset', '')}/{best_repeat_label.get('source', '')}: "
                f"{best_repeat_label.get('source_action', '')}, "
                f"dir15={best_repeat_label.get('directional_return_15m', '')}, "
                f"status={best_repeat_label.get('label_status', '')}"
            ),
            main_gap="fresh repeat label still excludes fees, funding PnL, slippage, and neutral baseline",
            next_step="compare source-specific repeat labels and promote only repeated winners",
        )
    repeat_observation_path = (
        root / "candidate_validation" / "current_followup_repeat_observations.csv"
    )
    okx_repeat_observation_path = (
        root / "candidate_validation" / "current_followup_okx_repeat_observations.csv"
    )
    repeat_summary = _followup_repeat_observation_summary(repeat_observation_path)
    okx_repeat_summary = _followup_repeat_observation_summary(okx_repeat_observation_path)
    if repeat_summary or okx_repeat_summary:
        joined_summary = "; ".join(
            summary for summary in (repeat_summary, okx_repeat_summary) if summary
        )
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_repeat_pending",
            strongest_current_signal=joined_summary,
            main_gap="fresh observations are recorded but not yet matured to 15m/1h labels",
            next_step="rerun followup repeat forward labels after 15m and 1h",
        )
    execution_context_path = (
        root / "candidate_validation" / "current_followup_execution_context.csv"
    )
    best_execution_context = _best_followup_execution_context_row(execution_context_path)
    if best_execution_context:
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_execution_context",
            strongest_current_signal=(
                f"{best_execution_context.get('asset', '')}: "
                f"{best_execution_context.get('action', '')}, "
                f"priority={best_execution_context.get('priority', '')}, "
                f"spread={best_execution_context.get('spread_bps', '')}bps, "
                f"depth10={best_execution_context.get('near_depth_10bps_notional', '')}"
            ),
            main_gap="current public venue context is rough and still excludes account fees, fills, queue position, and neutral baselines",
            next_step="repeat WLD/ONDO/XPL/PUMP labels with source-specific costs and venue checks",
        )
    queue_path = root / "candidate_validation" / "current_followup_queue.csv"
    best_queue = _best_followup_queue_row(queue_path)
    if best_queue:
        return ExplorationRow(
            lane="candidate_validation",
            status="followup_queue",
            strongest_current_signal=(
                f"{best_queue.get('asset', '')}: "
                f"{best_queue.get('followup_type', '')}, "
                f"source={best_queue.get('source', '')}, "
                f"priority={best_queue.get('priority', '')}"
            ),
            main_gap="queue prioritizes fresh observations but still lacks repeated live fills, costs, and neutral baselines",
            next_step="execute top follow-up queue items and update source-specific labels",
        )
    review_path = root / "candidate_validation" / "current_cross_lane_candidate_review.csv"
    family_path = root / "candidate_validation" / "current_signal_family_review.csv"
    best_review = _best_numeric_row(review_path, key="lead_score")
    if best_review:
        best_family = _best_numeric_row(family_path, key="support_score")
        family_note = ""
        if best_family:
            family_note = (
                f"; family={best_family.get('family', '')}, "
                f"hit15={best_family.get('hit_rate_15m', '')}, "
                f"mean15={best_family.get('mean_label_15m', '')}"
            )
        return ExplorationRow(
            lane="candidate_validation",
            status="cross_lane_review",
            strongest_current_signal=(
                f"{best_review.get('asset', '')}: score={best_review.get('lead_score', '')}, "
                f"lanes={best_review.get('lanes', '')}, "
                f"positive={best_review.get('positive_labels', '')}, "
                f"negative={best_review.get('negative_labels', '')}"
                f"{family_note}"
            ),
            main_gap="cross-lane score is still a triage heuristic, not a PnL model or execution test",
            next_step="repeat WLD and short-liquidation-squeeze labels with fees, funding, and venue depth",
        )
    label_path = root / "candidate_validation" / "current_hl_signal_forward_label_summary.csv"
    best_label = _best_forward_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="candidate_validation",
            status="forward_label_probe",
            strongest_current_signal=(
                f"{best_label.get('asset', '')}: {best_label.get('action', '')}, "
                f"source={best_label.get('source', '')}, "
                f"cov15={best_label.get('coverage_15m', '')}, "
                f"mean15={best_label.get('mean_return_15m', '')}, "
                f"hit15={best_label.get('positive_15m_rate', '')}"
            ),
            main_gap="15m price label excludes funding PnL, hedge PnL, fees, adverse selection, and neutral baselines",
            next_step="wait for 1h labels and compare direction-aware PnL against neutral baselines",
        )
    path = root / "candidate_validation" / "current_hl_candidate_return_context.csv"
    best = _best_numeric_row(path, key="score")
    signal = "not run yet"
    if best:
        signal = (
            f"{best.get('symbol', '')}: {best.get('action', '')}, "
            f"sources={best.get('sources', '')}, "
            f"1h={best.get('return_1h', '')}, "
            f"4h={best.get('return_4h', '')}"
        )
    return ExplorationRow(
        lane="candidate_validation",
        status="current_return_context",
        strongest_current_signal=signal,
        main_gap="recent return context is not forward-labeled alpha evidence",
        next_step="label candidate signals from their timestamps and compare against neutral baselines",
    )


def _stablecoin_liquidity_row(root: Path) -> ExplorationRow:
    peg_path = root / "stablecoin_liquidity" / "current_peg_stress_screen.csv"
    best_peg = _best_numeric_row(peg_path, key="score")
    if best_peg:
        return ExplorationRow(
            lane="stablecoin_liquidity",
            status=best_peg.get("status", "peg_stress_screen"),
            strongest_current_signal=(
                f"{best_peg.get('symbol', '')}/{best_peg.get('name', '')}: "
                f"price={best_peg.get('price', '')}, "
                f"peg_deviation={best_peg.get('peg_deviation', '')}, "
                f"score={best_peg.get('score', '')}"
            ),
            main_gap="stablecoin peg stress still needs redemption route, venue depth, custody, and repeated price checks",
            next_step=best_peg.get(
                "next_step",
                "check redemption path, tradable venues, and repeated peg snapshots before paper action",
            ),
        )
    forward_label_path = (
        root / "stablecoin_liquidity" / "current_supply_market_forward_labels.csv"
    )
    basket = _row_by_value(forward_label_path, field="asset", value="BASKET")
    if basket:
        return ExplorationRow(
            lane="stablecoin_liquidity",
            status="market_forward_label",
            strongest_current_signal=(
                f"BASKET: week_change={basket.get('week_change_usd', '')}, "
                f"dir4h={basket.get('directional_return_4h', '')}, "
                f"dir12h={basket.get('directional_return_12h', '')}, "
                f"action={basket.get('action', '')}"
            ),
            main_gap="stablecoin supply contraction contradicted the naive short-term risk-off direction in this window",
            next_step="treat supply change as a regime/divergence feature and combine it with funding, liquidation, and breadth sources",
        )
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


def _on_chain_flow_row(root: Path) -> ExplorationRow:
    summary_path = root / "on_chain_flow" / "chain_tvl_flow_market_context_summary.csv"
    best_summary = _best_chain_tvl_market_context_summary_row(summary_path)
    if best_summary:
        return ExplorationRow(
            lane="on_chain_flow",
            status="chain_tvl_flow_market_context_summary",
            strongest_current_signal=(
                f"{best_summary.get('group_key', '')}: "
                f"obs={best_summary.get('observations', '')}, "
                f"labeled={best_summary.get('labeled_observations', '')}, "
                f"mean_score={best_summary.get('mean_context_score', '')}, "
                f"action={best_summary.get('action', '')}"
            ),
            main_gap="chain TVL flow has only two labeled timestamps and still excludes costs, slippage, and 1h confirmation",
            next_step="repeat the ETH context and isolate second-sample POL/XLM/AVAX winners before promotion",
        )
    context_path = root / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv"
    best_context = _best_chain_tvl_market_context_row(context_path)
    if best_context:
        return ExplorationRow(
            lane="on_chain_flow",
            status="chain_tvl_flow_market_context",
            strongest_current_signal=(
                f"{best_context.get('venue', '')}/{best_context.get('token_symbol', '')}: "
                f"dir15={best_context.get('directional_return_15m', '')}, "
                f"funding_support={best_context.get('funding_support', '')}, "
                f"score={best_context.get('context_score', '')}"
            ),
            main_gap="chain TVL flow market context still uses one short price label and incomplete venue context",
            next_step="repeat MON/HYPE/SOL/ETH labels and fill missing KAT/POL OKX market context",
        )
    label_path = root / "on_chain_flow" / "current_chain_tvl_flow_forward_labels.csv"
    best_label = _best_chain_tvl_forward_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="on_chain_flow",
            status="chain_tvl_flow_forward_label",
            strongest_current_signal=(
                f"{best_label.get('venue', '')}/{best_label.get('token_symbol', '')}: "
                f"{best_label.get('action', '')}, "
                f"dir15={best_label.get('directional_return_15m', '')}, "
                f"week={best_label.get('week_change_pct', '')}"
            ),
            main_gap="chain TVL flow forward label is one short horizon and excludes costs, funding PnL, slippage, and stale-accounting checks",
            next_step="repeat HYPE/MEGA/STX/APT reversal labels and compare against liquidation plus funding context",
        )
    coverage_path = root / "on_chain_flow" / "current_chain_tvl_flow_venue_coverage.csv"
    best_coverage = _best_chain_tvl_venue_coverage_row(coverage_path)
    if best_coverage:
        return ExplorationRow(
            lane="on_chain_flow",
            status=best_coverage.get("action", "chain_tvl_flow_venue_coverage"),
            strongest_current_signal=(
                f"{best_coverage.get('chain', '')}/{best_coverage.get('token_symbol', '')}: "
                f"week={best_coverage.get('week_change_pct', '')}, "
                f"day={best_coverage.get('day_change_pct', '')}, "
                f"venues={best_coverage.get('venue_count', '')}"
            ),
            main_gap="chain TVL flow venue coverage is not yet joined to token forward returns, costs, or stale-accounting checks",
            next_step=best_coverage.get(
                "followup",
                "label covered chain-token behavior against market structure sources",
            ),
        )
    path = root / "on_chain_flow" / "current_chain_tvl_flow.csv"
    best = _best_chain_tvl_flow_row(path)
    if best:
        return ExplorationRow(
            lane="on_chain_flow",
            status=best.get("action", "chain_tvl_flow"),
            strongest_current_signal=(
                f"{best.get('chain', '')}/{best.get('token_symbol', '')}: "
                f"week={best.get('week_change_pct', '')}, "
                f"day={best.get('day_change_pct', '')}, "
                f"tvl={best.get('current_tvl_usd', '')}"
            ),
            main_gap="chain TVL flow is not yet joined to token forward returns, funding, liquidations, or stale-accounting checks",
            next_step=best.get(
                "followup",
                "label chain-token behavior against market structure sources",
            ),
        )
    return ExplorationRow(
        lane="on_chain_flow",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="wallet, bridge, exchange, and chain TVL flows are not connected",
        next_step="run chain TVL flow and label token follow-ups",
    )


def _protocol_fundamentals_row(root: Path) -> ExplorationRow:
    path = root / "protocol_fundamentals" / "current_protocol_fee_screen.csv"
    best = _best_protocol_fee_row(path)
    if best:
        return ExplorationRow(
            lane="protocol_fundamentals",
            status=best.get("status", "watch"),
            strongest_current_signal=(
                f"{best.get('token_symbol', '')}/{best.get('name', '')}: "
                f"{best.get('side', '')}, "
                f"fees7d={best.get('total_7d', '')}, "
                f"growth7d={best.get('change_7d_over_7d', '')}, "
                f"funding={best.get('funding', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap="protocol fee growth is not yet forward-labeled against token returns, funding, unlocks, and attention context",
            next_step=best.get(
                "next_step",
                "label token returns after protocol fee-growth snapshots",
            ),
        )
    return ExplorationRow(
        lane="protocol_fundamentals",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="protocol fee/revenue data is not connected to tradable token candidates",
        next_step="run protocol fee screen and label fee-growth token follow-ups",
    )


def _best_numeric_row(path: Path, *, key: str) -> dict[str, str] | None:
    rows = _csv_rows(path)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(key) or "-inf"))


def _csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _best_abs_numeric_row(path: Path, *, key: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: abs(float(row.get(key) or "0")))


def _best_protocol_fee_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            1.0 if row.get("status") == "paper_long_context" else 0.0,
            float(row.get("score") or "0"),
        ),
    )


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


def _best_crowding_monitor_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("observations") or "0"),
            float(row.get("mean_score") or "0"),
            float(row.get("min_score") or "0"),
            abs(float(row.get("mean_annualized_funding") or "0")),
        ),
    )


def _best_polymarket_monitor_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("observations") or "0"),
            float(row.get("mean_score") or "0"),
            float(row.get("mean_volume_24h") or "0"),
            float(row.get("mean_liquidity") or "0"),
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


def _best_forward_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if int(row.get("coverage_15m") or "0") > 0
            and row.get("mean_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("mean_return_15m") or "-inf"),
            float(row.get("positive_15m_rate") or "0"),
            int(row.get("coverage_15m") or "0"),
        ),
    )


def _best_paper_gate_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") == "small_paper_probe"
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("conservative_net_bps") or "-inf"),
            -float(row.get("visible_depth_usage") or "inf"),
            float(row.get("candidate_size_usd") or "0"),
        ),
    )


def _best_paper_outcome_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row.get("outcome_15m") == "paper_15m_win",
            float(row.get("net_15m_bps") or "-inf"),
            row.get("outcome_1h") == "paper_1h_win",
            float(row.get("net_1h_bps") or "-inf"),
        ),
    )


def _best_attention_forward_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("score") or "0"),
            float(row.get("directional_return_1h") or "-inf"),
        ),
    )


def _best_exchange_catalyst_forward_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("score") or "0"),
            float(row.get("directional_return_1h") or "-inf"),
        ),
    )


def _best_protocol_activity_forward_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("score") or "0"),
            float(row.get("directional_return_1h") or "-inf"),
        ),
    )


def _best_sector_tradable_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("label_status") == "tradable_labeled"
            and row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("score") or "0"),
            float(row.get("coin_volume_24h") or "0"),
        ),
    )


def _best_l2_imbalance_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("directional_return_1h") or "-inf"),
            abs(float(row.get("imbalance_10_bps") or "0")),
        ),
    )


def _best_l2_imbalance_paper_gate_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") == "small_paper_probe"
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("net_15m_bps") or "-inf"),
            float(row.get("net_1h_bps") or "-inf"),
            -float(row.get("visible_depth_usage") or "inf"),
        ),
    )


def _best_l2_imbalance_monitor_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("observations") or "0"),
            float(row.get("direction_persistence_rate") or "0"),
            float(row.get("mean_abs_imbalance_10_bps") or "0"),
            float(row.get("mean_near_depth_10bps_notional") or "0"),
        ),
    )


def _best_followup_queue_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("priority") or "-inf"))


def _best_followup_execution_context_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row for row in csv.DictReader(handle) if row.get("action") == "tradable_context_ok"
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("priority") or "-inf"))


def _best_followup_repeat_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            float(row.get("priority") or "0"),
        ),
    )


def _best_followup_repeat_summary_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("group_type") == "asset_source" and row.get("action") == "repeat_priority"
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("mean_dir15") or "-inf"),
            float(row.get("hit_rate_15m") or "0"),
            int(row.get("labeled_rows") or "0"),
        ),
    )


def _best_chain_tvl_flow_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    priority = {
        "chain_inflow_momentum_watch": 3,
        "chain_outflow_stress_watch": 2,
        "chain_flow_reversal_watch": 1,
        "chain_flow_context": 0,
    }
    return max(
        rows,
        key=lambda row: (
            priority.get(row.get("action", ""), 0),
            abs(float(row.get("week_change_pct") or "0")),
            float(row.get("current_tvl_usd") or "0"),
        ),
    )


def _best_chain_tvl_venue_coverage_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if int(row.get("venue_count") or "0") > 0
        )
    if not rows:
        return None
    priority = {
        "chain_inflow_momentum_watch": 3,
        "chain_outflow_stress_watch": 2,
        "chain_flow_reversal_watch": 1,
        "chain_flow_context": 0,
    }
    return max(
        rows,
        key=lambda row: (
            int(row.get("venue_count") or "0"),
            priority.get(row.get("action", ""), 0),
            abs(float(row.get("week_change_pct") or "0")),
        ),
    )


def _best_chain_tvl_forward_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_15m", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_15m") or "-inf"),
            abs(float(row.get("week_change_pct") or "0")),
        ),
    )


def _best_chain_tvl_market_context_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("context_score", "") != ""
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("context_score") or "-inf"))


def _best_chain_tvl_market_context_summary_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    action_rank = {
        "repeat_priority": 3,
        "keep_sampling": 2,
        "collect_repeat": 1,
        "deprioritize": -1,
    }
    return max(
        rows,
        key=lambda row: (
            action_rank.get(row.get("action", ""), 0),
            float(row.get("mean_context_score") or "-inf"),
            int(row.get("labeled_observations") or "0"),
        ),
    )


def _best_category_perp_context_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    action_rank = {
        "sector_perp_repeat_candidate": 3,
        "keep_sampling": 2,
        "wait_for_label": 1,
        "deprioritize": -1,
    }
    return max(
        rows,
        key=lambda row: (
            action_rank.get(row.get("action", ""), 0),
            float(row.get("context_score") or "-inf"),
        ),
    )


def _followup_repeat_observation_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    ready_rows = tuple(row for row in rows if row.get("observation_status") == "ready_for_label")
    if not ready_rows:
        return None
    top = max(ready_rows, key=lambda row: float(row.get("priority") or "0"))
    venue = "OKX" if "okx" in path.name else "HL"
    return (
        f"{venue} {len(ready_rows)} source-specific observations pending; "
        f"top={top.get('asset', '')}/{top.get('source', '')}, "
        f"dir={top.get('direction', '')}, priority={top.get('priority', '')}"
    )


def _okx_perp_pressure_signal(path: Path, label_path: Path) -> str:
    best = _best_numeric_row(path, key="pressure_score")
    if not best:
        return ""
    label = _label_row_for_asset(label_path, asset=best.get("asset", ""))
    label_note = ""
    if label and label.get("directional_return_15m", "") != "":
        label_note = f", dir15={label.get('directional_return_15m', '')}"
    return (
        f"; OKX {best.get('asset', '')}: {best.get('action', '')}, "
        f"ann_funding={best.get('annualized_funding', '')}, "
        f"premium={best.get('premium', '')}, score={best.get('pressure_score', '')}"
        f"{label_note}"
    )


def _label_row_for_asset(path: Path, *, asset: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("asset") == asset:
                return row
    return None


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
