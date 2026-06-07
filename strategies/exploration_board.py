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
        _liquidation_flow_row(root),
        _defi_yield_row(root),
        _market_making_row(root),
        _news_social_row(root),
        _prediction_markets_row(root),
        _candidate_validation_row(root),
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
            next_step="wait for 1h outcome and repeat JTO/ONDO/LTC gate on fresh liquidation events",
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
    paper_gate_path = root / "market_making" / "current_l2_imbalance_paper_gate.csv"
    best_gate = _best_l2_imbalance_paper_gate_row(paper_gate_path)
    if best_gate:
        return ExplorationRow(
            lane="market_making",
            status="l2_imbalance_paper_gate",
            strongest_current_signal=(
                f"{best_gate.get('asset', '')}: "
                f"size={best_gate.get('candidate_size_usd', '')}, "
                f"net15={best_gate.get('net_15m_bps', '')}bps, "
                f"net1h={best_gate.get('net_1h_bps', '')}bps, "
                f"depth_usage={best_gate.get('visible_depth_usage', '')}"
            ),
            main_gap="paper gate is directional and still excludes maker queue, fill probability, rebates, and repeated adverse-selection samples",
            next_step="repeat HYPE/SOL L2 gates on fresh snapshots and then design a maker-fill observation log",
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


def _prediction_markets_row(root: Path) -> ExplorationRow:
    depth_path = root / "prediction_markets" / "current_polymarket_clob_depth.csv"
    best_depth = _best_numeric_row(depth_path, key="visible_depth_score")
    if best_depth:
        signal = (
            f"{best_depth.get('question', '')} {best_depth.get('outcome', '')}: "
            f"spread={best_depth.get('spread', '')}, "
            f"bid_depth_5c={best_depth.get('bid_depth_to_5c', '')}, "
            f"ask_depth_5c={best_depth.get('ask_depth_to_5c', '')}"
        )
        return ExplorationRow(
            lane="prediction_markets",
            status="clob_depth_check",
            strongest_current_signal=signal,
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


def _candidate_validation_row(root: Path) -> ExplorationRow:
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
