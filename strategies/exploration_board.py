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
        _alpha_frontier_row(root),
        _alpha_source_gaps_row(root),
        _factor_hypothesis_templates_row(root),
        _factor_template_validation_queue_row(root),
        _ofi_execution_survival_row(root),
        _lob_policy_candidate_survival_row(root),
        _lob_maker_fill_survival_row(root),
        _crowded_positioning_survival_row(root),
        _alpha_method_frontier_row(root),
        _portable_microstructure_feature_frontier_row(root),
        _portable_microstructure_horizon_candidates_row(root),
        _portable_microstructure_horizon_tickets_row(root),
        _portable_microstructure_horizon_outcomes_row(root),
        _research_backed_alpha_expansion_plan_row(root),
        _exchange_stablecoin_inflow_readiness_row(root),
        _stablecoin_flow_probe_candidates_row(root),
        _stablecoin_flow_proxy_tickets_row(root),
        _stablecoin_flow_proxy_outcomes_row(root),
        _options_volatility_survival_row(root),
        _fundamental_sentiment_cross_section_row(root),
        _multimodal_btc_eth_feature_alignment_row(root),
        _sentiment_contagion_negative_control_row(root),
        _cross_modal_alpha_context_row(root),
        _cross_modal_source_split_row(root),
        _paper_probe_plan_row(root),
        _paper_tickets_row(root),
        _paper_ticket_outcomes_row(root),
        _paper_ticket_action_queue_row(root),
        _paper_ticket_fill_risk_check_row(root),
        _policy_expansion_outcome_frontier_row(root),
        _promoted_ticket_repeat_tickets_row(root),
        _promoted_ticket_repeat_outcomes_row(root),
        _promoted_ticket_repeat_action_queue_row(root),
        _promoted_ticket_repeat_fill_risk_check_row(root),
        _second_promoted_ticket_repeat_tickets_row(root),
        _second_promoted_ticket_repeat_outcomes_row(root),
        _second_promoted_ticket_repeat_action_queue_row(root),
        _second_promoted_ticket_repeat_fill_risk_check_row(root),
        _symbol_lane_paper_tickets_row(root),
        _symbol_lane_paper_outcomes_row(root),
        _symbol_lane_paper_action_queue_row(root),
        _symbol_lane_paper_fill_risk_check_row(root),
        _symbol_lane_promoted_repeat_tickets_row(root),
        _symbol_lane_promoted_repeat_outcomes_row(root),
        _symbol_lane_promoted_repeat_action_queue_row(root),
        _symbol_lane_promoted_repeat_fill_risk_check_row(root),
        _cost_adjusted_alpha_candidates_row(root),
        _cost_adjusted_alpha_clusters_row(root),
        _cost_survival_cross_section_row(root),
        _alpha_promotion_frontier_row(root),
        _alpha_promotion_worklist_row(root),
        _alpha_repeat_fill_survival_row(root),
        _surviving_alpha_path_risk_row(root),
        _surviving_alpha_fill_audit_tickets_row(root),
        _surviving_alpha_fill_audit_outcomes_row(root),
        _surviving_alpha_exit_regime_candidates_row(root),
        _surviving_alpha_exit_regime_tickets_row(root),
        _surviving_alpha_exit_regime_outcomes_row(root),
        _alpha_conflict_resolution_progress_row(root),
        _cost_adjusted_cluster_repeat_plan_row(root),
        _split_first_cluster_lane_plan_row(root),
        _split_first_lane_repeat_queue_row(root),
        _split_first_lane_label_progress_row(root),
        _split_first_lane_label_tickets_row(root),
        _split_first_lane_label_outcomes_row(root),
        _split_first_lane_repeat_tickets_row(root),
        _split_first_lane_repeat_outcomes_row(root),
        _symbol_opportunity_map_row(root),
        _symbol_cluster_conflicts_row(root),
        _symbol_cluster_label_queue_row(root),
        _symbol_lane_split_review_row(root),
        _policy_learning_row(root),
        _policy_context_frontier_row(root),
        _policy_action_preference_row(root),
        _policy_action_preference_oos_row(root),
        _wallet_entity_flow_row(root),
        _hyperliquid_seed_wallet_flow_row(root),
        _hyperliquid_seed_wallet_flow_actionability_row(root),
        _execution_edge_mode_row(root),
        _crypto_market_structure_row(root),
        _basis_term_structure_row(root),
        _cross_exchange_funding_row(root),
        _perp_market_map_row(root),
        _derivatives_positioning_row(root),
        _binance_derivatives_history_row(root),
        _binance_derivatives_intraday_live_gate_row(root),
        _binance_derivatives_intraday_paper_row(root),
        _binance_derivatives_intraday_repeat_row(root),
        _binance_derivatives_intraday_row(root),
        _macro_regime_row(root),
        _crypto_equity_proxy_row(root),
        _crypto_equity_factor_split_row(root),
        _speculative_beta_row(root),
        _event_flow_row(root),
        _liquidation_intensity_row(root),
        _liquidation_flow_row(root),
        _defi_yield_row(root),
        _defi_lending_row(root),
        _dex_pool_flow_row(root),
        _market_making_row(root),
        _options_volatility_row(root),
        _sector_rotation_row(root),
        _exchange_catalyst_row(root),
        _token_unlocks_row(root),
        _event_pressure_cluster_row(root),
        _ticker_attention_source_split_row(root),
        _news_social_row(root),
        _market_breadth_row(root),
        _event_crypto_hedge_survival_row(root),
        _prediction_market_crypto_hedge_row(root),
        _prediction_markets_row(root),
        _anomaly_stress_row(root),
        _tail_connectedness_regime_row(root),
        _protocol_activity_row(root),
        _institutional_flow_row(root),
        _candidate_validation_row(root),
        _stablecoin_liquidity_row(root),
        _stablecoin_exchange_inflow_proxy_row(root),
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


def _alpha_frontier_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_frontier.csv"
    best = _best_numeric_row(path, key="frontier_score")
    missing = _first_matching_row(path, key="current_status", value="missing_concrete_probe")
    if best:
        missing_lane = missing.get("lane", "") if missing else ""
        missing_step = missing.get("next_probe", "") if missing else ""
        return ExplorationRow(
            lane="alpha_frontier",
            status=best.get("current_status", "frontier_review"),
            strongest_current_signal=(
                f"{best.get('lane', '')}: "
                f"active={best.get('active_candidates', '')}, "
                f"best={best.get('best_opportunity', '')}, "
                f"frontier={best.get('frontier_score', '')}"
            ),
            main_gap=(
                f"missing concrete probe: {missing_lane}"
                if missing_lane
                else best.get("missing_work", "frontier lane still needs concrete probe evidence")
            ),
            next_step=missing_step or best.get("next_probe", "use frontier review to widen alpha discovery"),
        )
    return ExplorationRow(
        lane="alpha_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="broad alpha-source coverage is not summarized",
        next_step="run current alpha frontier after the alpha stack",
    )


def _alpha_source_gaps_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_source_gaps.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="alpha_source_gaps",
            status=best.get("status", "source_gap"),
            strongest_current_signal=(
                f"{best.get('gap_id', '')}: "
                f"{best.get('lane', '')}, "
                f"priority={best.get('priority', '')}, "
                f"coverage={best.get('current_coverage', '')}"
            ),
            main_gap=best.get("missing_work", "external alpha-source gap needs concrete probe"),
            next_step=best.get("next_probe", "turn the top source gap into a concrete probe"),
        )
    return ExplorationRow(
        lane="alpha_source_gaps",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="modern external alpha-source gaps are not summarized",
        next_step="run current alpha source gaps after data source probe",
    )


def _factor_hypothesis_templates_row(root: Path) -> ExplorationRow:
    path = root / "llm_factor_generation" / "current_factor_hypothesis_templates.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="factor_hypothesis_templates",
            status=best.get("status", "factor_hypothesis_template"),
            strongest_current_signal=(
                f"{best.get('template_id', '')}: "
                f"{best.get('lane', '')}, "
                f"priority={best.get('priority_score', '')}, "
                f"seeds={best.get('seed_opportunities', '')}"
            ),
            main_gap=best.get("failure_mode", "factor template still needs falsification route"),
            next_step=best.get("next_step", "route the top factor template to validation"),
        )
    return ExplorationRow(
        lane="factor_hypothesis_templates",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="factor hypotheses are not generated from current alpha stack rows",
        next_step="run current factor hypothesis templates after cross-modal source split",
    )


def _factor_template_validation_queue_row(root: Path) -> ExplorationRow:
    path = root / "llm_factor_generation" / "current_factor_template_validation_queue.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="factor_template_validation_queue",
            status=best.get("current_status", "factor_template_validation"),
            strongest_current_signal=(
                f"{best.get('template_id', '')}: "
                f"priority={best.get('priority_score', '')}, "
                f"artifact={best.get('current_artifact', '')}, "
                f"route={best.get('validation_route', '')}"
            ),
            main_gap=best.get("best_evidence", "template route still needs validation evidence"),
            next_step=best.get("next_step", "validate the top factor template route"),
        )
    return ExplorationRow(
        lane="factor_template_validation_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="factor templates are not routed to concrete validation artifacts",
        next_step="run current factor template validation queue after template generation",
    )


def _ofi_execution_survival_row(root: Path) -> ExplorationRow:
    path = root / "event_flow" / "current_ofi_execution_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="ofi_execution_survival",
            status=best.get("status", "ofi_execution_survival"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"{best.get('action', '')}, "
                f"mode={best.get('execution_mode', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"maker_net={best.get('maker_net_bps', '')}, "
                f"net15={best.get('l2_net_15m_bps', '')}, "
                f"net1h={best.get('l2_net_1h_bps', '')}"
            ),
            main_gap=best.get("missing_work", "OFI execution survival still needs fill and queue evidence"),
            next_step=best.get("next_probe", "paper-check the top OFI execution-survival candidate"),
        )
    return ExplorationRow(
        lane="ofi_execution_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="book-depth cost sweep is not joined to current L2 imbalance states",
        next_step="run current OFI execution survival after L2 paper gate and book-depth cost sweep",
    )


def _lob_policy_candidate_survival_row(root: Path) -> ExplorationRow:
    path = root / "event_flow" / "current_lob_policy_candidate_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="lob_policy_candidate_survival",
            status=best.get("survival_status", "lob_policy_candidate_survival"),
            strongest_current_signal=(
                f"{best.get('state_family', '')}: "
                f"{best.get('signal_action', '')}, "
                f"mode={best.get('execution_mode', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"world_net={best.get('world_net_bps', '')}, "
                f"seq_net={best.get('sequence_net_bps', '')}, "
                f"zero_seq={best.get('sequence_zero_cost_net_bps', '')}"
            ),
            main_gap=best.get("reason", "LOB policy candidate still needs execution survival"),
            next_step=best.get("next_step", "rerun LOB policy candidate survival after fresh snapshots"),
        )
    return ExplorationRow(
        lane="lob_policy_candidate_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="LOB world replay and rolling sequence probes have not been compared as policy candidates",
        next_step="run current LOB policy candidate survival after world replay and sequence state probes",
    )


def _lob_maker_fill_survival_row(root: Path) -> ExplorationRow:
    path = root / "event_flow" / "current_lob_maker_fill_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="lob_maker_fill_survival",
            status=best.get("survival_status", "lob_maker_fill_survival"),
            strongest_current_signal=(
                f"{best.get('state_family', '')}/{best.get('source_probe', '')}: "
                f"{best.get('signal_action', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"fill_rate={best.get('fill_rate', '')}, "
                f"filled_bps={best.get('filled_mark_reward_bps', '')}, "
                f"all_bps={best.get('all_state_reward_bps', '')}, "
                f"adverse={best.get('adverse_fill_rate', '')}"
            ),
            main_gap=best.get("reason", "maker policy still needs fill and adverse-selection checks"),
            next_step=best.get("next_step", "rerun maker fill survival after fresh book snapshots"),
        )
    return ExplorationRow(
        lane="lob_maker_fill_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="LOB maker policy candidates have not been checked against a passive-fill proxy",
        next_step="run current LOB maker fill survival after LOB policy candidate survival",
    )


def _crowded_positioning_survival_row(root: Path) -> ExplorationRow:
    path = root / "perp_market_map" / "current_crowded_positioning_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="crowded_positioning_survival",
            status=best.get("status", "crowded_positioning_survival"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"{best.get('side', '')}, "
                f"action={best.get('action', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"gate={best.get('label_gate_score', '')}, "
                f"net1h={best.get('net_directional_return_1h_proxy', '')}, "
                f"hit1h={best.get('positive_directional_1h_rate', '')}"
            ),
            main_gap=best.get("missing_work", "crowded positioning still needs unwind labels"),
            next_step=best.get("next_probe", "label crowded positioning continuation versus unwind"),
        )
    return ExplorationRow(
        lane="crowded_positioning_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="derivatives positioning rows are not joined to crowding-unwind labels",
        next_step="run current crowded positioning survival after crowding unwind label gate",
    )


def _alpha_method_frontier_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_method_frontier.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="alpha_method_frontier",
            status=best.get("decision", "method_frontier"),
            strongest_current_signal=(
                f"{best.get('method_id', '')}: "
                f"{best.get('family', '')}, "
                f"score={best.get('score', '')}, "
                f"data={best.get('data_evidence', '')}"
            ),
            main_gap=best.get("missing_link", "method frontier still needs a concrete probe"),
            next_step=best.get("first_probe", "turn the strongest method into a concrete alpha probe"),
        )
    return ExplorationRow(
        lane="alpha_method_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="modern alpha methods have not been mapped to current sources and candidates",
        next_step="run current alpha method frontier after source gaps and alpha frontier",
    )


def _portable_microstructure_feature_frontier_row(root: Path) -> ExplorationRow:
    path = root / "current_portable_microstructure_feature_frontier.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="portable_microstructure_feature_frontier",
            status=best.get("status", "portable_microstructure_feature_frontier"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"priority={best.get('priority', '')}, "
                f"15m={best.get('directional_return_15m', '')}, "
                f"1h={best.get('directional_return_1h', '')}"
            ),
            main_gap=best.get("missing_link", "shared microstructure feature table still needs validation"),
            next_step=best.get("next_step", "build the shared microstructure feature table"),
        )
    return ExplorationRow(
        lane="portable_microstructure_feature_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="BTC/ETH/SOL/HYPE microstructure snapshots and labels have not been joined",
        next_step="run current portable microstructure feature frontier after method frontier",
    )


def _portable_microstructure_horizon_candidates_row(root: Path) -> ExplorationRow:
    path = root / "current_portable_microstructure_horizon_candidates.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="portable_microstructure_horizon_candidates",
            status=best.get("status", "portable_microstructure_horizon_candidate"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"horizon={best.get('candidate_horizon', '')}, "
                f"return={best.get('candidate_directional_return', '')}, "
                f"priority={best.get('priority', '')}"
            ),
            main_gap=best.get("required_record", "horizon-specific microstructure candidate still needs repeat evidence"),
            next_step=best.get("next_step", "repeat the strongest horizon-specific microstructure candidate"),
        )
    return ExplorationRow(
        lane="portable_microstructure_horizon_candidates",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="portable microstructure frontier has not been split into candidate and rejected horizons",
        next_step="run current portable microstructure horizon candidates after the feature frontier",
    )


def _portable_microstructure_horizon_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_portable_microstructure_horizon_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="portable_microstructure_horizon_tickets",
            status=best.get("decision", "portable_microstructure_horizon_ticket"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('side', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"checkpoint={best.get('checkpoints', '')}"
            ),
            main_gap=best.get("required_record", "portable microstructure ticket needs repeat evidence"),
            next_step=best.get("next_step", "check the horizon-specific microstructure ticket"),
        )
    return ExplorationRow(
        lane="portable_microstructure_horizon_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="horizon-specific microstructure candidates have not been opened as paper observations",
        next_step="open portable microstructure horizon tickets after candidate generation",
    )


def _portable_microstructure_horizon_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_portable_microstructure_horizon_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="portable_microstructure_horizon_outcomes",
            status=best.get("outcome", "portable_microstructure_horizon_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "portable microstructure outcome still needs repeat evidence"),
            next_step=best.get("next_step", "refresh portable microstructure horizon outcomes"),
        )
    return ExplorationRow(
        lane="portable_microstructure_horizon_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="portable microstructure horizon tickets have not been checked against current marks",
        next_step="run portable microstructure horizon outcomes after checkpoint maturation",
    )


def _research_backed_alpha_expansion_plan_row(root: Path) -> ExplorationRow:
    path = root / "current_research_backed_alpha_expansion_plan.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="research_backed_alpha_expansion_plan",
            status=best.get("status", "research_backed_expansion"),
            strongest_current_signal=(
                f"{best.get('expansion_id', '')}: "
                f"{best.get('family', '')}, "
                f"priority={best.get('priority', '')}, "
                f"targets={best.get('target_assets', '')}"
            ),
            main_gap=best.get("missing_data", "research-backed expansion still needs data and labels"),
            next_step=best.get("first_probe", "turn the strongest research-backed expansion into a concrete probe"),
        )
    return ExplorationRow(
        lane="research_backed_alpha_expansion_plan",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="external research directions have not been mapped to current alpha-os coverage",
        next_step="run current research backed alpha expansion plan after method frontier",
    )


def _exchange_stablecoin_inflow_readiness_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_exchange_stablecoin_inflow_readiness.csv"
    best = _best_numeric_row(path, key="readiness_score")
    if best:
        return ExplorationRow(
            lane="exchange_stablecoin_inflow_readiness",
            status=best.get("status", "exchange_stablecoin_inflow_readiness"),
            strongest_current_signal=(
                f"{best.get('subject', '')}: "
                f"{best.get('alpha_kind', '')}, "
                f"score={best.get('readiness_score', '')}, "
                f"flow={best.get('flow_direction', '')}, "
                f"week={best.get('week_change_usd', '')}"
            ),
            main_gap=best.get("missing_data", "direct exchange stablecoin inflow still needs tagged deposits"),
            next_step=best.get("next_probe", "separate direct exchange inflow from chain liquidity proxy"),
        )
    return ExplorationRow(
        lane="exchange_stablecoin_inflow_readiness",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="direct exchange stablecoin inflow is not separated from chain stablecoin migration",
        next_step="run current exchange stablecoin inflow readiness after stablecoin exchange inflow proxy",
    )


def _stablecoin_flow_probe_candidates_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_stablecoin_flow_probe_candidates.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="stablecoin_flow_probe_candidates",
            status=best.get("status", "stablecoin_flow_probe_candidate"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"{best.get('candidate_type', '')}, "
                f"priority={best.get('priority', '')}, "
                f"flow={best.get('flow_direction', '')}"
            ),
            main_gap=best.get("required_record", "stablecoin-flow candidate still needs labels or direct flow data"),
            next_step=best.get("next_step", "run the strongest stablecoin-flow probe candidate"),
        )
    return ExplorationRow(
        lane="stablecoin_flow_probe_candidates",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="stablecoin-flow readiness has not been converted into data-probe and proxy-label candidates",
        next_step="run current stablecoin flow probe candidates after exchange inflow readiness",
    )


def _stablecoin_flow_proxy_tickets_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_stablecoin_flow_proxy_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="stablecoin_flow_proxy_tickets",
            status=best.get("decision", "stablecoin_flow_proxy_ticket"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('side', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"checkpoints={best.get('checkpoints', '')}"
            ),
            main_gap=best.get("required_record", "chain-liquidity proxy label needs controls"),
            next_step=best.get("next_step", "check the stablecoin proxy ticket"),
        )
    return ExplorationRow(
        lane="stablecoin_flow_proxy_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="stablecoin proxy candidates have not been opened as paper labels",
        next_step="open SOL/POL stablecoin proxy label tickets after probe candidates",
    )


def _stablecoin_flow_proxy_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_stablecoin_flow_proxy_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="stablecoin_flow_proxy_outcomes",
            status=best.get("outcome", "stablecoin_flow_proxy_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "stablecoin proxy outcome still needs controls"),
            next_step=best.get("next_step", "refresh stablecoin proxy outcomes"),
        )
    return ExplorationRow(
        lane="stablecoin_flow_proxy_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="stablecoin proxy tickets have not been checked against current marks",
        next_step="run stablecoin proxy outcomes after checkpoint maturation",
    )


def _options_volatility_survival_row(root: Path) -> ExplorationRow:
    path = root / "options_volatility" / "current_options_volatility_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="options_volatility_survival",
            status=best.get("status", "options_volatility_survival"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"score={best.get('survival_score', '')}, "
                f"iv={best.get('atm_iv', '')}, "
                f"rv24={best.get('realized_vol_24h', '')}, "
                f"premium={best.get('iv_premium_24h', '')}, "
                f"max_loss={best.get('max_loss_pct', '')}"
            ),
            main_gap=best.get("missing_work", "options candidate still needs hedge and quote evidence"),
            next_step=best.get("next_probe", "paper-check the top options volatility survival candidate"),
        )
    return ExplorationRow(
        lane="options_volatility_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cheap-IV rows are not separated by quote, premium, depth, and hedge survival",
        next_step="run current options volatility survival after volatility actionability",
    )


def _fundamental_sentiment_cross_section_row(root: Path) -> ExplorationRow:
    path = root / "current_fundamental_sentiment_cross_section.csv"
    best = _best_numeric_row(path, key="total_score")
    if best:
        return ExplorationRow(
            lane="fundamental_sentiment_cross_section",
            status=best.get("decision", "cross_section_feature_table"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"{best.get('side_hint', '')}, "
                f"score={best.get('total_score', '')}, "
                f"sources={best.get('source_count', '')}, "
                f"fund={best.get('fundamental_score', '')}, "
                f"sent={best.get('sentiment_score', '')}, "
                f"sector={best.get('sector_score', '')}, "
                f"funding={best.get('funding_score', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "cross-section feature table still needs neutral universe, timestamps, labels, and costs",
            ),
            next_step=best.get(
                "next_probe",
                "label the top cross-section row with leakage-safe rebalance timestamp",
            ),
        )
    return ExplorationRow(
        lane="fundamental_sentiment_cross_section",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="fundamental, sentiment, sector, and funding features are not joined cross-sectionally",
        next_step="build one cross-sectional rank table from current feature probes",
    )


def _multimodal_btc_eth_feature_alignment_row(root: Path) -> ExplorationRow:
    path = root / "current_multimodal_btc_eth_feature_alignment.csv"
    best = _best_numeric_row(path, key="alignment_score")
    if best:
        return ExplorationRow(
            lane="multimodal_btc_eth_feature_alignment",
            status=best.get("status", "multimodal_feature_alignment"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"features={best.get('feature_count', '')}, "
                f"score={best.get('alignment_score', '')}, "
                f"nlp={best.get('nlp_event_score', '')}, "
                f"attention={best.get('ticker_attention_score', '')}, "
                f"stablecoin={best.get('stablecoin_flow_score', '')}, "
                f"wallet={best.get('wallet_flow_score', '')}, "
                f"funding={best.get('funding_market_score', '')}, "
                f"equity={best.get('equity_factor_score', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "multimodal row still needs timestamp alignment, ablation, labels, and costs",
            ),
            next_step=best.get(
                "next_probe",
                "build a leakage-safe BTC/ETH multimodal feature row before model or trade action",
            ),
        )
    return ExplorationRow(
        lane="multimodal_btc_eth_feature_alignment",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="BTC/ETH multimodal features are not aligned into a single timestamp-aware row",
        next_step="join NLP/news, attention, stablecoin, wallet, funding, and equity-factor rows for BTC/ETH",
    )


def _sentiment_contagion_negative_control_row(root: Path) -> ExplorationRow:
    path = root / "current_sentiment_contagion_negative_control.csv"
    best = _best_numeric_row(path, key="control_gap")
    if best:
        return ExplorationRow(
            lane="sentiment_contagion_negative_control",
            status=best.get("status", "sentiment_contagion_control"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"belief={best.get('belief_proxy_score', '')}, "
                f"return={best.get('return_support_score', '')}, "
                f"gap={best.get('control_gap', '')}, "
                f"source={best.get('strongest_belief_source', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "sentiment control still needs social graph, belief outcome, beta attribution, and clean labels",
            ),
            next_step=best.get(
                "next_probe",
                "use social/event signal as a negative control before promotion",
            ),
        )
    return ExplorationRow(
        lane="sentiment_contagion_negative_control",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="social/event belief proxies are not separated from return-predictive alpha",
        next_step="build sentiment contagion negative controls for BTC/ETH/HYPE",
    )


def _cross_modal_alpha_context_row(root: Path) -> ExplorationRow:
    path = root / "current_cross_modal_alpha_context.csv"
    best = _best_numeric_row(path, key="total_score")
    if best:
        return ExplorationRow(
            lane="cross_modal_alpha_context",
            status=best.get("decision", "cross_modal_context"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"{best.get('aligned_direction', '')}, "
                f"score={best.get('total_score', '')}, "
                f"sources={best.get('aligned_sources', '')}"
            ),
            main_gap=best.get(
                "missing_work",
                "cross-modal context still needs timestamp, beta, execution, and source-quality controls",
            ),
            next_step=best.get("next_step", "label the strongest cross-modal context before paper probing"),
        )
    return ExplorationRow(
        lane="cross_modal_alpha_context",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="event, stablecoin, wallet, chain, and DEX contexts are not joined by tradable asset",
        next_step="run current cross-modal alpha context after source-specific probes",
    )


def _cross_modal_source_split_row(root: Path) -> ExplorationRow:
    path = root / "current_cross_modal_source_split.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="cross_modal_source_split",
            status=best.get("paper_action", "source_split"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}/{best.get('source', '')}: "
                f"{best.get('source_role', '')}, "
                f"{best.get('source_direction', '')}, "
                f"priority={best.get('priority_score', '')}"
            ),
            main_gap=best.get(
                "missing_work",
                "cross-modal source still needs source-level labels before collapsed paper action",
            ),
            next_step=best.get("next_step", "label cross-modal source components separately"),
        )
    return ExplorationRow(
        lane="cross_modal_source_split",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cross-modal context has not been split into source-level label tasks",
        next_step="run current cross-modal source split after cross-modal alpha context",
    )


def _policy_learning_row(root: Path) -> ExplorationRow:
    path = root / "policy_learning" / "current_policy_learning_samples.csv"
    best = _best_policy_learning_sample(path)
    if best:
        return ExplorationRow(
            lane="policy_learning_samples",
            status=best.get("reward_status", "sample_dataset"),
            strongest_current_signal=(
                f"{best.get('sample_id', '')}: "
                f"{best.get('asset', '')}, "
                f"action={best.get('action', '')}, "
                f"reward={best.get('reward_bps', '')}, "
                f"cost_adjusted={best.get('cost_adjusted_reward_bps', '')}"
            ),
            main_gap="samples are RL-shaped records only; no policy model, dataset split, or out-of-sample protocol exists yet",
            next_step="use samples to define a small observation/action/reward dataset before training any policy",
        )
    return ExplorationRow(
        lane="policy_learning_samples",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="paper outcomes are not converted into observation/action/reward samples",
        next_step="run policy-learning sample builder after paper outcomes and fill-risk checks",
    )


def _policy_action_preference_row(root: Path) -> ExplorationRow:
    path = root / "policy_learning" / "current_action_preference_candidates.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="policy_action_preferences",
            status=best.get("decision", "action_preference_candidate"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"samples={best.get('samples', '')}, "
                f"hit={best.get('hit_rate', '')}, "
                f"mean={best.get('mean_reward_bps', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap=(
                "action preferences are offline paper-sample aggregates; leakage-safe split, "
                "fill model, and policy evaluation are still missing"
            ),
            next_step=best.get(
                "next_step",
                "turn the strongest action preference into a leakage-safe policy-evaluation split",
            ),
        )
    return ExplorationRow(
        lane="policy_action_preferences",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="policy samples have not been aggregated into context/action preferences",
        next_step="run current action preference candidates after policy-learning samples",
    )


def _policy_context_frontier_row(root: Path) -> ExplorationRow:
    path = root / "policy_learning" / "current_policy_context_frontier.csv"
    best = _best_numeric_row(path, key="frontier_score")
    if best:
        return ExplorationRow(
            lane="policy_context_frontier",
            status=best.get("decision", "context_frontier"),
            strongest_current_signal=(
                f"{best.get('context', '')}: "
                f"records={best.get('records', '')}, "
                f"repeat={best.get('repeat_records', '')}, "
                f"mean={best.get('mean_reward_bps', '')}, "
                f"repeat_mean={best.get('repeat_mean_reward_bps', '')}, "
                f"score={best.get('frontier_score', '')}"
            ),
            main_gap=(
                "context frontier is still paper-only; observation state, action constraints, "
                "and stop/adverse-excursion fields are incomplete"
            ),
            next_step=best.get("next_step", "expand the strongest OAR-supported context"),
        )
    return ExplorationRow(
        lane="policy_context_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="OAR records have not been summarized by context",
        next_step="run policy context frontier after the OAR dataset",
    )


def _policy_action_preference_oos_row(root: Path) -> ExplorationRow:
    path = root / "policy_learning" / "current_action_preference_oos_check.csv"
    best = _best_numeric_row(path, key="oos_score")
    if best:
        return ExplorationRow(
            lane="policy_action_preference_oos",
            status=best.get("decision", "oos_check"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"train={best.get('train_samples', '')}/{best.get('train_mean_reward_bps', '')}, "
                f"test={best.get('test_samples', '')}/{best.get('test_mean_reward_bps', '')}, "
                f"score={best.get('oos_score', '')}"
            ),
            main_gap=(
                "OOS support is still paper-repeat only; live fills, stop behavior, and "
                "leakage-safe feature timestamps are not proven"
            ),
            next_step=best.get(
                "next_step",
                "paper-check OOS-supported action preference with explicit fill and stop rules",
            ),
        )
    return ExplorationRow(
        lane="policy_action_preference_oos",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="action preferences have not been checked against repeat samples",
        next_step="run action preference OOS check after action preference candidates",
    )


def _wallet_entity_flow_row(root: Path) -> ExplorationRow:
    path = root / "wallet_entity_flow" / "current_wallet_entity_flow_access.csv"
    rows = _csv_rows(path)
    access_ok = tuple(row for row in rows if row.get("status") in {"access_ok", "implemented_proxy"})
    best = access_ok[0] if access_ok else (rows[0] if rows else None)
    if best:
        return ExplorationRow(
            lane="wallet_entity_flow_access",
            status=best.get("status", "access_probe"),
            strongest_current_signal=(
                f"{best.get('source', '')}: "
                f"secret={best.get('requires_secret', '')}, "
                f"probe={best.get('probe_result', '')}"
            ),
            main_gap=best.get("limitation", "wallet/entity-flow data access still needs a concrete source"),
            next_step=best.get("next_step", "turn reachable wallet/entity data into forward labels"),
        )
    return ExplorationRow(
        lane="wallet_entity_flow_access",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="wallet/entity-flow source access has not been probed",
        next_step="run wallet/entity-flow access probe",
    )


def _hyperliquid_seed_wallet_flow_row(root: Path) -> ExplorationRow:
    path = root / "wallet_entity_flow" / "current_hyperliquid_seed_wallet_flow.csv"
    best = _best_tradable_seed_wallet_flow(path, root=root)
    if best:
        return ExplorationRow(
            lane="hyperliquid_seed_wallet_flow",
            status=best.get("action", "seed_wallet_flow"),
            strongest_current_signal=(
                f"{best.get('wallet_label', '')}/{best.get('coin', '')}: "
                f"net_buy={best.get('net_buy_notional', '')}, "
                f"net_pnl={best.get('net_closed_pnl_after_fees', '')}, "
                f"position={best.get('current_position', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap=best.get("caveat", "seed wallet is not a verified entity label"),
            next_step=best.get("next_step", "label wallet-flow pressure before treating it as alpha"),
        )
    return ExplorationRow(
        lane="hyperliquid_seed_wallet_flow",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="seed wallets have not been converted into flow observations",
        next_step="run Hyperliquid seed wallet flow probe",
    )


def _best_tradable_seed_wallet_flow(path: Path, *, root: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    tradable_assets = {
        row.get("asset", "").upper()
        for row in _csv_rows(root / "perp_market_map" / "current_hyperliquid_snapshot.csv")
        if row.get("asset")
    }
    rows = tuple(
        row
        for row in _csv_rows(path)
        if _seed_wallet_execution_asset(row.get("coin", ""), tradable_assets)
    )
    if not rows:
        return _best_numeric_row(path, key="score")
    return max(rows, key=lambda row: _safe_float(row.get("score")))


def _seed_wallet_execution_asset(source_coin: str, tradable_assets: set[str]) -> str:
    coin = source_coin.upper()
    if coin in tradable_assets:
        return coin
    if ":" in coin:
        suffix = coin.rsplit(":", 1)[-1]
        if suffix in tradable_assets:
            return suffix
    return ""


def _hyperliquid_seed_wallet_flow_actionability_row(root: Path) -> ExplorationRow:
    path = root / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv"
    best = _best_seed_wallet_flow_actionability(path)
    if best:
        return ExplorationRow(
            lane="hyperliquid_seed_wallet_flow_actionability",
            status=best.get("status", "wallet_flow_actionability"),
            strongest_current_signal=(
                f"{best.get('wallet_label', '')}/{best.get('execution_asset', '')}: "
                f"{best.get('side', '')}, "
                f"score={best.get('score', '')}, "
                f"fills={best.get('fills', '')}, "
                f"net_pnl={best.get('net_closed_pnl_after_fees', '')}, "
                f"position_usd={best.get('current_position_notional', '')}"
            ),
            main_gap=(
                "public wallet-flow candidates still need forward labels, funding, spread/depth, "
                "copycat-risk checks, and entity-quality controls"
            ),
            next_step=best.get(
                "next_step",
                "paper-label wallet-flow actionability before treating it as alpha",
            ),
        )
    return ExplorationRow(
        lane="hyperliquid_seed_wallet_flow_actionability",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="seed wallet flow has not been filtered into actionability candidates",
        next_step="run seed wallet flow actionability after seed wallet flow",
    )


def _execution_edge_mode_row(root: Path) -> ExplorationRow:
    path = root / "execution_edge" / "current_execution_mode_candidates.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="execution_edge_modes",
            status=best.get("action", ""),
            strongest_current_signal=(
                f"{best.get('asset', '')}: {best.get('execution_mode', '')}, "
                f"mode_net={best.get('estimated_mode_net_bps', '')}, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap=(
                "execution-mode candidates are paper-only and still need fill probability, "
                "queue position, latency, partial-fill, and adverse-selection evidence"
            ),
            next_step=best.get("next_step", "paper-repeat top candidate with explicit execution mode"),
        )
    return ExplorationRow(
        lane="execution_edge_modes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="paper wins are not compared across execution modes",
        next_step="run current execution mode candidates after fill-risk checks",
    )


def _paper_probe_plan_row(root: Path) -> ExplorationRow:
    path = root / "current_paper_probe_plan.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="paper_probe_plan",
            status=best.get("probe_type", "paper_probe_queue"),
            strongest_current_signal=(
                f"{best.get('opportunity', '')}: "
                f"{best.get('side', '')}, "
                f"priority={best.get('priority_score', '')}, "
                f"venue={best.get('venue', '')}, "
                f"size={best.get('candidate_size_usd', '')}"
            ),
            main_gap=best.get("missing_evidence", "paper probe still needs realized observation evidence"),
            next_step=best.get("next_step", "record the top paper observation"),
        )
    return ExplorationRow(
        lane="paper_probe_plan",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="current paper-probe candidates are not separated from the broader alpha stack",
        next_step="run current paper probe plan after the alpha stack",
    )


def _paper_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_paper_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="paper_tickets",
            status=best.get("decision", "paper_observation"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('side', '')}, "
                f"asset={best.get('asset', '')}, "
                f"venue={best.get('venue', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"checkpoints={best.get('checkpoints', '')}"
            ),
            main_gap=best.get("required_record", "paper ticket needs observation records"),
            next_step=best.get("next_step", "record paper-ticket checkpoint outcomes"),
        )
    return ExplorationRow(
        lane="paper_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="paper probe plan has not been opened into observation tickets",
        next_step="run current paper tickets after the paper probe plan",
    )


def _paper_ticket_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_paper_ticket_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="paper_ticket_outcomes",
            status=best.get("outcome", "paper_ticket_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('decision', '')}, "
                f"asset={best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "paper-ticket outcome still lacks complete evidence"),
            next_step=best.get("next_step", "refresh and record paper-ticket outcomes"),
        )
    return ExplorationRow(
        lane="paper_ticket_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="opened paper tickets have not been checked against current marks",
        next_step="run current paper ticket outcomes after ticket checkpoints mature",
    )


def _paper_ticket_action_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_paper_ticket_action_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="paper_ticket_action_queue",
            status=best.get("action", "paper_ticket_action"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}, "
                f"outcome={best.get('outcome', '')}"
            ),
            main_gap=best.get("reason", "paper-ticket action still needs follow-up evidence"),
            next_step=best.get("next_step", "run the top paper-ticket action"),
        )
    return ExplorationRow(
        lane="paper_ticket_action_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="paper-ticket outcomes have not been converted into next actions",
        next_step="run current paper ticket action queue after outcomes",
    )


def _paper_ticket_fill_risk_check_row(root: Path) -> ExplorationRow:
    path = root / "current_paper_ticket_fill_risk_check.csv"
    best = _best_numeric_row(path, key="estimated_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="paper_ticket_fill_risk_check",
            status=best.get("risk_action", "fill_risk_check"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}"
            ),
            main_gap=best.get("reason", "promoted paper ticket still needs fill and risk evidence"),
            next_step=best.get("next_step", "repeat cost-adjusted paper ticket and record risk evidence"),
        )
    return ExplorationRow(
        lane="paper_ticket_fill_risk_check",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promoted paper-ticket wins have not been checked against cost and depth",
        next_step="run current paper ticket fill risk check after action queue",
    )


def _policy_expansion_outcome_frontier_row(root: Path) -> ExplorationRow:
    path = root / "policy_learning" / "current_policy_expansion_outcome_frontier.csv"
    best = _best_numeric_row(path, key="frontier_score")
    if best:
        return ExplorationRow(
            lane="policy_expansion_outcome_frontier",
            status=best.get("decision", "policy_expansion_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"context={best.get('context', '')}, "
                f"dir={best.get('directional_return_bps', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}, "
                f"score={best.get('frontier_score', '')}"
            ),
            main_gap=best.get("evidence", "policy expansion outcome needs checkpoint and cost evidence"),
            next_step=best.get("next_step", "repeat or rework the strongest policy expansion outcome"),
        )
    return ExplorationRow(
        lane="policy_expansion_outcome_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="policy-expansion paper tickets have not been isolated from the broader paper queue",
        next_step="run policy expansion outcome frontier after paper-ticket fill risk check",
    )


def _promoted_ticket_repeat_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_promoted_ticket_repeat_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="promoted_ticket_repeat_tickets",
            status="repeat_ticket_open",
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"net_after_cost={best.get('estimated_net_after_cost_bps', '')}"
            ),
            main_gap=best.get("required_record", "repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "check repeat ticket outcome"),
        )
    return ExplorationRow(
        lane="promoted_ticket_repeat_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted paper probes have not been reopened as repeat tickets",
        next_step="open repeat tickets for cost-adjusted paper probes",
    )


def _promoted_ticket_repeat_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_promoted_ticket_repeat_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="promoted_ticket_repeat_outcomes",
            status=best.get("outcome", "repeat_ticket_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "refresh promoted repeat ticket outcomes"),
        )
    return ExplorationRow(
        lane="promoted_ticket_repeat_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promoted repeat tickets have not been checked against current marks",
        next_step="run promoted repeat ticket outcomes after checkpoint maturation",
    )


def _promoted_ticket_repeat_action_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_promoted_ticket_repeat_action_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="promoted_ticket_repeat_action_queue",
            status=best.get("action", "repeat_ticket_action"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}, "
                f"outcome={best.get('outcome', '')}"
            ),
            main_gap=best.get("reason", "repeat-ticket action still needs follow-up evidence"),
            next_step=best.get("next_step", "run the top repeat-ticket action"),
        )
    return ExplorationRow(
        lane="promoted_ticket_repeat_action_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="repeat-ticket outcomes have not been converted into next actions",
        next_step="run promoted repeat ticket action queue after outcomes",
    )


def _promoted_ticket_repeat_fill_risk_check_row(root: Path) -> ExplorationRow:
    path = root / "current_promoted_ticket_repeat_fill_risk_check.csv"
    best = _best_numeric_row(path, key="estimated_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="promoted_ticket_repeat_fill_risk_check",
            status=best.get("risk_action", "repeat_ticket_fill_risk_check"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}"
            ),
            main_gap=best.get("reason", "repeat-ticket paper ticket still needs fill and risk evidence"),
            next_step=best.get("next_step", "repeat cost-adjusted repeat ticket"),
        )
    return ExplorationRow(
        lane="promoted_ticket_repeat_fill_risk_check",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promoted repeat wins have not been checked against cost and depth",
        next_step="run repeat-ticket fill risk check after action queue",
    )


def _second_promoted_ticket_repeat_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_second_promoted_ticket_repeat_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="second_promoted_ticket_repeat_tickets",
            status="repeat_ticket_open",
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"net_after_cost={best.get('estimated_net_after_cost_bps', '')}"
            ),
            main_gap=best.get("required_record", "second repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "check second repeat ticket outcome"),
        )
    return ExplorationRow(
        lane="second_promoted_ticket_repeat_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="repeat winners have not been reopened as second repeat tickets",
        next_step="open second repeat tickets for repeat winners that survive costs",
    )


def _second_promoted_ticket_repeat_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_second_promoted_ticket_repeat_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="second_promoted_ticket_repeat_outcomes",
            status=best.get("outcome", "second_repeat_ticket_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "second repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "refresh second repeat ticket outcomes"),
        )
    return ExplorationRow(
        lane="second_promoted_ticket_repeat_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="second repeat tickets have not been checked against current marks",
        next_step="run second repeat ticket outcomes after checkpoint maturation",
    )


def _second_promoted_ticket_repeat_action_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_second_promoted_ticket_repeat_action_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="second_promoted_ticket_repeat_action_queue",
            status=best.get("action", "second_repeat_ticket_action"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}, "
                f"outcome={best.get('outcome', '')}"
            ),
            main_gap=best.get("reason", "second repeat-ticket action still needs follow-up evidence"),
            next_step=best.get("next_step", "run the top second repeat-ticket action"),
        )
    return ExplorationRow(
        lane="second_promoted_ticket_repeat_action_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="second repeat outcomes have not been converted into next actions",
        next_step="run second repeat action queue after outcomes",
    )


def _second_promoted_ticket_repeat_fill_risk_check_row(root: Path) -> ExplorationRow:
    path = root / "current_second_promoted_ticket_repeat_fill_risk_check.csv"
    best = _best_numeric_row(path, key="estimated_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="second_promoted_ticket_repeat_fill_risk_check",
            status=best.get("risk_action", "second_repeat_ticket_fill_risk_check"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}"
            ),
            main_gap=best.get("reason", "second repeat ticket still needs fill and risk evidence"),
            next_step=best.get("next_step", "repeat cost-adjusted second repeat ticket"),
        )
    return ExplorationRow(
        lane="second_promoted_ticket_repeat_fill_risk_check",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="second repeat wins have not been checked against cost and depth",
        next_step="run second repeat fill risk check after action queue",
    )


def _symbol_lane_paper_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_paper_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="symbol_lane_paper_tickets",
            status="lane_ticket_open",
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"{best.get('opportunity', '')}, "
                f"bias={best.get('lane_bias', '')}, "
                f"entry={best.get('entry_mark', '')}"
            ),
            main_gap=best.get("required_record", "symbol-lane ticket still needs outcome evidence"),
            next_step=best.get("next_step", "refresh symbol-lane paper outcomes"),
        )
    return ExplorationRow(
        lane="symbol_lane_paper_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="top symbol lanes have not been opened as separate paper tickets",
        next_step="open symbol-lane paper tickets for the top conflict cluster",
    )


def _symbol_lane_paper_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_paper_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="symbol_lane_paper_outcomes",
            status=best.get("outcome", "symbol_lane_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "symbol-lane outcome still needs follow-up evidence"),
            next_step=best.get("next_step", "refresh symbol-lane paper outcomes"),
        )
    return ExplorationRow(
        lane="symbol_lane_paper_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol-lane tickets have not been checked against current marks",
        next_step="run symbol-lane paper outcomes after checkpoint maturation",
    )


def _symbol_lane_paper_action_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_paper_action_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="symbol_lane_paper_action_queue",
            status=best.get("action", "symbol_lane_action"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}, "
                f"outcome={best.get('outcome', '')}"
            ),
            main_gap=best.get("reason", "symbol-lane action still needs follow-up evidence"),
            next_step=best.get("next_step", "run the top symbol-lane action"),
        )
    return ExplorationRow(
        lane="symbol_lane_paper_action_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol-lane outcomes have not been converted into next actions",
        next_step="run symbol-lane paper action queue after outcomes",
    )


def _symbol_lane_paper_fill_risk_check_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_paper_fill_risk_check.csv"
    best = _best_numeric_row(path, key="estimated_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="symbol_lane_paper_fill_risk_check",
            status=best.get("risk_action", "symbol_lane_fill_risk_check"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}"
            ),
            main_gap=best.get("reason", "symbol-lane paper ticket still needs fill and risk evidence"),
            next_step=best.get("next_step", "repeat cost-adjusted symbol-lane paper ticket"),
        )
    return ExplorationRow(
        lane="symbol_lane_paper_fill_risk_check",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promoted symbol-lane wins have not been checked against cost and depth",
        next_step="run symbol-lane fill risk check after action queue",
    )


def _symbol_lane_promoted_repeat_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_promoted_repeat_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="symbol_lane_promoted_repeat_tickets",
            status="repeat_ticket_open",
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"net_after_cost={best.get('estimated_net_after_cost_bps', '')}"
            ),
            main_gap=best.get("required_record", "symbol-lane repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "check symbol-lane repeat ticket outcome"),
        )
    return ExplorationRow(
        lane="symbol_lane_promoted_repeat_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted symbol-lane probes have not been reopened as repeat tickets",
        next_step="open repeat tickets for cost-adjusted symbol-lane probes",
    )


def _symbol_lane_promoted_repeat_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_promoted_repeat_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="symbol_lane_promoted_repeat_outcomes",
            status=best.get("outcome", "symbol_lane_repeat_ticket_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "symbol-lane repeat ticket still needs outcome evidence"),
            next_step=best.get("next_step", "refresh symbol-lane repeat ticket outcomes"),
        )
    return ExplorationRow(
        lane="symbol_lane_promoted_repeat_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol-lane repeat tickets have not been checked against current marks",
        next_step="run symbol-lane repeat ticket outcomes after checkpoint maturation",
    )


def _symbol_lane_promoted_repeat_action_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_promoted_repeat_action_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="symbol_lane_promoted_repeat_action_queue",
            status=best.get("action", "symbol_lane_repeat_action"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}, "
                f"outcome={best.get('outcome', '')}"
            ),
            main_gap=best.get("reason", "symbol-lane repeat action still needs follow-up evidence"),
            next_step=best.get("next_step", "run the top symbol-lane repeat action"),
        )
    return ExplorationRow(
        lane="symbol_lane_promoted_repeat_action_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol-lane repeat outcomes have not been converted into next actions",
        next_step="run symbol-lane repeat action queue after outcomes",
    )


def _symbol_lane_promoted_repeat_fill_risk_check_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_promoted_repeat_fill_risk_check.csv"
    best = _best_numeric_row(path, key="estimated_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="symbol_lane_promoted_repeat_fill_risk_check",
            status=best.get("risk_action", "symbol_lane_repeat_fill_risk_check"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"spread={best.get('spread_bps', '')}, "
                f"usage={best.get('visible_depth_usage', '')}"
            ),
            main_gap=best.get("reason", "symbol-lane repeat ticket still needs fill and risk evidence"),
            next_step=best.get("next_step", "repeat cost-adjusted symbol-lane repeat ticket"),
        )
    return ExplorationRow(
        lane="symbol_lane_promoted_repeat_fill_risk_check",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol-lane repeat wins have not been checked against cost and depth",
        next_step="run symbol-lane repeat fill risk check after action queue",
    )


def _cost_adjusted_alpha_candidates_row(root: Path) -> ExplorationRow:
    path = root / "current_cost_adjusted_alpha_candidates.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="cost_adjusted_alpha_candidates",
            status=best.get("status", "cost_adjusted_alpha_candidate"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"{best.get('asset', '')}, "
                f"net={best.get('estimated_net_after_cost_bps', '')}bps, "
                f"priority={best.get('priority_score', '')}"
            ),
            main_gap=best.get("missing_work", "cost-adjusted candidates still need repeat and fill evidence"),
            next_step=best.get("next_step", "repeat the strongest cost-adjusted alpha candidate"),
        )
    return ExplorationRow(
        lane="cost_adjusted_alpha_candidates",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted candidates are still scattered across fill-risk lanes",
        next_step="run current cost adjusted alpha candidates after fill-risk checks",
    )


def _cost_adjusted_alpha_clusters_row(root: Path) -> ExplorationRow:
    path = root / "current_cost_adjusted_alpha_clusters.csv"
    best = _best_numeric_row(path, key="cluster_score")
    if best:
        return ExplorationRow(
            lane="cost_adjusted_alpha_clusters",
            status=best.get("status", "cost_adjusted_alpha_cluster"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}: "
                f"candidates={best.get('candidate_count', '')}, "
                f"lanes={best.get('source_lane_count', '')}, "
                f"best_net={best.get('best_net_after_cost_bps', '')}bps, "
                f"score={best.get('cluster_score', '')}"
            ),
            main_gap=best.get("missing_work", "cluster still needs repeat and fill evidence"),
            next_step=best.get("next_step", "repeat the strongest cost-adjusted alpha cluster"),
        )
    return ExplorationRow(
        lane="cost_adjusted_alpha_clusters",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted candidates have not been grouped by asset and direction",
        next_step="run current cost adjusted alpha clusters after candidate aggregation",
    )


def _cost_survival_cross_section_row(root: Path) -> ExplorationRow:
    path = root / "current_cost_survival_cross_section.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="cost_survival_cross_section",
            status=best.get("status", "cost_survival_cross_section"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}: "
                f"score={best.get('survival_score', '')}, "
                f"net={best.get('mean_net_after_cost_bps', '')}bps, "
                f"wins={best.get('repeat_win_count', '')}, "
                f"lanes={best.get('source_lane_count', '')}, "
                f"dup={best.get('duplicate_pressure', '')}"
            ),
            main_gap=best.get("missing_work", "cost-survival row still needs fill and repeat evidence"),
            next_step=best.get("next_probe", "paper-check the top cost-surviving cluster"),
        )
    return ExplorationRow(
        lane="cost_survival_cross_section",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted clusters are not ranked by repeat, depth, and duplicate survival",
        next_step="run current cost survival cross section after cost-adjusted clustering",
    )


def _alpha_promotion_frontier_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_promotion_frontier.csv"
    best = _first_matching_row(path, key="lane", value="cost_survival") or _best_numeric_row(
        path,
        key="frontier_score",
    )
    if best:
        return ExplorationRow(
            lane="alpha_promotion_frontier",
            status=best.get("status", "alpha_promotion_frontier"),
            strongest_current_signal=(
                f"{best.get('frontier_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('action', '')}, "
                f"score={best.get('frontier_score', '')}, "
                f"edge={best.get('edge_bps', '')}"
            ),
            main_gap=best.get("blocker", "alpha frontier row still has an unresolved promotion blocker"),
            next_step=best.get("next_step", "clear the top promotion blocker or reject the row"),
        )
    return ExplorationRow(
        lane="alpha_promotion_frontier",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="paper candidates and modern-lane blockers are not summarized in one promotion view",
        next_step="run current alpha promotion frontier after cost survival and modern-lane survival checks",
    )


def _alpha_promotion_worklist_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_promotion_worklist.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="alpha_promotion_worklist",
            status=best.get("work_kind", "alpha_promotion_work"),
            strongest_current_signal=(
                f"{best.get('work_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('action', '')}, "
                f"priority={best.get('priority', '')}, "
                f"status={best.get('status', '')}"
            ),
            main_gap=best.get("required_evidence", "top promotion work still needs evidence"),
            next_step=best.get("next_step", "execute the top alpha promotion work item"),
        )
    return ExplorationRow(
        lane="alpha_promotion_worklist",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promotion frontier has not been reduced to a non-duplicate worklist",
        next_step="run current alpha promotion worklist after promotion frontier",
    )


def _alpha_repeat_fill_survival_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_repeat_fill_survival.csv"
    best = _best_numeric_row(path, key="survival_score")
    if best:
        return ExplorationRow(
            lane="alpha_repeat_fill_survival",
            status=best.get("status", "alpha_repeat_fill_survival"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"{best.get('decision', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"first_net={best.get('first_repeat_net_after_cost_bps', '')}, "
                f"second_net={best.get('second_repeat_net_after_cost_bps', '')}, "
                f"second={best.get('second_repeat_outcome', '')}"
            ),
            main_gap=best.get("required_evidence", "repeat row still lacks fill and stop evidence"),
            next_step=best.get("next_step", "rerun repeat fill survival after fresh repeat marks"),
        )
    return ExplorationRow(
        lane="alpha_repeat_fill_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="repeat/fill worklist rows have not been checked against repeat evidence",
        next_step="run current alpha repeat fill survival after promotion worklist",
    )


def _surviving_alpha_path_risk_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_path_risk.csv"
    best = _best_numeric_row(path, key="second_net_after_cost_bps")
    if best:
        return ExplorationRow(
            lane="surviving_alpha_path_risk",
            status=best.get("path_action", "surviving_alpha_path_risk"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"{best.get('decision', '')}, "
                f"net={best.get('second_net_after_cost_bps', '')}, "
                f"close={best.get('close_return_bps', '')}, "
                f"adverse={best.get('max_adverse_bps', '')}, "
                f"stop50={best.get('stop_50bps_status', '')}"
            ),
            main_gap=best.get("evidence", "surviving alpha still needs path-risk evidence"),
            next_step=best.get("next_step", "check public candle path risk before promotion"),
        )
    return ExplorationRow(
        lane="surviving_alpha_path_risk",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="second-repeat survivors have not been checked against candle path risk",
        next_step="run surviving alpha path risk after repeat fill survival",
    )


def _surviving_alpha_fill_audit_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_fill_audit_tickets.csv"
    rows = _csv_rows(path)
    if rows:
        row = rows[0]
        return ExplorationRow(
            lane="surviving_alpha_fill_audit_tickets",
            status="paper_fill_audit_ticket_open",
            strongest_current_signal=(
                f"{row.get('ticket_id', '')}: "
                f"{row.get('asset', '')} {row.get('side', '')}, "
                f"entry={row.get('entry_mark', '')}, "
                f"stop={row.get('stop_bps', '')}, "
                f"horizons={row.get('audit_horizons', '')}"
            ),
            main_gap=row.get("required_record", "fill audit still needs fresh public path evidence"),
            next_step=row.get("next_step", "wait for fill-audit outcome checkpoint"),
        )
    return ExplorationRow(
        lane="surviving_alpha_fill_audit_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="path-survived repeat candidates have not been opened as fresh fill-audit tickets",
        next_step="run fill-audit tickets after surviving alpha path risk",
    )


def _surviving_alpha_fill_audit_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_fill_audit_outcomes.csv"
    rows = _csv_rows(path)
    if rows:
        best = _best_fill_audit_outcome(rows)
        return ExplorationRow(
            lane="surviving_alpha_fill_audit_outcomes",
            status=best.get("outcome", "paper_fill_audit_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}/{best.get('horizon', '')}: "
                f"{best.get('checkpoint_status', '')}, "
                f"close={best.get('close_return_bps', '')}, "
                f"adverse={best.get('max_adverse_bps', '')}, "
                f"stop={best.get('stop_status', '')}"
            ),
            main_gap=best.get("evidence", "fill audit still needs path evidence"),
            next_step=best.get("next_step", "wait for or review fill-audit outcome"),
        )
    return ExplorationRow(
        lane="surviving_alpha_fill_audit_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="fresh fill-audit tickets have not been checked",
        next_step="run fill-audit outcomes after opening tickets",
    )


def _surviving_alpha_exit_regime_candidates_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_exit_regime_candidates.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="surviving_alpha_exit_regime_candidates",
            status=best.get("status", "surviving_alpha_exit_regime_candidate"),
            strongest_current_signal=(
                f"{best.get('candidate_id', '')}: "
                f"close={best.get('close_return_bps', '')}, "
                f"adverse={best.get('max_adverse_bps', '')}, "
                f"stop50={best.get('stop_50bps_status', '')}, "
                f"stop100={best.get('stop_100bps_status', '')}"
            ),
            main_gap=best.get("required_record", "exit regime still needs a fresh-trigger repeat"),
            next_step=best.get("next_step", "repeat the strongest exit regime candidate"),
        )
    return ExplorationRow(
        lane="surviving_alpha_exit_regime_candidates",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="path-risk blocked survivors have not been split by exit horizon",
        next_step="run exit-regime candidates after surviving alpha path risk",
    )


def _surviving_alpha_exit_regime_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_exit_regime_tickets.csv"
    rows = _csv_rows(path)
    if rows:
        row = rows[0]
        return ExplorationRow(
            lane="surviving_alpha_exit_regime_tickets",
            status="paper_exit_regime_ticket_open",
            strongest_current_signal=(
                f"{row.get('ticket_id', '')}: "
                f"{row.get('asset', '')} {row.get('side', '')}, "
                f"entry={row.get('entry_mark', '')}, "
                f"exit={row.get('exit_horizon_minutes', '')}m, "
                f"stop={row.get('stop_bps', '')}"
            ),
            main_gap=row.get("required_record", "exit-regime ticket needs a fresh paper path"),
            next_step=row.get("next_step", "wait for exit-regime outcome checkpoint"),
        )
    return ExplorationRow(
        lane="surviving_alpha_exit_regime_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="wide-stop exit candidates have not been opened as fresh paper tickets",
        next_step="run exit-regime tickets after exit-regime candidates",
    )


def _surviving_alpha_exit_regime_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_surviving_alpha_exit_regime_outcomes.csv"
    rows = _csv_rows(path)
    if rows:
        row = rows[0]
        return ExplorationRow(
            lane="surviving_alpha_exit_regime_outcomes",
            status=row.get("outcome", "paper_exit_regime_outcome"),
            strongest_current_signal=(
                f"{row.get('ticket_id', '')}: "
                f"{row.get('checkpoint_status', '')}, "
                f"close={row.get('close_return_bps', '')}, "
                f"adverse={row.get('max_adverse_bps', '')}, "
                f"stop={row.get('stop_status', '')}"
            ),
            main_gap=row.get("evidence", "exit-regime outcome still needs path evidence"),
            next_step=row.get("next_step", "wait for or review exit-regime outcome"),
        )
    return ExplorationRow(
        lane="surviving_alpha_exit_regime_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="fresh exit-regime tickets have not been checked",
        next_step="run exit-regime outcomes after opening tickets",
    )


def _best_fill_audit_outcome(rows: tuple[dict[str, str], ...]) -> dict[str, str]:
    outcome_rank = {
        "paper_fill_audit_win": 4,
        "pending": 3,
        "paper_fill_audit_loss": 2,
        "paper_fill_audit_stop_loss": 1,
        "missing_path": 0,
    }
    return max(
        rows,
        key=lambda row: (
            outcome_rank.get(row.get("outcome", ""), 0),
            _safe_float(row.get("close_return_bps")),
            -abs(_safe_float(row.get("max_adverse_bps"))),
        ),
    )


def _alpha_conflict_resolution_progress_row(root: Path) -> ExplorationRow:
    path = root / "current_alpha_conflict_resolution_progress.csv"
    best = _best_numeric_row(path, key="progress_score")
    if best:
        return ExplorationRow(
            lane="alpha_conflict_resolution_progress",
            status=best.get("status", "alpha_conflict_resolution_progress"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: "
                f"{best.get('decision', '')}, "
                f"action={best.get('cluster_action', '')}, "
                f"plans={best.get('lane_plan_count', '')}, "
                f"queued={best.get('queued_lane_count', '')}, "
                f"repeats={best.get('lane_repeat_count', '')}"
            ),
            main_gap=best.get("blocker", "conflict resolution still has an unresolved blocker"),
            next_step=best.get("next_step", "resolve top alpha cluster conflict"),
        )
    return ExplorationRow(
        lane="alpha_conflict_resolution_progress",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="promotion worklist dedupe and source split rows have not been reconciled with split artifacts",
        next_step="run current alpha conflict resolution progress after promotion worklist",
    )


def _cost_adjusted_cluster_repeat_plan_row(root: Path) -> ExplorationRow:
    path = root / "current_cost_adjusted_cluster_repeat_plan.csv"
    best = _best_numeric_row(path, key="cluster_score")
    if best:
        return ExplorationRow(
            lane="cost_adjusted_cluster_repeat_plan",
            status=best.get("action", "cluster_repeat_plan"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('decision', '')}, "
                f"net={best.get('best_net_after_cost_bps', '')}bps"
            ),
            main_gap=best.get("required_record", "cluster-level repeat still needs fill and risk notes"),
            next_step=best.get("next_step", "open the top consolidated cluster repeat probe"),
        )
    return ExplorationRow(
        lane="cost_adjusted_cluster_repeat_plan",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cost-adjusted clusters have not been converted into consolidated repeat actions",
        next_step="run current cost adjusted cluster repeat plan after clustering",
    )


def _split_first_cluster_lane_plan_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_cluster_lane_plan.csv"
    best = _best_numeric_row(path, key="resolution_score")
    if best:
        return ExplorationRow(
            lane="split_first_cluster_lane_plan",
            status=best.get("resolution_action", "split_first_lane_plan"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}/{best.get('lane_opportunity', '')}: "
                f"{best.get('lane_bias', '')}, "
                f"score={best.get('resolution_score', '')}"
            ),
            main_gap=best.get("required_record", "split-first lane still needs lane-specific evidence"),
            next_step=best.get("next_step", "resolve the strongest split-first cluster lane"),
        )
    return ExplorationRow(
        lane="split_first_cluster_lane_plan",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="mixed cost-adjusted clusters have not been decomposed into lane-level actions",
        next_step="run current split first cluster lane plan after cluster repeat planning",
    )


def _split_first_lane_repeat_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_repeat_queue.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="split_first_lane_repeat_queue",
            status=best.get("action", "split_first_lane_queue"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}/{best.get('lane_opportunity', '')}: "
                f"{best.get('lane_side', '')}, "
                f"priority={best.get('priority', '')}"
            ),
            main_gap=best.get("required_record", "lane queue still needs paper evidence"),
            next_step=best.get("next_step", "run the strongest split-first lane queue item"),
        )
    return ExplorationRow(
        lane="split_first_lane_repeat_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="split-first lane plan has not been converted into queueable lane paper work",
        next_step="run current split first lane repeat queue after lane planning",
    )


def _split_first_lane_label_progress_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_label_progress.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="split_first_lane_label_progress",
            status=best.get("progress_status", "split_first_lane_label_progress"),
            strongest_current_signal=(
                f"{best.get('cluster_id', '')}/{best.get('lane_opportunity', '')}: "
                f"{best.get('asset', '')}, "
                f"priority={best.get('priority', '')}"
            ),
            main_gap=best.get("required_record", "lane label work still needs a forward record"),
            next_step=best.get("next_step", "record the top split-first lane label"),
        )
    return ExplorationRow(
        lane="split_first_lane_label_progress",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="open lane-label rows are not tracked separately from repeat tickets",
        next_step="run current split first lane label progress after the split-first queue",
    )


def _split_first_lane_label_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_label_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="split_first_lane_label_tickets",
            status="lane_label_ticket_open",
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"{best.get('opportunity', '')}, "
                f"entry={best.get('entry_mark', '')}"
            ),
            main_gap=best.get("required_record", "lane label ticket still needs a forward outcome"),
            next_step=best.get("next_step", "refresh split-first lane label outcomes"),
        )
    return ExplorationRow(
        lane="split_first_lane_label_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="split-first lane label rows have not been opened as observation tickets",
        next_step="open lane-label tickets from split-first label progress",
    )


def _split_first_lane_label_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_label_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="split_first_lane_label_outcomes",
            status=best.get("outcome", "split_first_lane_label_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "lane label still needs mature forward evidence"),
            next_step=best.get("next_step", "refresh split-first lane label outcomes"),
        )
    return ExplorationRow(
        lane="split_first_lane_label_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="lane-label tickets have not been checked against current marks",
        next_step="run split-first lane label outcomes after ticket opening",
    )


def _split_first_lane_repeat_tickets_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_repeat_tickets.csv"
    rows = _csv_rows(path)
    best = rows[0] if rows else None
    if best:
        return ExplorationRow(
            lane="split_first_lane_repeat_tickets",
            status="lane_repeat_ticket_open",
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"net_after_cost={best.get('estimated_net_after_cost_bps', '')}"
            ),
            main_gap=best.get("required_record", "split-first lane repeat still needs outcome evidence"),
            next_step=best.get("next_step", "check split-first lane repeat outcome"),
        )
    return ExplorationRow(
        lane="split_first_lane_repeat_tickets",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="split-first repeat queue has not been opened as lane-level paper tickets",
        next_step="open split-first lane repeat tickets from the queue",
    )


def _split_first_lane_repeat_outcomes_row(root: Path) -> ExplorationRow:
    path = root / "current_split_first_lane_repeat_outcomes.csv"
    rows = _csv_rows(path)
    best = _best_paper_ticket_outcome(rows)
    if best:
        return ExplorationRow(
            lane="split_first_lane_repeat_outcomes",
            status=best.get("outcome", "split_first_lane_repeat_outcome"),
            strongest_current_signal=(
                f"{best.get('ticket_id', '')}: "
                f"{best.get('asset', '')}, "
                f"entry={best.get('entry_mark', '')}, "
                f"current={best.get('current_mark', '')}, "
                f"dir_bps={best.get('directional_return_bps', '')}"
            ),
            main_gap=best.get("missing_evidence", "split-first lane repeat still needs outcome evidence"),
            next_step=best.get("next_step", "refresh split-first lane repeat outcomes"),
        )
    return ExplorationRow(
        lane="split_first_lane_repeat_outcomes",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="split-first lane repeat tickets have not been checked against current marks",
        next_step="run split-first lane repeat outcomes after checkpoint maturation",
    )


def _symbol_opportunity_map_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_opportunity_map.csv"
    best = _best_numeric_row(path, key="cluster_score")
    if best:
        return ExplorationRow(
            lane="symbol_opportunity_map",
            status=best.get("status", "symbol_cluster"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"score={best.get('cluster_score', '')}, "
                f"sources={best.get('source_count', '')}, "
                f"candidates={best.get('candidate_count', '')}, "
                f"top={best.get('top_opportunities', '')}"
            ),
            main_gap="symbol clusters are prioritization only; they still need forward labels, costs, depth, and conflict checks",
            next_step=best.get("next_step", "label top symbol cluster against forward returns and execution feasibility"),
        )
    return ExplorationRow(
        lane="symbol_opportunity_map",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="current candidates are not grouped by symbol",
        next_step="run current symbol opportunity map to find cross-lane symbol clusters",
    )


def _symbol_cluster_conflicts_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_cluster_conflicts.csv"
    best = _best_numeric_row(path, key="cluster_score")
    if best:
        return ExplorationRow(
            lane="symbol_cluster_conflicts",
            status=best.get("status", "symbol_conflict"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"bias={best.get('dominant_bias', '')}, "
                f"L={best.get('long_count', '')}, "
                f"S={best.get('short_count', '')}, "
                f"RV={best.get('relative_value_count', '')}, "
                f"Y={best.get('yield_count', '')}, "
                f"R={best.get('risk_or_avoid_count', '')}, "
                f"score={best.get('cluster_score', '')}"
            ),
            main_gap=best.get(
                "conflicts",
                "symbol cluster direction is not resolved into a single action",
            ),
            next_step=best.get("next_step", "split symbol labels by lane before trading"),
        )
    return ExplorationRow(
        lane="symbol_cluster_conflicts",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol clusters have not been checked for directional conflicts",
        next_step="run current symbol cluster conflict screen after the symbol opportunity map",
    )


def _symbol_cluster_label_queue_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_cluster_label_queue.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="symbol_cluster_label_queue",
            status=best.get("queue_action", "symbol_label_queue"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"priority={best.get('priority_score', '')}, "
                f"cluster={best.get('cluster_status', '')}, "
                f"bias={best.get('dominant_bias', '')}, "
                f"sources={best.get('source_count', '')}, "
                f"candidates={best.get('candidate_count', '')}, "
                f"top={best.get('top_opportunities', '')}"
            ),
            main_gap=best.get("reason", "symbol-level labels are not yet separated by lane"),
            next_step=best.get("next_step", "run the top symbol-level label task"),
        )
    return ExplorationRow(
        lane="symbol_cluster_label_queue",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="symbol conflict output has not been turned into label tasks",
        next_step="run current symbol cluster label queue after conflict review",
    )


def _symbol_lane_split_review_row(root: Path) -> ExplorationRow:
    path = root / "current_symbol_lane_split_review.csv"
    best = _best_numeric_row(path, key="priority_score")
    if best:
        return ExplorationRow(
            lane="symbol_lane_split_review",
            status=best.get("lane_action", "lane_split_review"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"{best.get('opportunity', '')}, "
                f"bias={best.get('lane_bias', '')}, "
                f"support={best.get('support_state', '')}, "
                f"role={best.get('conflict_role', '')}, "
                f"priority={best.get('priority_score', '')}"
            ),
            main_gap="the same symbol can contain different alpha hypotheses with different labels and execution paths",
            next_step=best.get("next_step", "label this lane separately before combining symbol-level views"),
        )
    return ExplorationRow(
        lane="symbol_lane_split_review",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="top symbol queue has not been split into lane-level label tasks",
        next_step="run current symbol lane split review after the symbol label queue",
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
    dislocation_signal = _hyperliquid_dislocation_signal(
        root / "perp_market_map" / "current_hyperliquid_dislocation_candidates.csv"
    ) + _hyperliquid_dislocation_monitor_signal(
        root / "perp_market_map" / "current_hyperliquid_dislocation_monitor_summary.csv"
    ) + _hyperliquid_dislocation_label_signal(
        root / "perp_market_map" / "current_hyperliquid_dislocation_forward_labels.csv"
    ) + _hyperliquid_dislocation_execution_signal(
        root / "perp_market_map" / "current_hyperliquid_dislocation_execution_check.csv"
    )
    actionability_path = root / "perp_market_map" / "current_hyperliquid_dislocation_actionability.csv"
    actionability = _best_hyperliquid_dislocation_actionability_row(actionability_path)
    if actionability:
        return ExplorationRow(
            lane="perp_market_map",
            status=actionability.get("status", "hyperliquid_dislocation_actionability"),
            strongest_current_signal=(
                f"{actionability.get('asset', '')}: "
                f"{actionability.get('side', '')}, "
                f"score={actionability.get('score', '')}, "
                f"current1h={actionability.get('current_outcome_1h', '')} "
                f"{actionability.get('current_net_1h_bps', '')}, "
                f"gate={actionability.get('execution_gate', '')}, "
                f"hist_win1h={actionability.get('history_win_1h', '')}, "
                f"hist_mean1h={actionability.get('history_mean_net_1h_bps', '')}"
                f"{okx_signal}"
            ),
            main_gap=actionability.get(
                "reason",
                "dislocation candidate still needs repeated paper probes, stop behavior, and adverse-selection checks",
            ),
            next_step=actionability.get(
                "next_step",
                "repeat top Hyperliquid dislocation paper probe with current execution evidence",
            ),
        )
    outcome_path = root / "perp_market_map" / "current_crowding_reversion_paper_outcome.csv"
    best_outcome = _best_crowding_outcome_row(outcome_path)
    if best_outcome:
        return ExplorationRow(
            lane="perp_market_map",
            status=_crowding_outcome_status(best_outcome),
            strongest_current_signal=(
                f"{best_outcome.get('asset', '')}: "
                f"{best_outcome.get('action', '')}, "
                f"size={best_outcome.get('candidate_size_usd', '')}, "
                f"net15_bps={best_outcome.get('net_15m_bps', '')}, "
                f"out15={best_outcome.get('outcome_15m', '')}, "
                f"net1h_bps={best_outcome.get('net_1h_bps', '')}, "
                f"out1h={best_outcome.get('outcome_1h', '')}"
                f"{dislocation_signal}"
                f"{okx_signal}"
            ),
            main_gap=(
                "paper outcome is still not a live fill; queue position, partial fills, "
                "funding timing, stop behavior, and repeated samples are missing"
            ),
            next_step="wait for elapsed horizons, then repeat the gated probes on fresh HL crowding snapshots",
        )
    execution_path = root / "perp_market_map" / "current_crowding_reversion_execution_check.csv"
    best_execution = _best_crowding_execution_row(execution_path)
    if best_execution:
        return ExplorationRow(
            lane="perp_market_map",
            status=best_execution.get("gate_action", "crowding_execution_check"),
            strongest_current_signal=(
                f"{best_execution.get('asset', '')}: "
                f"{best_execution.get('action', '')}, "
                f"size={best_execution.get('candidate_size_usd', '')}, "
                f"net1h_bps={best_execution.get('net_1h_proxy_bps', '')}, "
                f"conservative_net1h_bps={best_execution.get('conservative_net_1h_bps', '')}, "
                f"spread_bps={best_execution.get('spread_bps', '')}, "
                f"depth_usage={best_execution.get('visible_depth_usage_10bps', '')}"
                f"{dislocation_signal}"
                f"{okx_signal}"
            ),
            main_gap=(
                "public-book gate is not a fill model; queue position, repeated adverse selection, "
                "stop behavior, and live paper fills are still missing"
            ),
            next_step="paper probe top gated HL carry-reversion candidates and record live fill/outcome evidence",
        )
    validated_path = root / "perp_market_map" / "current_crowding_reversion_validated_candidates.csv"
    best_validated = _best_crowding_validated_row(validated_path)
    if best_validated:
        return ExplorationRow(
            lane="perp_market_map",
            status=best_validated.get("status", "validated_carry_reversion_candidate"),
            strongest_current_signal=(
                f"{best_validated.get('asset', '')}: "
                f"{best_validated.get('action', '')}, "
                f"score={best_validated.get('validation_score', '')}, "
                f"dir15={best_validated.get('mean_directional_return_15m', '')}, "
                f"dir1h={best_validated.get('mean_directional_return_1h', '')}, "
                f"net1h={best_validated.get('net_directional_return_1h_proxy', '')}, "
                f"hit1h={best_validated.get('positive_directional_1h_rate', '')}"
                f"{dislocation_signal}"
                f"{okx_signal}"
            ),
            main_gap="validated carry-reversion labels are still tiny and do not include funding PnL, fees, spread, or stop behavior",
            next_step=best_validated.get(
                "next_step",
                "repeat top HL carry-reversion labels and add execution costs",
            ),
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
                f"{dislocation_signal}"
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
                f"{dislocation_signal}"
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
            f"{dislocation_signal}"
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


def _binance_derivatives_history_row(root: Path) -> ExplorationRow:
    compare_path = root / "p0_parallel" / "binance_derivatives_feature_regime_compare.csv"
    best_compare = _best_numeric_row(compare_path, key="combined_score")
    if best_compare:
        return ExplorationRow(
            lane="binance_derivatives_history",
            status=best_compare.get("status", "feature_regime_compare"),
            strongest_current_signal=(
                f"{best_compare.get('symbol', '')} "
                f"{best_compare.get('feature', '')}: "
                f"historical={best_compare.get('historical_score', '')}, "
                f"recent={best_compare.get('recent_score', '')}, "
                f"combined={best_compare.get('combined_score', '')}, "
                f"buckets={best_compare.get('historical_bucket', '')}/{best_compare.get('recent_bucket', '')}"
            ),
            main_gap="regime comparison is still daily and needs recent intraday labels plus execution costs",
            next_step=best_compare.get(
                "next_step",
                "rerun top Binance derivatives regime candidate with recent intraday labels",
            ),
        )
    symbol_feature_path = root / "p0_parallel" / "binance_derivatives_symbol_feature_candidates.csv"
    best_symbol_feature = _best_numeric_row(symbol_feature_path, key="edge_score")
    if best_symbol_feature:
        return ExplorationRow(
            lane="binance_derivatives_history",
            status=best_symbol_feature.get("status", "symbol_feature_queue"),
            strongest_current_signal=(
                f"{best_symbol_feature.get('symbol', '')} "
                f"{best_symbol_feature.get('feature', '')}: "
                f"bucket={best_symbol_feature.get('preferred_bucket', '')}, "
                f"score={best_symbol_feature.get('edge_score', '')}, "
                f"low_mean={best_symbol_feature.get('low_bucket_mean_next_return', '')}, "
                f"high_mean={best_symbol_feature.get('high_bucket_mean_next_return', '')}"
            ),
            main_gap="symbol-feature queue is historical and still needs recent-window reruns, regime splits, and execution costs",
            next_step=best_symbol_feature.get(
                "next_step",
                "rerun top Binance derivatives symbol-feature candidate on recent windows",
            ),
        )
    signal_path = root / "p0_parallel" / "binance_derivatives_signal_summary.csv"
    best_corr = _best_abs_numeric_row(signal_path, key="correlation_to_next_return")
    if best_corr:
        return ExplorationRow(
            lane="binance_derivatives_history",
            status="feature_prior_only",
            strongest_current_signal=(
                f"{best_corr.get('feature', '')}: "
                f"obs={best_corr.get('observations', '')}, "
                f"corr={best_corr.get('correlation_to_next_return', '')}, "
                f"high_mean={best_corr.get('high_bucket_mean_next_return', '')}"
            ),
            main_gap="feature summary is not split by symbol or recent regime",
            next_step="build symbol-feature candidates from the Binance derivatives history panel",
        )
    return ExplorationRow(
        lane="binance_derivatives_history",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="Binance derivatives history is not summarized into symbol-feature candidates",
        next_step="run Binance derivatives history and symbol-feature candidate queue",
    )


def _binance_derivatives_intraday_row(root: Path) -> ExplorationRow:
    path = root / "p0_parallel" / "binance_derivatives_intraday_feature_candidates.csv"
    best = _best_numeric_row(path, key="edge_score")
    if best:
        return ExplorationRow(
            lane="binance_derivatives_intraday",
            status=best.get("status", "intraday_feature_label"),
            strongest_current_signal=(
                f"{best.get('symbol', '')} "
                f"{best.get('feature', '')}: "
                f"bucket={best.get('preferred_bucket', '')}, "
                f"obs={best.get('observations', '')}, "
                f"low_1h={best.get('low_bucket_mean_next_1h_return', '')}, "
                f"high_1h={best.get('high_bucket_mean_next_1h_return', '')}, "
                f"score={best.get('edge_score', '')}"
            ),
            main_gap="5m-to-1h label screen still excludes fees, spread, fill probability, funding PnL, and repeat-window checks",
            next_step=best.get(
                "next_step",
                "repeat top intraday derivatives feature label on a fresh window with costs and fills",
            ),
        )
    return ExplorationRow(
        lane="binance_derivatives_intraday",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="daily derivatives candidates have not been checked on 5m-to-1h labels",
        next_step="run Binance derivatives intraday feature label screen for top recent symbol-feature candidates",
    )


def _binance_derivatives_intraday_repeat_row(root: Path) -> ExplorationRow:
    path = root / "p0_parallel" / "binance_derivatives_intraday_repeat_compare.csv"
    best = _best_numeric_row(path, key="combined_score")
    if best:
        return ExplorationRow(
            lane="binance_derivatives_intraday_repeat",
            status=best.get("status", "intraday_repeat_compare"),
            strongest_current_signal=(
                f"{best.get('symbol', '')} "
                f"{best.get('feature', '')}: "
                f"prior={best.get('prior_bucket', '')}/{best.get('prior_score', '')}, "
                f"recent={best.get('recent_bucket', '')}/{best.get('recent_score', '')}, "
                f"combined={best.get('combined_score', '')}"
            ),
            main_gap="repeat compare still excludes fees, spread, fill probability, funding PnL, stop behavior, and sizing",
            next_step=best.get(
                "next_step",
                "run the top repeated intraday derivatives feature with costs and fills",
            ),
        )
    return ExplorationRow(
        lane="binance_derivatives_intraday_repeat",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="recent 5m-to-1h labels have not been repeated on a non-overlapping window",
        next_step="run Binance derivatives intraday repeat compare before promoting any intraday feature",
    )


def _binance_derivatives_intraday_paper_row(root: Path) -> ExplorationRow:
    best_low_cost = _best_numeric_row(
        root / "p0_parallel" / "binance_derivatives_intraday_paper_labels_2bps.csv",
        key="score",
    )
    best_conservative = _best_numeric_row(
        root / "p0_parallel" / "binance_derivatives_intraday_paper_labels.csv",
        key="score",
    )
    best = best_low_cost or best_conservative
    if best:
        conservative_note = ""
        if best_conservative:
            conservative_note = (
                f"; 8bps_best={best_conservative.get('symbol', '')} "
                f"{best_conservative.get('feature', '')} "
                f"{best_conservative.get('status', '')} "
                f"net={best_conservative.get('combined_net_mean_1h', '')}"
            )
        return ExplorationRow(
            lane="binance_derivatives_intraday_paper",
            status=best.get("status", "intraday_paper_label"),
            strongest_current_signal=(
                f"{best.get('symbol', '')} "
                f"{best.get('feature', '')} "
                f"{best.get('action', '')}: "
                f"cost={best.get('round_trip_cost_bps', '')}bps, "
                f"prior_net={best.get('prior_net_mean_1h', '')}, "
                f"recent_net={best.get('recent_net_mean_1h', '')}, "
                f"combined_net={best.get('combined_net_mean_1h', '')}, "
                f"hit={best.get('combined_hit_rate', '')}"
                f"{conservative_note}"
            ),
            main_gap="paper label uses rough costs only and still lacks live spread, funding timestamp, fill delay, stop, and sizing",
            next_step=best.get(
                "next_step",
                "paper-check the top intraday feature with live spread, funding timing, fill delay, and stops",
            ),
        )
    return ExplorationRow(
        lane="binance_derivatives_intraday_paper",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="repeat intraday features have not been evaluated after rough costs",
        next_step="run Binance derivatives intraday paper labels with conservative and low-cost assumptions",
    )


def _binance_derivatives_intraday_live_gate_row(root: Path) -> ExplorationRow:
    path = root / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv"
    best = _best_numeric_row(path, key="estimated_low_fee_net_1h_bps")
    if best:
        return ExplorationRow(
            lane="binance_derivatives_intraday_live_gate",
            status=best.get("gate_action", "intraday_live_gate"),
            strongest_current_signal=(
                f"{best.get('symbol', '')} "
                f"{best.get('feature', '')} "
                f"{best.get('action', '')}: "
                f"source={best.get('source_status', '')}, "
                f"condition={best.get('live_condition', '')}, "
                f"spread={best.get('spread_bps', '')}, "
                f"depth5={best.get('side_depth_5bps_notional', '')}, "
                f"funding1h={best.get('funding_return_1h_bps', '')}, "
                f"low_fee_net={best.get('estimated_low_fee_net_1h_bps', '')}, "
                f"taker_net={best.get('estimated_taker_net_1h_bps', '')}"
            ),
            main_gap="Binance live feature source is blocked here; OKX book/funding is execution context only",
            next_step=best.get(
                "reason",
                "obtain live feature source and repeat live spread/funding/fill checks",
            ),
        )
    return ExplorationRow(
        lane="binance_derivatives_intraday_live_gate",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="low-cost intraday paper labels have not been checked against live book/funding context",
        next_step="run Binance intraday live execution gate for ARB candidates",
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
    best_context = _best_crypto_equity_proxy_context(context_path)
    signal = "not run yet"
    if best_context:
        relative_key = "vs_eth_5d" if best_context.get("group") == "eth_treasury_equity" else "vs_btc_5d"
        signal = (
            f"{best_context.get('symbol', '')}: "
            f"group={best_context.get('group', '')}, "
            f"{relative_key}={best_context.get(relative_key, '')}, "
            f"ret5d={best_context.get('return_5d', '')}"
        )
    return ExplorationRow(
        lane="crypto_equity_proxy",
        status="current_context",
        strongest_current_signal=signal,
        main_gap="crypto-linked equity proxies have not been converted into repeated labels",
        next_step="build paper tickets for proxy lead/lag, MSTR/BTC dislocation, and miner stress",
    )


def _crypto_equity_factor_split_row(root: Path) -> ExplorationRow:
    path = root / "crypto_equity_proxy" / "current_crypto_equity_factor_split.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="crypto_equity_factor_split",
            status=best.get("status", "crypto_equity_factor_split"),
            strongest_current_signal=(
                f"{best.get('factor_id', '')}: "
                f"role={best.get('factor_role', '')}, "
                f"target={best.get('target_asset', '')}, "
                f"side={best.get('side_hint', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "crypto-equity factor still needs hedge ratio, timestamp boundary, and residual labels",
            ),
            next_step=best.get(
                "next_probe",
                "split crypto-equity beta, residual, and market-hours factors before paper action",
            ),
        )
    return ExplorationRow(
        lane="crypto_equity_factor_split",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="crypto-equity proxy tickets are not split into beta, residual, and timing roles",
        next_step="run current crypto-equity factor split after proxy context",
    )


def _best_crypto_equity_proxy_context(path: Path) -> dict[str, str] | None:
    rows = _csv_rows(path)
    if not rows:
        return None
    return max(rows, key=_crypto_equity_proxy_context_score)


def _crypto_equity_proxy_context_score(row: dict[str, str]) -> float:
    if row.get("group") == "eth_treasury_equity":
        return abs(_safe_float(row.get("vs_eth_5d")))
    return abs(_safe_float(row.get("vs_btc_5d")))


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
    sequence_path = root / "event_flow" / "current_lob_sequence_state_probe.csv"
    sequence_rows = tuple(
        row
        for row in _csv_rows(sequence_path)
        if row.get("decision") not in {"reject_after_cost", "no_test_samples", "representation_only"}
    )
    if sequence_rows:
        best = max(sequence_rows, key=_lob_sequence_state_sort_key)
        return ExplorationRow(
            lane="event_flow",
            status=best.get("decision", "lob_sequence_state_probe"),
            strongest_current_signal=(
                f"{best.get('feature', '')} {best.get('bucket', '')} "
                f"{best.get('signal_action', '')}/{best.get('execution_mode', '')}, "
                f"net={best.get('test_net_bps', '')}bps, "
                f"hit={best.get('test_hit_rate', '')}"
            ),
            main_gap=(
                "LOB sequence-state probe is still a rolling-feature diagnostic; it needs queue/fill, "
                "adverse-selection, cancellation, and longer OOS evidence before execution"
            ),
            next_step=best.get(
                "next_step",
                "turn the strongest sequence state into a queue/fill-aware execution probe",
            ),
        )
    replay_path = root / "event_flow" / "current_lob_execution_world_replay.csv"
    replay_rows = tuple(
        row
        for row in _csv_rows(replay_path)
        if row.get("decision") not in {"hold_baseline", "worse_than_hold", "no_test_samples"}
    )
    if replay_rows:
        best = max(replay_rows, key=_lob_execution_replay_sort_key)
        return ExplorationRow(
            lane="event_flow",
            status=best.get("decision", "lob_execution_world_replay"),
            strongest_current_signal=(
                f"{best.get('feature', '')} {best.get('bucket', '')} "
                f"{best.get('signal_action', '')}/{best.get('execution_action', '')}, "
                f"net={best.get('net_reward_bps', '')}bps, "
                f"hit={best.get('hit_rate', '')}"
            ),
            main_gap=(
                "LOB replay is still a tiny diagnostic; maker/internalized actions need queue/fill, "
                "adverse-selection, cancellation, and longer OOS evidence"
            ),
            next_step=best.get(
                "next_step",
                "turn the strongest replay action into a queue/fill-aware execution probe",
            ),
        )
    cost_sweep_path = root / "event_flow" / "book_depth_execution_cost_sweep.csv"
    cost_sweep_rows = _csv_rows(cost_sweep_path)
    if cost_sweep_rows:
        best = max(cost_sweep_rows, key=lambda row: float(row.get("viability_score") or "-inf"))
        return ExplorationRow(
            lane="event_flow",
            status=best.get("viability_status", "book_depth_execution_cost_sweep"),
            strongest_current_signal=(
                f"{best.get('feature', '')} {best.get('bucket', '')} "
                f"{best.get('action', '')}/{best.get('execution_mode', '')}, "
                f"gross={best.get('test_gross_bps', '')}bps, "
                f"net={best.get('test_net_bps', '')}bps"
            ),
            main_gap=(
                "book-depth signal is too small for ordinary taker execution; maker/low-fee fill probability, "
                "queue position, adverse selection, and longer OOS windows are missing"
            ),
            next_step=best.get(
                "next_step",
                "test maker/low-fee execution and queue/adverse-selection controls",
            ),
        )
    walk_forward_path = root / "event_flow" / "book_depth_walk_forward_check.csv"
    walk_forward_rows = _csv_rows(walk_forward_path)
    if walk_forward_rows:
        best = max(walk_forward_rows, key=lambda row: float(row.get("test_net_bps") or "-inf"))
        return ExplorationRow(
            lane="event_flow",
            status=best.get("decision", "book_depth_walk_forward"),
            strongest_current_signal=(
                f"{best.get('feature', '')} {best.get('bucket', '')} "
                f"{best.get('action', '')}, "
                f"gross={best.get('test_gross_bps', '')}bps, "
                f"net={best.get('test_net_bps', '')}bps"
            ),
            main_gap="walk-forward is only recent-week and next-1m; liquidation timestamps, execution model, and longer OOS windows are missing",
            next_step="extend the walk-forward window and add liquidation/event timestamps before any paper action",
        )
    book_depth_path = root / "event_flow" / "book_depth_imbalance_screen.csv"
    book_depth_rows = _csv_rows(book_depth_path)
    if book_depth_rows:
        best = max(book_depth_rows, key=lambda row: float(row.get("mean_next_return") or "-inf"))
        return ExplorationRow(
            lane="event_flow",
            status="book_depth_context_probe",
            strongest_current_signal=(
                f"{best.get('feature', '')} {best.get('bucket', '')} "
                f"mean_next_return={best.get('mean_next_return', '')}, "
                f"hit_rate={best.get('hit_rate', '')}"
            ),
            main_gap="recent-week LOB/basis/positioning sample only; no costs, liquidation context, or purged walk-forward split",
            next_step="join liquidation timestamps and execution costs, then run a purged walk-forward check before any policy",
        )
    flow_path = root / "event_flow" / "flow_imbalance_screen.csv"
    top = _row_by_value(flow_path, field="bucket", value="top_20")
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


def _lob_sequence_state_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    decision_rank = {
        "market_sequence_candidate": 900,
        "low_fee_sequence_candidate": 800,
        "maker_sequence_candidate": 700,
        "market_sequence_tail_candidate": 600,
        "low_fee_sequence_tail_candidate": 500,
        "maker_sequence_tail_candidate": 400,
    }.get(row.get("decision", ""), 0)
    return (
        decision_rank,
        float(row.get("test_net_bps") or "-inf"),
        float(row.get("test_hit_rate") or "-inf"),
    )


def _lob_execution_replay_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    decision_rank = {
        "market_action_candidate": 800,
        "low_fee_action_candidate": 700,
        "maker_fill_model_needed": 600,
        "market_tail_candidate": 500,
        "low_fee_tail_candidate": 400,
        "maker_tail_or_queue_research": 300,
    }.get(row.get("decision", ""), 0)
    return (
        decision_rank,
        float(row.get("net_reward_bps") or "-inf"),
        float(row.get("hit_rate") or "-inf"),
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
            status="okx_monitor_actionability_review",
            strongest_current_signal=(
                f"{best_actionable.get('asset', '')}: {best_actionable.get('action', '')}, "
                f"score={best_actionable.get('actionability_score', '')}, "
                f"cont15={best_actionable.get('continuation_return_15m', '')}, "
                f"near_depth5={best_actionable.get('near_touch_depth_5bps', '')}, "
                f"note={best_actionable.get('note', '')}"
            ),
            main_gap="monitor actionability is historical monitor context, not the latest liquidation event; current-event intensity is tracked separately",
            next_step="refresh monitor samples or label the latest liquidation-intensity events before paper sizing",
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


def _liquidation_intensity_row(root: Path) -> ExplorationRow:
    gate_path = root / "liquidation_flow" / "current_okx_liquidation_intensity_paper_gate.csv"
    best_gate = _best_liquidation_intensity_paper_gate_row(gate_path)
    if best_gate:
        return ExplorationRow(
            lane="liquidation_intensity",
            status=best_gate.get("gate_action", "liquidation_intensity_paper_gate"),
            strongest_current_signal=(
                f"{best_gate.get('asset', '')}: {best_gate.get('trade_direction', '')} "
                f"{best_gate.get('action', '')}, label={best_gate.get('label_status', '')}, "
                f"size={best_gate.get('candidate_size_usd', '')}, "
                f"net_bps={best_gate.get('conservative_net_bps', '')}, "
                f"depth10={best_gate.get('depth_10bps_notional', '')}"
            ),
            main_gap="liquidation intensity paper gate still excludes real fills, funding PnL, stop behavior, repeat-event evidence, and adverse selection during bursts",
            next_step=best_gate.get("next_step", "paper-check the best liquidation intensity gate candidate"),
        )
    label_path = root / "liquidation_flow" / "current_okx_liquidation_intensity_forward_labels.csv"
    best_label = _best_liquidation_intensity_label_row(label_path)
    if best_label:
        return ExplorationRow(
            lane="liquidation_intensity",
            status=best_label.get("label_status", "liquidation_intensity_forward_label"),
            strongest_current_signal=(
                f"{best_label.get('asset', '')}: {best_label.get('action', '')}, "
                f"intensity={best_label.get('intensity_score', '')}, "
                f"cont15={best_label.get('continuation_return_15m', '')}, "
                f"rev15={best_label.get('reversal_return_15m', '')}, "
                f"cont1h={best_label.get('continuation_return_1h', '')}, "
                f"rev1h={best_label.get('reversal_return_1h', '')}"
            ),
            main_gap="liquidation intensity label still excludes spread, fees, funding PnL, fill probability, stop behavior, and repeat-event evidence",
            next_step=best_label.get("next_step", "gate the best liquidation intensity label with execution assumptions"),
        )
    path = root / "liquidation_flow" / "current_okx_liquidation_intensity.csv"
    best = _best_numeric_row(path, key="intensity_score")
    if best:
        return ExplorationRow(
            lane="liquidation_intensity",
            status=best.get("status", "liquidation_oi_context"),
            strongest_current_signal=(
                f"{best.get('asset', '')}: {best.get('action', '')}, "
                f"liq/OI={best.get('liquidation_to_open_interest', '')}, "
                f"liq={best.get('total_liquidation_notional', '')}, "
                f"OI={best.get('open_interest_usd', '')}, "
                f"imbalance={best.get('forced_buy_sell_imbalance', '')}"
            ),
            main_gap="liquidation intensity is not yet a direction; continuation versus reversal, depth, fees, funding, and adverse excursion are untested",
            next_step=best.get("next_step", "label top liquidation/OI event over 5m/15m/1h"),
        )
    return ExplorationRow(
        lane="liquidation_intensity",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="liquidation events are not normalized by open interest",
        next_step="run current OKX liquidation intensity after liquidation flow",
    )


def _defi_yield_row(root: Path) -> ExplorationRow:
    join_path = root / "defi_yield" / "current_yield_peg_risk_join.csv"
    best_join = _best_numeric_row(join_path, key="score")
    if best_join:
        return ExplorationRow(
            lane="defi_yield",
            status=best_join.get("status", "yield_peg_risk_join"),
            strongest_current_signal=(
                f"{best_join.get('chain', '')}/{best_join.get('project', '')} "
                f"{best_join.get('symbol', '')}: apy={best_join.get('apy', '')}, "
                f"base={best_join.get('apy_base', '')}, "
                f"peg={best_join.get('peg_symbol', '') or 'unmatched'}, "
                f"peg_deviation={best_join.get('peg_deviation', '')}"
            ),
            main_gap="yield candidates need peg, redemption, issuer, custody, APY-decay, and exit-liquidity separation",
            next_step=best_join.get(
                "next_step",
                "check whether peg or redemption risk explains the apparent yield edge",
            ),
        )
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


def _defi_lending_row(root: Path) -> ExplorationRow:
    actionability_path = root / "defi_lending" / "current_lending_stress_actionability.csv"
    actionability = _best_numeric_row(actionability_path, key="score")
    if actionability:
        return ExplorationRow(
            lane="defi_lending",
            status=actionability.get("status", "lending_stress_actionability"),
            strongest_current_signal=(
                f"{actionability.get('chain', '')} "
                f"{actionability.get('loan_asset', '')}/{actionability.get('collateral_asset', '')}: "
                f"util={actionability.get('utilization', '')}, "
                f"liquidity={actionability.get('liquidity_usd', '')}, "
                f"avg_supply={actionability.get('avg_net_supply_apy', '')}, "
                f"avg_borrow={actionability.get('avg_net_borrow_apy', '')}, "
                f"score={actionability.get('score', '')}"
            ),
            main_gap=actionability.get(
                "reason",
                "lending stress still needs capacity, exit liquidity, collateral, oracle, withdrawal, gas, and smart-contract checks",
            ),
            next_step=actionability.get("next_step", "run lending stress actionability check"),
        )
    path = root / "defi_lending" / "current_morpho_lending_rates.csv"
    rows = tuple(row for row in _csv_rows(path) if row.get("status") != "lending_context_watch")
    best = max(rows, key=lambda row: float(row.get("score") or "-inf")) if rows else None
    if best:
        return ExplorationRow(
            lane="defi_lending",
            status=best.get("status", "morpho_lending_rates"),
            strongest_current_signal=(
                f"{best.get('chain', '')} {best.get('loan_asset', '')}/{best.get('collateral_asset', '')}: "
                f"util={best.get('utilization', '')}, "
                f"liquidity={best.get('liquidity_usd', '')}, "
                f"avg_supply={best.get('avg_net_supply_apy', '')}, "
                f"avg_borrow={best.get('avg_net_borrow_apy', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap="lending pressure still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract checks",
            next_step=best.get(
                "next_step",
                "check Morpho rate persistence and collateral liquidation risk",
            ),
        )
    return ExplorationRow(
        lane="defi_lending",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="borrow and lending rates are not connected",
        next_step="run Morpho lending-rate screen",
    )


def _market_making_row(root: Path) -> ExplorationRow:
    microstructure_gate_path = root / "market_making" / "current_microstructure_flow_paper_gate.csv"
    best_microstructure_gate = _best_microstructure_flow_paper_gate_row(microstructure_gate_path)
    if best_microstructure_gate:
        return ExplorationRow(
            lane="market_making",
            status=best_microstructure_gate.get("gate_action", "microstructure_small_paper_probe"),
            strongest_current_signal=(
                f"{best_microstructure_gate.get('asset', '')}: "
                f"action={best_microstructure_gate.get('action', '')}, "
                f"size={best_microstructure_gate.get('candidate_size_usd', '')}, "
                f"net15={best_microstructure_gate.get('conservative_net_15m_bps', '')}bps, "
                f"net1h={best_microstructure_gate.get('conservative_net_1h_bps', '')}bps, "
                f"depth_usage={best_microstructure_gate.get('visible_depth_usage', '')}"
            ),
            main_gap="microstructure paper gate still excludes real fill, queue position, funding, maker/taker choice, and repeated adverse-selection samples",
            next_step=best_microstructure_gate.get(
                "next_step",
                "paper-check top microstructure flow with fill and adverse-selection logs",
            ),
        )
    microstructure_label_path = root / "market_making" / "current_microstructure_flow_forward_labels.csv"
    best_microstructure_label = _best_microstructure_flow_label_row(microstructure_label_path)
    if best_microstructure_label:
        return ExplorationRow(
            lane="market_making",
            status="microstructure_15m_1h_supported",
            strongest_current_signal=(
                f"{best_microstructure_label.get('asset', '')}: "
                f"action={best_microstructure_label.get('action', '')}, "
                f"pressure={best_microstructure_label.get('pressure_score', '')}, "
                f"dir15={best_microstructure_label.get('directional_return_15m', '')}, "
                f"dir1h={best_microstructure_label.get('directional_return_1h', '')}"
            ),
            main_gap="microstructure label is price-only and still lacks spread, taker/maker fee, queue position, fill probability, and repeated adverse-selection checks",
            next_step=(
                f"gate {best_microstructure_label.get('asset', '')} microstructure flow with fees, "
                "spread, queue, fill probability, and repeat snapshots"
            ),
        )
    microstructure_path = root / "market_making" / "current_microstructure_flow_snapshot.csv"
    best_microstructure = _best_microstructure_flow_row(microstructure_path)
    if best_microstructure:
        return ExplorationRow(
            lane="market_making",
            status=best_microstructure.get("action", "microstructure_flow_watch"),
            strongest_current_signal=(
                f"{best_microstructure.get('asset', '')}: "
                f"dir={best_microstructure.get('direction', '')}, "
                f"pressure={best_microstructure.get('pressure_score', '')}, "
                f"book={best_microstructure.get('book_imbalance_10bps', '')}, "
                f"trade={best_microstructure.get('trade_imbalance', '')}, "
                f"spread={best_microstructure.get('spread_bps', '')}bps"
            ),
            main_gap="microstructure snapshot still needs forward labels, maker/taker fee modeling, queue position, and adverse-selection checks",
            next_step=(
                f"label {best_microstructure.get('asset', '')} microstructure flow over 15m/1h "
                "and compare aligned pressure against book/trade divergence"
            ),
        )
    paper_gate_path = root / "market_making" / "current_l2_imbalance_paper_gate.csv"
    best_gate = _best_l2_imbalance_paper_gate_row(paper_gate_path)
    if best_gate:
        net_1h_note = ""
        if best_gate.get("net_1h_bps", ""):
            net_1h_note = f"net1h={best_gate.get('net_1h_bps', '')}bps, "
        return ExplorationRow(
            lane="market_making",
            status=_l2_imbalance_gate_status(best_gate),
            strongest_current_signal=(
                f"{best_gate.get('asset', '')}: "
                f"size={best_gate.get('candidate_size_usd', '')}, "
                f"net15={best_gate.get('net_15m_bps', '')}bps, "
                f"{net_1h_note}"
                f"depth_usage={best_gate.get('visible_depth_usage', '')}"
            ),
            main_gap="paper gate is directional and still excludes maker queue, fill probability, rebates, and repeated adverse-selection samples",
            next_step=f"repeat {best_gate.get('asset', '')} L2 imbalance on fresh snapshots and then design a maker-fill observation log",
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
    hedge_path = root / "options_volatility" / "current_volatility_hedge_candidates.csv"
    hedge = _best_numeric_row(hedge_path, key="score")
    if hedge:
        return ExplorationRow(
            lane="options_volatility",
            status=hedge.get("decision", "volatility_hedge_candidate"),
            strongest_current_signal=(
                f"{hedge.get('currency', '')} {hedge.get('expiry', '')}: "
                f"{hedge.get('hedge_profile', '')}, "
                f"score={hedge.get('score', '')}, "
                f"max_loss_pct={hedge.get('max_loss_pct', '')}, "
                f"premium_to_rv={hedge.get('premium_to_realized_move', '')}, "
                f"depth={hedge.get('top_ask_premium_depth_usd', '')}"
            ),
            main_gap=hedge.get(
                "reason",
                "volatility hedge candidate still needs sweep, hedge PnL, margin, and exit checks",
            ),
            next_step=hedge.get(
                "next_step",
                "paper-check option sweep depth, delta hedge marks, exit bid, margin, and stop",
            ),
        )
    actionability_path = root / "options_volatility" / "current_volatility_actionability.csv"
    actionability = _best_numeric_row(actionability_path, key="score")
    if actionability:
        return ExplorationRow(
            lane="options_volatility",
            status=actionability.get("status", "volatility_actionability"),
            strongest_current_signal=(
                f"{actionability.get('currency', '')} {actionability.get('expiry', '')}: "
                f"{actionability.get('structure', '')}, "
                f"prem24={actionability.get('iv_premium_24h', '')}, "
                f"spread={actionability.get('quote_spread_pct', '')}, "
                f"max_loss_pct={actionability.get('max_loss_pct', '')}, "
                f"prem_to_rv_move={actionability.get('premium_to_realized_move', '')}, "
                f"top_depth_usd={actionability.get('top_ask_premium_depth_usd', '')}"
            ),
            main_gap=actionability.get(
                "reason",
                "options candidate still needs multi-level sweep, hedge, margin, and exit checks",
            ),
            next_step=actionability.get(
                "next_step",
                "paper-check option sweep depth, delta hedge plan, max loss, margin, and exit bid",
            ),
        )
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
                f"max_loss_pct={best_ticket.get('max_loss_pct', '')}, "
                f"prem_to_rv_move={best_ticket.get('premium_to_realized_move', '')}, "
                f"top_depth_usd={best_ticket.get('top_ask_premium_depth_usd', '')}"
            ),
            main_gap="options paper ticket uses only top-of-book ATM straddle depth; it still lacks multi-level sweep, delta hedge PnL, margin, and realized-vol forecast",
            next_step="paper-check ATM straddle multi-level depth, max premium loss, delta hedge cost, margin, and expiry handling before any live action",
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
            "paper_long_vol_quote_candidate",
            "paper_calendar_spread_watch",
        }
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            {
                "paper_long_vol_quote_candidate": 1.0,
                "paper_short_put_spread_candidate": 0.9,
                "paper_calendar_spread_watch": 0.5,
            }.get(row.get("status"), 0.0),
            float(row.get("score") or "0"),
        ),
    )


def _sector_rotation_row(root: Path) -> ExplorationRow:
    context_path = root / "sector_rotation" / "current_category_perp_context.csv"
    best_context = _best_category_perp_context_row(context_path)
    if best_context:
        action = best_context.get("action", "")
        return ExplorationRow(
            lane="sector_rotation",
            status="category_perp_context",
            strongest_current_signal=(
                f"{best_context.get('category_name', '')}/{best_context.get('symbol', '')}: "
                f"action={best_context.get('action', '')}, "
                f"funding_support={best_context.get('best_funding_support', '')}, "
                f"score={best_context.get('context_score', '')}"
            ),
            main_gap=_sector_rotation_context_gap(action),
            next_step=_sector_rotation_context_next_step(best_context),
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


def _sector_rotation_context_gap(action: str) -> str:
    if action == "sector_perp_repeat_candidate":
        return "category-perp context has one supportive label; repeat labels, costs, and narrative conflicts are still missing"
    if action == "wait_for_label":
        return "category-perp context is tradable but still waiting for forward labels and execution checks"
    return "category-perp context is weak or mixed and needs repeat evidence before promotion"


def _sector_rotation_context_next_step(row: dict[str, str]) -> str:
    symbol = row.get("symbol", "")
    category = row.get("category_name", "")
    return (
        f"paper-label {category}/{symbol} over 15m/1h/4h with funding, "
        "spread/depth, and narrative-conflict checks"
    )


def _news_social_row(root: Path) -> ExplorationRow:
    survival_path = root / "news_social" / "current_news_event_survival.csv"
    best_survival = _best_numeric_row(survival_path, key="survival_score")
    if best_survival:
        return ExplorationRow(
            lane="news_social",
            status=best_survival.get("survival_status", "news_event_survival"),
            strongest_current_signal=(
                f"{best_survival.get('symbol', '')}: {best_survival.get('event_kind', '')}, "
                f"side={best_survival.get('side', '')}, "
                f"sources={best_survival.get('source_count', '')}, "
                f"labels={best_survival.get('label_count', '')}, "
                f"stories={best_survival.get('unique_story_count', '')}, "
                f"mean1h={best_survival.get('mean_directional_1h_bps', '')}, "
                f"mean4h={best_survival.get('mean_directional_4h_bps', '')}"
            ),
            main_gap=best_survival.get("reason", "news-event survival still needs source independence"),
            next_step=best_survival.get(
                "next_step",
                "repeat news-event labels with story-level dedupe and execution-cost checks",
            ),
        )
    source_independence_path = root / "news_social" / "current_news_event_source_independence.csv"
    source_independence_rows = tuple(
        row
        for row in _csv_rows(source_independence_path)
        if row.get("independence_status")
        in {
            "independent_multi_source_story",
            "same_story_multi_source_repeat",
            "pending_archive_before_independence",
            "single_source_supported_story",
            "weak_forward_story",
        }
    )
    best_independence = (
        max(source_independence_rows, key=lambda row: float(row.get("score") or "-inf"))
        if source_independence_rows
        else None
    )
    if best_independence:
        return ExplorationRow(
            lane="news_social",
            status=best_independence.get("independence_status", "news_event_source_independence"),
            strongest_current_signal=(
                f"{best_independence.get('symbol', '')}: {best_independence.get('event_kind', '')}, "
                f"side={best_independence.get('side', '')}, "
                f"sources={best_independence.get('source_count', '')}, "
                f"stories={best_independence.get('unique_story_count', '')}, "
                f"dominant={best_independence.get('dominant_story_terms', '')}, "
                f"mean1h={best_independence.get('mean_directional_1h_bps', '')}, "
                f"mean4h={best_independence.get('mean_directional_4h_bps', '')}"
            ),
            main_gap=best_independence.get(
                "reason",
                "news-event source-independence gate still needs execution costs and beta controls",
            ),
            next_step=best_independence.get(
                "next_step",
                "repeat news-event labels with story-level dedupe and execution-cost checks",
            ),
        )
    quality_gate_path = root / "news_social" / "current_news_event_quality_gate.csv"
    quality_rows = tuple(
        row
        for row in _csv_rows(quality_gate_path)
        if row.get("decision")
        in {
            "repeat_supported_multi_source_label",
            "repeat_after_pending_archive",
            "repeat_single_source_label",
            "watch_1h_only_news_label",
        }
    )
    best_quality = max(quality_rows, key=lambda row: float(row.get("score") or "-inf")) if quality_rows else None
    if best_quality:
        return ExplorationRow(
            lane="news_social",
            status=best_quality.get("decision", "news_event_quality_gate"),
            strongest_current_signal=(
                f"{best_quality.get('symbol', '')}: {best_quality.get('event_kind', '')}, "
                f"side={best_quality.get('side', '')}, "
                f"sources={best_quality.get('source_count', '')}, "
                f"support/reject={best_quality.get('supported_count', '')}/{best_quality.get('rejected_count', '')}, "
                f"mean1h={best_quality.get('mean_directional_1h_bps', '')}, "
                f"mean4h={best_quality.get('mean_directional_4h_bps', '')}"
            ),
            main_gap=best_quality.get(
                "reason",
                "news-event quality gate still needs manual duplicate-source review and execution costs",
            ),
            next_step=best_quality.get(
                "next_step",
                "repeat news-event quality gate with duplicate-source and execution-cost checks",
            ),
        )
    forward_label_path = root / "news_social" / "current_news_event_forward_labels.csv"
    forward_labels = tuple(
        row
        for row in _csv_rows(forward_label_path)
        if row.get("label_status") in {"direction_supported_1h_4h", "direction_supported_1h_only"}
    )
    best_forward = _best_news_event_forward_label(forward_labels)
    if best_forward:
        return ExplorationRow(
            lane="news_social",
            status=best_forward.get("label_status", "news_event_forward_label"),
            strongest_current_signal=(
                f"{best_forward.get('symbol', '')}: {best_forward.get('event_kind', '')}, "
                f"side={best_forward.get('side', '')}, "
                f"dir1h={best_forward.get('directional_1h_bps', '')}, "
                f"dir4h={best_forward.get('directional_4h_bps', '')}"
            ),
            main_gap="news-event label is timestamped but still needs duplicate-source checks, costs, and repeated OOS events",
            next_step=best_forward.get(
                "next_step",
                "repeat news-event forward labels with duplicate-source and execution-cost checks",
            ),
        )
    news_path = root / "news_social" / "current_news_event_screen.csv"
    news_rows = tuple(row for row in _csv_rows(news_path) if row.get("status") != "paper_news_context_watch")
    best_news = max(news_rows, key=lambda row: float(row.get("score") or "-inf")) if news_rows else None
    if best_news:
        return ExplorationRow(
            lane="news_social",
            status=best_news.get("status", "news_event_screen"),
            strongest_current_signal=(
                f"{best_news.get('symbol', '')}: {best_news.get('event_kind', '')}, "
                f"source={best_news.get('source', '')}, "
                f"age_h={best_news.get('age_hours', '')}, "
                f"funding={best_news.get('annualized_funding', '')}, "
                f"score={best_news.get('score', '')}"
            ),
            main_gap="news headlines are not leakage-safe labels and can be duplicated, stale, or already priced",
            next_step=best_news.get(
                "next_step",
                "label news-event reactions against forward returns and execution costs",
            ),
        )
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


def _best_news_event_forward_label(rows: tuple[dict[str, str], ...]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_4h_bps") or "-inf"),
            float(row.get("directional_1h_bps") or "-inf"),
        ),
    )


def _market_breadth_row(root: Path) -> ExplorationRow:
    path = root / "market_breadth" / "current_volume_price_dislocation.csv"
    execution = _best_market_breadth_execution_row(
        root / "market_breadth" / "current_volume_price_dislocation_execution_gate.csv"
    )
    if execution:
        return ExplorationRow(
            lane="market_breadth",
            status=execution.get("action", "volume_dislocation_execution_gate"),
            strongest_current_signal=(
                f"{execution.get('symbol', '')}/{execution.get('name', '')}: "
                f"{execution.get('side', '')}, "
                f"dir4h={execution.get('directional_return_4h', '')}, "
                f"net4h_bps={execution.get('conservative_net_4h_bps', '')}, "
                f"spread={execution.get('spread_bps', '')}, "
                f"depth_usage_250={execution.get('visible_depth_usage_250', '')}"
            ),
            main_gap="execution gate is public-book only; it still needs repeated paper fills, stop behavior, realized fees, and fresh labels",
            next_step=execution.get("next_step", "paper-probe market-breadth execution-gated candidates"),
        )
    label = _best_market_breadth_label_row(root / "market_breadth" / "current_volume_price_dislocation_labels.csv")
    if label:
        return ExplorationRow(
            lane="market_breadth",
            status=_market_breadth_label_status(label),
            strongest_current_signal=(
                f"{label.get('symbol', '')}/{label.get('name', '')}: "
                f"{label.get('side', '')}, "
                f"dir1h={label.get('directional_return_1h', '')}, "
                f"dir4h={label.get('directional_return_4h', '')}, "
                f"source={label.get('price_source', '')}, "
                f"score={label.get('score', '')}"
            ),
            main_gap="volume-price dislocation labels still need repeat windows, venue depth, fees, funding, stops, and false-breakout separation",
            next_step=f"repeat-label {label.get('symbol', '')} market-breadth dislocation with execution costs",
        )
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="market_breadth",
            status=best.get("status", "volume_price_dislocation"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}/{best.get('name', '')}: "
                f"{best.get('side', '')}, "
                f"vol_mcap={best.get('volume_to_market_cap', '')}, "
                f"price24h={best.get('price_change_24h', '')}, "
                f"price7d={best.get('price_change_7d', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap="volume-price dislocation is not yet labeled against forward returns, venue depth, fees, and false-breakout risk",
            next_step=best.get(
                "next_step",
                "label broad volume-price dislocation candidates over short horizons",
            ),
        )
    return ExplorationRow(
        lane="market_breadth",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="broad volume-price dislocations are not screened",
        next_step="run market breadth volume-price dislocation screen",
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


def _event_pressure_cluster_row(root: Path) -> ExplorationRow:
    path = root / "news_social" / "current_event_pressure_cluster.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="event_pressure_cluster",
            status=best.get("status", ""),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: {best.get('side', '')}, "
                f"sources={best.get('source_count', '')}, "
                f"events={best.get('event_count', '')}, "
                f"score={best.get('score', '')}"
            ),
            main_gap=(
                "event clusters can be stale, duplicated, already priced, or non-causal; "
                "they need leakage-safe forward labels and execution checks"
            ),
            next_step=best.get("next_step", "label top event-pressure cluster over short horizons"),
        )
    return ExplorationRow(
        lane="event_pressure_cluster",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="news, exchange catalysts, and attention are not grouped by symbol",
        next_step="run current event pressure cluster",
    )


def _ticker_attention_source_split_row(root: Path) -> ExplorationRow:
    path = root / "news_social" / "current_ticker_attention_source_split.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="ticker_attention_source_split",
            status=best.get("decision", "ticker_attention_source_split"),
            strongest_current_signal=(
                f"{best.get('symbol', '')}: "
                f"{best.get('source_specificity', '')}, "
                f"source={best.get('source', '')}, "
                f"priority={best.get('priority', '')}, "
                f"context={best.get('joined_context', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "ticker attention still needs source identity, dedupe controls, and forward labels",
            ),
            next_step=best.get(
                "next_probe",
                "split ticker-specific attention from broad sentiment before paper labels",
            ),
        )
    return ExplorationRow(
        lane="ticker_attention_source_split",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="ticker-specific attention is still mixed with broad sentiment and duplicated event clusters",
        next_step="run current ticker attention source split after event pressure source independence",
    )


def _token_unlocks_row(root: Path) -> ExplorationRow:
    actionability_path = root / "token_unlocks" / "current_token_unlock_actionability.csv"
    best_actionability = _best_token_unlock_actionability_row(actionability_path)
    if best_actionability:
        return ExplorationRow(
            lane="token_unlocks",
            status=best_actionability.get("status", "unlock_event_label_pending"),
            strongest_current_signal=(
                f"{best_actionability.get('symbol', '')}: side={best_actionability.get('side', '')}, "
                f"score={best_actionability.get('score', '')}, "
                f"ticket={best_actionability.get('ticket_status', '')}, "
                f"value={best_actionability.get('unlock_value_usd', '')}, "
                f"supply={best_actionability.get('percent_supply', '')}, "
                f"funding={best_actionability.get('annualized_funding', '')}, "
                f"impact={best_actionability.get('impact_spread', '')}"
            ),
            main_gap=best_actionability.get(
                "reason",
                "unlock event has no event-window label and can be already priced or crowded",
            ),
            next_step=best_actionability.get(
                "next_step",
                "label top unlock event window before promotion",
            ),
        )
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


def _best_token_unlock_actionability_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status")
            in {
                "unlock_event_label_pending",
                "unlock_event_crowded_squeeze_watch",
                "unlock_event_execution_blocked",
            }
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


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
    actionability_path = root / "prediction_markets" / "current_event_probability_actionability.csv"
    actionability = _best_prediction_market_probability_actionability(actionability_path)
    if actionability:
        return ExplorationRow(
            lane="prediction_markets",
            status=actionability.get("status", "event_probability_actionability"),
            strongest_current_signal=(
                f"{actionability.get('suggested_side', '')}: {actionability.get('question', '')}, "
                f"bid={actionability.get('current_bid', '')}, "
                f"ask={actionability.get('current_ask', '')}, "
                f"spread={actionability.get('spread', '')}, "
                f"edge_after_ask={actionability.get('current_edge_after_ask', '')}, "
                f"source_quality={actionability.get('source_quality_status', '')}, "
                f"refresh={actionability.get('refresh_status', '')}"
            ),
            main_gap=actionability.get(
                "reason",
                "event-probability candidate still lacks fill, fee, queue, resolution-risk, and adverse-selection proof",
            ),
            next_step=actionability.get(
                "next_step",
                "paper-check event probability candidate under explicit execution and resolution assumptions",
            ),
        )
    refresh_path = root / "prediction_markets" / "current_event_probability_paper_outcome_refresh.csv"
    best_refresh = _best_prediction_market_probability_refresh(refresh_path)
    if best_refresh:
        return ExplorationRow(
            lane="prediction_markets",
            status=best_refresh.get("status", "paper_outcome_survived_refresh"),
            strongest_current_signal=(
                f"{best_refresh.get('suggested_side', '')}: {best_refresh.get('question', '')}, "
                f"entry_ask={best_refresh.get('previous_entry_ask', '')}, "
                f"bid={best_refresh.get('current_bid', '')}, "
                f"ask={best_refresh.get('current_ask', '')}, "
                f"bid_pnl={best_refresh.get('mark_to_bid_pnl', '')}, "
                f"edge_now={best_refresh.get('current_edge_after_ask', '')}, "
                f"edge_change={best_refresh.get('edge_change', '')}"
            ),
            main_gap="survived refresh still lacks actual fill, queue position, fees, resolution-risk handling, and adverse-selection proof",
            next_step="repeat the refresh and only promote if edge survives another quote/news update with executable depth",
        )
    paper_outcome_path = root / "prediction_markets" / "current_event_probability_paper_outcome.csv"
    best_paper_outcome = _best_prediction_market_probability_paper_outcome(paper_outcome_path)
    if best_paper_outcome:
        return ExplorationRow(
            lane="prediction_markets",
            status=best_paper_outcome.get("status", "paper_outcome_active_watch"),
            strongest_current_signal=(
                f"{best_paper_outcome.get('suggested_side', '')}: {best_paper_outcome.get('question', '')}, "
                f"entry_ask={best_paper_outcome.get('entry_ask', '')}, "
                f"bid={best_paper_outcome.get('current_bid', '')}, "
                f"ask={best_paper_outcome.get('current_ask', '')}, "
                f"bid_pnl={best_paper_outcome.get('mark_to_bid_pnl', '')}, "
                f"edge_after_ask={best_paper_outcome.get('current_edge_after_ask', '')}"
            ),
            main_gap="paper outcome still lacks real fill, queue position, fees, refreshed news verification, and adverse-selection proof",
            next_step="refresh market/news snapshots and require the edge to survive quote movement before any live action",
        )
    paper_ticket_path = root / "prediction_markets" / "current_event_probability_paper_tickets.csv"
    best_paper_ticket = _best_prediction_market_probability_paper_ticket(paper_ticket_path)
    if best_paper_ticket:
        source_quality = _best_prediction_market_source_quality(
            root / "prediction_markets" / "current_event_source_quality.csv",
            best_paper_ticket.get("market_id", ""),
        )
        quality_suffix = ""
        if source_quality:
            quality_suffix = (
                f", source_quality={source_quality.get('status', '')}, "
                f"sources={source_quality.get('source_count_72h', '')}, "
                f"recent_articles={source_quality.get('article_count_24h', '')}"
            )
        return ExplorationRow(
            lane="prediction_markets",
            status=best_paper_ticket.get("status", "paper_event_probability_ticket"),
            strongest_current_signal=(
                f"{best_paper_ticket.get('suggested_side', '')}: {best_paper_ticket.get('question', '')}, "
                f"ask={best_paper_ticket.get('entry_ask', '')}, "
                f"edge_after_ask={best_paper_ticket.get('edge_after_ask', '')}, "
                f"ask_depth_5c={best_paper_ticket.get('ask_depth_to_5c', '')}"
                f"{quality_suffix}"
            ),
            main_gap="paper ticket still uses a rough headline probability and has no fill, fee, queue, or adverse-selection proof",
            next_step="paper-check source freshness, duplicate headlines, queue/fill assumptions, and outcome movement before any live action",
        )
    gap_path = root / "prediction_markets" / "current_event_probability_gap.csv"
    best_gap = _best_prediction_market_probability_gap(gap_path)
    if best_gap:
        return ExplorationRow(
            lane="prediction_markets",
            status=best_gap.get("status", "paper_probability_gap_candidate"),
            strongest_current_signal=(
                f"{best_gap.get('suggested_side', '')}: {best_gap.get('question', '')}, "
                f"market_yes={best_gap.get('market_yes_probability', '')}, "
                f"estimated_yes={best_gap.get('estimated_yes_probability', '')}, "
                f"gap={best_gap.get('probability_gap', '')}, "
                f"confidence={best_gap.get('confidence_score', '')}"
            ),
            main_gap="probability gap is headline-derived and uncalibrated; source timing, duplication, costs, and adverse selection are unresolved",
            next_step="paper-check the probability gap with source-level verification and stale-news filtering before any live action",
        )
    ticket_path = root / "prediction_markets" / "current_prediction_market_paper_tickets.csv"
    best_ticket = _best_prediction_market_paper_ticket(ticket_path)
    if best_ticket:
        news = _best_prediction_market_news_pressure(
            root / "prediction_markets" / "current_event_news_pressure.csv",
            best_ticket.get("market_id", ""),
        )
        news_suffix = ""
        if news:
            news_suffix = (
                f", news={news.get('status', '')}, "
                f"articles24={news.get('article_count_24h', '')}, "
                f"sources={news.get('source_count_72h', '')}"
            )
        return ExplorationRow(
            lane="prediction_markets",
            status=best_ticket.get("status", "paper_event_model_candidate"),
            strongest_current_signal=(
                f"{best_ticket.get('question', '')} {best_ticket.get('outcome', '')}: "
                f"{best_ticket.get('structure', '')}, "
                f"spread={best_ticket.get('spread', '')}, "
                f"depth={best_ticket.get('visible_depth_score', '')}, "
                f"vol24={best_ticket.get('volume_24h', '')}"
                f"{news_suffix}"
            ),
            main_gap="prediction-market paper ticket still lacks true-probability model, news feed, latency, and adverse-selection checks",
            next_step="compare external news-flow evidence against market-implied odds before any live action",
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


def _prediction_market_crypto_hedge_row(root: Path) -> ExplorationRow:
    alignment_path = root / "prediction_markets" / "current_event_crypto_hedge_event_alignment.csv"
    alignment = _best_event_crypto_hedge_event_alignment(alignment_path)
    if alignment:
        return ExplorationRow(
            lane="prediction_market_crypto_hedge",
            status=alignment.get("alignment_status", "event_crypto_hedge_event_alignment"),
            strongest_current_signal=(
                f"{alignment.get('asset', '')} {alignment.get('hedge_action', '')}: "
                f"{alignment.get('event_bias', '')}, "
                f"asset_bps={alignment.get('asset_directional_return_bps', '')}, "
                f"event_bps={alignment.get('event_mark_return_bps', '')}, "
                f"control_count={alignment.get('same_asset_control_count', '')}, "
                f"control_gap={alignment.get('same_asset_control_gap_bps', '')}, "
                f"market={alignment.get('market_id', '')}"
            ),
            main_gap=(
                "event-market probability did not move with the crypto hedge; this is currently a crypto beta "
                "or same-asset control move, not proven event alpha"
            ),
            next_step=alignment.get(
                "next_step",
                "require event-market movement and same-asset controls before promotion",
            ),
        )
    attribution_path = root / "prediction_markets" / "current_event_crypto_hedge_beta_attribution.csv"
    attribution = _best_event_crypto_hedge_beta_attribution(attribution_path)
    if attribution:
        return ExplorationRow(
            lane="prediction_market_crypto_hedge",
            status=attribution.get("attribution_status", "event_crypto_hedge_beta_attribution"),
            strongest_current_signal=(
                f"{attribution.get('asset', '')} {attribution.get('hedge_action', '')}: "
                f"{attribution.get('event_bias', '')}, "
                f"asset_bps={attribution.get('asset_directional_return_bps', '')}, "
                f"basket_bps={attribution.get('basket_directional_return_bps', '')}, "
                f"residual_bps={attribution.get('residual_vs_basket_bps', '')}, "
                f"market={attribution.get('market_id', '')}"
            ),
            main_gap=(
                "event-crypto hedge reaction is currently explained as common crypto beta; it still needs "
                "funding, spread/depth, event timestamp, and repeated market controls before promotion"
            ),
            next_step=attribution.get(
                "next_step",
                "repeat event crypto hedge labels with explicit beta and cost controls",
            ),
        )
    reaction_path = root / "prediction_markets" / "current_event_crypto_hedge_reaction_labels.csv"
    reaction = _best_event_crypto_hedge_reaction_label(reaction_path)
    if reaction:
        return ExplorationRow(
            lane="prediction_market_crypto_hedge",
            status=reaction.get("reaction_status", "event_crypto_hedge_reaction_labels"),
            strongest_current_signal=(
                f"{reaction.get('asset', '')} {reaction.get('hedge_action', '')}: "
                f"{reaction.get('event_bias', '')}, "
                f"dir_bps={reaction.get('directional_return_bps', '')}, "
                f"elapsed_min={reaction.get('elapsed_minutes', '')}, "
                f"gap={reaction.get('probability_gap', '')}, "
                f"edge={reaction.get('current_edge_after_ask', '')}, "
                f"market={reaction.get('market_id', '')}"
            ),
            main_gap=(
                "event-crypto hedge reaction label still needs funding, spread/depth, beta attribution, "
                "and event-timestamp controls before promotion"
            ),
            next_step=reaction.get(
                "next_step",
                "repeat event crypto hedge labels with explicit costs and failure regimes",
            ),
        )
    path = root / "prediction_markets" / "current_event_crypto_hedge_candidates.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="prediction_market_crypto_hedge",
            status=best.get("status", "event_crypto_hedge_candidates"),
            strongest_current_signal=(
                f"{best.get('asset', '')} {best.get('hedge_action', '')}: "
                f"{best.get('event_bias', '')}, "
                f"score={best.get('score', '')}, "
                f"gap={best.get('probability_gap', '')}, "
                f"edge_after_ask={best.get('current_edge_after_ask', '')}, "
                f"depth={best.get('ask_depth_to_5c', '')}, "
                f"market={best.get('market_id', '')}"
            ),
            main_gap=(
                "event probability is only event-state evidence; hedge still needs timestamp alignment, "
                "funding, spread/depth, beta attribution, and failure-regime labels"
            ),
            next_step=best.get(
                "next_step",
                "paper-label the crypto hedge around the event market before promotion",
            ),
        )
    return ExplorationRow(
        lane="prediction_market_crypto_hedge",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="prediction-market event states have not been mapped into crypto hedge candidates",
        next_step="run current_event_crypto_hedge_candidates after probability refresh/actionability",
    )


def _event_crypto_hedge_survival_row(root: Path) -> ExplorationRow:
    path = root / "prediction_markets" / "current_event_crypto_hedge_survival.csv"
    best = _best_event_crypto_hedge_survival(path)
    if best:
        return ExplorationRow(
            lane="event_crypto_hedge_survival",
            status=best.get("survival_status", "event_crypto_hedge_survival"),
            strongest_current_signal=(
                f"{best.get('asset', '')} {best.get('hedge_action', '')}: "
                f"{best.get('event_bias', '')}, "
                f"score={best.get('survival_score', '')}, "
                f"asset_bps={best.get('asset_directional_return_bps', '')}, "
                f"event_bps={best.get('event_mark_return_bps', '')}, "
                f"residual={best.get('residual_vs_basket_bps', '')}, "
                f"market={best.get('market_id', '')}"
            ),
            main_gap=best.get("reason", "event crypto hedge still needs survival checks"),
            next_step=best.get("next_step", "rerun event crypto hedge survival after fresh marks"),
        )
    return ExplorationRow(
        lane="event_crypto_hedge_survival",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="event crypto hedges have not been checked against reaction, beta, controls, and event alignment",
        next_step="run current_event_crypto_hedge_survival after event alignment",
    )


def _best_event_crypto_hedge_survival(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    status_rank = {
        "event_crypto_hedge_survived_alignment": 5.0,
        "event_crypto_hedge_residual_watch": 4.0,
        "event_crypto_hedge_rejected_event_flat": 3.0,
        "event_crypto_hedge_rejected_same_asset_control": 3.0,
        "event_crypto_hedge_rejected_event_contradiction": 3.0,
        "event_crypto_hedge_pending_mark": 2.0,
        "event_crypto_hedge_candidate_unproven": 1.0,
    }
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("survival_status", ""), 0.0),
            abs(_safe_float(row.get("survival_score"))),
        ),
    )


def _best_event_crypto_hedge_event_alignment(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    status_rank = {
        "event_probability_and_crypto_aligned": 4.0,
        "same_asset_control_explains_return": 3.0,
        "event_alignment_inconclusive": 2.0,
        "event_probability_flat_crypto_moved": 1.0,
        "event_probability_contradicts_crypto": 0.0,
    }
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("alignment_status", ""), 0.0),
            _safe_float(row.get("asset_directional_return_bps")),
        ),
    )


def _best_event_crypto_hedge_beta_attribution(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    status_rank = {
        "event_crypto_residual_outperformance": 4.0,
        "event_crypto_beta_move_supported": 3.0,
        "event_crypto_beta_attribution_pending": 2.0,
        "event_crypto_residual_contradiction": 1.0,
        "event_crypto_beta_attribution_negative": 0.0,
    }
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("attribution_status", ""), 0.0),
            _safe_float(row.get("asset_directional_return_bps")),
        ),
    )


def _best_event_crypto_hedge_reaction_label(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    status_rank = {
        "event_crypto_hedge_reaction_win": 4.0,
        "event_crypto_hedge_reaction_pending": 3.0,
        "event_crypto_hedge_reaction_flat": 2.0,
        "event_crypto_hedge_reaction_loss": 1.0,
        "event_crypto_hedge_reaction_missing_mark": 0.0,
    }
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("reaction_status", ""), 0.0),
            _safe_float(row.get("directional_return_bps")),
        ),
    )


def _best_seed_wallet_flow_actionability(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    status_rank = {
        "wallet_position_follow_candidate": 4.0,
        "wallet_recent_flow_candidate": 3.0,
        "wallet_flow_watch": 2.0,
        "wallet_flow_deprioritize": 1.0,
        "wallet_flow_blocked_untradable_asset": 0.0,
        "wallet_flow_reject_negative_seed_pnl": -1.0,
    }
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("status", ""), 0.0),
            _safe_float(row.get("score")),
        ),
    )


def _safe_float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _anomaly_stress_row(root: Path) -> ExplorationRow:
    tradeability_path = root / "anomaly_stress" / "current_peg_anomaly_tradeability.csv"
    tradeability = _best_numeric_row(tradeability_path, key="score")
    if tradeability:
        return ExplorationRow(
            lane="anomaly_stress",
            status=tradeability.get("status", "peg_anomaly_tradeability"),
            strongest_current_signal=(
                f"stablecoin_liquidity: {tradeability.get('symbol', '')}, "
                f"side={tradeability.get('side', '')}, score={tradeability.get('score', '')}, "
                f"pool_matches={tradeability.get('dex_pool_match_count', '')}, "
                f"yield_conflicts={tradeability.get('yield_conflict_count', '')}"
            ),
            main_gap=tradeability.get(
                "reason",
                "peg anomaly needs route, quote freshness, redemption, and executable depth checks",
            ),
            next_step=tradeability.get("next_step", "run peg anomaly tradeability checks"),
        )
    path = root / "anomaly_stress" / "current_cross_market_stress_anomaly.csv"
    best = _best_numeric_row(path, key="score")
    if best:
        return ExplorationRow(
            lane="anomaly_stress",
            status=best.get("status", "cross_market_stress_anomaly"),
            strongest_current_signal=(
                f"{best.get('source_lane', '')}: {best.get('subject', '')}, "
                f"side={best.get('side', '')}, score={best.get('score', '')}, "
                f"severity={best.get('severity', '')}"
            ),
            main_gap=best.get(
                "failure_mode",
                "anomaly can be stale, untradable, or explained by risk rather than alpha",
            ),
            next_step=best.get("next_step", "run a specific falsification test for the top anomaly"),
        )
    return ExplorationRow(
        lane="anomaly_stress",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="cross-market anomaly states are not joined across lanes",
        next_step="run current cross-market stress anomaly screen",
    )


def _tail_connectedness_regime_row(root: Path) -> ExplorationRow:
    path = root / "anomaly_stress" / "current_tail_connectedness_regime.csv"
    best = _best_numeric_row(path, key="connectedness_score")
    if best:
        return ExplorationRow(
            lane="tail_connectedness_regime",
            status=best.get("status", "tail_connectedness_regime"),
            strongest_current_signal=(
                f"{best.get('regime_id', '')}: "
                f"role={best.get('regime_role', '')}, "
                f"sources={best.get('source_count', '')}, "
                f"severity={best.get('severity_score', '')}, "
                f"connectedness={best.get('connectedness_score', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "tail regime still needs rolling connectedness, timestamps, and cross-asset labels",
            ),
            next_step=best.get(
                "next_probe",
                "condition downstream alpha labels on the strongest tail regime",
            ),
        )
    return ExplorationRow(
        lane="tail_connectedness_regime",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="tail and connectedness regimes are not separated from directional alpha candidates",
        next_step="run current tail connectedness regime after anomaly stress and event pressure",
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


def _best_prediction_market_news_pressure(path: Path, market_id: str) -> dict[str, str] | None:
    if not path.exists() or not market_id:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    matches = tuple(row for row in rows if row.get("market_id") == market_id)
    if not matches:
        return None
    return max(matches, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_probability_gap(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status") in {"paper_probability_gap_candidate", "probability_gap_watch"}
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_probability_paper_ticket(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status") in {"paper_event_probability_ticket", "event_probability_watch"}
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_probability_paper_outcome(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status") in {"paper_outcome_active_watch", "paper_outcome_edge_watch"}
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_probability_refresh(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status") in {"paper_outcome_survived_refresh", "paper_outcome_weak_refresh"}
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_probability_actionability(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status")
            in {
                "event_probability_candidate_after_refresh_check",
                "event_probability_candidate_after_current_quote_check",
                "event_probability_edge_watch",
            }
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_prediction_market_source_quality(path: Path, market_id: str) -> dict[str, str] | None:
    if not path.exists() or not market_id:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(row for row in csv.DictReader(handle) if row.get("market_id") == market_id)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("quality_score") or "0"))


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
    best_treasury = _current_public_treasury_context(root / "institutional_flow" / "current_public_treasury_context.csv")
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
                f"{_public_treasury_note(best_treasury)}"
            ),
            main_gap="current watch survived coarse 1h entry and adverse-excursion stress, but still lacks stop/fill and venue-specific mark/index checks",
            next_step="paper-check BTC short venue choice, stop criteria, mark/index basis, and actual account fee/fill assumptions before any live action",
        )
    if best_treasury:
        return ExplorationRow(
            lane="institutional_flow",
            status=best_treasury.get("action", "public_treasury_context"),
            strongest_current_signal=(
                f"{best_treasury.get('asset', '')}: dominance={best_treasury.get('market_cap_dominance', '')}, "
                f"top={best_treasury.get('top_holder_name', '')}/{best_treasury.get('top_holder_symbol', '')}, "
                f"top_supply_pct={best_treasury.get('top_holder_supply_pct', '')}, "
                f"funding={best_treasury.get('annualized_funding', '')}"
            ),
            main_gap="public treasury holdings are slow structural context and are not yet separated from equity issuance, corporate news, or perp positioning",
            next_step=best_treasury.get(
                "next_step",
                "join public treasury context to equity proxies, funding, and forward labels",
            ),
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


def _current_public_treasury_context(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = tuple(csv.DictReader(handle))
    rows = tuple(
        row
        for row in source_rows
        if row.get("action")
        in {
            "public_treasury_accumulation_vs_short_perp_watch",
            "public_treasury_crowded_long_watch",
            "public_treasury_concentration_watch",
        }
    )
    if not rows:
        return None
    return max(rows, key=lambda row: _safe_float(row.get("score")))


def _public_treasury_note(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    return (
        f"; treasury={row.get('asset', '')}/{row.get('action', '')}, "
        f"dominance={row.get('market_cap_dominance', '')}, "
        f"top={row.get('top_holder_symbol', '')}"
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
    repeat_gate_path = root / "candidate_validation" / "current_repeat_execution_gate.csv"
    best_repeat_gate = _best_repeat_execution_gate_row(repeat_gate_path)
    if best_repeat_gate:
        return ExplorationRow(
            lane="candidate_validation",
            status=best_repeat_gate.get("gate_action", "repeat_execution_gate"),
            strongest_current_signal=(
                f"{best_repeat_gate.get('venue', '')} "
                f"{best_repeat_gate.get('asset', '')}/{best_repeat_gate.get('source', '')}: "
                f"labels={best_repeat_gate.get('label_count', '')}, "
                f"hit15={best_repeat_gate.get('hit_rate_15m', '')}, "
                f"mean15_bps={best_repeat_gate.get('mean_dir15_bps', '')}, "
                f"rough_net15_bps={best_repeat_gate.get('rough_net15_bps', '')}"
            ),
            main_gap="repeat execution gate still lacks 1h confirmation, realized fill, funding PnL, stop behavior, and adverse-selection checks",
            next_step=best_repeat_gate.get("next_step", "paper-check the best repeat execution gate candidate"),
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
    chain_label_path = root / "stablecoin_liquidity" / "current_chain_stablecoin_migration_forward_labels.csv"
    best_chain_label = _best_chain_stablecoin_migration_label_row(chain_label_path)
    if best_chain_label:
        return ExplorationRow(
            lane="stablecoin_liquidity",
            status=best_chain_label.get("label_status", "chain_stablecoin_migration_forward_label"),
            strongest_current_signal=(
                f"{best_chain_label.get('chain', '')}/{best_chain_label.get('token_symbol', '')}: "
                f"migration={best_chain_label.get('migration_status', '')}, "
                f"week_change={best_chain_label.get('week_change_usd', '')}, "
                f"dir4h={best_chain_label.get('directional_return_4h', '')}, "
                f"dir12h={best_chain_label.get('directional_return_12h', '')}"
            ),
            main_gap="chain stablecoin migration has only short-horizon token labels and still excludes bridge route, venue depth, funding, and execution costs",
            next_step=best_chain_label.get(
                "next_step",
                "repeat chain-migration labels and add venue plus execution context",
            ),
        )
    migration_path = root / "stablecoin_liquidity" / "current_chain_stablecoin_migration.csv"
    migration_rows = tuple(row for row in _csv_rows(migration_path) if row.get("status") != "chain_stablecoin_context")
    best_migration = max(migration_rows, key=lambda row: float(row.get("score") or "-inf")) if migration_rows else None
    if best_migration:
        return ExplorationRow(
            lane="stablecoin_liquidity",
            status=best_migration.get("status", "chain_stablecoin_migration"),
            strongest_current_signal=(
                f"{best_migration.get('chain', '')}/{best_migration.get('token_symbol', '') or '-'}: "
                f"week_change={best_migration.get('week_change_usd', '')}, "
                f"week_pct={best_migration.get('week_change_pct', '')}, "
                f"top_asset={best_migration.get('top_asset', '')}, "
                f"score={best_migration.get('score', '')}"
            ),
            main_gap="stablecoin chain migration still needs bridge route checks, chain-token mapping, venue coverage, and forward labels",
            next_step=best_migration.get(
                "next_step",
                "label chain-token returns after stablecoin migration and check venue coverage",
            ),
        )
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


def _stablecoin_exchange_inflow_proxy_row(root: Path) -> ExplorationRow:
    path = root / "stablecoin_liquidity" / "current_stablecoin_exchange_inflow_proxy.csv"
    best = _best_numeric_row(path, key="priority")
    if best:
        return ExplorationRow(
            lane="stablecoin_exchange_inflow_proxy",
            status=best.get("status", "stablecoin_exchange_inflow_proxy"),
            strongest_current_signal=(
                f"{best.get('chain', '')}/{best.get('token_symbol', '') or '-'}: "
                f"{best.get('stablecoin_flow_direction', '')}, "
                f"week_change={best.get('week_change_usd', '')}, "
                f"priority={best.get('priority', '')}"
            ),
            main_gap=best.get(
                "missing_data",
                "exchange-inflow proxy still needs exchange wallet map and tagged stablecoin deposits",
            ),
            next_step=best.get(
                "next_probe",
                "separate exchange-inflow alpha from chain-level stablecoin liquidity proxy",
            ),
        )
    return ExplorationRow(
        lane="stablecoin_exchange_inflow_proxy",
        status="not_run",
        strongest_current_signal="not run yet",
        main_gap="chain stablecoin migration has not been separated from direct exchange-inflow alpha",
        next_step="run current stablecoin exchange inflow proxy after chain migration labels",
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
    actionability_path = root / "protocol_fundamentals" / "current_protocol_fee_actionability.csv"
    best_actionability = _best_numeric_row(actionability_path, key="score")
    if best_actionability:
        return ExplorationRow(
            lane="protocol_fundamentals",
            status=best_actionability.get("status", "protocol_fee_actionability"),
            strongest_current_signal=(
                f"{best_actionability.get('token_symbol', '')}/{best_actionability.get('protocol', '')}: "
                f"{best_actionability.get('side', '')}, "
                f"thesis={best_actionability.get('thesis_status', '')}, "
                f"labels={best_actionability.get('label_observations', '')}, "
                f"wins4h={best_actionability.get('wins_4h', '')}, "
                f"mean4h={best_actionability.get('mean_directional_4h', '')}, "
                f"score={best_actionability.get('score', '')}"
            ),
            main_gap=best_actionability.get(
                "reason",
                "protocol fee context needs repeated labels and execution checks",
            ),
            next_step=best_actionability.get(
                "next_step",
                "repeat protocol-fee labels and refresh execution context",
            ),
        )
    price_context_path = root / "protocol_fundamentals" / "current_protocol_fee_price_context.csv"
    best_price_context = _best_numeric_row(price_context_path, key="score")
    if best_price_context:
        return ExplorationRow(
            lane="protocol_fundamentals",
            status=best_price_context.get("status", "fee_price_context"),
            strongest_current_signal=(
                f"{best_price_context.get('token_symbol', '')}/{best_price_context.get('protocol', '')}: "
                f"{best_price_context.get('side', '')}, "
                f"fee_growth7d={best_price_context.get('fee_growth_7d', '')}, "
                f"price7d={best_price_context.get('price_change_7d', '')}, "
                f"fee_mcap={best_price_context.get('fee_to_market_cap', '')}, "
                f"score={best_price_context.get('score', '')}"
            ),
            main_gap=(
                "protocol fee-growth price context is still a current snapshot; it needs forward labels, "
                "funding PnL, unlock context, and execution checks"
            ),
            next_step=best_price_context.get(
                "next_step",
                "paper-label protocol fee-growth price-context candidates",
            ),
        )
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


def _first_matching_row(path: Path, *, key: str, value: str) -> dict[str, str] | None:
    for row in _csv_rows(path):
        if row.get(key) == value:
            return row
    return None


def _best_paper_ticket_outcome(rows: tuple[dict[str, str], ...]) -> dict[str, str] | None:
    numeric_rows = tuple(row for row in rows if row.get("directional_return_bps"))
    if numeric_rows:
        return max(numeric_rows, key=lambda row: float(row.get("directional_return_bps") or "-inf"))
    return rows[0] if rows else None


def _best_abs_numeric_row(path: Path, *, key: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: abs(float(row.get(key) or "0")))


def _best_policy_learning_sample(path: Path) -> dict[str, str] | None:
    rows = _csv_rows(path)
    if not rows:
        return None
    status_rank = {
        "cost_adjusted_win": 5,
        "mark_win_without_cost": 4,
        "cost_adjusted_edge_failed": 3,
        "depth_too_thin_for_probe": 2,
        "mark_loss": 1,
    }
    return max(
        rows,
        key=lambda row: (
            status_rank.get(row.get("reward_status", ""), 0),
            float(row.get("cost_adjusted_reward_bps") or row.get("reward_bps") or "0"),
        ),
    )


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


def _best_crowding_validated_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "paper_validated_carry_reversion_candidate"
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("validation_score") or "0"))


def _hyperliquid_dislocation_signal(path: Path) -> str:
    best = _best_numeric_row(path, key="score")
    if not best:
        return ""
    return (
        f"; HL dislocation {best.get('asset', '')} "
        f"{best.get('status', '')} "
        f"{best.get('side', '')} "
        f"score={best.get('score', '')}"
    )


def _hyperliquid_dislocation_monitor_signal(path: Path) -> str:
    best = _best_dislocation_monitor_row(path)
    if not best:
        return ""
    return (
        f"; monitor {best.get('asset', '')} "
        f"{best.get('status', '')} "
        f"{best.get('side', '')} "
        f"obs={best.get('observations', '')} "
        f"mean_score={best.get('mean_score', '')}"
    )


def _hyperliquid_dislocation_label_signal(path: Path) -> str:
    best = _best_dislocation_label_row(path)
    if not best:
        return ""
    return (
        f"; label {best.get('asset', '')} "
        f"{best.get('status', '')} "
        f"{best.get('side', '')} "
        f"net15={best.get('net_15m_bps', '')} "
        f"out15={best.get('outcome_15m', '')}"
    )


def _hyperliquid_dislocation_execution_signal(path: Path) -> str:
    best = _best_dislocation_execution_row(path)
    if not best:
        return ""
    return (
        f"; exec {best.get('asset', '')} "
        f"{best.get('status', '')} "
        f"{best.get('side', '')} "
        f"size={best.get('candidate_size_usd', '')} "
        f"net15={best.get('conservative_net_15m_bps', '')} "
        f"gate={best.get('gate_action', '')}"
    )


def _best_dislocation_execution_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row.get("gate_action") == "paper_execution_probe",
            float(row.get("conservative_net_15m_bps") or "-1000000"),
            -float(row.get("candidate_size_usd") or "0"),
        ),
    )


def _best_hyperliquid_dislocation_actionability_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("status")
            in {
                "dislocation_repeat_execution_candidate",
                "dislocation_repeat_needs_execution_check",
                "dislocation_single_snapshot_1h_watch",
            }
        )
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("score") or "0"))


def _best_dislocation_monitor_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row.get("monitor_action") == "repeat_label_priority",
            int(row.get("observations") or "0"),
            float(row.get("mean_score") or "0"),
            abs(float(row.get("mean_annualized_funding") or "0")),
        ),
    )


def _best_dislocation_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row.get("outcome_1h") == "paper_1h_win",
            row.get("outcome_15m") == "paper_15m_win",
            float(row.get("net_15m_bps") or "-1000000"),
            float(row.get("score") or "0"),
        ),
    )


def _best_crowding_execution_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            1 if row.get("gate_action") == "paper_execution_probe" else 0,
            float(row.get("conservative_net_1h_bps") or "0"),
            -float(row.get("candidate_size_usd") or "0"),
        ),
    )


def _best_crowding_outcome_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _crowding_outcome_rank(row),
            float(row.get("net_15m_bps") or "-1000000"),
            float(row.get("net_1h_bps") or "-1000000"),
            -float(row.get("candidate_size_usd") or "0"),
        ),
    )


def _crowding_outcome_rank(row: dict[str, str]) -> int:
    if row.get("outcome_1h") == "paper_1h_win":
        return 4
    if row.get("outcome_15m") == "paper_15m_win":
        return 3
    if row.get("outcome_15m") == "pending_15m" or row.get("outcome_1h") == "pending_1h":
        return 2
    return 1


def _crowding_outcome_status(row: dict[str, str]) -> str:
    if row.get("outcome_1h") == "paper_1h_win":
        return "paper_outcome_supported_carry_reversion_probe"
    if row.get("outcome_15m") == "paper_15m_win":
        return "paper_short_horizon_supported_carry_reversion_probe"
    if row.get("outcome_15m") == "pending_15m" or row.get("outcome_1h") == "pending_1h":
        return "paper_outcome_pending"
    return "paper_outcome_failed_carry_reversion_probe"


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


def _best_microstructure_flow_paper_gate_row(path: Path) -> dict[str, str] | None:
    rows = tuple(
        row
        for row in _csv_rows(path)
        if row.get("gate_action") == "microstructure_small_paper_probe"
    )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("conservative_net_15m_bps") or "-inf"),
            float(row.get("conservative_net_1h_bps") or "-inf"),
            -float(row.get("visible_depth_usage") or "inf"),
        ),
    )


def _best_microstructure_flow_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    action_rank = {
        "aligned_pressure_watch": 4,
        "book_trade_divergence_watch": 3,
        "one_sided_pressure_watch": 2,
        "no_clear_pressure": 1,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            action_rank.get(row.get("action", ""), 0),
            abs(float(row.get("pressure_score") or "0")),
            int(row.get("trade_count") or "0"),
            -float(row.get("spread_bps") or "0"),
        ),
    )


def _best_microstructure_flow_label_row(path: Path) -> dict[str, str] | None:
    rows = tuple(
        row
        for row in _csv_rows(path)
        if float(row.get("directional_return_15m") or "0") > 0.0
        and float(row.get("directional_return_1h") or "0") > 0.0
    )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("directional_return_1h") or "0"),
            float(row.get("directional_return_15m") or "0"),
            abs(float(row.get("pressure_score") or "0")),
        ),
    )


def _l2_imbalance_gate_status(row: dict[str, str]) -> str:
    if float(row.get("net_1h_bps") or "0") > 0.0:
        return "l2_imbalance_15m_1h_supported_probe"
    return "l2_imbalance_15m_only_probe"


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


def _best_liquidation_intensity_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("label_status")
            in {
                "continuation_15m_1h_supported",
                "reversal_15m_1h_supported",
                "continuation_15m_supported_pending_1h",
                "reversal_15m_supported_pending_1h",
            }
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            {
                "continuation_15m_1h_supported": 4,
                "reversal_15m_1h_supported": 4,
                "continuation_15m_supported_pending_1h": 3,
                "reversal_15m_supported_pending_1h": 3,
            }.get(row.get("label_status", ""), 0),
            max(
                float(row.get("continuation_return_15m") or "0"),
                float(row.get("reversal_return_15m") or "0"),
            ),
            float(row.get("intensity_score") or "0"),
        ),
    )


def _best_liquidation_intensity_paper_gate_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") in {"small_paper_probe", "small_paper_probe_pending_1h"}
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row.get("gate_action") == "small_paper_probe",
            float(row.get("conservative_net_bps") or "0"),
            -float(row.get("visible_depth_usage") or "0"),
        ),
    )


def _best_repeat_execution_gate_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("gate_action") == "small_repeat_paper_check"
            and row.get("rough_net15_bps", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("rough_net15_bps") or "-inf"),
            float(row.get("mean_dir15_bps") or "0"),
            float(row.get("label_count") or "0"),
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


def _best_chain_stablecoin_migration_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_4h", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _chain_stablecoin_label_rank(row.get("label_status", "")),
            float(row.get("directional_return_4h") or "-inf"),
            abs(float(row.get("week_change_usd") or "0")),
        ),
    )


def _chain_stablecoin_label_rank(status: str) -> int:
    return {
        "chain_migration_direction_supported": 4,
        "labeled_4h_pending_12h": 3,
        "mixed_chain_migration_direction": 2,
        "chain_migration_direction_contradicted": 1,
    }.get(status, 0)


def _best_market_breadth_label_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if row.get("directional_return_1h", "") != ""
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _market_breadth_label_rank(row),
            float(row.get("directional_return_4h") or "-inf"),
            float(row.get("directional_return_1h") or "-inf"),
            float(row.get("score") or "0"),
        ),
    )


def _best_market_breadth_execution_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _market_breadth_execution_rank(row.get("action", "")),
            float(row.get("conservative_net_4h_bps") or "-inf"),
            float(row.get("score") or "0"),
        ),
    )


def _market_breadth_execution_rank(action: str) -> int:
    return {
        "paper_execution_probe": 6,
        "thin_volume_watch": 4,
        "wide_spread_watch": 3,
        "too_large_for_visible_depth": 3,
        "no_edge_after_rough_cost": 2,
        "label_contradicted": 1,
        "not_hyperliquid": 0,
        "missing_l2_context": 0,
    }.get(action, 0)


def _market_breadth_label_rank(row: dict[str, str]) -> int:
    dir_4h = float(row.get("directional_return_4h") or "0")
    dir_1h = float(row.get("directional_return_1h") or "0")
    if dir_4h > 0.0 and dir_1h > 0.0:
        return 4
    if dir_4h > 0.0:
        return 3
    if dir_1h > 0.0:
        return 2
    return 1


def _market_breadth_label_status(row: dict[str, str]) -> str:
    rank = _market_breadth_label_rank(row)
    if rank == 4:
        return "volume_dislocation_4h_supported_pending_12h"
    if rank == 3:
        return "volume_dislocation_delayed_4h_support"
    if rank == 2:
        return "volume_dislocation_1h_only_watch"
    return "volume_dislocation_4h_contradicted_pending_12h"


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
