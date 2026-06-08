from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_COMMANDS = (
    ("strategies.wallet_entity_flow.current_wallet_entity_flow_access",),
    ("strategies.wallet_entity_flow.current_hyperliquid_seed_wallet_flow",),
    ("strategies.wallet_entity_flow.current_seed_wallet_flow_actionability",),
    ("strategies.news_social.current_news_event_screen",),
    ("strategies.news_social.current_exchange_catalyst_snapshot",),
    ("strategies.news_social.current_exchange_catalyst_market_join",),
    ("strategies.news_social.current_attention_snapshot",),
    ("strategies.news_social.current_attention_market_join",),
    ("strategies.news_social.current_attention_price_context",),
    ("strategies.basis_term_structure.current_deribit_futures_basis",),
    ("strategies.cross_exchange_funding.current_funding_spread",),
    ("strategies.cross_exchange_funding.current_funding_feasibility",),
    ("strategies.cross_exchange_funding.current_okx_hl_funding_spread",),
    ("strategies.cross_exchange_funding.current_dislocation_watchlist",),
    ("strategies.institutional_flow.current_btc_etf_flow_snapshot",),
    ("strategies.institutional_flow.current_btc_etf_market_join",),
    ("strategies.institutional_flow.current_btc_etf_funding_candidate",),
    ("strategies.institutional_flow.current_btc_etf_funding_paper_ticket",),
    ("strategies.dex_pool_flow.current_geckoterminal_pool_flow",),
    ("strategies.derivatives_positioning.current_coingecko_derivatives_positioning",),
    ("strategies.stablecoin_liquidity.current_supply_snapshot",),
    ("strategies.stablecoin_liquidity.current_peg_stress_screen",),
    ("strategies.stablecoin_liquidity.current_supply_market_forward_labels",),
    ("strategies.stablecoin_liquidity.current_chain_stablecoin_migration",),
    ("strategies.stablecoin_liquidity.current_chain_stablecoin_migration_forward_labels",),
    ("strategies.stablecoin_liquidity.current_stablecoin_exchange_inflow_proxy",),
    ("strategies.defi_yield.current_yield_screen",),
    ("strategies.defi_yield.current_yield_quality_screen",),
    ("strategies.defi_yield.current_yield_peg_risk_join",),
    ("strategies.defi_lending.current_morpho_lending_rates",),
    ("strategies.defi_lending.current_lending_stress_actionability",),
    ("strategies.protocol_fundamentals.current_protocol_fee_screen",),
    ("strategies.protocol_fundamentals.current_protocol_fee_valuation",),
    ("strategies.protocol_fundamentals.current_protocol_fee_price_context",),
    ("strategies.protocol_fundamentals.current_protocol_fee_actionability",),
    ("strategies.protocol_fundamentals.current_protocol_fee_candidate_review",),
    ("strategies.protocol_fundamentals.current_protocol_fee_execution_context",),
    ("strategies.protocol_fundamentals.current_protocol_fee_price_lag_labels",),
    ("strategies.protocol_fundamentals.current_protocol_fee_paper_tickets",),
    ("strategies.protocol_activity.current_coingecko_protocol_activity",),
    ("strategies.protocol_activity.current_protocol_activity_market_join",),
    ("strategies.protocol_activity.current_protocol_activity_forward_labels",),
    ("strategies.on_chain_flow.current_chain_tvl_flow",),
    ("strategies.on_chain_flow.current_chain_tvl_flow_venue_coverage",),
    ("strategies.on_chain_flow.current_chain_tvl_flow_market_context",),
    ("strategies.on_chain_flow.current_chain_tvl_flow_forward_labels",),
    ("strategies.token_unlocks.current_token_unlock_snapshot",),
    ("strategies.token_unlocks.current_token_unlock_market_join",),
    ("strategies.token_unlocks.current_token_unlock_actionability",),
    ("strategies.token_unlocks.current_token_unlock_paper_tickets",),
    ("strategies.market_breadth.current_volume_price_dislocation",),
    ("strategies.market_breadth.current_volume_price_dislocation_labels",),
    ("strategies.market_breadth.current_volume_price_dislocation_execution_gate",),
    ("strategies.perp_market_map.current_hyperliquid_dislocation_candidates",),
    ("strategies.perp_market_map.current_hyperliquid_dislocation_forward_labels",),
    ("strategies.perp_market_map.current_hyperliquid_dislocation_execution_check",),
    ("strategies.perp_market_map.current_hyperliquid_dislocation_actionability",),
    ("strategies.perp_market_map.current_hyperliquid_oi_shift_candidates",),
    ("strategies.perp_market_map.current_okx_perp_pressure",),
    ("strategies.perp_market_map.current_okx_perp_pressure_forward_labels",),
    ("strategies.perp_market_map.current_crowding_reversion_screen",),
    ("strategies.perp_market_map.current_crowding_reversion_execution_check",),
    ("strategies.perp_market_map.current_crowding_reversion_validated_candidates",),
    ("strategies.liquidation_flow.current_okx_liquidation_flow",),
    ("strategies.liquidation_flow.current_okx_liquidation_forward_labels",),
    ("strategies.liquidation_flow.current_okx_liquidation_depth_check",),
    ("strategies.liquidation_flow.current_okx_liquidation_actionability_review",),
    ("strategies.liquidation_flow.current_okx_liquidation_paper_gate",),
    ("strategies.liquidation_flow.current_okx_liquidation_intensity",),
    ("strategies.liquidation_flow.current_okx_liquidation_intensity_forward_labels",),
    ("strategies.liquidation_flow.current_okx_liquidation_intensity_paper_gate",),
    ("strategies.market_making.current_microstructure_flow_snapshot",),
    ("strategies.market_making.current_microstructure_flow_forward_labels",),
    ("strategies.market_making.current_microstructure_flow_paper_gate",),
    ("strategies.market_making.current_l2_imbalance_monitor",),
    ("strategies.market_making.current_l2_imbalance_forward_labels",),
    ("strategies.market_making.current_l2_imbalance_paper_gate",),
    ("strategies.sector_rotation.current_coingecko_category_rotation",),
    ("strategies.sector_rotation.current_category_tradable_forward_labels",),
    ("strategies.sector_rotation.current_category_perp_context",),
    ("strategies.institutional_flow.current_public_treasury_context",),
    ("strategies.macro_regime.current_macro_crypto_context",),
    ("strategies.speculative_beta.current_speculative_beta_context",),
    ("strategies.crypto_equity_proxy.current_crypto_equity_proxy_context",),
    ("strategies.crypto_equity_proxy.current_crypto_equity_factor_split",),
    ("strategies.news_social.current_event_pressure_cluster",),
    ("strategies.news_social.current_news_event_source_independence",),
    ("strategies.news_social.current_ticker_attention_source_split",),
    ("strategies.options_volatility.current_deribit_options_surface",),
    ("strategies.options_volatility.current_deribit_options_realized_vol_labels",),
    ("strategies.options_volatility.current_options_volatility_paper_tickets",),
    ("strategies.options_volatility.current_volatility_actionability",),
    ("strategies.options_volatility.current_volatility_hedge_candidates",),
    ("strategies.prediction_markets.current_event_probability_refresh",),
    ("strategies.prediction_markets.current_event_probability_actionability",),
    ("strategies.prediction_markets.current_event_crypto_hedge_candidates",),
    ("strategies.anomaly_stress.current_cross_market_stress_anomaly",),
    ("strategies.anomaly_stress.current_peg_anomaly_tradeability",),
    ("strategies.anomaly_stress.current_tail_connectedness_regime",),
    ("strategies.execution_edge.current_execution_mode_candidates",),
    ("strategies.event_flow.current_lob_execution_world_replay",),
    ("strategies.event_flow.current_lob_sequence_state_probe",),
    ("strategies.current_cross_modal_alpha_context",),
    ("strategies.current_cross_modal_source_split",),
    ("strategies.current_alpha_stack",),
    ("strategies.current_paper_probe_plan",),
    ("strategies.current_paper_ticket_outcomes",),
    ("strategies.prediction_markets.current_event_crypto_hedge_reaction_labels",),
    ("strategies.prediction_markets.current_event_crypto_hedge_beta_attribution",),
    ("strategies.prediction_markets.current_event_crypto_hedge_event_alignment",),
    ("strategies.current_paper_ticket_action_queue",),
    ("strategies.current_paper_ticket_fill_risk_check",),
    ("strategies.policy_learning.current_policy_expansion_outcome_frontier",),
    ("strategies.current_promoted_ticket_repeat_tickets", "--preserve-opened-at"),
    ("strategies.current_promoted_ticket_repeat_outcomes",),
    (
        "strategies.current_paper_ticket_action_queue",
        "--outcomes-path",
        "strategies/current_promoted_ticket_repeat_outcomes.csv",
        "--output-path",
        "strategies/current_promoted_ticket_repeat_action_queue.csv",
        "--md-output-path",
        "strategies/current_promoted_ticket_repeat_action_queue.md",
    ),
    (
        "strategies.current_paper_ticket_fill_risk_check",
        "--action-queue-path",
        "strategies/current_promoted_ticket_repeat_action_queue.csv",
        "--tickets-path",
        "strategies/current_promoted_ticket_repeat_tickets.csv",
        "--output-path",
        "strategies/current_promoted_ticket_repeat_fill_risk_check.csv",
        "--md-output-path",
        "strategies/current_promoted_ticket_repeat_fill_risk_check.md",
    ),
    (
        "strategies.current_promoted_ticket_repeat_tickets",
        "--fill-risk-path",
        "strategies/current_promoted_ticket_repeat_fill_risk_check.csv",
        "--outcomes-path",
        "strategies/current_promoted_ticket_repeat_outcomes.csv",
        "--output-path",
        "strategies/current_second_promoted_ticket_repeat_tickets.csv",
        "--md-output-path",
        "strategies/current_second_promoted_ticket_repeat_tickets.md",
        "--preserve-opened-at",
    ),
    (
        "strategies.current_promoted_ticket_repeat_outcomes",
        "--tickets-path",
        "strategies/current_second_promoted_ticket_repeat_tickets.csv",
        "--output-path",
        "strategies/current_second_promoted_ticket_repeat_outcomes.csv",
        "--md-output-path",
        "strategies/current_second_promoted_ticket_repeat_outcomes.md",
    ),
    (
        "strategies.current_paper_ticket_action_queue",
        "--outcomes-path",
        "strategies/current_second_promoted_ticket_repeat_outcomes.csv",
        "--output-path",
        "strategies/current_second_promoted_ticket_repeat_action_queue.csv",
        "--md-output-path",
        "strategies/current_second_promoted_ticket_repeat_action_queue.md",
    ),
    (
        "strategies.current_paper_ticket_fill_risk_check",
        "--action-queue-path",
        "strategies/current_second_promoted_ticket_repeat_action_queue.csv",
        "--tickets-path",
        "strategies/current_second_promoted_ticket_repeat_tickets.csv",
        "--output-path",
        "strategies/current_second_promoted_ticket_repeat_fill_risk_check.csv",
        "--md-output-path",
        "strategies/current_second_promoted_ticket_repeat_fill_risk_check.md",
    ),
    ("strategies.current_symbol_lane_paper_tickets", "--preserve-opened-at", "--top-symbols", "20"),
    ("strategies.current_symbol_lane_paper_outcomes",),
    (
        "strategies.current_paper_ticket_action_queue",
        "--outcomes-path",
        "strategies/current_symbol_lane_paper_outcomes.csv",
        "--output-path",
        "strategies/current_symbol_lane_paper_action_queue.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_paper_action_queue.md",
    ),
    (
        "strategies.current_paper_ticket_fill_risk_check",
        "--action-queue-path",
        "strategies/current_symbol_lane_paper_action_queue.csv",
        "--tickets-path",
        "strategies/current_symbol_lane_paper_tickets.csv",
        "--output-path",
        "strategies/current_symbol_lane_paper_fill_risk_check.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_paper_fill_risk_check.md",
    ),
    (
        "strategies.current_promoted_ticket_repeat_tickets",
        "--fill-risk-path",
        "strategies/current_symbol_lane_paper_fill_risk_check.csv",
        "--outcomes-path",
        "strategies/current_symbol_lane_paper_outcomes.csv",
        "--output-path",
        "strategies/current_symbol_lane_promoted_repeat_tickets.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_promoted_repeat_tickets.md",
        "--preserve-opened-at",
    ),
    (
        "strategies.current_promoted_ticket_repeat_outcomes",
        "--tickets-path",
        "strategies/current_symbol_lane_promoted_repeat_tickets.csv",
        "--output-path",
        "strategies/current_symbol_lane_promoted_repeat_outcomes.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_promoted_repeat_outcomes.md",
    ),
    (
        "strategies.current_paper_ticket_action_queue",
        "--outcomes-path",
        "strategies/current_symbol_lane_promoted_repeat_outcomes.csv",
        "--output-path",
        "strategies/current_symbol_lane_promoted_repeat_action_queue.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_promoted_repeat_action_queue.md",
    ),
    (
        "strategies.current_paper_ticket_fill_risk_check",
        "--action-queue-path",
        "strategies/current_symbol_lane_promoted_repeat_action_queue.csv",
        "--tickets-path",
        "strategies/current_symbol_lane_promoted_repeat_tickets.csv",
        "--output-path",
        "strategies/current_symbol_lane_promoted_repeat_fill_risk_check.csv",
        "--md-output-path",
        "strategies/current_symbol_lane_promoted_repeat_fill_risk_check.md",
    ),
    ("strategies.current_cost_adjusted_alpha_candidates",),
    ("strategies.current_cost_adjusted_alpha_clusters",),
    ("strategies.current_cost_adjusted_cluster_repeat_plan",),
    ("strategies.current_split_first_cluster_lane_plan",),
    ("strategies.current_split_first_lane_repeat_queue",),
    ("strategies.current_split_first_lane_repeat_tickets", "--preserve-opened-at"),
    (
        "strategies.current_promoted_ticket_repeat_outcomes",
        "--tickets-path",
        "strategies/current_split_first_lane_repeat_tickets.csv",
        "--output-path",
        "strategies/current_split_first_lane_repeat_outcomes.csv",
        "--md-output-path",
        "strategies/current_split_first_lane_repeat_outcomes.md",
    ),
    ("strategies.current_symbol_opportunity_map",),
    ("strategies.current_symbol_cluster_conflicts",),
    ("strategies.current_symbol_cluster_label_queue",),
    ("strategies.current_symbol_lane_split_review",),
    ("strategies.policy_learning.current_policy_learning_samples",),
    ("strategies.policy_learning.current_observation_action_reward_dataset",),
    ("strategies.policy_learning.current_policy_context_frontier",),
    ("strategies.policy_learning.current_action_preference_candidates",),
    ("strategies.policy_learning.current_action_preference_oos_check",),
    ("strategies.policy_learning.current_policy_expansion_targets",),
    ("strategies.current_paper_probe_plan",),
    ("strategies.current_alpha_source_gaps",),
    ("strategies.current_alpha_frontier",),
    ("strategies.current_alpha_method_frontier",),
    ("strategies.current_research_backed_alpha_expansion_plan",),
    ("strategies.current_fundamental_sentiment_cross_section",),
    ("strategies.current_multimodal_btc_eth_feature_alignment",),
    ("strategies.exploration_board",),
)

PUBLIC_MARK_MODULES = (
    "strategies.perp_market_map.current_hyperliquid_snapshot",
    "strategies.candidate_validation.current_followup_execution_context",
    "strategies.candidate_validation.current_followup_okx_execution_context",
    "strategies.p0_parallel.binance_derivatives_intraday_live_execution_gate",
)

PUBLIC_MARK_COMMANDS = tuple((module,) for module in PUBLIC_MARK_MODULES)

OPEN_TICKET_MODULE = "strategies.current_paper_tickets"
OPEN_TICKET_COMMAND = (OPEN_TICKET_MODULE, "--preserve-opened-at", "--top", "80")


def run_observation_cycle(*, open_new_tickets: bool = False, refresh_public_marks: bool = False) -> None:
    commands: list[tuple[str, ...]] = []
    if refresh_public_marks:
        commands.extend(PUBLIC_MARK_COMMANDS)
    if open_new_tickets:
        pre_ticket_commands = _commands_through_paper_probe_plan()
        commands.extend(pre_ticket_commands)
        commands.append(OPEN_TICKET_COMMAND)
        commands.extend(DEFAULT_COMMANDS[len(pre_ticket_commands) :])
    else:
        commands.extend(DEFAULT_COMMANDS)
    for command in commands:
        module, *args = command
        print(f"== {module}")
        subprocess.run((sys.executable, "-m", module, *args), check=True)


def _commands_through_paper_probe_plan() -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for command in DEFAULT_COMMANDS:
        commands.append(command)
        if command[0] == "strategies.current_paper_probe_plan":
            break
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-new-tickets",
        action="store_true",
        help="Open more current paper tickets while preserving existing ticket entries.",
    )
    parser.add_argument(
        "--refresh-public-marks",
        action="store_true",
        help="Refresh public mark sources before checking paper-ticket outcomes.",
    )
    args = parser.parse_args()
    run_observation_cycle(
        open_new_tickets=args.open_new_tickets,
        refresh_public_marks=args.refresh_public_marks,
    )


if __name__ == "__main__":
    main()
