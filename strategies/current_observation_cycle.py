from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_COMMANDS = (
    ("strategies.wallet_entity_flow.current_wallet_entity_flow_access",),
    ("strategies.wallet_entity_flow.current_hyperliquid_seed_wallet_flow",),
    ("strategies.news_social.current_news_event_screen",),
    ("strategies.news_social.current_exchange_catalyst_snapshot",),
    ("strategies.news_social.current_exchange_catalyst_market_join",),
    ("strategies.news_social.current_attention_snapshot",),
    ("strategies.news_social.current_attention_market_join",),
    ("strategies.news_social.current_attention_price_context",),
    ("strategies.news_social.current_event_pressure_cluster",),
    ("strategies.options_volatility.current_deribit_options_surface",),
    ("strategies.options_volatility.current_deribit_options_realized_vol_labels",),
    ("strategies.options_volatility.current_options_volatility_paper_tickets",),
    ("strategies.options_volatility.current_volatility_actionability",),
    ("strategies.options_volatility.current_volatility_hedge_candidates",),
    ("strategies.execution_edge.current_execution_mode_candidates",),
    ("strategies.current_alpha_stack",),
    ("strategies.current_paper_probe_plan",),
    ("strategies.current_paper_ticket_outcomes",),
    ("strategies.current_paper_ticket_action_queue",),
    ("strategies.current_paper_ticket_fill_risk_check",),
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
    ("strategies.current_symbol_lane_paper_tickets", "--preserve-opened-at", "--top-symbols", "5"),
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
    ("strategies.current_symbol_opportunity_map",),
    ("strategies.current_symbol_cluster_conflicts",),
    ("strategies.current_symbol_cluster_label_queue",),
    ("strategies.current_symbol_lane_split_review",),
    ("strategies.policy_learning.current_policy_learning_samples",),
    ("strategies.policy_learning.current_action_preference_candidates",),
    ("strategies.policy_learning.current_action_preference_oos_check",),
    ("strategies.current_alpha_frontier",),
    ("strategies.exploration_board",),
)

PUBLIC_MARK_MODULES = (
    "strategies.perp_market_map.current_hyperliquid_snapshot",
    "strategies.candidate_validation.current_followup_execution_context",
    "strategies.candidate_validation.current_followup_okx_execution_context",
    "strategies.p0_parallel.binance_derivatives_intraday_live_execution_gate",
    "strategies.prediction_markets.current_event_probability_refresh",
)

PUBLIC_MARK_COMMANDS = tuple((module,) for module in PUBLIC_MARK_MODULES)

OPEN_TICKET_MODULE = "strategies.current_paper_tickets"
OPEN_TICKET_COMMAND = (OPEN_TICKET_MODULE, "--preserve-opened-at", "--top", "50")


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
