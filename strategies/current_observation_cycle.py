from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_COMMANDS = (
    ("strategies.current_alpha_stack",),
    ("strategies.current_paper_probe_plan",),
    ("strategies.current_paper_ticket_outcomes",),
    ("strategies.current_paper_ticket_action_queue",),
    ("strategies.current_paper_ticket_fill_risk_check",),
    ("strategies.current_promoted_ticket_repeat_tickets", "--preserve-opened-at"),
    ("strategies.current_promoted_ticket_repeat_outcomes",),
    ("strategies.current_symbol_lane_paper_tickets", "--preserve-opened-at"),
    ("strategies.current_symbol_lane_paper_outcomes",),
    ("strategies.current_symbol_opportunity_map",),
    ("strategies.current_symbol_cluster_conflicts",),
    ("strategies.current_symbol_cluster_label_queue",),
    ("strategies.current_symbol_lane_split_review",),
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
OPEN_TICKET_COMMAND = (OPEN_TICKET_MODULE,)


def run_observation_cycle(*, open_new_tickets: bool = False, refresh_public_marks: bool = False) -> None:
    commands: list[tuple[str, ...]] = []
    if refresh_public_marks:
        commands.extend(PUBLIC_MARK_COMMANDS)
    if open_new_tickets:
        commands.append(OPEN_TICKET_COMMAND)
    commands.extend(DEFAULT_COMMANDS)
    for command in commands:
        module, *args = command
        print(f"== {module}")
        subprocess.run((sys.executable, "-m", module, *args), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-new-tickets",
        action="store_true",
        help="Recreate current paper tickets. Omit this when preserving opened-at timestamps.",
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
