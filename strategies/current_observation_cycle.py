from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_MODULES = (
    "strategies.current_alpha_stack",
    "strategies.current_paper_probe_plan",
    "strategies.current_paper_ticket_outcomes",
    "strategies.current_symbol_opportunity_map",
    "strategies.current_symbol_cluster_conflicts",
    "strategies.current_symbol_cluster_label_queue",
    "strategies.current_symbol_lane_split_review",
    "strategies.exploration_board",
)

OPEN_TICKET_MODULE = "strategies.current_paper_tickets"


def run_observation_cycle(*, open_new_tickets: bool = False) -> None:
    modules = list(DEFAULT_MODULES)
    if open_new_tickets:
        modules.insert(2, OPEN_TICKET_MODULE)
    for module in modules:
        print(f"== {module}")
        subprocess.run((sys.executable, "-m", module), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-new-tickets",
        action="store_true",
        help="Recreate current paper tickets. Omit this when preserving opened-at timestamps.",
    )
    args = parser.parse_args()
    run_observation_cycle(open_new_tickets=args.open_new_tickets)


if __name__ == "__main__":
    main()
