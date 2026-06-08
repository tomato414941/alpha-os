from __future__ import annotations

import argparse
from pathlib import Path

from strategies.current_paper_ticket_outcomes import (
    build_paper_ticket_outcomes,
    write_paper_ticket_outcomes_csv,
    write_paper_ticket_outcomes_md,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickets-path",
        type=Path,
        default=ROOT / "current_broad_alpha_paper_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_paper_outcomes.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_paper_outcomes.md",
    )
    args = parser.parse_args()

    rows = build_paper_ticket_outcomes(tickets_path=args.tickets_path)
    write_paper_ticket_outcomes_csv(rows, output_path=args.output_path)
    write_paper_ticket_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.ticket_id, row.checkpoint_status, row.outcome, row.directional_return_bps)


if __name__ == "__main__":
    main()
