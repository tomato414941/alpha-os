from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

from strategies.daily_close.yahoo import fetch_yahoo_daily_rows, write_daily_rows
from strategies.equity_index.data import DEFAULT_SYMBOLS, LOCAL_DATASET_DIR


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _default_end_date() -> date:
    return datetime.now(UTC).date()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=_default_end_date())
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DATASET_DIR)
    args = parser.parse_args()

    for symbol in args.symbols:
        rows = fetch_yahoo_daily_rows(
            symbol,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(write_daily_rows(symbol, rows, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
