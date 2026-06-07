from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABEL_STATUSES = {
    "fee_growth_price_lag_candidate",
    "fee_growth_price_confirmation",
    "fee_growth_price_chase_risk",
    "fee_decay_price_weakness_context",
}


@dataclass(frozen=True)
class ProtocolFeePriceLagHistoryRow:
    observed_at: str
    token_symbol: str
    protocol: str
    status: str
    side: str
    direction: int
    current_price: float
    fee_to_market_cap: float
    fee_to_fdv: float
    fee_growth_7d: float
    funding: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    score: float
    observation_status: str
    next_step: str


def build_protocol_fee_price_lag_history_rows(
    *,
    history_path: Path = ROOT / "protocol_fee_price_lag_observation_history.csv",
    context_path: Path = ROOT / "current_protocol_fee_price_context.csv",
    observed_at: datetime | None = None,
) -> tuple[ProtocolFeePriceLagHistoryRow, ...]:
    timestamp = (observed_at or datetime.now(UTC)).replace(second=0, microsecond=0).isoformat()
    existing = tuple(_history_row(row) for row in _read_rows(history_path))
    current = tuple(_context_row(row, observed_at=timestamp) for row in _read_rows(context_path) if row.get("status") in LABEL_STATUSES)
    rows_by_key = {_key(row): row for row in existing}
    for row in current:
        rows_by_key.setdefault(_key(row), row)
    return tuple(sorted(rows_by_key.values(), key=lambda row: (row.observed_at, row.token_symbol)))


def write_protocol_fee_price_lag_history_csv(
    rows: tuple[ProtocolFeePriceLagHistoryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observed_at",
                "token_symbol",
                "protocol",
                "status",
                "side",
                "direction",
                "current_price",
                "fee_to_market_cap",
                "fee_to_fdv",
                "fee_growth_7d",
                "funding",
                "price_change_24h",
                "price_change_7d",
                "price_change_30d",
                "score",
                "observation_status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.observed_at,
                    row.token_symbol,
                    row.protocol,
                    row.status,
                    row.side,
                    row.direction,
                    f"{row.current_price:.12f}",
                    f"{row.fee_to_market_cap:.8f}",
                    f"{row.fee_to_fdv:.8f}",
                    f"{row.fee_growth_7d:.8f}",
                    f"{row.funding:.8f}",
                    f"{row.price_change_24h:.8f}",
                    f"{row.price_change_7d:.8f}",
                    f"{row.price_change_30d:.8f}",
                    f"{row.score:.8f}",
                    row.observation_status,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_price_lag_history_md(
    rows: tuple[ProtocolFeePriceLagHistoryRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready_rows = tuple(row for row in rows if row.observation_status == "ready_for_label")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Protocol Fee Price-Lag Observation History\n\n")
        handle.write(
            "This stores current protocol fee-growth price-lag observations so later runs can attach "
            "4h, 12h, 24h, and 7d forward labels. It is a sample store, not a trade log.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- ready rows: `{len(ready_rows)}`\n\n")
        handle.write("| observed at | token | status | dir | price | fee growth 7d | price 7d | score | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in tuple(sorted(rows, key=lambda item: item.observed_at, reverse=True))[:top]:
            handle.write(
                f"| {row.observed_at} | {row.token_symbol} | {row.status} | {row.direction} | "
                f"{row.current_price:.8f} | {row.fee_growth_7d:.2f} | {row.price_change_7d:.2f} | "
                f"{row.score:.4f} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _context_row(row: dict[str, str], *, observed_at: str) -> ProtocolFeePriceLagHistoryRow:
    direction = _direction(row.get("side", ""))
    current_price = _float(row.get("current_price"))
    return ProtocolFeePriceLagHistoryRow(
        observed_at=observed_at,
        token_symbol=row.get("token_symbol", ""),
        protocol=row.get("protocol", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=direction,
        current_price=current_price,
        fee_to_market_cap=_float(row.get("fee_to_market_cap")),
        fee_to_fdv=_float(row.get("fee_to_fdv")),
        fee_growth_7d=_float(row.get("fee_growth_7d")),
        funding=_float(row.get("funding")),
        price_change_24h=_float(row.get("price_change_24h")),
        price_change_7d=_float(row.get("price_change_7d")),
        price_change_30d=_float(row.get("price_change_30d")),
        score=_float(row.get("score")),
        observation_status="ready_for_label" if current_price > 0.0 and direction != 0 else "context_only",
        next_step=row.get("next_step", ""),
    )


def _history_row(row: dict[str, str]) -> ProtocolFeePriceLagHistoryRow:
    return ProtocolFeePriceLagHistoryRow(
        observed_at=row.get("observed_at", ""),
        token_symbol=row.get("token_symbol", ""),
        protocol=row.get("protocol", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=int(row.get("direction") or "0"),
        current_price=_float(row.get("current_price")),
        fee_to_market_cap=_float(row.get("fee_to_market_cap")),
        fee_to_fdv=_float(row.get("fee_to_fdv")),
        fee_growth_7d=_float(row.get("fee_growth_7d")),
        funding=_float(row.get("funding")),
        price_change_24h=_float(row.get("price_change_24h")),
        price_change_7d=_float(row.get("price_change_7d")),
        price_change_30d=_float(row.get("price_change_30d")),
        score=_float(row.get("score")),
        observation_status=row.get("observation_status", ""),
        next_step=row.get("next_step", ""),
    )


def _direction(side: str) -> int:
    if side in {"long_token", "wait_or_pullback_long"}:
        return 1
    if side == "watch_or_short":
        return -1
    return 0


def _key(row: ProtocolFeePriceLagHistoryRow) -> tuple[str, str, str, str]:
    return (row.observed_at, row.token_symbol, row.protocol, row.status)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path", type=Path, default=ROOT / "protocol_fee_price_lag_observation_history.csv")
    parser.add_argument("--context-path", type=Path, default=ROOT / "current_protocol_fee_price_context.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "protocol_fee_price_lag_observation_history.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_protocol_fee_price_lag_history_rows(history_path=args.history_path, context_path=args.context_path)
    write_protocol_fee_price_lag_history_csv(rows, output_path=args.history_path)
    write_protocol_fee_price_lag_history_md(rows, output_path=args.markdown_output_path, top=args.top)
    print(
        f"rows={len(rows)}",
        f"ready={sum(row.observation_status == 'ready_for_label' for row in rows)}",
    )


if __name__ == "__main__":
    main()
