from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABEL_STATUSES = {
    "attention_price_lag_candidate",
    "attention_breakout_continuation_watch",
    "attention_capitulation_reversal_watch",
    "attention_chase_risk",
}


@dataclass(frozen=True)
class AttentionPriceHistoryRow:
    observed_at: str
    symbol: str
    name: str
    coin_id: str
    status: str
    side: str
    direction: int
    attention_rank: int
    current_price: float
    volume_to_market_cap: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    score: float
    observation_status: str
    next_step: str


def build_attention_price_history_rows(
    *,
    history_path: Path = ROOT / "attention_price_observation_history.csv",
    context_path: Path = ROOT / "current_attention_price_context.csv",
    observed_at: datetime | None = None,
) -> tuple[AttentionPriceHistoryRow, ...]:
    timestamp = (observed_at or datetime.now(UTC)).replace(second=0, microsecond=0).isoformat()
    existing = tuple(_history_row(row) for row in _read_rows(history_path))
    current = tuple(
        _context_row(row, observed_at=timestamp)
        for row in _read_rows(context_path)
        if row.get("status") in LABEL_STATUSES
    )
    rows_by_key = {_key(row): row for row in existing}
    for row in current:
        rows_by_key.setdefault(_key(row), row)
    return tuple(sorted(rows_by_key.values(), key=lambda row: (row.observed_at, row.symbol, row.status)))


def write_attention_price_history_csv(
    rows: tuple[AttentionPriceHistoryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observed_at",
                "symbol",
                "name",
                "coin_id",
                "status",
                "side",
                "direction",
                "attention_rank",
                "current_price",
                "volume_to_market_cap",
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
                    row.symbol,
                    row.name,
                    row.coin_id,
                    row.status,
                    row.side,
                    row.direction,
                    row.attention_rank,
                    f"{row.current_price:.12f}",
                    f"{row.volume_to_market_cap:.8f}",
                    f"{row.price_change_24h:.8f}",
                    f"{row.price_change_7d:.8f}",
                    f"{row.price_change_30d:.8f}",
                    f"{row.score:.8f}",
                    row.observation_status,
                    row.next_step,
                )
            )
    return output_path


def write_attention_price_history_md(
    rows: tuple[AttentionPriceHistoryRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready_rows = tuple(row for row in rows if row.observation_status == "ready_for_label")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Attention Price Observation History\n\n")
        handle.write(
            "This stores attention-price observations so later runs can attach 1h, 4h, 12h, and 24h "
            "forward labels. It is a sample store, not a trade log.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- ready rows: `{len(ready_rows)}`\n\n")
        handle.write("| observed at | symbol | status | dir | rank | price | price 7d | score | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in tuple(sorted(rows, key=lambda item: item.observed_at, reverse=True))[:top]:
            handle.write(
                f"| {row.observed_at} | {row.symbol} | {row.status} | {row.direction} | "
                f"{row.attention_rank} | {row.current_price:.8f} | {row.price_change_7d:.2f} | "
                f"{row.score:.4f} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _context_row(row: dict[str, str], *, observed_at: str) -> AttentionPriceHistoryRow:
    direction = _direction(row.get("side", ""))
    current_price = _float(row.get("current_price"))
    return AttentionPriceHistoryRow(
        observed_at=observed_at,
        symbol=row.get("symbol", ""),
        name=row.get("name", ""),
        coin_id=row.get("coin_id", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=direction,
        attention_rank=int(row.get("attention_rank") or "0"),
        current_price=current_price,
        volume_to_market_cap=_float(row.get("volume_to_market_cap")),
        price_change_24h=_float(row.get("price_change_24h")),
        price_change_7d=_float(row.get("price_change_7d")),
        price_change_30d=_float(row.get("price_change_30d")),
        score=_float(row.get("score")),
        observation_status="ready_for_label" if current_price > 0.0 and direction != 0 else "context_only",
        next_step=row.get("next_step", ""),
    )


def _history_row(row: dict[str, str]) -> AttentionPriceHistoryRow:
    return AttentionPriceHistoryRow(
        observed_at=row.get("observed_at", ""),
        symbol=row.get("symbol", ""),
        name=row.get("name", ""),
        coin_id=row.get("coin_id", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=int(row.get("direction") or "0"),
        attention_rank=int(row.get("attention_rank") or "0"),
        current_price=_float(row.get("current_price")),
        volume_to_market_cap=_float(row.get("volume_to_market_cap")),
        price_change_24h=_float(row.get("price_change_24h")),
        price_change_7d=_float(row.get("price_change_7d")),
        price_change_30d=_float(row.get("price_change_30d")),
        score=_float(row.get("score")),
        observation_status=row.get("observation_status", ""),
        next_step=row.get("next_step", ""),
    )


def _direction(side: str) -> int:
    if side in {"long_attention_lag", "long_momentum_watch"}:
        return 1
    if side == "wait_or_fade_watch":
        return -1
    return 0


def _key(row: AttentionPriceHistoryRow) -> tuple[str, str, str]:
    return (row.observed_at, row.symbol, row.status)


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
    parser.add_argument("--history-path", type=Path, default=ROOT / "attention_price_observation_history.csv")
    parser.add_argument("--context-path", type=Path, default=ROOT / "current_attention_price_context.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "attention_price_observation_history.md")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_attention_price_history_rows(history_path=args.history_path, context_path=args.context_path)
    write_attention_price_history_csv(rows, output_path=args.history_path)
    write_attention_price_history_md(rows, output_path=args.markdown_output_path, top=args.top)
    print(
        f"rows={len(rows)}",
        f"ready={sum(row.observation_status == 'ready_for_label' for row in rows)}",
    )


if __name__ == "__main__":
    main()
