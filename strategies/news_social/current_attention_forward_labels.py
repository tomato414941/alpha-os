from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AttentionForwardLabel:
    timestamp: str
    symbol: str
    action: str
    direction: int
    score: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None


def build_attention_forward_labels(
    *,
    input_path: Path = ROOT / "current_attention_market_join.csv",
) -> tuple[AttentionForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_symbol = {
        symbol: _fetch_hyperliquid_candles(symbol)
        for symbol in sorted({row["symbol"] for row in rows})
    }
    return tuple(
        _build_label(row=row, candles=candles_by_symbol.get(row["symbol"], ()))
        for row in rows
    )


def write_attention_forward_labels(
    rows: tuple[AttentionForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "action",
                "direction",
                "score",
                "raw_return_15m",
                "raw_return_1h",
                "directional_return_15m",
                "directional_return_1h",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.symbol,
                    row.action,
                    row.direction,
                    f"{row.score:.8f}",
                    "" if row.raw_return_15m is None else f"{row.raw_return_15m:.8f}",
                    "" if row.raw_return_1h is None else f"{row.raw_return_1h:.8f}",
                    (
                        ""
                        if row.directional_return_15m is None
                        else f"{row.directional_return_15m:.8f}"
                    ),
                    (
                        ""
                        if row.directional_return_1h is None
                        else f"{row.directional_return_1h:.8f}"
                    ),
                )
            )
    return output_path


def write_attention_forward_labels_md(
    rows: tuple[AttentionForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.directional_return_15m is not None,
                row.directional_return_15m or -1.0,
                row.directional_return_1h is not None,
                row.directional_return_1h or -1.0,
            ),
            reverse=True,
        )
    )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Attention Forward Labels\n\n")
        handle.write(
            "This labels attention/perp-overlap candidates with subsequent "
            "Hyperliquid returns. Positive directional return means the carry or "
            "funding direction was right over that horizon.\n\n"
        )
        handle.write(
            "| symbol | action | dir | score | raw 15m | dir 15m | raw 1h | dir 1h |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in ranked[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.action} | "
                f"{row.direction} | "
                f"{row.score:.6f} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is a label on the attention/perp overlap only. It does not include "
            "fees, spread, funding PnL, social-news causality, or a neutral baseline.\n"
        )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _build_label(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
) -> AttentionForwardLabel:
    timestamp = _parse_datetime(row["timestamp"])
    direction = _direction_for_row(row)
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return AttentionForwardLabel(
        timestamp=timestamp.isoformat(),
        symbol=row["symbol"],
        action=row["action"],
        direction=direction,
        score=float(row.get("score") or "0"),
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=(
            None if raw_return_15m is None or direction == 0 else raw_return_15m * direction
        ),
        directional_return_1h=(
            None if raw_return_1h is None or direction == 0 else raw_return_1h * direction
        ),
    )


def _direction_for_row(row: dict[str, str]) -> int:
    carry_action = row.get("carry_reversion_action", "")
    if carry_action.startswith("long_"):
        return 1
    if carry_action.startswith("short_"):
        return -1
    funding = float(row.get("annualized_funding") or "0")
    if row.get("action") == "attention_funding_watch" and funding < 0.0:
        return 1
    if row.get("action") == "attention_funding_watch" and funding > 0.0:
        return -1
    return 0


def _fetch_hyperliquid_candles(symbol: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=8)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": "15m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple(
        {
            "timestamp": float(row["t"]),
            "end_timestamp": float(row["T"]),
            "close": float(row["c"]),
        }
        for row in response.json()
    )


def _forward_return(
    candles: tuple[dict[str, float], ...],
    start: datetime,
    target: datetime,
) -> float | None:
    start_close = _close_at_or_after(candles, start)
    end_close = _close_at_or_after(candles, target)
    if start_close is None or end_close is None:
        return None
    return (end_close / start_close) - 1.0 if start_close > 0.0 else None


def _close_at_or_after(candles: tuple[dict[str, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for candle in candles:
        if candle["timestamp"] <= target_ms <= candle["end_timestamp"]:
            return candle["close"]
        if candle["timestamp"] >= target_ms:
            return candle["close"]
    return None


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_attention_market_join.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_attention_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_attention_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_attention_forward_labels(input_path=args.input_path)
    write_attention_forward_labels(rows, output_path=args.output_path)
    write_attention_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.action,
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.4f}'}",
            f"dir1h={'' if row.directional_return_1h is None else f'{row.directional_return_1h:.4f}'}",
        )


if __name__ == "__main__":
    main()
