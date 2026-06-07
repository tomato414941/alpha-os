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
class ExchangeCatalystForwardLabel:
    published_at: str
    symbol: str
    catalyst_kind: str
    action: str
    direction_hint: int
    score: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    label_status: str


def build_exchange_catalyst_forward_labels(
    *,
    input_path: Path = ROOT / "current_exchange_catalyst_market_join.csv",
) -> tuple[ExchangeCatalystForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_symbol = {
        symbol: _fetch_hyperliquid_candles(symbol, rows)
        for symbol in sorted({row["symbol"] for row in rows})
    }
    return tuple(
        _build_label(row=row, candles=candles_by_symbol.get(row["symbol"], ()))
        for row in rows
    )


def write_exchange_catalyst_forward_labels_csv(
    rows: tuple[ExchangeCatalystForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "published_at",
                "symbol",
                "catalyst_kind",
                "action",
                "direction_hint",
                "score",
                "raw_return_15m",
                "raw_return_1h",
                "directional_return_15m",
                "directional_return_1h",
                "label_status",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.published_at,
                    row.symbol,
                    row.catalyst_kind,
                    row.action,
                    row.direction_hint,
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
                    row.label_status,
                )
            )
    return output_path


def write_exchange_catalyst_forward_labels_md(
    rows: tuple[ExchangeCatalystForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 25,
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
        handle.write("# Current Exchange Catalyst Forward Labels\n\n")
        handle.write(
            "This labels exchange-announcement catalysts with subsequent Hyperliquid "
            "returns. Positive directional return means the event direction hint was "
            "right before costs.\n\n"
        )
        handle.write(
            "| published | symbol | kind | dir | score | raw 15m | dir 15m | raw 1h | dir 1h | status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in ranked[:top]:
            handle.write(
                "| "
                f"{row.published_at} | "
                f"{row.symbol} | "
                f"{row.catalyst_kind} | "
                f"{row.direction_hint} | "
                f"{row.score:.6f} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{row.label_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is an event-reaction label only. It does not prove a reusable "
            "strategy and does not include fees, slippage, funding PnL, or venue "
            "latency.\n"
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
) -> ExchangeCatalystForwardLabel:
    published_at = _parse_datetime(row["published_at"])
    direction = int(row.get("direction_hint") or "0")
    raw_return_15m = _forward_return(candles, published_at, published_at + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, published_at, published_at + timedelta(hours=1))
    return ExchangeCatalystForwardLabel(
        published_at=published_at.isoformat(),
        symbol=row["symbol"],
        catalyst_kind=row["catalyst_kind"],
        action=row["action"],
        direction_hint=direction,
        score=float(row.get("score") or "0"),
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=(
            None if raw_return_15m is None or direction == 0 else raw_return_15m * direction
        ),
        directional_return_1h=(
            None if raw_return_1h is None or direction == 0 else raw_return_1h * direction
        ),
        label_status=_label_status(raw_return_15m=raw_return_15m, raw_return_1h=raw_return_1h),
    )


def _fetch_hyperliquid_candles(
    symbol: str,
    rows: tuple[dict[str, str], ...],
) -> tuple[dict[str, float], ...]:
    published_times = (
        _parse_datetime(row["published_at"])
        for row in rows
        if row["symbol"] == symbol and row.get("published_at")
    )
    times = tuple(published_times)
    if not times:
        return ()
    start = min(times) - timedelta(minutes=30)
    end = max(max(times) + timedelta(hours=2), datetime.now(UTC))
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


def _label_status(*, raw_return_15m: float | None, raw_return_1h: float | None) -> str:
    if raw_return_15m is None:
        return "pending_15m"
    if raw_return_1h is None:
        return "labeled_15m_pending_1h"
    return "labeled"


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_market_join.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_exchange_catalyst_forward_labels(input_path=args.input_path)
    write_exchange_catalyst_forward_labels_csv(rows, output_path=args.output_path)
    write_exchange_catalyst_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.catalyst_kind,
            f"dir15={row.directional_return_15m}",
            row.label_status,
        )


if __name__ == "__main__":
    main()
