from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from strategies.protocol_fundamentals.current_protocol_fee_valuation import COINGECKO_IDS


ROOT = Path(__file__).resolve().parent
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"


@dataclass(frozen=True)
class ProtocolFeePriceLagLabelRow:
    observed_at: str
    token_symbol: str
    protocol: str
    status: str
    side: str
    direction: int
    priority: float
    start_price: float
    raw_return_4h: float | None
    raw_return_12h: float | None
    raw_return_24h: float | None
    raw_return_7d: float | None
    directional_return_4h: float | None
    directional_return_12h: float | None
    directional_return_24h: float | None
    directional_return_7d: float | None
    label_status: str


def build_protocol_fee_price_lag_label_rows(
    *,
    history_path: Path = ROOT / "protocol_fee_price_lag_observation_history.csv",
) -> tuple[ProtocolFeePriceLagLabelRow, ...]:
    observations = tuple(row for row in _read_rows(history_path) if row.get("observation_status") == "ready_for_label")
    prices = {
        token: _fetch_prices(token)
        for token in sorted({row.get("token_symbol", "") for row in observations if row.get("token_symbol", "") in COINGECKO_IDS})
    }
    rows = tuple(
        _build_label(row=row, prices=prices.get(row.get("token_symbol", ""), ()))
        for row in observations
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_protocol_fee_price_lag_labels_csv(
    rows: tuple[ProtocolFeePriceLagLabelRow, ...],
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
                "priority",
                "start_price",
                "raw_return_4h",
                "raw_return_12h",
                "raw_return_24h",
                "raw_return_7d",
                "directional_return_4h",
                "directional_return_12h",
                "directional_return_24h",
                "directional_return_7d",
                "label_status",
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
                    f"{row.priority:.8f}",
                    f"{row.start_price:.12f}",
                    _format_optional(row.raw_return_4h),
                    _format_optional(row.raw_return_12h),
                    _format_optional(row.raw_return_24h),
                    _format_optional(row.raw_return_7d),
                    _format_optional(row.directional_return_4h),
                    _format_optional(row.directional_return_12h),
                    _format_optional(row.directional_return_24h),
                    _format_optional(row.directional_return_7d),
                    row.label_status,
                )
            )
    return output_path


def write_protocol_fee_price_lag_labels_md(
    rows: tuple[ProtocolFeePriceLagLabelRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_4h = tuple(row for row in rows if row.directional_return_4h is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Protocol Fee Price-Lag Forward Labels\n\n")
        handle.write(
            "This labels stored fee-growth price-lag observations. Positive directional return means "
            "the observation's direction was right before fees, funding PnL, and slippage.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- labeled 4h rows: `{len(labeled_4h)}`\n\n")
        handle.write(
            "| observed at | token | status | dir | priority | dir 4h | dir 12h | dir 24h | dir 7d | label status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.observed_at} | {row.token_symbol} | {row.status} | {row.direction} | "
                f"{row.priority:.4f} | {_format_optional(row.directional_return_4h)} | "
                f"{_format_optional(row.directional_return_12h)} | {_format_optional(row.directional_return_24h)} | "
                f"{_format_optional(row.directional_return_7d)} | {row.label_status} |\n"
            )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    prices: tuple[tuple[float, float], ...],
) -> ProtocolFeePriceLagLabelRow:
    observed_at = _parse_datetime(row.get("observed_at", ""))
    direction = int(row.get("direction") or "0")
    start_price = _float(row.get("current_price"))
    raw_4h = _forward_return(prices, start_price, observed_at + timedelta(hours=4))
    raw_12h = _forward_return(prices, start_price, observed_at + timedelta(hours=12))
    raw_24h = _forward_return(prices, start_price, observed_at + timedelta(hours=24))
    raw_7d = _forward_return(prices, start_price, observed_at + timedelta(days=7))
    return ProtocolFeePriceLagLabelRow(
        observed_at=observed_at.isoformat(),
        token_symbol=row.get("token_symbol", ""),
        protocol=row.get("protocol", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=direction,
        priority=_float(row.get("score")),
        start_price=start_price,
        raw_return_4h=raw_4h,
        raw_return_12h=raw_12h,
        raw_return_24h=raw_24h,
        raw_return_7d=raw_7d,
        directional_return_4h=_directional(raw_4h, direction),
        directional_return_12h=_directional(raw_12h, direction),
        directional_return_24h=_directional(raw_24h, direction),
        directional_return_7d=_directional(raw_7d, direction),
        label_status=_label_status(raw_4h=raw_4h, raw_12h=raw_12h, raw_24h=raw_24h, raw_7d=raw_7d),
    )


def _fetch_prices(token: str) -> tuple[tuple[float, float], ...]:
    coin_id = COINGECKO_IDS[token].coin_id
    try:
        response = requests.get(
            COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id),
            params={"vs_currency": "usd", "days": "10"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ()
    return tuple((float(timestamp), float(price)) for timestamp, price in response.json().get("prices", ()))


def _forward_return(
    prices: tuple[tuple[float, float], ...],
    start_price: float,
    target: datetime,
) -> float | None:
    if start_price <= 0.0:
        return None
    target_price = _price_at_or_after(prices, target)
    if target_price is None:
        return None
    return target_price / start_price - 1.0


def _price_at_or_after(prices: tuple[tuple[float, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for timestamp_ms, price in prices:
        if timestamp_ms >= target_ms:
            return price
    return None


def _directional(raw_return: float | None, direction: int) -> float | None:
    return None if raw_return is None or direction == 0 else raw_return * direction


def _label_status(
    *,
    raw_4h: float | None,
    raw_12h: float | None,
    raw_24h: float | None,
    raw_7d: float | None,
) -> str:
    if raw_4h is None:
        return "pending_4h"
    if raw_12h is None:
        return "labeled_4h_pending_12h"
    if raw_24h is None:
        return "labeled_12h_pending_24h"
    if raw_7d is None:
        return "labeled_24h_pending_7d"
    return "labeled_7d"


def _sort_key(row: ProtocolFeePriceLagLabelRow) -> tuple[bool, float, float]:
    return (
        row.directional_return_4h is not None,
        row.directional_return_4h or -1.0,
        row.priority,
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "protocol_fee_price_lag_observation_history.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_price_lag_labels.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_price_lag_labels.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_protocol_fee_price_lag_label_rows(history_path=args.history_path)
    write_protocol_fee_price_lag_labels_csv(rows, output_path=args.output_path)
    write_protocol_fee_price_lag_labels_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.label_status, row.token_symbol, f"dir4h={_format_optional(row.directional_return_4h)}")


if __name__ == "__main__":
    main()
