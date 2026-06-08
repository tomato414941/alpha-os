from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class VolumePriceDislocationLabelRow:
    observed_at: str
    symbol: str
    name: str
    coin_id: str
    status: str
    side: str
    direction: int
    score: float
    start_price: float
    raw_return_1h: float | None
    raw_return_4h: float | None
    raw_return_12h: float | None
    raw_return_24h: float | None
    directional_return_1h: float | None
    directional_return_4h: float | None
    directional_return_12h: float | None
    directional_return_24h: float | None
    price_source: str
    label_status: str


def build_volume_price_dislocation_label_rows(
    *,
    history_path: Path = ROOT / "volume_price_dislocation_observation_history.csv",
) -> tuple[VolumePriceDislocationLabelRow, ...]:
    observations = tuple(row for row in _read_rows(history_path) if row.get("observation_status") == "ready_for_label")
    prices = {
        coin_id: _fetch_prices(coin_id)
        for coin_id in sorted({row.get("coin_id", "") for row in observations if row.get("coin_id", "")})
    }
    candles = {
        symbol: _fetch_hyperliquid_candles(asset=symbol, start=_earliest_observed_at(observations))
        for symbol in sorted({row.get("symbol", "") for row in observations if row.get("symbol", "")})
    }
    rows = tuple(
        _build_label(
            row=row,
            prices=prices.get(row.get("coin_id", ""), ()),
            candles=candles.get(row.get("symbol", ""), ()),
        )
        for row in observations
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_volume_price_dislocation_labels_csv(
    rows: tuple[VolumePriceDislocationLabelRow, ...],
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
                "score",
                "start_price",
                "raw_return_1h",
                "raw_return_4h",
                "raw_return_12h",
                "raw_return_24h",
                "directional_return_1h",
                "directional_return_4h",
                "directional_return_12h",
                "directional_return_24h",
                "price_source",
                "label_status",
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
                    f"{row.score:.8f}",
                    f"{row.start_price:.12f}",
                    _format_optional(row.raw_return_1h),
                    _format_optional(row.raw_return_4h),
                    _format_optional(row.raw_return_12h),
                    _format_optional(row.raw_return_24h),
                    _format_optional(row.directional_return_1h),
                    _format_optional(row.directional_return_4h),
                    _format_optional(row.directional_return_12h),
                    _format_optional(row.directional_return_24h),
                    row.price_source,
                    row.label_status,
                )
            )
    return output_path


def write_volume_price_dislocation_labels_md(
    rows: tuple[VolumePriceDislocationLabelRow, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_1h = tuple(row for row in rows if row.directional_return_1h is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Volume Price Dislocation Forward Labels\n\n")
        handle.write(
            "This labels stored volume-price dislocation observations. Positive directional return "
            "means the observation direction was right before fees, spread, funding PnL, and slippage.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- labeled 1h rows: `{len(labeled_1h)}`\n\n")
        handle.write(
            "| observed at | symbol | status | dir | score | dir 1h | dir 4h | dir 12h | "
            "dir 24h | source | label status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.observed_at} | {row.symbol} | {row.status} | {row.direction} | "
                f"{row.score:.4f} | {_format_optional(row.directional_return_1h)} | "
                f"{_format_optional(row.directional_return_4h)} | {_format_optional(row.directional_return_12h)} | "
                f"{_format_optional(row.directional_return_24h)} | {row.price_source} | {row.label_status} |\n"
            )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    prices: tuple[tuple[float, float], ...],
    candles: tuple[dict[str, float], ...],
) -> VolumePriceDislocationLabelRow:
    observed_at = _parse_datetime(row.get("observed_at", ""))
    direction = int(row.get("direction") or "0")
    start_price = _float(row.get("current_price"))
    raw_1h = _forward_return(prices, start_price, observed_at + timedelta(hours=1))
    raw_4h = _forward_return(prices, start_price, observed_at + timedelta(hours=4))
    raw_12h = _forward_return(prices, start_price, observed_at + timedelta(hours=12))
    raw_24h = _forward_return(prices, start_price, observed_at + timedelta(hours=24))
    price_source = "coingecko"
    if raw_1h is None:
        start_price = _close_at_or_after(candles, observed_at) or start_price
        raw_1h = _candle_forward_return(candles, observed_at, observed_at + timedelta(hours=1))
        raw_4h = _candle_forward_return(candles, observed_at, observed_at + timedelta(hours=4))
        raw_12h = _candle_forward_return(candles, observed_at, observed_at + timedelta(hours=12))
        raw_24h = _candle_forward_return(candles, observed_at, observed_at + timedelta(hours=24))
        price_source = "hyperliquid" if raw_1h is not None else "unavailable"
    return VolumePriceDislocationLabelRow(
        observed_at=observed_at.isoformat(),
        symbol=row.get("symbol", ""),
        name=row.get("name", ""),
        coin_id=row.get("coin_id", ""),
        status=row.get("status", ""),
        side=row.get("side", ""),
        direction=direction,
        score=_float(row.get("score")),
        start_price=start_price,
        raw_return_1h=raw_1h,
        raw_return_4h=raw_4h,
        raw_return_12h=raw_12h,
        raw_return_24h=raw_24h,
        directional_return_1h=_directional(raw_1h, direction),
        directional_return_4h=_directional(raw_4h, direction),
        directional_return_12h=_directional(raw_12h, direction),
        directional_return_24h=_directional(raw_24h, direction),
        price_source=price_source,
        label_status=_label_status(raw_1h=raw_1h, raw_4h=raw_4h, raw_12h=raw_12h, raw_24h=raw_24h),
    )


def _fetch_prices(coin_id: str) -> tuple[tuple[float, float], ...]:
    try:
        response = requests.get(
            COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id),
            params={"vs_currency": "usd", "days": "3"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ()
    return tuple((float(timestamp), float(price)) for timestamp, price in response.json().get("prices", ()))


def _fetch_hyperliquid_candles(
    *,
    asset: str,
    start: datetime,
) -> tuple[dict[str, float], ...]:
    try:
        response = requests.post(
            HYPERLIQUID_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": "15m",
                    "startTime": int((start - timedelta(minutes=30)).timestamp() * 1000),
                    "endTime": int(datetime.now(UTC).timestamp() * 1000),
                },
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ()
    payload = response.json()
    if not isinstance(payload, list):
        return ()
    return tuple(
        {
            "timestamp": float(row["t"]),
            "end_timestamp": float(row["T"]),
            "close": float(row["c"]),
        }
        for row in payload
    )


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


def _candle_forward_return(
    candles: tuple[dict[str, float], ...],
    start: datetime,
    target: datetime,
) -> float | None:
    start_close = _close_at_or_after(candles, start)
    end_close = _close_at_or_after(candles, target)
    if start_close is None or end_close is None:
        return None
    return end_close / start_close - 1.0


def _price_at_or_after(prices: tuple[tuple[float, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for timestamp_ms, price in prices:
        if timestamp_ms >= target_ms:
            return price
    return None


def _close_at_or_after(candles: tuple[dict[str, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for candle in candles:
        if candle["timestamp"] >= target_ms:
            return candle["close"]
    return None


def _directional(raw_return: float | None, direction: int) -> float | None:
    return None if raw_return is None or direction == 0 else raw_return * direction


def _label_status(
    *,
    raw_1h: float | None,
    raw_4h: float | None,
    raw_12h: float | None,
    raw_24h: float | None,
) -> str:
    if raw_1h is None:
        return "pending_1h"
    if raw_4h is None:
        return "labeled_1h_pending_4h"
    if raw_12h is None:
        return "labeled_4h_pending_12h"
    if raw_24h is None:
        return "labeled_12h_pending_24h"
    return "labeled_24h"


def _sort_key(row: VolumePriceDislocationLabelRow) -> tuple[bool, float, float]:
    return (
        row.directional_return_1h is not None,
        row.directional_return_1h or -1.0,
        row.score,
    )


def _earliest_observed_at(rows: tuple[dict[str, str], ...]) -> datetime:
    parsed = tuple(_parse_datetime(row.get("observed_at", "")) for row in rows if row.get("observed_at", ""))
    if not parsed:
        return datetime.now(UTC) - timedelta(days=3)
    return min(parsed)


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
    parser.add_argument("--history-path", type=Path, default=ROOT / "volume_price_dislocation_observation_history.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_volume_price_dislocation_labels.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_volume_price_dislocation_labels.md")
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    rows = build_volume_price_dislocation_label_rows(history_path=args.history_path)
    write_volume_price_dislocation_labels_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_labels_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.label_status, row.symbol, f"dir1h={_format_optional(row.directional_return_1h)}")


if __name__ == "__main__":
    main()
