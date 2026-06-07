from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CategoryTradableForwardLabel:
    timestamp: str
    category_id: str
    category_name: str
    category_action: str
    category_change_24h: float
    coin_id: str
    symbol: str
    coin_name: str
    coin_market_cap: float
    coin_volume_24h: float
    direction: int
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    label_status: str
    score: float


def build_category_tradable_forward_labels(
    *,
    input_path: Path = ROOT / "current_coingecko_category_rotation.csv",
    max_categories: int = 30,
) -> tuple[CategoryTradableForwardLabel, ...]:
    category_rows = _read_rows(input_path)[:max_categories]
    coin_ids = _coin_ids_from_category_rows(category_rows)
    markets = _fetch_coingecko_markets(coin_ids)
    tradable_assets = _fetch_hyperliquid_assets()
    symbols = {
        _coin_symbol(markets.get(coin_id, {}))
        for coin_id in coin_ids
        if _coin_symbol(markets.get(coin_id, {})) in tradable_assets
    }
    candles_by_symbol = {
        symbol: _fetch_hyperliquid_candles(symbol) for symbol in sorted(symbols)
    }
    rows: list[CategoryTradableForwardLabel] = []
    for category_row in category_rows:
        for coin_id in _split_coin_ids(category_row.get("top_3_coins_id", "")):
            rows.append(
                _build_label(
                    category_row=category_row,
                    coin_id=coin_id,
                    market=markets.get(coin_id),
                    tradable_assets=tradable_assets,
                    candles_by_symbol=candles_by_symbol,
                )
            )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_category_tradable_forward_labels(
    rows: tuple[CategoryTradableForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "category_id",
                "category_name",
                "category_action",
                "category_change_24h",
                "coin_id",
                "symbol",
                "coin_name",
                "coin_market_cap",
                "coin_volume_24h",
                "direction",
                "raw_return_15m",
                "raw_return_1h",
                "directional_return_15m",
                "directional_return_1h",
                "label_status",
                "score",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.category_id,
                    row.category_name,
                    row.category_action,
                    f"{row.category_change_24h:.8f}",
                    row.coin_id,
                    row.symbol,
                    row.coin_name,
                    f"{row.coin_market_cap:.4f}",
                    f"{row.coin_volume_24h:.4f}",
                    row.direction,
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
                    f"{row.score:.8f}",
                )
            )
    return output_path


def write_category_tradable_forward_labels_md(
    rows: tuple[CategoryTradableForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Category Tradable Forward Labels\n\n")
        handle.write(
            "This maps CoinGecko category rotation into Hyperliquid-tradable "
            "constituents and labels subsequent 15m/1h returns. It is a sector "
            "rotation label, not a trade instruction.\n\n"
        )
        handle.write(
            "| category | coin | action | change24 | dir | raw 15m | dir 15m | raw 1h | dir 1h | status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.category_name} | "
                f"{row.symbol or row.coin_id} | "
                f"{row.category_action} | "
                f"{row.category_change_24h:.4f} | "
                f"{row.direction} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{row.label_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is still one snapshot. It does not model constituent weighting, "
            "category membership quality, liquidity, costs, or repeated evidence. "
            "Rows with `not_hyperliquid` are useful only as context because they are "
            "not directly tradable through this venue.\n"
        )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _coin_ids_from_category_rows(rows: tuple[dict[str, str], ...]) -> tuple[str, ...]:
    coin_ids: list[str] = []
    for row in rows:
        for coin_id in _split_coin_ids(row.get("top_3_coins_id", "")):
            if coin_id not in coin_ids:
                coin_ids.append(coin_id)
    return tuple(coin_ids)


def _split_coin_ids(value: str) -> tuple[str, ...]:
    return tuple(coin_id for coin_id in value.split(";") if coin_id)


def _fetch_coingecko_markets(coin_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    if not coin_ids:
        return {}
    response = requests.get(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": str(max(len(coin_ids), 1)),
            "page": "1",
            "sparkline": "false",
        },
        timeout=30,
    )
    response.raise_for_status()
    return {str(row["id"]): row for row in response.json()}


def _fetch_hyperliquid_assets() -> frozenset[str]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=30,
    )
    response.raise_for_status()
    meta, _contexts = response.json()
    return frozenset(str(row["name"]).upper() for row in meta["universe"])


def _build_label(
    *,
    category_row: dict[str, str],
    coin_id: str,
    market: dict[str, object] | None,
    tradable_assets: frozenset[str],
    candles_by_symbol: dict[str, tuple[dict[str, float], ...]],
) -> CategoryTradableForwardLabel:
    timestamp = _parse_datetime(category_row["timestamp"])
    direction = _direction_for_category(category_row)
    symbol = _coin_symbol(market or {})
    status = _label_status(market=market, symbol=symbol, tradable_assets=tradable_assets)
    candles = candles_by_symbol.get(symbol, ())
    raw_return_15m = (
        _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
        if status != "not_hyperliquid"
        else None
    )
    raw_return_1h = (
        _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
        if status != "not_hyperliquid"
        else None
    )
    if status == "tradable_pending_label" and (
        raw_return_15m is not None or raw_return_1h is not None
    ):
        status = "tradable_labeled"
    return CategoryTradableForwardLabel(
        timestamp=timestamp.isoformat(),
        category_id=category_row.get("category_id", ""),
        category_name=category_row.get("name", ""),
        category_action=category_row.get("action", ""),
        category_change_24h=_float(category_row.get("market_cap_change_24h")),
        coin_id=coin_id,
        symbol=symbol,
        coin_name=str((market or {}).get("name") or ""),
        coin_market_cap=_float((market or {}).get("market_cap")),
        coin_volume_24h=_float((market or {}).get("total_volume")),
        direction=direction,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=(
            None if raw_return_15m is None or direction == 0 else raw_return_15m * direction
        ),
        directional_return_1h=(
            None if raw_return_1h is None or direction == 0 else raw_return_1h * direction
        ),
        label_status=status,
        score=_score(
            category_change=_float(category_row.get("market_cap_change_24h")),
            category_score=_float(category_row.get("score")),
            coin_market_cap=_float((market or {}).get("market_cap")),
            directional_return_15m=raw_return_15m * direction
            if raw_return_15m is not None and direction != 0
            else None,
        ),
    )


def _direction_for_category(row: dict[str, str]) -> int:
    change = _float(row.get("market_cap_change_24h"))
    if row.get("action") == "sector_stress_watch" or change < 0.0:
        return -1
    if change > 0.0:
        return 1
    return 0


def _label_status(
    *,
    market: dict[str, object] | None,
    symbol: str,
    tradable_assets: frozenset[str],
) -> str:
    if not market:
        return "missing_market_data"
    if symbol not in tradable_assets:
        return "not_hyperliquid"
    return "tradable_pending_label"


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


def _sort_key(row: CategoryTradableForwardLabel) -> tuple[bool, float, float, float]:
    return (
        row.directional_return_15m is not None,
        row.directional_return_15m or -1.0,
        row.directional_return_1h or -1.0,
        row.score,
    )


def _score(
    *,
    category_change: float,
    category_score: float,
    coin_market_cap: float,
    directional_return_15m: float | None,
) -> float:
    label_component = 0.0 if directional_return_15m is None else directional_return_15m * 1000.0
    return abs(category_change) + category_score / 100.0 + coin_market_cap / 1_000_000_000.0 + label_component


def _coin_symbol(row: dict[str, object]) -> str:
    return str(row.get("symbol") or "").upper()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_coingecko_category_rotation.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_category_tradable_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_category_tradable_forward_labels.md",
    )
    parser.add_argument("--max-categories", type=int, default=30)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_category_tradable_forward_labels(
        input_path=args.input_path,
        max_categories=args.max_categories,
    )
    write_category_tradable_forward_labels(rows, output_path=args.output_path)
    write_category_tradable_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.category_name,
            row.symbol or row.coin_id,
            row.label_status,
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.4f}'}",
            f"dir1h={'' if row.directional_return_1h is None else f'{row.directional_return_1h:.4f}'}",
        )


if __name__ == "__main__":
    main()
