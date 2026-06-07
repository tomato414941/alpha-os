from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ("BTC", "ETH", "SOL", "HYPE")


@dataclass(frozen=True)
class SupplyMarketForwardLabel:
    timestamp: str
    liquidity_group: str
    total_supply_usd: float
    week_change_usd: float
    week_change_pct: float
    expected_risk_direction: int
    asset: str
    raw_return_1h: float | None
    raw_return_4h: float | None
    raw_return_12h: float | None
    directional_return_1h: float | None
    directional_return_4h: float | None
    directional_return_12h: float | None
    action: str


def build_supply_market_forward_labels(
    *,
    supply_path: Path = ROOT / "current_supply_snapshot.csv",
    assets: tuple[str, ...] = DEFAULT_ASSETS,
) -> tuple[SupplyMarketForwardLabel, ...]:
    supply_rows = _read_rows(supply_path)
    timestamp = _parse_datetime(supply_rows[0]["timestamp"])
    group = _major_stablecoin_group(supply_rows)
    expected_direction = 1 if group["week_change_usd"] >= 0.0 else -1
    candles_by_asset = {
        asset: _fetch_hyperliquid_candles(asset=asset, start=timestamp)
        for asset in assets
    }
    asset_labels = tuple(
        _build_label(
            timestamp=timestamp,
            liquidity_group="major_stablecoins",
            total_supply_usd=group["total_supply_usd"],
            week_change_usd=group["week_change_usd"],
            expected_direction=expected_direction,
            asset=asset,
            candles=candles,
        )
        for asset, candles in candles_by_asset.items()
    )
    basket_label = _basket_label(asset_labels)
    return asset_labels + (basket_label,)


def write_supply_market_forward_labels_csv(
    rows: tuple[SupplyMarketForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "liquidity_group",
                "total_supply_usd",
                "week_change_usd",
                "week_change_pct",
                "expected_risk_direction",
                "asset",
                "raw_return_1h",
                "raw_return_4h",
                "raw_return_12h",
                "directional_return_1h",
                "directional_return_4h",
                "directional_return_12h",
                "action",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.liquidity_group,
                    f"{row.total_supply_usd:.2f}",
                    f"{row.week_change_usd:.2f}",
                    f"{row.week_change_pct:.8f}",
                    row.expected_risk_direction,
                    row.asset,
                    "" if row.raw_return_1h is None else f"{row.raw_return_1h:.8f}",
                    "" if row.raw_return_4h is None else f"{row.raw_return_4h:.8f}",
                    "" if row.raw_return_12h is None else f"{row.raw_return_12h:.8f}",
                    (
                        ""
                        if row.directional_return_1h is None
                        else f"{row.directional_return_1h:.8f}"
                    ),
                    (
                        ""
                        if row.directional_return_4h is None
                        else f"{row.directional_return_4h:.8f}"
                    ),
                    (
                        ""
                        if row.directional_return_12h is None
                        else f"{row.directional_return_12h:.8f}"
                    ),
                    row.action,
                )
            )
    return output_path


def write_supply_market_forward_labels_md(
    rows: tuple[SupplyMarketForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Supply Market Forward Labels\n\n")
        handle.write(
            "This labels the current major-stablecoin liquidity snapshot against "
            "subsequent market returns. Positive directional return means the "
            "risk-on/risk-off direction implied by stablecoin supply was right.\n\n"
        )
        handle.write(
            "| asset | week change USD | week change % | expected dir | raw 1h | dir 1h | raw 4h | dir 4h | raw 12h | dir 12h | action |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.week_change_usd:.0f} | "
                f"{row.week_change_pct:.6f} | "
                f"{row.expected_risk_direction} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{'' if row.raw_return_4h is None else f'{row.raw_return_4h:.6f}'} | "
                f"{'' if row.directional_return_4h is None else f'{row.directional_return_4h:.6f}'} | "
                f"{'' if row.raw_return_12h is None else f'{row.raw_return_12h:.6f}'} | "
                f"{'' if row.directional_return_12h is None else f'{row.directional_return_12h:.6f}'} | "
                f"{row.action} |\n"
            )
    return output_path


def _build_label(
    *,
    timestamp: datetime,
    liquidity_group: str,
    total_supply_usd: float,
    week_change_usd: float,
    expected_direction: int,
    asset: str,
    candles: tuple[dict[str, float], ...],
) -> SupplyMarketForwardLabel:
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    raw_return_4h = _forward_return(candles, timestamp, timestamp + timedelta(hours=4))
    raw_return_12h = _forward_return(candles, timestamp, timestamp + timedelta(hours=12))
    return SupplyMarketForwardLabel(
        timestamp=timestamp.isoformat(),
        liquidity_group=liquidity_group,
        total_supply_usd=total_supply_usd,
        week_change_usd=week_change_usd,
        week_change_pct=week_change_usd / total_supply_usd if total_supply_usd > 0.0 else 0.0,
        expected_risk_direction=expected_direction,
        asset=asset,
        raw_return_1h=raw_return_1h,
        raw_return_4h=raw_return_4h,
        raw_return_12h=raw_return_12h,
        directional_return_1h=_directional_return(raw_return_1h, expected_direction),
        directional_return_4h=_directional_return(raw_return_4h, expected_direction),
        directional_return_12h=_directional_return(raw_return_12h, expected_direction),
        action=_action(
            directional_return_4h=_directional_return(raw_return_4h, expected_direction),
            directional_return_12h=_directional_return(raw_return_12h, expected_direction),
        ),
    )


def _basket_label(rows: tuple[SupplyMarketForwardLabel, ...]) -> SupplyMarketForwardLabel:
    first = rows[0]
    raw_return_1h = _mean_present(tuple(row.raw_return_1h for row in rows))
    raw_return_4h = _mean_present(tuple(row.raw_return_4h for row in rows))
    raw_return_12h = _mean_present(tuple(row.raw_return_12h for row in rows))
    return SupplyMarketForwardLabel(
        timestamp=first.timestamp,
        liquidity_group=first.liquidity_group,
        total_supply_usd=first.total_supply_usd,
        week_change_usd=first.week_change_usd,
        week_change_pct=first.week_change_pct,
        expected_risk_direction=first.expected_risk_direction,
        asset="BASKET",
        raw_return_1h=raw_return_1h,
        raw_return_4h=raw_return_4h,
        raw_return_12h=raw_return_12h,
        directional_return_1h=_directional_return(raw_return_1h, first.expected_risk_direction),
        directional_return_4h=_directional_return(raw_return_4h, first.expected_risk_direction),
        directional_return_12h=_directional_return(raw_return_12h, first.expected_risk_direction),
        action=_action(
            directional_return_4h=_directional_return(raw_return_4h, first.expected_risk_direction),
            directional_return_12h=_directional_return(
                raw_return_12h, first.expected_risk_direction
            ),
        ),
    )


def _major_stablecoin_group(rows: tuple[dict[str, str], ...]) -> dict[str, float]:
    major_rows = tuple(
        row
        for row in rows
        if float(row.get("current_supply_usd") or "0") >= 1_000_000_000.0
        and 0.95 <= float(row.get("price") or "0") <= 1.05
    )
    return {
        "total_supply_usd": sum(float(row["current_supply_usd"]) for row in major_rows),
        "week_change_usd": sum(float(row["week_change_usd"]) for row in major_rows),
    }


def _fetch_hyperliquid_candles(
    *,
    asset: str,
    start: datetime,
) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
                "interval": "15m",
                "startTime": int((start - timedelta(minutes=30)).timestamp() * 1000),
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


def _directional_return(raw_return: float | None, direction: int) -> float | None:
    if raw_return is None:
        return None
    return raw_return * direction


def _action(
    *,
    directional_return_4h: float | None,
    directional_return_12h: float | None,
) -> str:
    if directional_return_4h is None:
        return "pending_4h"
    if directional_return_12h is None:
        return "labeled_4h_pending_12h"
    if directional_return_4h > 0.0 and directional_return_12h > 0.0:
        return "liquidity_direction_supported"
    if directional_return_4h < 0.0 and directional_return_12h < 0.0:
        return "liquidity_direction_contradicted"
    return "mixed_liquidity_direction"


def _mean_present(values: tuple[float | None, ...]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return sum(present) / len(present)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supply-path",
        type=Path,
        default=ROOT / "current_supply_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_supply_market_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_supply_market_forward_labels.md",
    )
    args = parser.parse_args()

    rows = build_supply_market_forward_labels(supply_path=args.supply_path)
    write_supply_market_forward_labels_csv(rows, output_path=args.output_path)
    write_supply_market_forward_labels_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(
            row.asset,
            row.action,
            f"dir4h={'' if row.directional_return_4h is None else f'{row.directional_return_4h:.4f}'}",
            f"dir12h={'' if row.directional_return_12h is None else f'{row.directional_return_12h:.4f}'}",
        )


if __name__ == "__main__":
    main()
