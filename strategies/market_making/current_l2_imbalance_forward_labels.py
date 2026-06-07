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
class L2ImbalanceForwardLabel:
    timestamp: str
    asset: str
    spread_bps: float
    imbalance_10_bps: float
    direction: int
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None


def build_l2_imbalance_forward_labels(
    *,
    input_path: Path = ROOT / "current_l2_snapshot.csv",
) -> tuple[L2ImbalanceForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_asset = {
        asset: _fetch_hyperliquid_candles(asset)
        for asset in sorted({row["asset"] for row in rows})
    }
    return tuple(
        _build_label(row=row, candles=candles_by_asset.get(row["asset"], ()))
        for row in rows
    )


def write_l2_imbalance_forward_labels(
    rows: tuple[L2ImbalanceForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset",
                "spread_bps",
                "imbalance_10_bps",
                "direction",
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
                    row.asset,
                    f"{row.spread_bps:.8f}",
                    f"{row.imbalance_10_bps:.8f}",
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
                )
            )
    return output_path


def write_l2_imbalance_forward_labels_md(
    rows: tuple[L2ImbalanceForwardLabel, ...],
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
        handle.write("# Current L2 Imbalance Forward Labels\n\n")
        handle.write(
            "This labels whether the visible 10 bps book imbalance matched "
            "subsequent Hyperliquid price direction. It is an imbalance alpha "
            "probe, not a market-making fill model.\n\n"
        )
        handle.write(
            "| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in ranked[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.spread_bps:.4f} | "
                f"{row.imbalance_10_bps:.4f} | "
                f"{row.direction} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Positive directional return means the snapshot's visible imbalance "
            "pointed in the right price direction. A market-making strategy still "
            "needs queue position, fill probability, maker/taker fees, and adverse "
            "selection estimates.\n"
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
) -> L2ImbalanceForwardLabel:
    timestamp = _parse_datetime(row["timestamp"])
    imbalance = float(row["imbalance_10_bps"])
    direction = _direction_for_imbalance(imbalance)
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return L2ImbalanceForwardLabel(
        timestamp=timestamp.isoformat(),
        asset=row["asset"],
        spread_bps=float(row["spread_bps"]),
        imbalance_10_bps=imbalance,
        direction=direction,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=(
            None if raw_return_15m is None or direction == 0 else raw_return_15m * direction
        ),
        directional_return_1h=(
            None if raw_return_1h is None or direction == 0 else raw_return_1h * direction
        ),
    )


def _direction_for_imbalance(imbalance: float) -> int:
    if imbalance > 0.0:
        return 1
    if imbalance < 0.0:
        return -1
    return 0


def _fetch_hyperliquid_candles(asset: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=24)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
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
        default=ROOT / "current_l2_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_l2_imbalance_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_l2_imbalance_forward_labels(input_path=args.input_path)
    write_l2_imbalance_forward_labels(rows, output_path=args.output_path)
    write_l2_imbalance_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            f"imbalance={row.imbalance_10_bps:.4f}",
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.4f}'}",
            f"dir1h={'' if row.directional_return_1h is None else f'{row.directional_return_1h:.4f}'}",
        )


if __name__ == "__main__":
    main()
