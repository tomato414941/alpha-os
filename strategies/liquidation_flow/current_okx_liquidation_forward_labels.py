from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationForwardLabel:
    timestamp: str
    asset: str
    action: str
    direction: int
    raw_return_15m: float | None
    raw_return_1h: float | None
    continuation_return_15m: float | None
    continuation_return_1h: float | None


def build_liquidation_forward_labels(
    *,
    input_path: Path = ROOT / "current_okx_liquidation_flow.csv",
) -> tuple[LiquidationForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_asset = {
        row["asset"]: _fetch_okx_candles(f"{row['asset']}-USDT-SWAP")
        for row in rows
    }
    return tuple(
        _build_label(row=row, candles=candles_by_asset.get(row["asset"], ()))
        for row in rows
    )


def write_liquidation_forward_labels(
    labels: tuple[LiquidationForwardLabel, ...],
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
                "action",
                "direction",
                "raw_return_15m",
                "raw_return_1h",
                "continuation_return_15m",
                "continuation_return_1h",
            )
        )
        for label in labels:
            writer.writerow(
                (
                    label.timestamp,
                    label.asset,
                    label.action,
                    label.direction,
                    "" if label.raw_return_15m is None else f"{label.raw_return_15m:.8f}",
                    "" if label.raw_return_1h is None else f"{label.raw_return_1h:.8f}",
                    (
                        ""
                        if label.continuation_return_15m is None
                        else f"{label.continuation_return_15m:.8f}"
                    ),
                    (
                        ""
                        if label.continuation_return_1h is None
                        else f"{label.continuation_return_1h:.8f}"
                    ),
                )
            )
    return output_path


def write_liquidation_forward_label_md(
    labels: tuple[LiquidationForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = tuple(
        sorted(
            labels,
            key=lambda row: (
                row.continuation_return_15m is not None,
                row.continuation_return_15m or -1.0,
                row.continuation_return_1h or -1.0,
            ),
            reverse=True,
        )
    )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Forward Labels\n\n")
        handle.write(
            "This labels liquidation-flow candidates with continuation returns. "
            "Positive continuation return means the forced-flow direction "
            "continued over that horizon.\n\n"
        )
        handle.write(
            "| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for label in ranked[:top]:
            handle.write(
                "| "
                f"{label.asset} | "
                f"{label.action} | "
                f"{label.direction} | "
                f"{'' if label.raw_return_15m is None else f'{label.raw_return_15m:.6f}'} | "
                f"{'' if label.continuation_return_15m is None else f'{label.continuation_return_15m:.6f}'} | "
                f"{'' if label.raw_return_1h is None else f'{label.raw_return_1h:.6f}'} | "
                f"{'' if label.continuation_return_1h is None else f'{label.continuation_return_1h:.6f}'} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is price-only continuation labeling. It does not decide whether "
            "a liquidation event should be traded as continuation, reversal, or "
            "ignored without further regime and execution checks.\n"
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
) -> LiquidationForwardLabel:
    timestamp = _parse_datetime(row["latest_liquidation_at"])
    direction = _direction_for_action(row["action"])
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return LiquidationForwardLabel(
        timestamp=timestamp.isoformat(),
        asset=row["asset"],
        action=row["action"],
        direction=direction,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        continuation_return_15m=(
            None if raw_return_15m is None or direction == 0 else raw_return_15m * direction
        ),
        continuation_return_1h=(
            None if raw_return_1h is None or direction == 0 else raw_return_1h * direction
        ),
    )


def _fetch_okx_candles(inst_id: str) -> tuple[dict[str, float], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/candles",
        params={"instId": inst_id, "bar": "15m", "limit": "100"},
        timeout=30,
    )
    response.raise_for_status()
    rows = tuple(
        {
            "timestamp": float(row[0]),
            "end_timestamp": float(row[0]) + 15 * 60 * 1000,
            "close": float(row[4]),
        }
        for row in response.json().get("data", ())
    )
    return tuple(sorted(rows, key=lambda row: row["timestamp"]))


def _forward_return(
    candles: tuple[dict[str, float], ...],
    start: datetime,
    target: datetime,
) -> float | None:
    start_close = _close_at(candles, start)
    end_close = _close_at(candles, target)
    if start_close is None or end_close is None:
        return None
    return (end_close / start_close) - 1.0 if start_close > 0.0 else None


def _close_at(candles: tuple[dict[str, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for candle in candles:
        if candle["timestamp"] <= target_ms <= candle["end_timestamp"]:
            return candle["close"]
        if candle["timestamp"] >= target_ms:
            return candle["close"]
    return None


def _direction_for_action(action: str) -> int:
    if action.startswith("long_liquidation"):
        return -1
    if action.startswith("short_liquidation"):
        return 1
    return 0


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_flow.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    labels = build_liquidation_forward_labels(input_path=args.input_path)
    write_liquidation_forward_labels(labels, output_path=args.output_path)
    write_liquidation_forward_label_md(labels, output_path=args.md_output_path, top=args.top)
    for label in labels[: args.top]:
        print(
            label.asset,
            label.action,
            f"cont15={'' if label.continuation_return_15m is None else f'{label.continuation_return_15m:.4f}'}",
            f"cont1h={'' if label.continuation_return_1h is None else f'{label.continuation_return_1h:.4f}'}",
        )


if __name__ == "__main__":
    main()
