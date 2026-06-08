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
class LiquidationIntensityForwardLabel:
    event_timestamp: str
    asset: str
    inst_id: str
    action: str
    intensity_status: str
    intensity_score: float
    direction: int
    raw_return_5m: float | None
    raw_return_15m: float | None
    raw_return_1h: float | None
    continuation_return_5m: float | None
    continuation_return_15m: float | None
    continuation_return_1h: float | None
    reversal_return_5m: float | None
    reversal_return_15m: float | None
    reversal_return_1h: float | None
    label_status: str
    next_step: str


def build_liquidation_intensity_forward_labels(
    *,
    intensity_path: Path = ROOT / "current_okx_liquidation_intensity.csv",
    flow_path: Path = ROOT / "current_okx_liquidation_flow.csv",
) -> tuple[LiquidationIntensityForwardLabel, ...]:
    flow_by_key = {
        (row.get("asset", ""), row.get("action", "")): row
        for row in _read_rows(flow_path)
    }
    intensity_rows = tuple(
        row
        for row in _read_rows(intensity_path)
        if row.get("status") != "low_liquidation_intensity_context"
    )
    candles_by_asset = {
        asset: _fetch_okx_candles(f"{asset}-USDT-SWAP")
        for asset in sorted({row.get("asset", "") for row in intensity_rows if row.get("asset")})
    }
    labels: list[LiquidationIntensityForwardLabel] = []
    for row in intensity_rows:
        asset = row.get("asset", "")
        action = row.get("action", "")
        flow = flow_by_key.get((asset, action), {})
        timestamp = _parse_datetime(flow.get("latest_liquidation_at") or row.get("timestamp", ""))
        labels.append(
            _build_label(
                row=row,
                timestamp=timestamp,
                candles=candles_by_asset.get(asset, ()),
            )
        )
    return tuple(sorted(labels, key=_sort_key, reverse=True))


def write_liquidation_intensity_forward_labels(
    labels: tuple[LiquidationIntensityForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "event_timestamp",
                "asset",
                "inst_id",
                "action",
                "intensity_status",
                "intensity_score",
                "direction",
                "raw_return_5m",
                "raw_return_15m",
                "raw_return_1h",
                "continuation_return_5m",
                "continuation_return_15m",
                "continuation_return_1h",
                "reversal_return_5m",
                "reversal_return_15m",
                "reversal_return_1h",
                "label_status",
                "next_step",
            )
        )
        for label in labels:
            writer.writerow(
                (
                    label.event_timestamp,
                    label.asset,
                    label.inst_id,
                    label.action,
                    label.intensity_status,
                    f"{label.intensity_score:.8f}",
                    label.direction,
                    _format_optional(label.raw_return_5m),
                    _format_optional(label.raw_return_15m),
                    _format_optional(label.raw_return_1h),
                    _format_optional(label.continuation_return_5m),
                    _format_optional(label.continuation_return_15m),
                    _format_optional(label.continuation_return_1h),
                    _format_optional(label.reversal_return_5m),
                    _format_optional(label.reversal_return_15m),
                    _format_optional(label.reversal_return_1h),
                    label.label_status,
                    label.next_step,
                )
            )
    return output_path


def write_liquidation_intensity_forward_label_md(
    labels: tuple[LiquidationIntensityForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Intensity Forward Labels\n\n")
        handle.write(
            "This labels high liquidation/OI events from the current intensity screen. "
            "It is a continuation-versus-reversal check, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for label in labels[:top]:
            handle.write(
                "| "
                f"{label.asset} | "
                f"{label.action} | "
                f"{label.intensity_status} | "
                f"{label.direction} | "
                f"{label.intensity_score:.4f} | "
                f"{_format_optional(label.continuation_return_5m, digits=6)} | "
                f"{_format_optional(label.continuation_return_15m, digits=6)} | "
                f"{_format_optional(label.continuation_return_1h, digits=6)} | "
                f"{_format_optional(label.reversal_return_5m, digits=6)} | "
                f"{_format_optional(label.reversal_return_15m, digits=6)} | "
                f"{_format_optional(label.reversal_return_1h, digits=6)} | "
                f"{label.label_status} | "
                f"{label.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Continuation means price moved in the forced-flow direction implied by the liquidation event. "
            "Reversal means price moved against that direction. These labels still exclude spread, fees, "
            "funding PnL, fill probability, and adverse-excursion stops.\n"
        )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    timestamp: datetime,
    candles: tuple[dict[str, float], ...],
) -> LiquidationIntensityForwardLabel:
    action = row.get("action", "")
    direction = _direction_for_action(action)
    raw_5m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=5))
    raw_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    cont_5m = _directional(raw_5m, direction)
    cont_15m = _directional(raw_15m, direction)
    cont_1h = _directional(raw_1h, direction)
    rev_5m = _reversal(raw_5m, direction)
    rev_15m = _reversal(raw_15m, direction)
    rev_1h = _reversal(raw_1h, direction)
    label_status = _label_status(
        direction=direction,
        continuation_15m=cont_15m,
        continuation_1h=cont_1h,
        reversal_15m=rev_15m,
        reversal_1h=rev_1h,
    )
    asset = row.get("asset", "")
    return LiquidationIntensityForwardLabel(
        event_timestamp=timestamp.isoformat(),
        asset=asset,
        inst_id=row.get("inst_id", f"{asset}-USDT-SWAP"),
        action=action,
        intensity_status=row.get("status", ""),
        intensity_score=_float(row.get("intensity_score")),
        direction=direction,
        raw_return_5m=raw_5m,
        raw_return_15m=raw_15m,
        raw_return_1h=raw_1h,
        continuation_return_5m=cont_5m,
        continuation_return_15m=cont_15m,
        continuation_return_1h=cont_1h,
        reversal_return_5m=rev_5m,
        reversal_return_15m=rev_15m,
        reversal_return_1h=rev_1h,
        label_status=label_status,
        next_step=_next_step(asset=asset, action=action, label_status=label_status),
    )


def _fetch_okx_candles(inst_id: str) -> tuple[dict[str, float], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/candles",
        params={"instId": inst_id, "bar": "1m", "limit": "300"},
        timeout=30,
    )
    response.raise_for_status()
    rows = tuple(
        {
            "timestamp": float(row[0]),
            "end_timestamp": float(row[0]) + 60 * 1000,
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


def _directional(value: float | None, direction: int) -> float | None:
    if value is None or direction == 0:
        return None
    return value * direction


def _reversal(value: float | None, direction: int) -> float | None:
    if value is None or direction == 0:
        return None
    return value * -direction


def _label_status(
    *,
    direction: int,
    continuation_15m: float | None,
    continuation_1h: float | None,
    reversal_15m: float | None,
    reversal_1h: float | None,
) -> str:
    if direction == 0:
        return "mixed_direction_unlabeled"
    if continuation_15m is None and reversal_15m is None:
        return "label_pending"
    if continuation_15m is not None and continuation_15m > 0.0:
        if continuation_1h is None:
            return "continuation_15m_supported_pending_1h"
        if continuation_1h > 0.0:
            return "continuation_15m_1h_supported"
    if reversal_15m is not None and reversal_15m > 0.0:
        if reversal_1h is None:
            return "reversal_15m_supported_pending_1h"
        if reversal_1h > 0.0:
            return "reversal_15m_1h_supported"
    return "label_weak_or_conflicting"


def _next_step(*, asset: str, action: str, label_status: str) -> str:
    if label_status.endswith("pending_1h"):
        return f"wait for {asset} {action} 1h label, then add depth, fees, funding, and adverse-excursion checks"
    if label_status.endswith("supported"):
        return f"gate {asset} {action} with OKX depth, fees, funding, fill, and stop assumptions"
    if label_status == "mixed_direction_unlabeled":
        return f"do not promote {asset} until mixed liquidation direction is separated"
    return f"repeat {asset} {action} on a fresh liquidation/OI event before promotion"


def _sort_key(label: LiquidationIntensityForwardLabel) -> tuple[int, float, float]:
    status_rank = {
        "continuation_15m_1h_supported": 5,
        "reversal_15m_1h_supported": 5,
        "continuation_15m_supported_pending_1h": 4,
        "reversal_15m_supported_pending_1h": 4,
        "label_pending": 3,
        "label_weak_or_conflicting": 2,
        "mixed_direction_unlabeled": 1,
    }.get(label.label_status, 0)
    best_15m = max(
        label.continuation_return_15m if label.continuation_return_15m is not None else -1.0,
        label.reversal_return_15m if label.reversal_return_15m is not None else -1.0,
    )
    return (status_rank, best_15m, label.intensity_score)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_optional(value: float | None, *, digits: int = 8) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity-path", type=Path, default=ROOT / "current_okx_liquidation_intensity.csv")
    parser.add_argument("--flow-path", type=Path, default=ROOT / "current_okx_liquidation_flow.csv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_intensity_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_intensity_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    labels = build_liquidation_intensity_forward_labels(
        intensity_path=args.intensity_path,
        flow_path=args.flow_path,
    )
    write_liquidation_intensity_forward_labels(labels, output_path=args.output_path)
    write_liquidation_intensity_forward_label_md(labels, output_path=args.md_output_path, top=args.top)
    for label in labels[: args.top]:
        print(
            label.asset,
            label.action,
            label.label_status,
            f"cont15={_format_optional(label.continuation_return_15m, digits=4)}",
            f"rev15={_format_optional(label.reversal_return_15m, digits=4)}",
        )


if __name__ == "__main__":
    main()
