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
class FollowupOkxRepeatForwardLabel:
    timestamp: str
    asset: str
    source: str
    source_action: str
    direction: int
    priority: float
    inst_id: str
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    label_status: str


def build_followup_okx_repeat_forward_labels(
    *,
    input_path: Path = ROOT / "current_followup_okx_repeat_observations.csv",
) -> tuple[FollowupOkxRepeatForwardLabel, ...]:
    observations = tuple(
        row for row in _read_rows(input_path) if row.get("observation_status") == "ready_for_label"
    )
    candles_by_inst = {
        inst_id: _fetch_okx_candles(inst_id)
        for inst_id in sorted({row["inst_id"] for row in observations})
    }
    rows = tuple(
        _build_label(row=row, candles=candles_by_inst.get(row["inst_id"], ()))
        for row in observations
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_followup_okx_repeat_forward_labels(
    rows: tuple[FollowupOkxRepeatForwardLabel, ...],
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
                "source",
                "source_action",
                "direction",
                "priority",
                "inst_id",
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
                    row.timestamp,
                    row.asset,
                    row.source,
                    row.source_action,
                    row.direction,
                    f"{row.priority:.4f}",
                    row.inst_id,
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


def write_followup_okx_repeat_forward_labels_md(
    rows: tuple[FollowupOkxRepeatForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up OKX Repeat Forward Labels\n\n")
        handle.write(
            "This labels OKX source-specific follow-up observations against OKX "
            "15m candles. It is not net PnL.\n\n"
        )
        handle.write(
            "| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.source_action} | "
                f"{row.direction} | "
                f"{row.priority:.4f} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{row.label_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Pending rows have not matured. Positive directional return means the "
            "source-specific direction was right on OKX before fees and slippage.\n"
        )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
) -> FollowupOkxRepeatForwardLabel:
    timestamp = _parse_datetime(row["timestamp"])
    direction = int(row.get("direction") or "0")
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return FollowupOkxRepeatForwardLabel(
        timestamp=timestamp.isoformat(),
        asset=row["asset"],
        source=row["source"],
        source_action=row.get("source_action", ""),
        direction=direction,
        priority=float(row.get("priority") or "0"),
        inst_id=row["inst_id"],
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


def _label_status(*, raw_return_15m: float | None, raw_return_1h: float | None) -> str:
    if raw_return_15m is None:
        return "pending_15m"
    if raw_return_1h is None:
        return "labeled_15m_pending_1h"
    return "labeled_1h"


def _sort_key(row: FollowupOkxRepeatForwardLabel) -> tuple[bool, float, float]:
    return (
        row.directional_return_15m is not None,
        row.directional_return_15m or -1.0,
        row.priority,
    )


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
        "--input-path",
        type=Path,
        default=ROOT / "current_followup_okx_repeat_observations.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_okx_repeat_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_okx_repeat_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_followup_okx_repeat_forward_labels(input_path=args.input_path)
    write_followup_okx_repeat_forward_labels(rows, output_path=args.output_path)
    write_followup_okx_repeat_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.source,
            row.label_status,
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.4f}'}",
        )


if __name__ == "__main__":
    main()
