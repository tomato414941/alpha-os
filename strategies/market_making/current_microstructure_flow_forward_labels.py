from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from strategies.market_making.current_l2_imbalance_forward_labels import (
    _fetch_hyperliquid_candles,
    _forward_return,
    _parse_datetime,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MicrostructureFlowForwardLabel:
    timestamp: str
    asset: str
    action: str
    direction: int
    pressure_score: float
    book_imbalance_10bps: float
    trade_imbalance: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    label_status: str


def build_microstructure_flow_forward_labels(
    *,
    input_path: Path = ROOT / "current_microstructure_flow_snapshot.csv",
) -> tuple[MicrostructureFlowForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_asset = {
        asset: _fetch_hyperliquid_candles(asset)
        for asset in sorted({row["asset"] for row in rows})
    }
    labels = tuple(
        _build_label(row=row, candles=candles_by_asset.get(row["asset"], ()))
        for row in rows
    )
    return tuple(sorted(labels, key=_sort_key, reverse=True))


def write_microstructure_flow_forward_labels(
    rows: tuple[MicrostructureFlowForwardLabel, ...],
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
                "pressure_score",
                "book_imbalance_10bps",
                "trade_imbalance",
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
                    row.action,
                    row.direction,
                    f"{row.pressure_score:.8f}",
                    f"{row.book_imbalance_10bps:.8f}",
                    f"{row.trade_imbalance:.8f}",
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


def write_microstructure_flow_forward_labels_md(
    rows: tuple[MicrostructureFlowForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Microstructure Flow Forward Labels\n\n")
        handle.write(
            "This labels book-plus-trade microstructure observations against "
            "Hyperliquid 15m and 1h forward returns. It is not net PnL.\n\n"
        )
        handle.write(
            "| asset | action | dir | pressure | book imb | trade imb | raw 15m | dir 15m | raw 1h | dir 1h | status |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.direction} | "
                f"{row.pressure_score:.4f} | "
                f"{row.book_imbalance_10bps:.4f} | "
                f"{row.trade_imbalance:.4f} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{row.label_status} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Positive directional return means the microstructure direction was "
            "right before fees and slippage. Compare aligned pressure rows against "
            "book/trade divergence rows before promoting the feature.\n"
        )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
) -> MicrostructureFlowForwardLabel:
    timestamp = _parse_datetime(row["timestamp"])
    direction = int(row["direction"])
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return MicrostructureFlowForwardLabel(
        timestamp=timestamp.isoformat(),
        asset=row["asset"],
        action=row["action"],
        direction=direction,
        pressure_score=float(row["pressure_score"]),
        book_imbalance_10bps=float(row["book_imbalance_10bps"]),
        trade_imbalance=float(row["trade_imbalance"]),
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


def _label_status(*, raw_return_15m: float | None, raw_return_1h: float | None) -> str:
    if raw_return_15m is None:
        return "pending_15m"
    if raw_return_1h is None:
        return "labeled_15m_pending_1h"
    return "labeled_1h"


def _sort_key(row: MicrostructureFlowForwardLabel) -> tuple[bool, float, float]:
    return (
        row.directional_return_15m is not None,
        row.directional_return_15m or -1.0,
        abs(row.pressure_score),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_microstructure_flow_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_microstructure_flow_forward_labels(input_path=args.input_path)
    write_microstructure_flow_forward_labels(rows, output_path=args.output_path)
    write_microstructure_flow_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            row.label_status,
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'}",
        )


if __name__ == "__main__":
    main()
