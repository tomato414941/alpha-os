from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChainTvlFlowForwardLabel:
    timestamp: str
    venue: str
    chain: str
    token_symbol: str
    action: str
    direction: int
    week_change_pct: float
    day_change_pct: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    label_status: str


def build_chain_tvl_flow_forward_labels(
    *,
    coverage_path: Path = ROOT / "current_chain_tvl_flow_venue_coverage.csv",
    max_rows: int = 24,
) -> tuple[ChainTvlFlowForwardLabel, ...]:
    coverage_rows = tuple(
        row
        for row in _read_rows(coverage_path)
        if int(row.get("venue_count") or "0") > 0
    )[:max_rows]
    labels: list[ChainTvlFlowForwardLabel] = []
    for row in coverage_rows:
        direction = _direction_for_action(row["action"])
        if direction == 0:
            continue
        timestamp = _parse_datetime(row["timestamp"])
        token = row["token_symbol"]
        if row.get("hyperliquid_perp") == "True":
            labels.append(
                _build_label(
                    timestamp=timestamp,
                    venue="HL",
                    chain=row["chain"],
                    token_symbol=token,
                    action=row["action"],
                    direction=direction,
                    week_change_pct=float(row.get("week_change_pct") or "0"),
                    day_change_pct=float(row.get("day_change_pct") or "0"),
                    candles=_fetch_hyperliquid_candles(token),
                )
            )
        if row.get("okx_swap") == "True":
            labels.append(
                _build_label(
                    timestamp=timestamp,
                    venue="OKX",
                    chain=row["chain"],
                    token_symbol=token,
                    action=row["action"],
                    direction=direction,
                    week_change_pct=float(row.get("week_change_pct") or "0"),
                    day_change_pct=float(row.get("day_change_pct") or "0"),
                    candles=_fetch_okx_candles(f"{token}-USDT-SWAP"),
                )
            )
    return tuple(sorted(labels, key=_sort_key, reverse=True))


def write_chain_tvl_flow_forward_labels_csv(
    rows: tuple[ChainTvlFlowForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "venue",
                "chain",
                "token_symbol",
                "action",
                "direction",
                "week_change_pct",
                "day_change_pct",
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
                    row.venue,
                    row.chain,
                    row.token_symbol,
                    row.action,
                    row.direction,
                    f"{row.week_change_pct:.8f}",
                    f"{row.day_change_pct:.8f}",
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


def write_chain_tvl_flow_forward_labels_md(
    rows: tuple[ChainTvlFlowForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = tuple(row for row in rows if row.directional_return_15m is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain TVL Flow Forward Labels\n\n")
        handle.write(
            "This labels tradable chain TVL flow candidates against subsequent token returns. "
            "Positive directional return means the action-specific direction was right before costs.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- labeled 15m rows: `{len(labeled)}`\n\n")
        handle.write(
            "| venue | chain | token | action | dir | week % | day % | raw 15m | dir 15m | raw 1h | dir 1h | status |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.venue} | "
                f"{row.chain} | "
                f"{row.token_symbol} | "
                f"{row.action} | "
                f"{row.direction} | "
                f"{row.week_change_pct:.4f} | "
                f"{row.day_change_pct:.4f} | "
                f"{'' if row.raw_return_15m is None else f'{row.raw_return_15m:.6f}'} | "
                f"{'' if row.directional_return_15m is None else f'{row.directional_return_15m:.6f}'} | "
                f"{'' if row.raw_return_1h is None else f'{row.raw_return_1h:.6f}'} | "
                f"{'' if row.directional_return_1h is None else f'{row.directional_return_1h:.6f}'} | "
                f"{row.label_status} |\n"
            )
    return output_path


def _build_label(
    *,
    timestamp: datetime,
    venue: str,
    chain: str,
    token_symbol: str,
    action: str,
    direction: int,
    week_change_pct: float,
    day_change_pct: float,
    candles: tuple[dict[str, float], ...],
) -> ChainTvlFlowForwardLabel:
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    return ChainTvlFlowForwardLabel(
        timestamp=timestamp.isoformat(),
        venue=venue,
        chain=chain,
        token_symbol=token_symbol,
        action=action,
        direction=direction,
        week_change_pct=week_change_pct,
        day_change_pct=day_change_pct,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=(
            None if raw_return_15m is None else raw_return_15m * direction
        ),
        directional_return_1h=None if raw_return_1h is None else raw_return_1h * direction,
        label_status=_label_status(raw_return_15m=raw_return_15m, raw_return_1h=raw_return_1h),
    )


def _fetch_hyperliquid_candles(asset: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=3)
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


def _direction_for_action(action: str) -> int:
    if action == "chain_inflow_momentum_watch":
        return 1
    if action == "chain_outflow_stress_watch":
        return -1
    if action == "chain_flow_reversal_watch":
        return 1
    return 0


def _label_status(*, raw_return_15m: float | None, raw_return_1h: float | None) -> str:
    if raw_return_15m is None:
        return "pending_15m"
    if raw_return_1h is None:
        return "labeled_15m_pending_1h"
    return "labeled_1h"


def _sort_key(row: ChainTvlFlowForwardLabel) -> tuple[bool, float, float]:
    return (
        row.directional_return_15m is not None,
        row.directional_return_15m or -1.0,
        abs(row.week_change_pct),
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
        "--coverage-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_venue_coverage.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_forward_labels.md",
    )
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_chain_tvl_flow_forward_labels(
        coverage_path=args.coverage_path,
        max_rows=args.max_rows,
    )
    write_chain_tvl_flow_forward_labels_csv(rows, output_path=args.output_path)
    write_chain_tvl_flow_forward_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.venue,
            row.token_symbol,
            row.action,
            row.label_status,
            f"dir15={'' if row.directional_return_15m is None else f'{row.directional_return_15m:.4f}'}",
        )


if __name__ == "__main__":
    main()
