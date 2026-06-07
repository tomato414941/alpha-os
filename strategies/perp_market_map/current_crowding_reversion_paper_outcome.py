from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class CrowdingPaperOutcomeRow:
    entry_timestamp: str
    asset: str
    action: str
    direction: int
    candidate_size_usd: float
    entry_mid_price: float
    conservative_cost_bps: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    net_15m_bps: float | None
    net_1h_bps: float | None
    outcome_15m: str
    outcome_1h: str


def build_crowding_paper_outcomes(
    *,
    execution_check_path: Path = ROOT / "current_crowding_reversion_execution_check.csv",
) -> tuple[CrowdingPaperOutcomeRow, ...]:
    gate_rows = _selected_execution_rows(execution_check_path)
    candles_by_asset = {
        asset: _fetch_hyperliquid_candles(asset)
        for asset in sorted({row["asset"] for row in gate_rows})
    }
    rows = tuple(
        _build_outcome(row=row, candles=candles_by_asset.get(row["asset"], ()))
        for row in gate_rows
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_crowding_paper_outcomes_csv(
    rows: tuple[CrowdingPaperOutcomeRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "entry_timestamp",
                "asset",
                "action",
                "direction",
                "candidate_size_usd",
                "entry_mid_price",
                "conservative_cost_bps",
                "raw_return_15m",
                "raw_return_1h",
                "directional_return_15m",
                "directional_return_1h",
                "net_15m_bps",
                "net_1h_bps",
                "outcome_15m",
                "outcome_1h",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.entry_timestamp,
                    row.asset,
                    row.action,
                    row.direction,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.entry_mid_price:.12f}",
                    f"{row.conservative_cost_bps:.8f}",
                    _optional_float(row.raw_return_15m),
                    _optional_float(row.raw_return_1h),
                    _optional_float(row.directional_return_15m),
                    _optional_float(row.directional_return_1h),
                    _optional_float(row.net_15m_bps),
                    _optional_float(row.net_1h_bps),
                    row.outcome_15m,
                    row.outcome_1h,
                )
            )
    return output_path


def write_crowding_paper_outcomes_md(
    rows: tuple[CrowdingPaperOutcomeRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    covered_15m = tuple(row for row in rows if row.net_15m_bps is not None)
    covered_1h = tuple(row for row in rows if row.net_1h_bps is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Reversion Paper Outcome\n\n")
        handle.write(
            "This labels depth-gated Hyperliquid carry-reversion probes after the "
            "same rough cost proxy used by the execution check. It is still a "
            "paper observation, not a live fill.\n\n"
        )
        handle.write(f"- rows: `{len(rows)}`\n")
        handle.write(f"- covered 15m: `{len(covered_15m)}`\n")
        handle.write(f"- covered 1h: `{len(covered_1h)}`\n\n")
        handle.write(
            "| entry | asset | action | size | cost bps | net15 bps | out15 | net1h bps | out1h |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.entry_timestamp} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{row.conservative_cost_bps:.2f} | "
                f"{_optional_display(row.net_15m_bps)} | "
                f"{row.outcome_15m} | "
                f"{_optional_display(row.net_1h_bps)} | "
                f"{row.outcome_1h} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_15m_win` or `paper_1h_win` only means the price moved in the "
            "candidate direction after the rough cost proxy. It still excludes "
            "actual order placement, queue position, partial fills, funding timing, "
            "mark/index basis, and stop behavior.\n"
        )
    return output_path


def _selected_execution_rows(path: Path) -> tuple[dict[str, str], ...]:
    rows = tuple(
        row
        for row in _read_rows(path)
        if row.get("gate_action") == "paper_execution_probe"
    )
    best_by_asset_action: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=_execution_sort_key, reverse=True):
        key = (row.get("asset", ""), row.get("action", ""))
        if not key[0] or not key[1] or key in best_by_asset_action:
            continue
        best_by_asset_action[key] = row
    return tuple(best_by_asset_action.values())


def _build_outcome(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
) -> CrowdingPaperOutcomeRow:
    entry_timestamp = _parse_datetime(row["timestamp"])
    direction = _direction_for_action(row["action"])
    raw_return_15m = _forward_return(candles, entry_timestamp, entry_timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, entry_timestamp, entry_timestamp + timedelta(hours=1))
    directional_return_15m = None if raw_return_15m is None else raw_return_15m * direction
    directional_return_1h = None if raw_return_1h is None else raw_return_1h * direction
    cost_bps = _float(row.get("conservative_cost_bps"))
    net_15m_bps = _net_bps(directional_return_15m, cost_bps=cost_bps)
    net_1h_bps = _net_bps(directional_return_1h, cost_bps=cost_bps)
    return CrowdingPaperOutcomeRow(
        entry_timestamp=entry_timestamp.isoformat(),
        asset=row["asset"],
        action=row["action"],
        direction=direction,
        candidate_size_usd=_float(row.get("candidate_size_usd")),
        entry_mid_price=_float(row.get("mid_price")),
        conservative_cost_bps=cost_bps,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        directional_return_15m=directional_return_15m,
        directional_return_1h=directional_return_1h,
        net_15m_bps=net_15m_bps,
        net_1h_bps=net_1h_bps,
        outcome_15m=_outcome(net_15m_bps, horizon="15m"),
        outcome_1h=_outcome(net_1h_bps, horizon="1h"),
    )


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
            "timestamp": float(candle["t"]),
            "end_timestamp": float(candle["T"]),
            "close": float(candle["c"]),
        }
        for candle in response.json()
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


def _execution_sort_key(row: dict[str, str]) -> tuple[float, float]:
    return (
        _float(row.get("conservative_net_1h_bps")),
        -_float(row.get("candidate_size_usd")),
    )


def _sort_key(row: CrowdingPaperOutcomeRow) -> tuple[bool, float, bool, float]:
    return (
        row.net_15m_bps is not None,
        row.net_15m_bps or -1_000_000.0,
        row.net_1h_bps is not None,
        row.net_1h_bps or -1_000_000.0,
    )


def _net_bps(value: float | None, *, cost_bps: float) -> float | None:
    return None if value is None else (value * 10_000.0) - cost_bps


def _outcome(value: float | None, *, horizon: str) -> str:
    if value is None:
        return f"pending_{horizon}"
    if value > 0.0:
        return f"paper_{horizon}_win"
    return f"paper_{horizon}_loss"


def _direction_for_action(action: str) -> int:
    if action.startswith("long_"):
        return 1
    if action.startswith("short_"):
        return -1
    return 0


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _optional_display(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-check-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_execution_check.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_paper_outcome.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_paper_outcome.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_crowding_paper_outcomes(execution_check_path=args.execution_check_path)
    write_crowding_paper_outcomes_csv(rows, output_path=args.output_path)
    write_crowding_paper_outcomes_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"size={row.candidate_size_usd:.0f}",
            f"net15={_optional_display(row.net_15m_bps)}",
            row.outcome_15m,
            f"net1h={_optional_display(row.net_1h_bps)}",
            row.outcome_1h,
        )


if __name__ == "__main__":
    main()
