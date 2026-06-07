from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests


BITBO_BTC_ETF_FLOWS_URL = "https://bitbo.io/treasuries/etf-flows"


@dataclass(frozen=True)
class BtcEtfFlowDay:
    timestamp: str
    flow_date: str
    flow_btc: float
    flow_usd: float


@dataclass(frozen=True)
class BtcEtfFlowSnapshot:
    timestamp: str
    latest_date: str
    latest_flow_btc: float
    latest_flow_usd: float
    rolling_5d_flow_btc: float
    rolling_10d_flow_btc: float
    inflow_streak_days: int
    outflow_streak_days: int
    action: str
    score: float


def fetch_btc_etf_flow_days(
    *,
    url: str = BITBO_BTC_ETF_FLOWS_URL,
    now: datetime | None = None,
) -> tuple[BtcEtfFlowDay, ...]:
    now = now or datetime.now(tz=UTC)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return _parse_bitbo_history_usd(response.text, timestamp=now.isoformat())


def build_snapshot(days: tuple[BtcEtfFlowDay, ...]) -> BtcEtfFlowSnapshot:
    ordered = tuple(sorted(days, key=lambda row: row.flow_date))
    if not ordered:
        return BtcEtfFlowSnapshot("", "", 0.0, 0.0, 0.0, 0.0, 0, 0, "missing_etf_flow", 0.0)
    latest = ordered[-1]
    rolling_5d = sum(row.flow_btc for row in ordered[-5:])
    rolling_10d = sum(row.flow_btc for row in ordered[-10:])
    inflow_streak = _signed_streak(ordered, positive=True)
    outflow_streak = _signed_streak(ordered, positive=False)
    action = _action_for_snapshot(
        latest_flow_btc=latest.flow_btc,
        rolling_5d_flow_btc=rolling_5d,
        inflow_streak_days=inflow_streak,
        outflow_streak_days=outflow_streak,
    )
    score = _score_snapshot(
        latest_flow_btc=latest.flow_btc,
        rolling_5d_flow_btc=rolling_5d,
        rolling_10d_flow_btc=rolling_10d,
        inflow_streak_days=inflow_streak,
        outflow_streak_days=outflow_streak,
    )
    return BtcEtfFlowSnapshot(
        timestamp=latest.timestamp,
        latest_date=latest.flow_date,
        latest_flow_btc=latest.flow_btc,
        latest_flow_usd=latest.flow_usd,
        rolling_5d_flow_btc=rolling_5d,
        rolling_10d_flow_btc=rolling_10d,
        inflow_streak_days=inflow_streak,
        outflow_streak_days=outflow_streak,
        action=action,
        score=score,
    )


def write_flow_days(days: tuple[BtcEtfFlowDay, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "flow_date", "flow_btc", "flow_usd"))
        for row in days:
            writer.writerow(
                (
                    row.timestamp,
                    row.flow_date,
                    f"{row.flow_btc:.8f}",
                    f"{row.flow_usd:.2f}",
                )
            )
    return output_path


def write_snapshot(snapshot: BtcEtfFlowSnapshot, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "latest_date",
                "latest_flow_btc",
                "latest_flow_usd",
                "rolling_5d_flow_btc",
                "rolling_10d_flow_btc",
                "inflow_streak_days",
                "outflow_streak_days",
                "action",
                "score",
            )
        )
        writer.writerow(
            (
                snapshot.timestamp,
                snapshot.latest_date,
                f"{snapshot.latest_flow_btc:.8f}",
                f"{snapshot.latest_flow_usd:.2f}",
                f"{snapshot.rolling_5d_flow_btc:.8f}",
                f"{snapshot.rolling_10d_flow_btc:.8f}",
                snapshot.inflow_streak_days,
                snapshot.outflow_streak_days,
                snapshot.action,
                f"{snapshot.score:.8f}",
            )
        )
    return output_path


def write_markdown(snapshot: BtcEtfFlowSnapshot, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current BTC ETF Flow Snapshot\n\n")
        handle.write(
            "This extracts Bitcoin spot ETF flow context from Bitbo. It is institutional demand context, not a trade instruction.\n\n"
        )
        handle.write("| latest date | latest BTC | latest USD | 5d BTC | 10d BTC | inflow streak | outflow streak | action | score |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
        handle.write(
            f"| {snapshot.latest_date} | {snapshot.latest_flow_btc:.8f} | "
            f"{snapshot.latest_flow_usd:.2f} | {snapshot.rolling_5d_flow_btc:.8f} | "
            f"{snapshot.rolling_10d_flow_btc:.8f} | {snapshot.inflow_streak_days} | "
            f"{snapshot.outflow_streak_days} | {snapshot.action} | {snapshot.score:.6f} |\n"
        )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "ETF flows can indicate slower institutional accumulation or distribution. The next step is to join this context to BTC/ETH market state and label forward returns by regime.\n"
        )
    return output_path


def _parse_bitbo_history_usd(html: str, *, timestamp: str) -> tuple[BtcEtfFlowDay, ...]:
    block_match = re.search(r"const historyUsd = \[(.*?)\n\s*\];", html, re.S)
    if block_match is None:
        return ()
    price_match = re.search(r"truncate\([^*]+\*\s*([0-9.]+),\s*2\)", block_match.group(1))
    price = float(price_match.group(1)) if price_match else 0.0
    pattern = re.compile(
        r"getPreviousBusinessDay\((\d+)\),\s*truncate\(([+-]?\d+(?:\.\d+)?)\s*\*\s*([0-9.]+),\s*2\)"
    )
    rows = []
    for match in pattern.finditer(block_match.group(1)):
        ms_timestamp = int(match.group(1))
        flow_btc = float(match.group(2))
        row_price = float(match.group(3)) if match.group(3) else price
        rows.append(
            BtcEtfFlowDay(
                timestamp=timestamp,
                flow_date=_previous_business_day(ms_timestamp).isoformat(),
                flow_btc=flow_btc,
                flow_usd=flow_btc * row_price,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.flow_date))


def _previous_business_day(ms_timestamp: int) -> date:
    day = datetime.fromtimestamp(ms_timestamp / 1000, tz=UTC).date()
    weekday = day.weekday()
    if weekday == 6:
        return day - timedelta(days=2)
    if weekday == 0:
        return day - timedelta(days=3)
    return day - timedelta(days=1)


def _signed_streak(days: tuple[BtcEtfFlowDay, ...], *, positive: bool) -> int:
    streak = 0
    for row in reversed(days):
        if positive and row.flow_btc > 0.0:
            streak += 1
        elif not positive and row.flow_btc < 0.0:
            streak += 1
        else:
            break
    return streak


def _action_for_snapshot(
    *,
    latest_flow_btc: float,
    rolling_5d_flow_btc: float,
    inflow_streak_days: int,
    outflow_streak_days: int,
) -> str:
    if latest_flow_btc > 0.0 and rolling_5d_flow_btc > 0.0 and inflow_streak_days >= 2:
        return "btc_etf_accumulation_regime"
    if latest_flow_btc < 0.0 and rolling_5d_flow_btc < 0.0 and outflow_streak_days >= 2:
        return "btc_etf_distribution_regime"
    if rolling_5d_flow_btc > 0.0:
        return "btc_etf_inflow_context"
    if rolling_5d_flow_btc < 0.0:
        return "btc_etf_outflow_context"
    return "btc_etf_neutral_context"


def _score_snapshot(
    *,
    latest_flow_btc: float,
    rolling_5d_flow_btc: float,
    rolling_10d_flow_btc: float,
    inflow_streak_days: int,
    outflow_streak_days: int,
) -> float:
    magnitude = (
        abs(latest_flow_btc) / 1_000.0
        + abs(rolling_5d_flow_btc) / 3_000.0
        + abs(rolling_10d_flow_btc) / 5_000.0
    )
    streak = max(inflow_streak_days, outflow_streak_days) * 0.25
    return magnitude + streak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BITBO_BTC_ETF_FLOWS_URL)
    parser.add_argument(
        "--history-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_flow_history.csv",
    )
    parser.add_argument(
        "--snapshot-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_flow_snapshot.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_flow_snapshot.md",
    )
    args = parser.parse_args()

    days = fetch_btc_etf_flow_days(url=args.url)
    snapshot = build_snapshot(days)
    write_flow_days(days, output_path=args.history_output_path)
    write_snapshot(snapshot, output_path=args.snapshot_output_path)
    write_markdown(snapshot, output_path=args.markdown_output_path)
    print(
        snapshot.latest_date,
        snapshot.action,
        f"latest_btc={snapshot.latest_flow_btc:.2f}",
        f"five_day_btc={snapshot.rolling_5d_flow_btc:.2f}",
        f"ten_day_btc={snapshot.rolling_10d_flow_btc:.2f}",
        f"score={snapshot.score:.4f}",
    )


if __name__ == "__main__":
    main()
