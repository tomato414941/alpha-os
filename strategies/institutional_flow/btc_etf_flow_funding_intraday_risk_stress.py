from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests

from btc_etf_flow_funding_candidate_rule import _is_large_outflow_start_funding_signal
from btc_etf_flow_funding_regime_summary import ROOT, build_funding_enriched_label_rows


BINANCE_UM_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"


@dataclass(frozen=True)
class HourlyBar:
    timestamp: datetime
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class IntradayRiskStressRow:
    group_key: str
    trades: int
    mean_price_net_return: float
    hit_rate: float
    mean_max_adverse_excursion: float
    max_adverse_excursion: float
    liquidation_risk_2x: int
    liquidation_risk_3x: int
    liquidation_risk_5x: int
    action: str


def build_intraday_risk_stress_rows(
    *,
    labels_path: Path,
    entry_offset_hours: tuple[int, ...],
    fee_bps_per_side: float,
    max_workers: int = 24,
) -> tuple[IntradayRiskStressRow, ...]:
    signal_rows = tuple(
        row
        for row in build_funding_enriched_label_rows(
            labels_path=labels_path,
            max_workers=max_workers,
        )
        if _is_large_outflow_start_funding_signal(row)
    )
    if not signal_rows:
        return ()
    needed_days = _needed_hourly_days(signal_rows, entry_offset_hours=entry_offset_hours)
    bars_by_time = _fetch_btc_hourly_bars(days=needed_days, max_workers=max_workers)
    return tuple(
        _risk_stress_row(
            group_key=f"entry_offset_hours_{offset_hours}",
            trades=_non_overlapping_risk_trades(
                signal_rows=signal_rows,
                entry_offset_hours=offset_hours,
                bars_by_time=bars_by_time,
                fee_bps_per_side=fee_bps_per_side,
            ),
        )
        for offset_hours in entry_offset_hours
    )


def write_risk_csv(
    rows: tuple[IntradayRiskStressRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_key",
                "trades",
                "mean_price_net_return",
                "hit_rate",
                "mean_max_adverse_excursion",
                "max_adverse_excursion",
                "liquidation_risk_2x",
                "liquidation_risk_3x",
                "liquidation_risk_5x",
                "action",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.group_key,
                    row.trades,
                    f"{row.mean_price_net_return:.8f}",
                    f"{row.hit_rate:.8f}",
                    f"{row.mean_max_adverse_excursion:.8f}",
                    f"{row.max_adverse_excursion:.8f}",
                    row.liquidation_risk_2x,
                    row.liquidation_risk_3x,
                    row.liquidation_risk_5x,
                    row.action,
                )
            )
    return output_path


def write_risk_md(
    rows: tuple[IntradayRiskStressRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Funding Intraday Risk Stress\n\n")
        handle.write(
            "This measures 1h mark-to-market adverse excursion for the BTC ETF-flow/funding short candidate. "
            "For a short, adverse excursion is the largest high above the entry close during the 120-hour hold. "
            "Liquidation columns are rough flags using 50%, 33.3%, and 20% adverse moves for 2x, 3x, and 5x leverage.\n\n"
        )
        handle.write(
            "| group | trades | mean price net | hit | mean adverse | max adverse | liq 2x | liq 3x | liq 5x | action |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.group_key} | {row.trades} | {row.mean_price_net_return:.8f} | "
                f"{row.hit_rate:.4f} | {row.mean_max_adverse_excursion:.8f} | "
                f"{row.max_adverse_excursion:.8f} | {row.liquidation_risk_2x} | "
                f"{row.liquidation_risk_3x} | {row.liquidation_risk_5x} | {row.action} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "This is not an exchange liquidation model. Maintenance margin, mark/index divergence, funding timestamps, and stop fills are still ignored. "
            "The purpose is to see whether the candidate requires obviously unsafe leverage.\n"
        )
    return output_path


def _needed_hourly_days(
    signal_rows: tuple[dict[str, str], ...],
    *,
    entry_offset_hours: tuple[int, ...],
) -> tuple[date, ...]:
    days: set[date] = set()
    max_offset = max(entry_offset_hours)
    for row in signal_rows:
        start = datetime.combine(
            date.fromisoformat(row["label_start_date"]),
            datetime.min.time(),
            tzinfo=UTC,
        )
        first_day = start.date()
        last_day = (start + timedelta(hours=max_offset + 120)).date()
        days.update(first_day + timedelta(days=offset) for offset in range((last_day - first_day).days + 1))
    return tuple(sorted(days))


def _non_overlapping_risk_trades(
    *,
    signal_rows: tuple[dict[str, str], ...],
    entry_offset_hours: int,
    bars_by_time: dict[datetime, HourlyBar],
    fee_bps_per_side: float,
) -> tuple[tuple[float, float], ...]:
    trades: list[tuple[float, float]] = []
    next_available_entry = datetime.min.replace(tzinfo=UTC)
    round_trip_fee = (fee_bps_per_side * 2.0) / 10_000.0
    for row in sorted(signal_rows, key=lambda item: item["label_start_date"]):
        entry_at = datetime.combine(
            date.fromisoformat(row["label_start_date"]),
            datetime.min.time(),
            tzinfo=UTC,
        ) + timedelta(hours=entry_offset_hours)
        if entry_at < next_available_entry:
            continue
        exit_at = entry_at + timedelta(hours=120)
        entry_bar = bars_by_time.get(entry_at)
        exit_bar = bars_by_time.get(exit_at)
        if entry_bar is None or exit_bar is None or entry_bar.close <= 0.0:
            continue
        bars = tuple(
            bar
            for timestamp, bar in bars_by_time.items()
            if entry_at <= timestamp <= exit_at
        )
        if not bars:
            continue
        raw_return = (exit_bar.close / entry_bar.close) - 1.0
        net_return = (raw_return * int(row["direction_hint"])) - round_trip_fee
        max_adverse_excursion = max((bar.high / entry_bar.close) - 1.0 for bar in bars)
        trades.append((net_return, max_adverse_excursion))
        next_available_entry = exit_at
    return tuple(trades)


def _fetch_btc_hourly_bars(
    *,
    days: tuple[date, ...],
    max_workers: int,
) -> dict[datetime, HourlyBar]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows_by_day = tuple(executor.map(_fetch_btc_hourly_bar_rows, days))
    return {bar.timestamp: bar for rows in rows_by_day for bar in rows}


def _fetch_btc_hourly_bar_rows(day: date) -> tuple[HourlyBar, ...]:
    response = requests.get(_hourly_kline_url(day), timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    rows: list[HourlyBar] = []
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            for item in csv.reader(TextIOWrapper(handle, encoding="utf-8")):
                if not item or item[0] == "open_time":
                    continue
                rows.append(
                    HourlyBar(
                        timestamp=_ms_to_datetime(int(item[0])),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                    )
                )
    return tuple(rows)


def _hourly_kline_url(day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/klines/BTCUSDT/1h/BTCUSDT-1h-{day:%Y-%m-%d}.zip"


def _risk_stress_row(
    *,
    group_key: str,
    trades: tuple[tuple[float, float], ...],
) -> IntradayRiskStressRow:
    net_returns = tuple(trade[0] for trade in trades)
    adverse_excursions = tuple(trade[1] for trade in trades)
    row = IntradayRiskStressRow(
        group_key=group_key,
        trades=len(trades),
        mean_price_net_return=_mean(net_returns),
        hit_rate=_hit_rate(net_returns),
        mean_max_adverse_excursion=_mean(adverse_excursions),
        max_adverse_excursion=max(adverse_excursions, default=0.0),
        liquidation_risk_2x=sum(1 for value in adverse_excursions if value >= 0.50),
        liquidation_risk_3x=sum(1 for value in adverse_excursions if value >= 1.0 / 3.0),
        liquidation_risk_5x=sum(1 for value in adverse_excursions if value >= 0.20),
        action="",
    )
    return IntradayRiskStressRow(
        **{
            **row.__dict__,
            "action": _action_for_risk(row),
        }
    )


def _action_for_risk(row: IntradayRiskStressRow) -> str:
    if row.trades >= 10 and row.liquidation_risk_5x == 0 and row.max_adverse_excursion < 0.20:
        return "survives_5x_buffer"
    if row.trades >= 10 and row.liquidation_risk_3x == 0:
        return "survives_3x_buffer"
    if row.trades >= 10 and row.liquidation_risk_2x == 0:
        return "survives_2x_buffer"
    return "leverage_risk"


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return sum(1.0 for value in values if value > 0.0) / len(values) if values else 0.0


def _ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ROOT / "btc_etf_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_intraday_risk_stress.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_intraday_risk_stress.md",
    )
    parser.add_argument("--entry-offset-hours", nargs="+", type=int, default=[0, 8, 16, 24, 32, 48])
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()

    rows = build_intraday_risk_stress_rows(
        labels_path=args.labels_path,
        entry_offset_hours=tuple(args.entry_offset_hours),
        fee_bps_per_side=args.fee_bps_per_side,
        max_workers=args.max_workers,
    )
    write_risk_csv(rows, output_path=args.output_path)
    write_risk_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(
            row.group_key,
            f"trades={row.trades}",
            f"mean_adverse={row.mean_max_adverse_excursion:.8f}",
            f"max_adverse={row.max_adverse_excursion:.8f}",
            f"liq5x={row.liquidation_risk_5x}",
            row.action,
        )


if __name__ == "__main__":
    main()
