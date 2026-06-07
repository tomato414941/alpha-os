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
from btc_etf_flow_funding_regime_summary import ROOT, _fetch_btc_funding_by_day
from btc_etf_flow_funding_regime_summary import build_funding_enriched_label_rows


BINANCE_UM_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"


@dataclass(frozen=True)
class IntradayEntryStressRow:
    group_key: str
    trades: int
    skipped_overlap_signals: int
    total_return: float
    mean_net_return: float
    hit_rate: float
    max_drawdown: float
    fee_bps_per_side: float
    action: str


def build_intraday_entry_stress_rows(
    *,
    labels_path: Path,
    entry_offset_hours: tuple[int, ...],
    fee_bps_per_side: float,
    max_workers: int = 24,
) -> tuple[IntradayEntryStressRow, ...]:
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
    hourly_closes = _fetch_btc_hourly_closes(days=needed_days, max_workers=max_workers)
    funding_by_day = _fetch_btc_funding_by_day(
        start=min(needed_days),
        end=max(needed_days),
        max_workers=max_workers,
    )
    return tuple(
        _stress_row(
            group_key=f"entry_offset_hours_{offset_hours}",
            net_returns=_non_overlapping_net_returns(
                signal_rows=signal_rows,
                entry_offset_hours=offset_hours,
                hourly_closes=hourly_closes,
                funding_by_day=funding_by_day,
                fee_bps_per_side=fee_bps_per_side,
            ),
            total_signal_count=len(signal_rows),
            fee_bps_per_side=fee_bps_per_side,
        )
        for offset_hours in entry_offset_hours
    )


def write_stress_csv(
    rows: tuple[IntradayEntryStressRow, ...],
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
                "skipped_overlap_signals",
                "total_return",
                "mean_net_return",
                "hit_rate",
                "max_drawdown",
                "fee_bps_per_side",
                "action",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.group_key,
                    row.trades,
                    row.skipped_overlap_signals,
                    f"{row.total_return:.8f}",
                    f"{row.mean_net_return:.8f}",
                    f"{row.hit_rate:.8f}",
                    f"{row.max_drawdown:.8f}",
                    f"{row.fee_bps_per_side:.4f}",
                    row.action,
                )
            )
    return output_path


def write_stress_md(
    rows: tuple[IntradayEntryStressRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Funding Intraday Entry Stress\n\n")
        handle.write(
            "This retests the BTC ETF-flow/funding paper rule with Binance BTCUSDT 1h closes. "
            "Entry is shifted by fixed hour offsets from the label-start UTC day and held for 120 hours. "
            "Funding PnL remains a rough daily approximation.\n\n"
        )
        handle.write(
            "| group | trades | skipped | total return | mean net | hit | max drawdown | fee bps/side | action |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.group_key} | {row.trades} | {row.skipped_overlap_signals} | "
                f"{row.total_return:.8f} | {row.mean_net_return:.8f} | "
                f"{row.hit_rate:.4f} | {row.max_drawdown:.8f} | "
                f"{row.fee_bps_per_side:.4f} | {row.action} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "This is still not a live execution model. It does not simulate order books, liquidation, stop logic, or mark/index basis. "
            "Its purpose is only to test whether the daily close result is hypersensitive to entry timing.\n"
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


def _non_overlapping_net_returns(
    *,
    signal_rows: tuple[dict[str, str], ...],
    entry_offset_hours: int,
    hourly_closes: dict[datetime, float],
    funding_by_day: dict[date, float],
    fee_bps_per_side: float,
) -> tuple[float, ...]:
    net_returns: list[float] = []
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
        entry_close = hourly_closes.get(entry_at)
        exit_close = hourly_closes.get(exit_at)
        funding_support = _funding_support(
            funding_by_day=funding_by_day,
            entry_day=entry_at.date(),
            direction=int(row["direction_hint"]),
        )
        if entry_close is None or exit_close is None or entry_close <= 0.0 or funding_support is None:
            continue
        raw_return = (exit_close / entry_close) - 1.0
        directional_return = raw_return * int(row["direction_hint"])
        net_returns.append(directional_return + funding_support - round_trip_fee)
        next_available_entry = exit_at
    return tuple(net_returns)


def _fetch_btc_hourly_closes(
    *,
    days: tuple[date, ...],
    max_workers: int,
) -> dict[datetime, float]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows_by_day = tuple(executor.map(_fetch_btc_hourly_close_rows, days))
    return {timestamp: close for rows in rows_by_day for timestamp, close in rows}


def _fetch_btc_hourly_close_rows(day: date) -> tuple[tuple[datetime, float], ...]:
    response = requests.get(_hourly_kline_url(day), timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    rows: list[tuple[datetime, float]] = []
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            for item in csv.reader(TextIOWrapper(handle, encoding="utf-8")):
                if not item or item[0] == "open_time":
                    continue
                rows.append((_ms_to_datetime(int(item[0])), float(item[4])))
    return tuple(rows)


def _hourly_kline_url(day: date) -> str:
    return f"{BINANCE_UM_DAILY_URL}/klines/BTCUSDT/1h/BTCUSDT-1h-{day:%Y-%m-%d}.zip"


def _funding_support(
    *,
    funding_by_day: dict[date, float],
    entry_day: date,
    direction: int,
) -> float | None:
    days = tuple(entry_day + timedelta(days=offset) for offset in range(5))
    if any(day not in funding_by_day for day in days):
        return None
    return -direction * sum(funding_by_day[day] for day in days)


def _stress_row(
    *,
    group_key: str,
    net_returns: tuple[float, ...],
    total_signal_count: int,
    fee_bps_per_side: float,
) -> IntradayEntryStressRow:
    row = IntradayEntryStressRow(
        group_key=group_key,
        trades=len(net_returns),
        skipped_overlap_signals=max(0, total_signal_count - len(net_returns)),
        total_return=_compounded_return(net_returns),
        mean_net_return=_mean(net_returns),
        hit_rate=_hit_rate(net_returns),
        max_drawdown=_max_drawdown(net_returns),
        fee_bps_per_side=fee_bps_per_side,
        action="",
    )
    return IntradayEntryStressRow(
        **{
            **row.__dict__,
            "action": _action_for_stress(row),
        }
    )


def _action_for_stress(row: IntradayEntryStressRow) -> str:
    if row.trades >= 10 and row.total_return > 0.0 and row.hit_rate >= 0.55:
        return "survives"
    if row.trades >= 5 and row.total_return > 0.0:
        return "thin_positive"
    return "fails_or_too_thin"


def _compounded_return(values: tuple[float, ...]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return sum(1.0 for value in values if value > 0.0) / len(values) if values else 0.0


def _max_drawdown(values: tuple[float, ...]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak) - 1.0 if peak > 0.0 else 0.0)
    return max_drawdown


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
        default=ROOT / "btc_etf_flow_funding_intraday_entry_stress.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_intraday_entry_stress.md",
    )
    parser.add_argument("--entry-offset-hours", nargs="+", type=int, default=[0, 8, 16, 24, 32, 48])
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()

    rows = build_intraday_entry_stress_rows(
        labels_path=args.labels_path,
        entry_offset_hours=tuple(args.entry_offset_hours),
        fee_bps_per_side=args.fee_bps_per_side,
        max_workers=args.max_workers,
    )
    write_stress_csv(rows, output_path=args.output_path)
    write_stress_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(
            row.group_key,
            f"trades={row.trades}",
            f"total={row.total_return:.8f}",
            f"mean={row.mean_net_return:.8f}",
            f"hit={row.hit_rate:.4f}",
            f"mdd={row.max_drawdown:.8f}",
            row.action,
        )


if __name__ == "__main__":
    main()
