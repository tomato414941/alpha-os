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


BINANCE_UM_MONTHLY_URL = "https://data.binance.vision/data/futures/um/monthly"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BtcEtfFlowFundingRegimeSummary:
    group_key: str
    observations: int
    mean_flow_btc: float
    mean_rolling_5d_flow_btc: float
    mean_start_funding_rate: float
    mean_forward_5d_funding_rate: float
    mean_funding_support_5d: float
    mean_directional_5d: float
    mean_directional_5d_with_funding: float
    hit_rate_5d: float
    hit_rate_5d_with_funding: float
    action: str


def build_funding_regime_summaries(
    *,
    labels_path: Path,
    max_workers: int = 12,
) -> tuple[BtcEtfFlowFundingRegimeSummary, ...]:
    rows = tuple(row for row in _read_rows(labels_path) if row.get("directional_return_5d"))
    if not rows:
        return ()
    start = min(date.fromisoformat(row["label_start_date"]) for row in rows)
    end = max(date.fromisoformat(row["label_start_date"]) + timedelta(days=4) for row in rows)
    funding_by_day = _fetch_btc_funding_by_day(start=start, end=end, max_workers=max_workers)

    enriched_rows = tuple(
        row
        for row in (_enrich_row_with_funding(row, funding_by_day=funding_by_day) for row in rows)
        if row is not None
    )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in enriched_rows:
        funding_group = row["funding_alignment"]
        action = row["action"]
        rolling_group = _rolling_size_group(row)
        grouped.setdefault(action, []).append(row)
        grouped.setdefault(rolling_group, []).append(row)
        grouped.setdefault(f"{action}__{funding_group}", []).append(row)
        grouped.setdefault(f"{rolling_group}__{funding_group}", []).append(row)

    summaries = tuple(_summarize_group(group_key=key, rows=tuple(value)) for key, value in grouped.items())
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.mean_directional_5d_with_funding,
                row.hit_rate_5d_with_funding,
                row.observations,
            ),
            reverse=True,
        )
    )


def write_summaries_csv(
    summaries: tuple[BtcEtfFlowFundingRegimeSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_key",
                "observations",
                "mean_flow_btc",
                "mean_rolling_5d_flow_btc",
                "mean_start_funding_rate",
                "mean_forward_5d_funding_rate",
                "mean_funding_support_5d",
                "mean_directional_5d",
                "mean_directional_5d_with_funding",
                "hit_rate_5d",
                "hit_rate_5d_with_funding",
                "action",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.group_key,
                    summary.observations,
                    f"{summary.mean_flow_btc:.8f}",
                    f"{summary.mean_rolling_5d_flow_btc:.8f}",
                    f"{summary.mean_start_funding_rate:.12f}",
                    f"{summary.mean_forward_5d_funding_rate:.12f}",
                    f"{summary.mean_funding_support_5d:.12f}",
                    f"{summary.mean_directional_5d:.8f}",
                    f"{summary.mean_directional_5d_with_funding:.8f}",
                    f"{summary.hit_rate_5d:.8f}",
                    f"{summary.hit_rate_5d_with_funding:.8f}",
                    summary.action,
                )
            )
    return output_path


def write_summaries_md(
    summaries: tuple[BtcEtfFlowFundingRegimeSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Funding Regime Summary\n\n")
        handle.write(
            "This joins leakage-safe BTC ETF flow labels to Binance BTCUSDT perp funding. "
            "Positive funding means BTC perp shorts receive funding; negative funding means longs receive funding. "
            "Funding PnL is a rough daily notional proxy, not an execution-ready PnL.\n\n"
        )
        handle.write(
            "| group | obs | mean 5d flow BTC | start funding | 5d funding | "
            "funding support | dir 5d | dir 5d + funding | hit 5d + funding | action |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for summary in summaries:
            handle.write(
                f"| {summary.group_key} | {summary.observations} | "
                f"{summary.mean_rolling_5d_flow_btc:.2f} | "
                f"{summary.mean_start_funding_rate:.8f} | "
                f"{summary.mean_forward_5d_funding_rate:.8f} | "
                f"{summary.mean_funding_support_5d:.8f} | "
                f"{summary.mean_directional_5d:.8f} | "
                f"{summary.mean_directional_5d_with_funding:.8f} | "
                f"{summary.hit_rate_5d_with_funding:.4f} | "
                f"{summary.action} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The main tradable question is whether ETF flow direction and perp funding carry point the same way. "
            "For example, large ETF outflow plus positive BTCUSDT funding means the short BTC view also receives funding.\n"
        )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _enrich_row_with_funding(
    row: dict[str, str],
    *,
    funding_by_day: dict[date, float],
) -> dict[str, str] | None:
    label_start = date.fromisoformat(row["label_start_date"])
    funding_days = tuple(label_start + timedelta(days=offset) for offset in range(5))
    if any(day not in funding_by_day for day in funding_days):
        return None
    direction = int(row["direction_hint"])
    forward_5d_funding = sum(funding_by_day[day] for day in funding_days)
    funding_support = -direction * forward_5d_funding
    directional_5d = float(row["directional_return_5d"])
    return {
        **row,
        "start_funding_rate": f"{funding_by_day[label_start]:.12f}",
        "forward_5d_funding_rate": f"{forward_5d_funding:.12f}",
        "funding_support_5d": f"{funding_support:.12f}",
        "directional_return_5d_with_funding": f"{directional_5d + funding_support:.12f}",
        "funding_alignment": _funding_alignment(funding_support),
    }


def _fetch_btc_funding_by_day(
    *,
    start: date,
    end: date,
    max_workers: int,
) -> dict[date, float]:
    months = _months_between(start=start, end=end)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows_by_month = tuple(executor.map(_fetch_funding_rate_month, months))
    funding_rows = tuple(row for rows in rows_by_month for row in rows)
    by_day: dict[date, list[float]] = {}
    for timestamp, funding_rate in funding_rows:
        day = timestamp.date()
        if start <= day <= end:
            by_day.setdefault(day, []).append(funding_rate)
    return {day: sum(values) for day, values in by_day.items()}


def _fetch_funding_rate_month(month: date) -> tuple[tuple[datetime, float], ...]:
    rows: list[tuple[datetime, float]] = []
    for item in _download_zip_csv(_funding_rate_monthly_url(month)):
        if not item or item[0] == "calc_time":
            continue
        rows.append((_ms_to_datetime(int(item[0])), float(item[2])))
    return tuple(rows)


def _download_zip_csv(url: str) -> tuple[list[str], ...]:
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            return tuple(list(row) for row in csv.reader(TextIOWrapper(handle, encoding="utf-8")))


def _funding_rate_monthly_url(month: date) -> str:
    return (
        f"{BINANCE_UM_MONTHLY_URL}/fundingRate/BTCUSDT/"
        f"BTCUSDT-fundingRate-{month:%Y-%m}.zip"
    )


def _months_between(*, start: date, end: date) -> tuple[date, ...]:
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    while cursor <= end_month:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return tuple(months)


def _ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _rolling_size_group(row: dict[str, str]) -> str:
    flow = float(row["rolling_5d_flow_btc"])
    if flow >= 15_000.0:
        return "large_5d_inflow"
    if flow <= -15_000.0:
        return "large_5d_outflow"
    return "mixed_5d_flow"


def _funding_alignment(funding_support: float) -> str:
    if funding_support > 0.0:
        return "funding_aligned"
    if funding_support < 0.0:
        return "funding_against"
    return "funding_flat"


def _summarize_group(
    *,
    group_key: str,
    rows: tuple[dict[str, str], ...],
) -> BtcEtfFlowFundingRegimeSummary:
    summary = BtcEtfFlowFundingRegimeSummary(
        group_key=group_key,
        observations=len(rows),
        mean_flow_btc=_mean(tuple(float(row["flow_btc"]) for row in rows)),
        mean_rolling_5d_flow_btc=_mean(tuple(float(row["rolling_5d_flow_btc"]) for row in rows)),
        mean_start_funding_rate=_mean(tuple(float(row["start_funding_rate"]) for row in rows)),
        mean_forward_5d_funding_rate=_mean(tuple(float(row["forward_5d_funding_rate"]) for row in rows)),
        mean_funding_support_5d=_mean(tuple(float(row["funding_support_5d"]) for row in rows)),
        mean_directional_5d=_mean(tuple(float(row["directional_return_5d"]) for row in rows)),
        mean_directional_5d_with_funding=_mean(
            tuple(float(row["directional_return_5d_with_funding"]) for row in rows)
        ),
        hit_rate_5d=_hit_rate(tuple(float(row["directional_return_5d"]) for row in rows)),
        hit_rate_5d_with_funding=_hit_rate(
            tuple(float(row["directional_return_5d_with_funding"]) for row in rows)
        ),
        action="",
    )
    return BtcEtfFlowFundingRegimeSummary(
        **{
            **summary.__dict__,
            "action": _action_for_summary(summary),
        }
    )


def _action_for_summary(summary: BtcEtfFlowFundingRegimeSummary) -> str:
    if (
        summary.observations >= 20
        and summary.mean_directional_5d_with_funding >= 0.015
        and summary.hit_rate_5d_with_funding >= 0.60
        and summary.mean_funding_support_5d > 0.0
    ):
        return "funding_regime_candidate"
    if (
        summary.observations >= 20
        and summary.mean_directional_5d_with_funding > 0.0
        and summary.mean_funding_support_5d > 0.0
    ):
        return "funding_regime_watch"
    return "weak_or_insufficient"


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return sum(1.0 for value in values if value > 0.0) / len(values) if values else 0.0


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
        default=ROOT / "btc_etf_flow_funding_regime_summary.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_regime_summary.md",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    summaries = build_funding_regime_summaries(
        labels_path=args.labels_path,
        max_workers=args.max_workers,
    )
    write_summaries_csv(summaries, output_path=args.output_path)
    write_summaries_md(summaries, output_path=args.markdown_output_path)
    for summary in summaries[:10]:
        print(
            summary.group_key,
            f"obs={summary.observations}",
            f"dir5_funding={summary.mean_directional_5d_with_funding:.8f}",
            f"funding_support={summary.mean_funding_support_5d:.8f}",
            f"hit={summary.hit_rate_5d_with_funding:.4f}",
            summary.action,
        )


if __name__ == "__main__":
    main()
