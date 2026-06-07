from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests


BINANCE_UM_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BtcEtfFlowLabel:
    flow_date: str
    label_start_date: str
    flow_btc: float
    rolling_5d_flow_btc: float
    direction_hint: int
    action: str
    raw_return_1d: float | None
    raw_return_3d: float | None
    raw_return_5d: float | None
    directional_return_1d: float | None
    directional_return_3d: float | None
    directional_return_5d: float | None
    label_status: str


def build_btc_etf_flow_forward_labels(
    *,
    history_path: Path,
    max_workers: int = 24,
) -> tuple[BtcEtfFlowLabel, ...]:
    flow_rows = _read_flow_rows(history_path)
    if not flow_rows:
        return ()
    flow_dates = tuple(date.fromisoformat(row["flow_date"]) for row in flow_rows)
    start = min(flow_dates) + timedelta(days=1)
    end = max(flow_dates) + timedelta(days=6)
    closes = _fetch_btc_daily_closes(start=start, end=end, max_workers=max_workers)
    labeled_rows: list[BtcEtfFlowLabel] = []
    for index, row in enumerate(flow_rows):
        flow_date = date.fromisoformat(row["flow_date"])
        label_start = flow_date + timedelta(days=1)
        flow_btc = float(row["flow_btc"])
        rolling_5d = sum(float(item["flow_btc"]) for item in flow_rows[max(0, index - 4) : index + 1])
        direction = _direction_for_flow(flow_btc=flow_btc, rolling_5d_flow_btc=rolling_5d)
        raw_1d = _forward_return(closes, start=label_start, days=1)
        raw_3d = _forward_return(closes, start=label_start, days=3)
        raw_5d = _forward_return(closes, start=label_start, days=5)
        labeled_rows.append(
            BtcEtfFlowLabel(
                flow_date=row["flow_date"],
                label_start_date=label_start.isoformat(),
                flow_btc=flow_btc,
                rolling_5d_flow_btc=rolling_5d,
                direction_hint=direction,
                action=_action_for_flow(flow_btc=flow_btc, rolling_5d_flow_btc=rolling_5d),
                raw_return_1d=raw_1d,
                raw_return_3d=raw_3d,
                raw_return_5d=raw_5d,
                directional_return_1d=_directional_return(raw_1d, direction),
                directional_return_3d=_directional_return(raw_3d, direction),
                directional_return_5d=_directional_return(raw_5d, direction),
                label_status=_label_status(raw_return_1d=raw_1d, raw_return_5d=raw_5d),
            )
        )
    return tuple(labeled_rows)


def write_labels_csv(rows: tuple[BtcEtfFlowLabel, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "flow_date",
                "label_start_date",
                "flow_btc",
                "rolling_5d_flow_btc",
                "direction_hint",
                "action",
                "raw_return_1d",
                "raw_return_3d",
                "raw_return_5d",
                "directional_return_1d",
                "directional_return_3d",
                "directional_return_5d",
                "label_status",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.flow_date,
                    row.label_start_date,
                    f"{row.flow_btc:.8f}",
                    f"{row.rolling_5d_flow_btc:.8f}",
                    row.direction_hint,
                    row.action,
                    _format_float(row.raw_return_1d),
                    _format_float(row.raw_return_3d),
                    _format_float(row.raw_return_5d),
                    _format_float(row.directional_return_1d),
                    _format_float(row.directional_return_3d),
                    _format_float(row.directional_return_5d),
                    row.label_status,
                )
            )
    return output_path


def write_labels_md(rows: tuple[BtcEtfFlowLabel, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = tuple(row for row in rows if row.directional_return_5d is not None)
    ranked = tuple(
        sorted(
            labeled,
            key=lambda row: row.directional_return_5d if row.directional_return_5d is not None else -1.0,
            reverse=True,
        )
    )
    summary = _summarize_labels(labeled)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Forward Labels\n\n")
        handle.write(
            "This labels Bitcoin ETF flow after the flow date to avoid using same-day price movement. It is not net PnL.\n\n"
        )
        handle.write("## Summary\n\n")
        handle.write(
            f"- labeled rows: `{summary['observations']:.0f}`\n"
            f"- mean directional 1d: `{summary['mean_directional_1d']:.8f}`\n"
            f"- mean directional 3d: `{summary['mean_directional_3d']:.8f}`\n"
            f"- mean directional 5d: `{summary['mean_directional_5d']:.8f}`\n"
            f"- 5d hit rate: `{summary['hit_rate_5d']:.4f}`\n\n"
        )
        handle.write("## Top Directional 5d Rows\n\n")
        handle.write(
            "| flow date | action | flow BTC | 5d flow BTC | dir | raw 1d | dir 1d | raw 3d | dir 3d | raw 5d | dir 5d |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in ranked[:25]:
            handle.write(
                f"| {row.flow_date} | {row.action} | {row.flow_btc:.2f} | "
                f"{row.rolling_5d_flow_btc:.2f} | {row.direction_hint} | "
                f"{_format_float(row.raw_return_1d)} | {_format_float(row.directional_return_1d)} | "
                f"{_format_float(row.raw_return_3d)} | {_format_float(row.directional_return_3d)} | "
                f"{_format_float(row.raw_return_5d)} | {_format_float(row.directional_return_5d)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Positive directional return means ETF inflow was treated as long BTC context and ETF outflow as short BTC context. This is a coarse daily regime label and excludes fees, funding PnL, and intraday timing.\n"
        )
    return output_path


def _read_flow_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _fetch_btc_daily_closes(
    *,
    start: date,
    end: date,
    max_workers: int,
) -> dict[date, float]:
    days = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = tuple(executor.map(_fetch_btc_daily_close, days))
    return {day: close for day, close in rows if close is not None}


def _fetch_btc_daily_close(day: date) -> tuple[date, float | None]:
    url = f"{BINANCE_UM_DAILY_URL}/klines/BTCUSDT/1d/BTCUSDT-1d-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return day, None
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            reader = csv.reader(TextIOWrapper(handle, encoding="utf-8"))
            for item in reader:
                if not item or item[0] == "open_time":
                    continue
                return day, float(item[4])
    return day, None


def _forward_return(closes: dict[date, float], *, start: date, days: int) -> float | None:
    start_close = closes.get(start)
    end_close = closes.get(start + timedelta(days=days))
    if start_close is None or end_close is None or start_close <= 0.0:
        return None
    return (end_close / start_close) - 1.0


def _direction_for_flow(*, flow_btc: float, rolling_5d_flow_btc: float) -> int:
    if flow_btc == 0.0 and rolling_5d_flow_btc == 0.0:
        return 0
    return 1 if (flow_btc + rolling_5d_flow_btc) >= 0.0 else -1


def _action_for_flow(*, flow_btc: float, rolling_5d_flow_btc: float) -> str:
    if flow_btc > 1_000.0 and rolling_5d_flow_btc > 3_000.0:
        return "btc_etf_accumulation_label"
    if flow_btc < -1_000.0 and rolling_5d_flow_btc < -3_000.0:
        return "btc_etf_distribution_label"
    if rolling_5d_flow_btc > 0.0:
        return "btc_etf_inflow_context_label"
    if rolling_5d_flow_btc < 0.0:
        return "btc_etf_outflow_context_label"
    return "btc_etf_neutral_context_label"


def _directional_return(raw_return: float | None, direction: int) -> float | None:
    if raw_return is None or direction == 0:
        return None
    return raw_return * direction


def _label_status(*, raw_return_1d: float | None, raw_return_5d: float | None) -> str:
    if raw_return_1d is None:
        return "pending_1d"
    if raw_return_5d is None:
        return "labeled_1d_pending_5d"
    return "labeled_5d"


def _summarize_labels(rows: tuple[BtcEtfFlowLabel, ...]) -> dict[str, float]:
    return {
        "observations": float(len(rows)),
        "mean_directional_1d": _mean_present(tuple(row.directional_return_1d for row in rows)),
        "mean_directional_3d": _mean_present(tuple(row.directional_return_3d for row in rows)),
        "mean_directional_5d": _mean_present(tuple(row.directional_return_5d for row in rows)),
        "hit_rate_5d": _hit_rate(tuple(row.directional_return_5d for row in rows)),
    }


def _mean_present(values: tuple[float | None, ...]) -> float:
    present = tuple(value for value in values if value is not None)
    return sum(present) / len(present) if present else 0.0


def _hit_rate(values: tuple[float | None, ...]) -> float:
    present = tuple(value for value in values if value is not None)
    return sum(1.0 for value in present if value > 0.0) / len(present) if present else 0.0


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "current_btc_etf_flow_history.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_forward_labels.md",
    )
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()

    rows = build_btc_etf_flow_forward_labels(
        history_path=args.history_path,
        max_workers=args.max_workers,
    )
    write_labels_csv(rows, output_path=args.output_path)
    write_labels_md(rows, output_path=args.markdown_output_path)
    summary = _summarize_labels(tuple(row for row in rows if row.directional_return_5d is not None))
    print(
        "summary",
        f"observations={summary['observations']:.0f}",
        f"mean_dir_1d={summary['mean_directional_1d']:.8f}",
        f"mean_dir_3d={summary['mean_directional_3d']:.8f}",
        f"mean_dir_5d={summary['mean_directional_5d']:.8f}",
        f"hit5={summary['hit_rate_5d']:.4f}",
    )


if __name__ == "__main__":
    main()
