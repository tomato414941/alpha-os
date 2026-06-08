from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests


ROOT = Path(__file__).resolve().parent
BINANCE_UM_DAILY_KLINES_URL = "https://data.binance.vision/data/futures/um/daily/klines"
HORIZONS = (("15m", 15), ("1h", 60), ("4h", 240))


@dataclass(frozen=True)
class NewsEventForwardLabel:
    source: str
    published_at: str
    symbol: str
    event_kind: str
    side: str
    title: str
    entry_close: float
    return_15m_bps: float | None
    return_1h_bps: float | None
    return_4h_bps: float | None
    directional_15m_bps: float | None
    directional_1h_bps: float | None
    directional_4h_bps: float | None
    label_status: str
    next_step: str


def build_news_event_forward_labels(
    *,
    news_event_path: Path = ROOT / "current_news_event_screen.csv",
    top: int = 80,
) -> tuple[NewsEventForwardLabel, ...]:
    rows = _read_rows(news_event_path)[:top]
    kline_cache: dict[tuple[str, datetime.date], dict[int, float]] = {}
    output = []
    for row in rows:
        output.append(_label_row(row, kline_cache=kline_cache))
    return tuple(output)


def write_news_event_forward_labels_csv(
    rows: tuple[NewsEventForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source",
                "published_at",
                "symbol",
                "event_kind",
                "side",
                "title",
                "entry_close",
                "return_15m_bps",
                "return_1h_bps",
                "return_4h_bps",
                "directional_15m_bps",
                "directional_1h_bps",
                "directional_4h_bps",
                "label_status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.source,
                    row.published_at,
                    row.symbol,
                    row.event_kind,
                    row.side,
                    row.title,
                    _format_float(row.entry_close),
                    _format_optional(row.return_15m_bps),
                    _format_optional(row.return_1h_bps),
                    _format_optional(row.return_4h_bps),
                    _format_optional(row.directional_15m_bps),
                    _format_optional(row.directional_1h_bps),
                    _format_optional(row.directional_4h_bps),
                    row.label_status,
                    row.next_step,
                )
            )
    return output_path


def write_news_event_forward_labels_md(
    rows: tuple[NewsEventForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current News Event Forward Labels\n\n")
        handle.write(
            "This labels RSS headline timestamps against Binance USD-M 1m forward returns. "
            "It is a timestamp check, not a trade instruction.\n\n"
        )
        handle.write(
            "| source | published | symbol | kind | side | entry | dir 15m | dir 1h | dir 4h | status | title |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.source} | "
                f"{row.published_at} | "
                f"{row.symbol} | "
                f"{row.event_kind} | "
                f"{row.side} | "
                f"{_format_float(row.entry_close)} | "
                f"{_format_optional(row.directional_15m_bps)} | "
                f"{_format_optional(row.directional_1h_bps)} | "
                f"{_format_optional(row.directional_4h_bps)} | "
                f"{row.label_status} | "
                f"{_escape(row.title)} |\n"
            )
    return output_path


def _label_row(
    row: dict[str, str],
    *,
    kline_cache: dict[tuple[str, datetime.date], dict[int, float]],
) -> NewsEventForwardLabel:
    symbol = row.get("symbol", "")
    published_at = _parse_datetime(row.get("published_at", ""))
    entry_ms = _minute_ms(published_at)
    symbol_pair = f"{symbol}USDT"
    closes = _closes_for_window(symbol_pair, published_at, kline_cache=kline_cache)
    entry_close = closes.get(entry_ms, 0.0)
    returns: dict[str, float | None] = {}
    directional: dict[str, float | None] = {}
    side_multiplier = _side_multiplier(row)
    for horizon, minutes in HORIZONS:
        forward_close = closes.get(entry_ms + (minutes * 60_000), 0.0)
        value = ((forward_close / entry_close) - 1.0) * 10_000.0 if entry_close > 0.0 and forward_close > 0.0 else None
        returns[horizon] = value
        directional[horizon] = value * side_multiplier if value is not None and side_multiplier != 0.0 else None
    status = _label_status(
        entry_close=entry_close,
        published_at=published_at,
        directional=directional,
        side_multiplier=side_multiplier,
    )
    return NewsEventForwardLabel(
        source=row.get("source", ""),
        published_at=row.get("published_at", ""),
        symbol=symbol,
        event_kind=row.get("event_kind", ""),
        side=_label_side(side_multiplier),
        title=row.get("title", ""),
        entry_close=entry_close,
        return_15m_bps=returns["15m"],
        return_1h_bps=returns["1h"],
        return_4h_bps=returns["4h"],
        directional_15m_bps=directional["15m"],
        directional_1h_bps=directional["1h"],
        directional_4h_bps=directional["4h"],
        label_status=status,
        next_step=_next_step(symbol=symbol, status=status),
    )


def _closes_for_window(
    symbol_pair: str,
    published_at: datetime,
    *,
    kline_cache: dict[tuple[str, datetime.date], dict[int, float]],
) -> dict[int, float]:
    output: dict[int, float] = {}
    for day_offset in (0, 1):
        day = (published_at + timedelta(days=day_offset)).date()
        key = (symbol_pair, day)
        if key not in kline_cache:
            kline_cache[key] = _fetch_1m_closes(symbol_pair, day)
        output.update(kline_cache[key])
    return output


def _fetch_1m_closes(symbol_pair: str, day: datetime.date) -> dict[int, float]:
    url = f"{BINANCE_UM_DAILY_KLINES_URL}/{symbol_pair}/1m/{symbol_pair}-1m-{day:%Y-%m-%d}.zip"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    closes: dict[int, float] = {}
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            for row in csv.DictReader(TextIOWrapper(handle, encoding="utf-8")):
                closes[int(row["open_time"])] = float(row["close"])
    return closes


def _label_status(
    *,
    entry_close: float,
    published_at: datetime,
    directional: dict[str, float | None],
    side_multiplier: float,
) -> str:
    if entry_close <= 0.0:
        if published_at.date() >= datetime.now(UTC).date():
            return "pending_forward_archive"
        return "missing_archive_or_symbol"
    if side_multiplier == 0.0:
        return "context_only_label"
    if directional["1h"] is None:
        return "pending_forward_archive"
    if directional["1h"] > 0.0 and (directional["4h"] or 0.0) > 0.0:
        return "direction_supported_1h_4h"
    if directional["1h"] > 0.0:
        return "direction_supported_1h_only"
    return "direction_rejected_1h"


def _side_multiplier(row: dict[str, str]) -> float:
    side = row.get("side", "")
    if side in {"long_event_follow", "paper_long", "collect_label"} and _int(row.get("direction_hint")) > 0:
        return 1.0
    if side in {"short_or_avoid", "paper_short"} or _int(row.get("direction_hint")) < 0:
        return -1.0
    return 0.0


def _label_side(side_multiplier: float) -> str:
    if side_multiplier > 0.0:
        return "paper_long_news_event"
    if side_multiplier < 0.0:
        return "paper_short_news_event"
    return "context_label"


def _next_step(*, symbol: str, status: str) -> str:
    if status == "direction_supported_1h_4h":
        return f"repeat {symbol} news-event label with duplicate-source and execution-cost checks"
    if status == "direction_supported_1h_only":
        return f"watch {symbol} news-event decay and require repeat support before paper action"
    if status == "missing_archive_or_symbol":
        return f"skip {symbol} until a tradable futures archive exists"
    if status == "pending_forward_archive":
        return f"wait for {symbol} forward archive before judging this headline"
    return f"keep {symbol} headline as context only"


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _minute_ms(value: datetime) -> int:
    minute = value.replace(second=0, microsecond=0)
    return int(minute.timestamp() * 1000)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _int(value: object) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _format_float(value: float) -> str:
    return f"{value:.8f}" if value else ""


def _format_optional(value: float | None) -> str:
    return f"{value:.8f}" if value is not None else ""


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-event-path", type=Path, default=ROOT / "current_news_event_screen.csv")
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_news_event_forward_labels.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_news_event_forward_labels.md")
    args = parser.parse_args()

    rows = build_news_event_forward_labels(news_event_path=args.news_event_path, top=args.top)
    write_news_event_forward_labels_csv(rows, output_path=args.output_path)
    write_news_event_forward_labels_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.symbol, row.event_kind, row.side, row.label_status, _format_optional(row.directional_1h_bps))


if __name__ == "__main__":
    main()
