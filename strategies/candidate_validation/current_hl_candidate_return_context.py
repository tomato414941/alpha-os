from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class Candidate:
    symbol: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ReturnContextRow:
    symbol: str
    sources: str
    latest_close: float
    return_1h: float
    return_4h: float
    return_24h: float
    volume_24h: float
    candle_count: int
    action: str
    score: float
    reason: str


def build_return_context_rows(
    *,
    candidates: tuple[Candidate, ...] | None = None,
) -> tuple[ReturnContextRow, ...]:
    rows = tuple(
        _build_return_context_row(candidate)
        for candidate in (candidates or collect_candidates())
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def collect_candidates() -> tuple[Candidate, ...]:
    sources_by_symbol: dict[str, set[str]] = {}
    _add_stable_candidate_sources(sources_by_symbol)
    _add_crowding_candidate_sources(sources_by_symbol)
    _add_attention_candidate_sources(sources_by_symbol)
    return tuple(
        Candidate(symbol=symbol, sources=tuple(sorted(sources)))
        for symbol, sources in sorted(sources_by_symbol.items())
    )


def write_return_context_csv(
    rows: tuple[ReturnContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "sources",
                "latest_close",
                "return_1h",
                "return_4h",
                "return_24h",
                "volume_24h",
                "candle_count",
                "action",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.sources,
                    f"{row.latest_close:.12f}",
                    f"{row.return_1h:.8f}",
                    f"{row.return_4h:.8f}",
                    f"{row.return_24h:.8f}",
                    f"{row.volume_24h:.8f}",
                    row.candle_count,
                    row.action,
                    f"{row.score:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_return_context_md(
    rows: tuple[ReturnContextRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current HL Candidate Return Context\n\n")
        handle.write(
            "This joins current candidate screens to recent Hyperliquid candle returns. "
            "It is context, not a causal alpha test.\n\n"
        )
        handle.write(
            "| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.sources} | "
                f"{row.latest_close:.8f} | "
                f"{row.return_1h:.6f} | "
                f"{row.return_4h:.6f} | "
                f"{row.return_24h:.6f} | "
                f"{row.volume_24h:.2f} | "
                f"{row.action} | "
                f"{row.score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`multi_source_momentum_context` means a candidate appears in more than one "
            "research lane and has a recent directional move. `single_source_context` "
            "keeps a candidate visible but lower priority. Future-return labels are "
            "still needed before this becomes evidence of alpha.\n"
        )
    return output_path


def _add_stable_candidate_sources(sources_by_symbol: dict[str, set[str]]) -> None:
    path = STRATEGIES_ROOT / "cross_exchange_funding" / "stable_12_sample_monitor_summary.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") == "okx_hl_current" and row.get("asset"):
                sources_by_symbol.setdefault(row["asset"], set()).add("cross_exchange_funding")


def _add_crowding_candidate_sources(sources_by_symbol: dict[str, set[str]]) -> None:
    path = STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    for row in rows[:15]:
        sources_by_symbol.setdefault(row["asset"], set()).add("perp_carry_reversion")


def _add_attention_candidate_sources(sources_by_symbol: dict[str, set[str]]) -> None:
    path = STRATEGIES_ROOT / "news_social" / "current_attention_market_join.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sources_by_symbol.setdefault(row["symbol"], set()).add("attention_market_join")


def _build_return_context_row(candidate: Candidate) -> ReturnContextRow:
    candles = _fetch_candles(candidate.symbol)
    latest_close = candles[-1]["close"] if candles else 0.0
    return_1h = _window_return(candles, hours=1)
    return_4h = _window_return(candles, hours=4)
    return_24h = _window_return(candles, hours=24)
    volume_24h = sum(candle["volume"] for candle in candles[-96:])
    action = _action(candidate=candidate, return_1h=return_1h, return_4h=return_4h)
    return ReturnContextRow(
        symbol=candidate.symbol,
        sources=";".join(candidate.sources),
        latest_close=latest_close,
        return_1h=return_1h,
        return_4h=return_4h,
        return_24h=return_24h,
        volume_24h=volume_24h,
        candle_count=len(candles),
        action=action,
        score=_score(
            source_count=len(candidate.sources),
            return_1h=return_1h,
            return_4h=return_4h,
            volume_24h=volume_24h,
        ),
        reason=_reason(action),
    )


def _fetch_candles(symbol: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=30)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
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
            "open": float(row["o"]),
            "close": float(row["c"]),
            "volume": float(row["v"]),
        }
        for row in response.json()
    )


def _window_return(candles: tuple[dict[str, float], ...], *, hours: int) -> float:
    needed = (hours * 4) + 1
    if len(candles) < needed:
        return 0.0
    start = candles[-needed]["close"]
    end = candles[-1]["close"]
    return (end / start) - 1.0 if start > 0.0 else 0.0


def _action(*, candidate: Candidate, return_1h: float, return_4h: float) -> str:
    if len(candidate.sources) > 1 and abs(return_1h) >= 0.01:
        return "multi_source_momentum_context"
    if len(candidate.sources) > 1:
        return "multi_source_watch"
    if abs(return_4h) >= 0.03:
        return "single_source_momentum_context"
    return "single_source_context"


def _score(*, source_count: int, return_1h: float, return_4h: float, volume_24h: float) -> float:
    volume_score = min(volume_24h / 1_000_000.0, 10.0)
    return (source_count * 5.0) + (abs(return_1h) * 100.0) + (abs(return_4h) * 50.0) + volume_score


def _reason(action: str) -> str:
    if action == "multi_source_momentum_context":
        return "candidate has multiple sources and a recent 1h move"
    if action == "multi_source_watch":
        return "candidate has multiple sources but no large recent move"
    if action == "single_source_momentum_context":
        return "candidate has a single source and a recent 4h move"
    return "candidate remains visible but needs stronger labels"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_hl_candidate_return_context.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hl_candidate_return_context.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_return_context_rows()
    write_return_context_csv(rows, output_path=args.csv_output_path)
    write_return_context_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.action,
            f"1h={row.return_1h:.4f}",
            f"4h={row.return_4h:.4f}",
            f"sources={row.sources}",
        )


if __name__ == "__main__":
    main()
