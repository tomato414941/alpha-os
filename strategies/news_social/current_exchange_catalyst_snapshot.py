from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


BINANCE_ANNOUNCEMENTS_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
)
SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,14})(?:USDT|USDC|FDUSD|BTC|ETH|BNB)\b")
TOKEN_LIST_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,14}\b")
IGNORED_SYMBOLS = {
    "AED",
    "API",
    "BNB",
    "BTC",
    "ETF",
    "ETFS",
    "FDUSD",
    "OTC",
    "USD",
    "USDC",
    "USDT",
    "VIP",
}
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExchangeCatalystRow:
    timestamp: str
    source: str
    published_at: str
    catalog: str
    symbol: str
    catalyst_kind: str
    direction_hint: int
    score: float
    title: str
    reason: str


def fetch_binance_announcements(
    url: str = BINANCE_ANNOUNCEMENTS_URL,
    *,
    page_size: int = 50,
) -> dict[str, object]:
    response = requests.get(
        url,
        params={"type": 1, "pageNo": 1, "pageSize": page_size},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def build_exchange_catalyst_rows(
    *,
    binance_payload: dict[str, object],
    timestamp: str | None = None,
    max_age_days: int = 90,
) -> tuple[ExchangeCatalystRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    rows: list[ExchangeCatalystRow] = []
    seen: set[tuple[str, str, str]] = set()
    data = binance_payload.get("data") if isinstance(binance_payload, dict) else {}
    catalogs = data.get("catalogs") if isinstance(data, dict) else ()
    for catalog in catalogs or ():
        catalog_name = str(catalog.get("catalogName") or "")
        for article in catalog.get("articles") or ():
            published_at = _published_at(article)
            if not published_at or _parse_datetime(published_at) < cutoff:
                continue
            title = str(article.get("title") or "")
            kind = _catalyst_kind(title=title, catalog=catalog_name)
            if kind == "ignore":
                continue
            for symbol in _symbols_from_title(title):
                key = (symbol, title, kind)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    ExchangeCatalystRow(
                        timestamp=observed_at,
                        source="binance_announcements",
                        published_at=published_at,
                        catalog=catalog_name,
                        symbol=symbol,
                        catalyst_kind=kind,
                        direction_hint=_direction_hint(kind),
                        score=_catalyst_score(kind=kind, catalog=catalog_name, title=title),
                        title=title,
                        reason=_reason(kind),
                    )
                )
    return tuple(sorted(rows, key=lambda row: (row.score, row.published_at), reverse=True))


def write_exchange_catalyst_rows(
    rows: tuple[ExchangeCatalystRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "source",
                "published_at",
                "catalog",
                "symbol",
                "catalyst_kind",
                "direction_hint",
                "score",
                "title",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.source,
                    row.published_at,
                    row.catalog,
                    row.symbol,
                    row.catalyst_kind,
                    row.direction_hint,
                    f"{row.score:.8f}",
                    row.title,
                    row.reason,
                )
            )
    return output_path


def write_exchange_catalyst_rows_md(
    rows: tuple[ExchangeCatalystRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Exchange Catalyst Snapshot\n\n")
        handle.write(
            "This extracts current exchange-announcement catalysts from Binance. "
            "It is an external-event screen, not a trade instruction.\n\n"
        )
        handle.write("| published | catalog | symbol | kind | dir | score | title |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.published_at} | "
                f"{row.catalog} | "
                f"{row.symbol} | "
                f"{row.catalyst_kind} | "
                f"{row.direction_hint} | "
                f"{row.score:.4f} | "
                f"{row.title} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Exchange listings, futures launches, removals, and network events can "
            "move prices independently from pure price history. Rows still need "
            "venue overlap, future-return labels, and execution checks.\n"
        )
    return output_path


def _published_at(article: dict[str, object]) -> str:
    value = article.get("releaseDate") or article.get("publishDate") or 0
    if not value:
        return ""
    return datetime.fromtimestamp(float(value) / 1000.0, UTC).isoformat()


def _catalyst_kind(*, title: str, catalog: str) -> str:
    lowered = title.lower()
    if any(
        token in lowered
        for token in ("tradfi", "stocks", "stock ", "etf", "equity", "pre-ipo")
    ):
        return "ignore"
    if "delist" in lowered or "remove" in lowered or "removal" in lowered:
        return "exchange_removal_watch"
    if "futures will launch" in lowered or "perpetual contract" in lowered:
        return "perp_listing_watch"
    if "new cryptocurrency listing" in catalog.lower() or "will list" in lowered:
        return "spot_listing_watch"
    if "alpha trading competition" in lowered:
        return "attention_campaign_watch"
    if "airdrop" in lowered or "hodler" in lowered:
        return "airdrop_attention_watch"
    if "network upgrade" in lowered or "hard fork" in lowered:
        return "network_event_watch"
    return "ignore"


def _symbols_from_title(title: str) -> tuple[str, ...]:
    symbols = {match.group(1) for match in SYMBOL_RE.finditer(title)}
    if not symbols:
        symbol_section = title.split(":", maxsplit=1)[-1]
        symbols.update(TOKEN_LIST_RE.findall(symbol_section))
    return tuple(
        sorted(
            symbol
            for symbol in symbols
            if symbol not in IGNORED_SYMBOLS and not symbol.isdigit()
        )
    )


def _direction_hint(kind: str) -> int:
    if kind == "exchange_removal_watch":
        return -1
    if kind in {
        "perp_listing_watch",
        "spot_listing_watch",
        "attention_campaign_watch",
        "airdrop_attention_watch",
        "network_event_watch",
    }:
        return 1
    return 0


def _catalyst_score(*, kind: str, catalog: str, title: str) -> float:
    base = {
        "perp_listing_watch": 5.0,
        "spot_listing_watch": 4.0,
        "exchange_removal_watch": 4.0,
        "attention_campaign_watch": 2.5,
        "airdrop_attention_watch": 2.0,
        "network_event_watch": 1.5,
    }.get(kind, 0.0)
    if "New Cryptocurrency Listing" in catalog:
        base += 1.0
    if "Futures" in title:
        base += 1.0
    if "Alpha" in title:
        base += 0.5
    return base


def _reason(kind: str) -> str:
    if kind == "exchange_removal_watch":
        return "exchange removal can create forced selling or liquidity withdrawal"
    if kind == "perp_listing_watch":
        return "new perp venue can change leverage, liquidity, and attention"
    if kind == "spot_listing_watch":
        return "new spot venue can change access and attention"
    if kind == "attention_campaign_watch":
        return "exchange campaign can create short-lived attention and flow"
    if kind == "airdrop_attention_watch":
        return "airdrop announcement can create attention and positioning"
    if kind == "network_event_watch":
        return "network event can alter narrative and operational risk"
    return "ignored catalyst"


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_snapshot.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_snapshot.md",
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--max-age-days", type=int, default=90)
    args = parser.parse_args()

    rows = build_exchange_catalyst_rows(
        binance_payload=fetch_binance_announcements(),
        max_age_days=args.max_age_days,
    )
    write_exchange_catalyst_rows(rows, output_path=args.output_path)
    write_exchange_catalyst_rows_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.catalyst_kind, f"score={row.score:.2f}", row.title)


if __name__ == "__main__":
    main()
