from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import requests


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent
FEEDS = (
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss"),
    ("decrypt", "https://decrypt.co/feed"),
)
TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,12}\b")
IGNORED_SYMBOLS = {
    "AI",
    "API",
    "CEO",
    "CEX",
    "CFTC",
    "DAO",
    "DEX",
    "ETF",
    "ETFS",
    "FED",
    "IPO",
    "NFT",
    "SEC",
    "TVL",
    "USD",
}
BASE_ALLOWED_SYMBOLS = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "BNB",
    "USDT",
    "USDC",
}
KEYWORD_SYMBOLS = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "bnb": "BNB",
    "hyperliquid": "HYPE",
    "zcash": "ZEC",
    "zec": "ZEC",
    "microstrategy": "BTC",
    "strategy": "BTC",
    "tether": "USDT",
    "usdt": "USDT",
    "circle": "USDC",
    "usdc": "USDC",
    "usd coin": "USDC",
}


@dataclass(frozen=True)
class NewsEventRow:
    observed_at: str
    source: str
    published_at: str
    age_hours: float
    symbol: str
    event_kind: str
    direction_hint: int
    status: str
    side: str
    score: float
    title: str
    category: str
    link: str
    perp_action: str
    annualized_funding: float
    impact_spread: float
    reason: str
    next_step: str


def fetch_feed_items(name: str, url: str, *, limit: int = 30) -> tuple[dict[str, str], ...]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    items: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = _node_text(item, "title")
        published_at = _node_text(item, "pubDate")
        link = _node_text(item, "link")
        category = _node_text(item, "category")
        if not title or not published_at:
            continue
        items.append(
            {
                "source": name,
                "published_at": _iso_datetime(published_at),
                "title": _clean_text(title),
                "category": _clean_text(category),
                "link": _clean_text(link),
            }
        )
        if len(items) >= limit:
            break
    return tuple(items)


def build_news_event_rows(
    feed_items: tuple[dict[str, str], ...],
    *,
    perp_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_screen.csv",
    observed_at: str | None = None,
    max_age_hours: float = 72.0,
) -> tuple[NewsEventRow, ...]:
    now = datetime.now(UTC)
    timestamp = observed_at or now.isoformat()
    perp_by_symbol = _rows_by_symbol(perp_path)
    allowed_symbols = set(perp_by_symbol) | set(KEYWORD_SYMBOLS.values()) | BASE_ALLOWED_SYMBOLS
    output: list[NewsEventRow] = []
    seen: set[tuple[str, str, str]] = set()
    for item in feed_items:
        published_at = _parse_datetime(item["published_at"])
        age_hours = max((now - published_at).total_seconds() / 3600.0, 0.0)
        if age_hours > max_age_hours:
            continue
        event_kind, direction_hint = _event_kind(item["title"], item["category"])
        if event_kind == "ignore":
            continue
        for symbol in _symbols(item["title"], allowed_symbols=allowed_symbols):
            key = (item["source"], item["title"], symbol)
            if key in seen:
                continue
            seen.add(key)
            perp = perp_by_symbol.get(symbol, {})
            row = _build_row(
                item=item,
                symbol=symbol,
                event_kind=event_kind,
                direction_hint=direction_hint,
                age_hours=age_hours,
                observed_at=timestamp,
                perp=perp,
            )
            output.append(row)
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_news_event_csv(rows: tuple[NewsEventRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observed_at",
                "source",
                "published_at",
                "age_hours",
                "symbol",
                "event_kind",
                "direction_hint",
                "status",
                "side",
                "score",
                "title",
                "category",
                "link",
                "perp_action",
                "annualized_funding",
                "impact_spread",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.observed_at,
                    row.source,
                    row.published_at,
                    f"{row.age_hours:.6f}",
                    row.symbol,
                    row.event_kind,
                    row.direction_hint,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    row.title,
                    row.category,
                    row.link,
                    row.perp_action,
                    f"{row.annualized_funding:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_news_event_md(rows: tuple[NewsEventRow, ...], *, output_path: Path, top: int = 25) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current News Event Screen\n\n")
        handle.write(
            "This classifies current crypto RSS headlines into event candidates and joins them to current perp state. "
            "It stores headline metadata only and is not a trade instruction.\n\n"
        )
        handle.write(
            "| source | published | symbol | kind | status | side | score | funding | perp action | title |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.source} | {row.published_at} | {row.symbol} | {row.event_kind} | "
                f"{row.status} | {row.side} | {row.score:.4f} | "
                f"{row.annualized_funding:.6f} | {row.perp_action or '-'} | {_escape_md(row.title)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "News is a catalyst source, not an edge by itself. Rows need timestamp leakage checks, "
            "duplicate-source checks, venue depth, and forward-return labels before paper action.\n"
        )
    return output_path


def _build_row(
    *,
    item: dict[str, str],
    symbol: str,
    event_kind: str,
    direction_hint: int,
    age_hours: float,
    observed_at: str,
    perp: dict[str, str],
) -> NewsEventRow:
    annualized_funding = _float(perp.get("annualized_funding"))
    impact_spread = _float(perp.get("impact_spread"))
    perp_action = perp.get("action", "")
    status, side = _status_side(event_kind=event_kind, direction_hint=direction_hint, perp_action=perp_action)
    score = _score(
        event_kind=event_kind,
        direction_hint=direction_hint,
        age_hours=age_hours,
        has_perp=bool(perp),
        annualized_funding=annualized_funding,
        impact_spread=impact_spread,
    )
    return NewsEventRow(
        observed_at=observed_at,
        source=item["source"],
        published_at=item["published_at"],
        age_hours=age_hours,
        symbol=symbol,
        event_kind=event_kind,
        direction_hint=direction_hint,
        status=status,
        side=side,
        score=score,
        title=item["title"],
        category=item["category"],
        link=item["link"],
        perp_action=perp_action,
        annualized_funding=annualized_funding,
        impact_spread=impact_spread,
        reason=_reason(event_kind=event_kind, direction_hint=direction_hint, perp_action=perp_action),
        next_step=(
            f"label {symbol} returns after this {event_kind} headline and check duplicate-source timing, "
            "venue depth, spread, and funding cost"
        ),
    )


def _event_kind(title: str, category: str) -> tuple[str, int]:
    text = f"{title} {category}".lower()
    if any(word in text for word in ("hack", "exploit", "breach", "stolen", "scam", "phishing")):
        return "security_risk", -1
    if any(word in text for word in ("sec", "cftc", "lawsuit", "sued", "ban", "sanction", "probe", "regulat")):
        return "regulatory_risk", -1
    if any(word in text for word in ("etf", "treasury", "blackrock", "strategy", "microstrategy", "institutional")):
        return "institutional_flow", 1
    if any(word in text for word in ("listing", "lists", "launches trading", "perpetual", "futures")):
        return "listing_or_liquidity", 1
    if any(word in text for word in ("nasdaq", "stock", "fed", "rates", "inflation", "dollar", "yields")):
        return "macro_crypto", 0
    if any(word in text for word in ("stablecoin", "depeg", "reserve", "tether", "circle")):
        return "stablecoin_event", 0
    if any(word in text for word in ("revenue", "fees", "earnings", "users", "tvl")):
        return "fundamental_event", 1
    if any(word in text for word in ("ai", "rwa", "tokenization", "memecoin", "narrative")):
        return "narrative_event", 1
    return "ignore", 0


def _symbols(title: str, *, allowed_symbols: set[str]) -> tuple[str, ...]:
    text = title.lower()
    symbols = {symbol for keyword, symbol in KEYWORD_SYMBOLS.items() if keyword in text}
    symbols.update(
        match.group(0)
        for match in TOKEN_RE.finditer(title)
        if match.group(0) not in IGNORED_SYMBOLS and match.group(0) in allowed_symbols
    )
    return tuple(sorted(symbols))


def _status_side(*, event_kind: str, direction_hint: int, perp_action: str) -> tuple[str, str]:
    if event_kind == "security_risk":
        return "paper_news_security_risk_watch", "short_or_avoid"
    if event_kind == "regulatory_risk":
        return "paper_news_regulatory_risk_watch", "short_or_avoid"
    if event_kind == "macro_crypto":
        return "paper_news_macro_crypto_watch", "risk_context"
    if direction_hint > 0 and "long" in perp_action:
        return "paper_news_event_reaction_watch", "long_event_follow"
    if direction_hint < 0 and "short" in perp_action:
        return "paper_news_event_reaction_watch", "short_event_follow"
    return "paper_news_context_watch", "collect_label"


def _score(
    *,
    event_kind: str,
    direction_hint: int,
    age_hours: float,
    has_perp: bool,
    annualized_funding: float,
    impact_spread: float,
) -> float:
    event_score = {
        "security_risk": 32.0,
        "regulatory_risk": 30.0,
        "institutional_flow": 28.0,
        "macro_crypto": 24.0,
        "listing_or_liquidity": 22.0,
        "stablecoin_event": 20.0,
        "fundamental_event": 18.0,
        "narrative_event": 15.0,
    }.get(event_kind, 0.0)
    recency_score = max(24.0 - age_hours, 0.0)
    perp_score = 8.0 if has_perp else 0.0
    funding_alignment = abs(annualized_funding) * 2.0 if direction_hint != 0 else abs(annualized_funding)
    friction_penalty = impact_spread * 100.0
    return event_score + recency_score + perp_score + funding_alignment - friction_penalty


def _reason(*, event_kind: str, direction_hint: int, perp_action: str) -> str:
    if perp_action:
        return f"{event_kind} headline overlaps with current perp state {perp_action}"
    if direction_hint == 0:
        return f"{event_kind} headline is market context without a single-direction signal"
    return f"{event_kind} headline needs forward labels and execution checks"


def _node_text(item: ElementTree.Element, tag: str) -> str:
    node = item.find(tag)
    if node is None or node.text is None:
        return ""
    return node.text


def _clean_text(value: str) -> str:
    return " ".join(html.unescape(value).split())


def _iso_datetime(value: str) -> str:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _rows_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row.get("asset", "").upper(): row for row in csv.DictReader(handle)}


def _float(value: str | None) -> float:
    return float(value or "0")


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_news_event_screen.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_news_event_screen.md")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    items: list[dict[str, str]] = []
    for name, url in FEEDS:
        items.extend(fetch_feed_items(name, url))
    rows = build_news_event_rows(tuple(items))
    write_news_event_csv(rows, output_path=args.output_path)
    write_news_event_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.symbol, row.event_kind, f"score={row.score:.4f}", row.source)


if __name__ == "__main__":
    main()
