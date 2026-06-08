from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://www.sciencedirect.com/science/article/abs/pii/S0378426625001384"


@dataclass(frozen=True)
class TickerAttentionSourceSplitRow:
    timestamp: str
    symbol: str
    name: str
    source: str
    source_specificity: str
    attention_rank: int
    attention_signal: str
    joined_context: str
    event_cluster_status: str
    source_independence_status: str
    decision: str
    priority: float
    evidence: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_ticker_attention_source_split_rows(
    *,
    attention_path: Path = ROOT / "current_attention_snapshot.csv",
    attention_price_context_path: Path = ROOT / "current_attention_price_context.csv",
    attention_market_join_path: Path = ROOT / "current_attention_market_join.csv",
    event_pressure_cluster_path: Path = ROOT / "current_event_pressure_cluster.csv",
    source_independence_path: Path = ROOT / "current_news_event_source_independence.csv",
) -> tuple[TickerAttentionSourceSplitRow, ...]:
    price_context_by_symbol = _rows_by_symbol(attention_price_context_path)
    market_join_by_symbol = _rows_by_symbol(attention_market_join_path)
    event_cluster_by_symbol = _rows_by_symbol(event_pressure_cluster_path)
    source_independence_by_symbol = _best_rows_by_symbol(source_independence_path, score_key="score")
    rows = tuple(
        _build_row(
            attention=row,
            price_context=price_context_by_symbol.get(row.get("symbol", "").upper()),
            market_join=market_join_by_symbol.get(row.get("symbol", "").upper()),
            event_cluster=event_cluster_by_symbol.get(row.get("symbol", "").upper()),
            source_independence=source_independence_by_symbol.get(row.get("symbol", "").upper()),
        )
        for row in _read_rows(attention_path)
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_ticker_attention_source_split_csv(
    rows: tuple[TickerAttentionSourceSplitRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "name",
                "source",
                "source_specificity",
                "attention_rank",
                "attention_signal",
                "joined_context",
                "event_cluster_status",
                "source_independence_status",
                "decision",
                "priority",
                "evidence",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.symbol,
                    row.name,
                    row.source,
                    row.source_specificity,
                    row.attention_rank,
                    row.attention_signal,
                    row.joined_context,
                    row.event_cluster_status,
                    row.source_independence_status,
                    row.decision,
                    f"{row.priority:.8f}",
                    row.evidence,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_ticker_attention_source_split_md(
    rows: tuple[TickerAttentionSourceSplitRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Ticker Attention Source Split\n\n")
        handle.write(
            "This separates ticker-specific attention from broad market sentiment and duplicated "
            "news/event clusters before paper labels. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | source | specificity | decision | priority | context | event cluster | next probe |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.source} | {row.source_specificity} | {row.decision} | "
                f"{row.priority:.4f} | {_escape(row.joined_context)} | "
                f"{_escape(row.event_cluster_status)} | {_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`ticker_specific_attention_alpha_candidate` means the attention observation is mapped to a "
            "single ticker and has market context. `dedupe_news_before_attention_label` means the ticker "
            "attention is mixed with RSS or exchange-event pressure and must be separated before labeling. "
            "`broad_market_sentiment_control` is a control input, not a ticker alpha candidate.\n"
        )
    return output_path


def _build_row(
    *,
    attention: dict[str, str],
    price_context: dict[str, str] | None,
    market_join: dict[str, str] | None,
    event_cluster: dict[str, str] | None,
    source_independence: dict[str, str] | None,
) -> TickerAttentionSourceSplitRow:
    symbol = attention.get("symbol", "").upper()
    source = attention.get("source", "")
    rank = int(attention.get("rank") or "0")
    attention_signal = _attention_signal(attention)
    joined_context = _joined_context(price_context=price_context, market_join=market_join)
    event_status = _event_cluster_status(event_cluster)
    independence_status = (source_independence or {}).get("independence_status", "")
    decision, priority, missing_data, next_probe = _decision_priority_missing_next(
        symbol=symbol,
        source=source,
        rank=rank,
        price_context=price_context,
        market_join=market_join,
        event_cluster=event_cluster,
        source_independence=source_independence,
    )
    return TickerAttentionSourceSplitRow(
        timestamp=attention.get("timestamp", ""),
        symbol=symbol,
        name=attention.get("name", ""),
        source=source,
        source_specificity=_source_specificity(attention),
        attention_rank=rank,
        attention_signal=attention_signal,
        joined_context=joined_context,
        event_cluster_status=event_status,
        source_independence_status=independence_status,
        decision=decision,
        priority=priority,
        evidence=_evidence(attention=attention, price_context=price_context, market_join=market_join),
        missing_data=missing_data,
        next_probe=next_probe,
    )


def _decision_priority_missing_next(
    *,
    symbol: str,
    source: str,
    rank: int,
    price_context: dict[str, str] | None,
    market_join: dict[str, str] | None,
    event_cluster: dict[str, str] | None,
    source_independence: dict[str, str] | None,
) -> tuple[str, float, str, str]:
    if symbol == "MARKET" or source == "alternative_me_fear_greed":
        return (
            "broad_market_sentiment_control",
            20.0 - rank,
            "asset-level mapping and ticker-specific source identity",
            "keep fear/greed as market regime control, not as ticker paper alpha",
        )
    has_market_context = price_context is not None or market_join is not None
    top_sources = (event_cluster or {}).get("top_sources", "")
    mixed_with_news = any(token in top_sources for token in ("rss:", "exchange_catalyst"))
    base_priority = max(30.0, 120.0 - (rank * 3.0))
    context_bonus = _float((price_context or {}).get("score")) + _float((market_join or {}).get("score"))
    event_bonus = min(_float((event_cluster or {}).get("score")), 25.0)
    independence_bonus = min(max(_float((source_independence or {}).get("score")), 0.0), 10.0)
    priority = base_priority + (context_bonus * 0.25) + event_bonus + independence_bonus
    if not has_market_context:
        return (
            "source_quality_required",
            priority - 30.0,
            "market context, venue support, and forward labels",
            f"collect market context for {symbol} before treating attention as alpha",
        )
    if mixed_with_news:
        return (
            "dedupe_news_before_attention_label",
            priority,
            "ticker-level source account identity, RSS/event dedupe, and timestamp control",
            f"label {symbol} ticker attention separately from RSS/exchange event pressure",
        )
    return (
        "ticker_specific_attention_alpha_candidate",
        priority,
        "source-account identity, repeated observations, and 15m/1h/4h return labels",
        f"paper-label {symbol} ticker-specific attention against price, funding, and depth",
    )


def _attention_signal(row: dict[str, str]) -> str:
    if row.get("source") == "alternative_me_fear_greed":
        return f"fear_greed={row.get('score', '')}; label={row.get('label', '')}"
    return f"trending_rank={row.get('rank', '')}; 24h_change={row.get('value', '')}"


def _joined_context(
    *,
    price_context: dict[str, str] | None,
    market_join: dict[str, str] | None,
) -> str:
    parts: list[str] = []
    if price_context:
        parts.append(
            "price="
            f"{price_context.get('status', '')}; "
            f"side={price_context.get('side', '')}; "
            f"score={price_context.get('score', '')}"
        )
    if market_join:
        parts.append(
            "perp="
            f"{market_join.get('action', '')}; "
            f"score={market_join.get('score', '')}"
        )
    return " || ".join(parts) if parts else "no joined market context"


def _event_cluster_status(row: dict[str, str] | None) -> str:
    if not row:
        return "no event cluster"
    return (
        f"{row.get('status', '')}; "
        f"sources={row.get('source_count', '')}; "
        f"top={row.get('top_sources', '')}"
    )


def _source_specificity(row: dict[str, str]) -> str:
    if row.get("symbol", "").upper() == "MARKET":
        return "broad_market"
    if row.get("source") == "coingecko_trending" and row.get("asset_id"):
        return "ticker_mapped"
    return "unknown"


def _evidence(
    *,
    attention: dict[str, str],
    price_context: dict[str, str] | None,
    market_join: dict[str, str] | None,
) -> str:
    parts = [_attention_signal(attention)]
    if price_context:
        parts.append((price_context.get("evidence") or "").replace("\n", " "))
    if market_join:
        parts.append(
            f"funding={market_join.get('annualized_funding', '')}; "
            f"spread={market_join.get('impact_spread', '')}"
        )
    return " || ".join(part for part in parts if part)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _rows_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("symbol", "").upper(): row for row in _read_rows(path)}


def _best_rows_by_symbol(path: Path, *, score_key: str) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        symbol = row.get("symbol", "").upper()
        if not symbol:
            continue
        if symbol not in best or _float(row.get(score_key)) > _float(best[symbol].get(score_key)):
            best[symbol] = row
    return best


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-path", type=Path, default=ROOT / "current_attention_snapshot.csv")
    parser.add_argument(
        "--attention-price-context-path",
        type=Path,
        default=ROOT / "current_attention_price_context.csv",
    )
    parser.add_argument(
        "--attention-market-join-path",
        type=Path,
        default=ROOT / "current_attention_market_join.csv",
    )
    parser.add_argument(
        "--event-pressure-cluster-path",
        type=Path,
        default=ROOT / "current_event_pressure_cluster.csv",
    )
    parser.add_argument(
        "--source-independence-path",
        type=Path,
        default=ROOT / "current_news_event_source_independence.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_ticker_attention_source_split.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_ticker_attention_source_split.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_ticker_attention_source_split_rows(
        attention_path=args.attention_path,
        attention_price_context_path=args.attention_price_context_path,
        attention_market_join_path=args.attention_market_join_path,
        event_pressure_cluster_path=args.event_pressure_cluster_path,
        source_independence_path=args.source_independence_path,
    )
    write_ticker_attention_source_split_csv(rows, output_path=args.output_path)
    write_ticker_attention_source_split_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.decision, row.symbol, f"priority={row.priority:.4f}", row.next_probe)


if __name__ == "__main__":
    main()
