from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5079335"


@dataclass(frozen=True)
class CrossSectionRow:
    symbol: str
    decision: str
    side_hint: str
    total_score: float
    fundamental_score: float
    sentiment_score: float
    sector_score: float
    funding_score: float
    source_count: int
    evidence: str
    conflict: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_cross_section_rows(
    *,
    protocol_fee_path: Path = ROOT / "protocol_fundamentals" / "current_protocol_fee_actionability.csv",
    ticker_attention_path: Path = ROOT / "news_social" / "current_ticker_attention_source_split.csv",
    sector_context_path: Path = ROOT / "sector_rotation" / "current_category_perp_context.csv",
    funding_spread_path: Path = ROOT / "cross_exchange_funding" / "current_funding_spread.csv",
    okx_hl_funding_path: Path = ROOT / "cross_exchange_funding" / "current_okx_hl_funding_spread.csv",
) -> tuple[CrossSectionRow, ...]:
    protocol_by_symbol = _best_rows_by_symbol(protocol_fee_path, symbol_key="token_symbol", score_key="score")
    attention_by_symbol = _best_rows_by_symbol(ticker_attention_path, symbol_key="symbol", score_key="priority")
    sector_by_symbol = _best_rows_by_symbol(sector_context_path, symbol_key="symbol", score_key="context_score")
    funding_by_symbol = _best_rows_by_symbol(funding_spread_path, symbol_key="asset", score_key="annualized_spread")
    okx_hl_by_symbol = _best_rows_by_symbol(okx_hl_funding_path, symbol_key="asset", score_key="net_24h_proxy")
    symbols = sorted(
        set(protocol_by_symbol)
        | set(attention_by_symbol)
        | set(sector_by_symbol)
        | set(funding_by_symbol)
        | set(okx_hl_by_symbol)
    )
    rows = tuple(
        _build_row(
            symbol=symbol,
            protocol=protocol_by_symbol.get(symbol),
            attention=attention_by_symbol.get(symbol),
            sector=sector_by_symbol.get(symbol),
            funding=funding_by_symbol.get(symbol),
            okx_hl=okx_hl_by_symbol.get(symbol),
        )
        for symbol in symbols
    )
    return tuple(sorted(rows, key=lambda row: row.total_score, reverse=True))


def write_cross_section_csv(rows: tuple[CrossSectionRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "decision",
                "side_hint",
                "total_score",
                "fundamental_score",
                "sentiment_score",
                "sector_score",
                "funding_score",
                "source_count",
                "evidence",
                "conflict",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.decision,
                    row.side_hint,
                    f"{row.total_score:.8f}",
                    f"{row.fundamental_score:.8f}",
                    f"{row.sentiment_score:.8f}",
                    f"{row.sector_score:.8f}",
                    f"{row.funding_score:.8f}",
                    row.source_count,
                    row.evidence,
                    row.conflict,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_cross_section_md(rows: tuple[CrossSectionRow, ...], *, output_path: Path, top: int = 25) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Fundamental Sentiment Cross Section\n\n")
        handle.write(
            "This is a first cross-sectional feature table from existing fundamental, sentiment, "
            "sector, and funding probes. It ranks research candidates; it is not a rebalance rule.\n\n"
        )
        handle.write(
            "| symbol | decision | side | total | fundamental | sentiment | sector | funding | sources | conflict | next probe |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.decision} | {row.side_hint} | {row.total_score:.4f} | "
                f"{row.fundamental_score:.4f} | {row.sentiment_score:.4f} | {row.sector_score:.4f} | "
                f"{row.funding_score:.4f} | {row.source_count} | {_escape(row.conflict)} | "
                f"{_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows with multiple sources are better cross-section candidates than single-lane rows. "
            "`split_conflicting_features_before_label` means the feature signs disagree and should not be "
            "collapsed into one trade. The next step is leakage-safe forward labeling by rebalance timestamp.\n"
        )
    return output_path


def _build_row(
    *,
    symbol: str,
    protocol: dict[str, str] | None,
    attention: dict[str, str] | None,
    sector: dict[str, str] | None,
    funding: dict[str, str] | None,
    okx_hl: dict[str, str] | None,
) -> CrossSectionRow:
    fundamental_score, fundamental_evidence = _fundamental_score(protocol)
    sentiment_score, sentiment_evidence = _sentiment_score(attention)
    sector_score, sector_evidence = _sector_score(sector)
    funding_score, funding_evidence = _funding_score(funding=funding, okx_hl=okx_hl)
    directional_scores = tuple(
        score
        for score in (fundamental_score, sentiment_score, sector_score)
        if abs(score) >= 10.0
    )
    source_count = sum(component is not None for component in (protocol, attention, sector, funding or okx_hl))
    conflict = _conflict(directional_scores)
    total_score = (
        abs(fundamental_score)
        + abs(sentiment_score)
        + abs(sector_score)
        + max(funding_score, 0.0)
        + (source_count * 8.0)
        - (25.0 if conflict else 0.0)
    )
    if source_count < 2:
        total_score = min(total_score, 65.0)
    decision = _decision(total_score=total_score, source_count=source_count, conflict=conflict)
    side_hint = _side_hint(directional_scores)
    return CrossSectionRow(
        symbol=symbol,
        decision=decision,
        side_hint=side_hint,
        total_score=total_score,
        fundamental_score=fundamental_score,
        sentiment_score=sentiment_score,
        sector_score=sector_score,
        funding_score=funding_score,
        source_count=source_count,
        evidence=" || ".join(
            part for part in (fundamental_evidence, sentiment_evidence, sector_evidence, funding_evidence) if part
        ),
        conflict=conflict or "none",
        missing_data="neutral universe, rebalance timestamp, forward labels, feature ablation, and transaction costs",
        next_probe=_next_probe(symbol=symbol, decision=decision, side_hint=side_hint),
    )


def _fundamental_score(row: dict[str, str] | None) -> tuple[float, str]:
    if not row:
        return 0.0, ""
    score = _float(row.get("score"))
    side = row.get("side", "")
    sign = 1.0 if side == "long_token" else -0.7 if side == "watch_or_short" else 0.0
    directional_score = score * sign
    evidence = (
        f"fundamental={row.get('protocol', '')}; side={side}; "
        f"status={row.get('status', '')}; score={row.get('score', '')}"
    )
    return directional_score, evidence


def _sentiment_score(row: dict[str, str] | None) -> tuple[float, str]:
    if not row:
        return 0.0, ""
    decision = row.get("decision", "")
    priority = _float(row.get("priority"))
    if decision == "ticker_specific_attention_alpha_candidate":
        score = priority * 0.45
    elif decision == "dedupe_news_before_attention_label":
        score = priority * 0.25
    elif decision == "source_quality_required":
        score = priority * 0.10
    else:
        score = 0.0
    evidence = f"sentiment={decision}; source={row.get('source', '')}; priority={row.get('priority', '')}"
    return score, evidence


def _sector_score(row: dict[str, str] | None) -> tuple[float, str]:
    if not row:
        return 0.0, ""
    direction = _float(row.get("direction"))
    score = _float(row.get("context_score")) * 40.0 * direction
    evidence = (
        f"sector={row.get('category_name', '')}; action={row.get('action', '')}; "
        f"context={row.get('context_score', '')}"
    )
    return score, evidence


def _funding_score(
    *,
    funding: dict[str, str] | None,
    okx_hl: dict[str, str] | None,
) -> tuple[float, str]:
    spread_score = min(abs(_float((funding or {}).get("annualized_spread"))) * 8.0, 40.0)
    okx_net_score = min(max(_float((okx_hl or {}).get("net_24h_proxy")), 0.0) * 4000.0, 25.0)
    score = spread_score + okx_net_score
    evidence_parts: list[str] = []
    if funding:
        evidence_parts.append(
            f"funding_spread={funding.get('annualized_spread', '')}; "
            f"long={funding.get('long_venue', '')}; short={funding.get('short_venue', '')}"
        )
    if okx_hl:
        evidence_parts.append(
            f"okx_hl_net24={okx_hl.get('net_24h_proxy', '')}; "
            f"capacity={okx_hl.get('capacity_proxy_notional', '')}"
        )
    return score, " || ".join(evidence_parts)


def _decision(*, total_score: float, source_count: int, conflict: str) -> str:
    if conflict and source_count >= 2:
        return "split_conflicting_features_before_label"
    if source_count >= 3 and total_score >= 120.0:
        return "cross_section_label_priority"
    if source_count >= 2 and total_score >= 70.0:
        return "cross_section_watchlist"
    return "insufficient_cross_section_context"


def _side_hint(scores: tuple[float, ...]) -> str:
    if not scores:
        return "none"
    total = sum(scores)
    if total > 10.0:
        return "long_bias"
    if total < -10.0:
        return "short_bias"
    return "mixed_or_flat"


def _conflict(scores: tuple[float, ...]) -> str:
    positive = any(score > 10.0 for score in scores)
    negative = any(score < -10.0 for score in scores)
    return "long_short_feature_conflict" if positive and negative else ""


def _next_probe(*, symbol: str, decision: str, side_hint: str) -> str:
    if decision == "split_conflicting_features_before_label":
        return f"split {symbol} features by sign before any cross-section label"
    if decision == "cross_section_label_priority":
        return f"label {symbol} {side_hint} cross-section row at the next rebalance timestamp"
    if decision == "cross_section_watchlist":
        return f"collect one more independent {symbol} feature before cross-section labeling"
    return f"keep {symbol} as context until another feature source appears"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _best_rows_by_symbol(path: Path, *, symbol_key: str, score_key: str) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        symbol = row.get(symbol_key, "").upper()
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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_fundamental_sentiment_cross_section.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_fundamental_sentiment_cross_section.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_cross_section_rows()
    write_cross_section_csv(rows, output_path=args.output_path)
    write_cross_section_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.decision, row.symbol, row.side_hint, f"score={row.total_score:.4f}", row.conflict)


if __name__ == "__main__":
    main()
