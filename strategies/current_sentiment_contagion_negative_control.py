from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://www.dallasfed.org/research/papers/2026/wp2605"
TARGET_SYMBOLS = ("BTC", "ETH", "HYPE")


@dataclass(frozen=True)
class SentimentContagionControlRow:
    symbol: str
    status: str
    belief_proxy_score: float
    return_support_score: float
    control_gap: float
    strongest_belief_source: str
    price_evidence: str
    control_reason: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_sentiment_contagion_control_rows(
    *,
    ticker_attention_path: Path = ROOT / "news_social" / "current_ticker_attention_source_split.csv",
    event_pressure_path: Path = ROOT / "news_social" / "current_event_pressure_cluster.csv",
    news_quality_path: Path = ROOT / "news_social" / "current_news_event_quality_gate.csv",
    attention_label_path: Path = ROOT / "news_social" / "current_attention_price_labels.csv",
    event_hedge_path: Path = ROOT / "prediction_markets" / "current_event_crypto_hedge_candidates.csv",
    event_alignment_path: Path = ROOT / "prediction_markets" / "current_event_crypto_hedge_event_alignment.csv",
    beta_attribution_path: Path = ROOT / "prediction_markets" / "current_event_crypto_hedge_beta_attribution.csv",
    cross_modal_split_path: Path = ROOT / "current_cross_modal_source_split.csv",
) -> tuple[SentimentContagionControlRow, ...]:
    attention_by_symbol = _best_rows_by_symbol(ticker_attention_path, symbol_key="symbol", score_key="priority")
    event_by_symbol = _best_rows_by_symbol(event_pressure_path, symbol_key="symbol", score_key="score")
    news_by_symbol = _best_rows_by_symbol(news_quality_path, symbol_key="symbol", score_key="score")
    attention_label_by_symbol = _best_rows_by_symbol(attention_label_path, symbol_key="symbol", score_key="priority")
    hedge_by_symbol = _best_rows_by_symbol(event_hedge_path, symbol_key="asset", score_key="score")
    alignment_by_symbol = _best_rows_by_symbol(event_alignment_path, symbol_key="asset", score_key="same_asset_control_gap_bps")
    beta_by_symbol = _best_rows_by_symbol(beta_attribution_path, symbol_key="asset", score_key="residual_vs_basket_bps")
    cross_modal_controls = _cross_modal_control_rows(cross_modal_split_path)
    rows = tuple(
        _build_row(
            symbol=symbol,
            attention=attention_by_symbol.get(symbol),
            event=event_by_symbol.get(symbol),
            news=news_by_symbol.get(symbol),
            attention_label=attention_label_by_symbol.get(symbol),
            hedge=hedge_by_symbol.get(symbol),
            alignment=alignment_by_symbol.get(symbol),
            beta=beta_by_symbol.get(symbol),
            cross_modal_control=cross_modal_controls.get(symbol),
        )
        for symbol in TARGET_SYMBOLS
    )
    return tuple(sorted(rows, key=lambda row: row.control_gap, reverse=True))


def write_sentiment_contagion_control_csv(
    rows: tuple[SentimentContagionControlRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "belief_proxy_score",
                "return_support_score",
                "control_gap",
                "strongest_belief_source",
                "price_evidence",
                "control_reason",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.status,
                    f"{row.belief_proxy_score:.8f}",
                    f"{row.return_support_score:.8f}",
                    f"{row.control_gap:.8f}",
                    row.strongest_belief_source,
                    row.price_evidence,
                    row.control_reason,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_sentiment_contagion_control_md(
    rows: tuple[SentimentContagionControlRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Sentiment Contagion Negative Control\n\n")
        handle.write(
            "This separates belief, attention, and event-probability movement from return-predictive alpha. "
            "It is a negative-control table, not a trade instruction.\n\n"
        )
        handle.write("| symbol | status | belief proxy | return support | gap | strongest source | reason | next probe |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.belief_proxy_score:.4f} | "
                f"{row.return_support_score:.4f} | {row.control_gap:.4f} | "
                f"{_escape(row.strongest_belief_source)} | {_escape(row.control_reason)} | "
                f"{_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A high control gap means attention or event-belief evidence is stronger than clean return evidence. "
            "Those rows should be used as controls or falsification tests before promoting a social/event signal.\n"
        )
    return output_path


def _build_row(
    *,
    symbol: str,
    attention: dict[str, str] | None,
    event: dict[str, str] | None,
    news: dict[str, str] | None,
    attention_label: dict[str, str] | None,
    hedge: dict[str, str] | None,
    alignment: dict[str, str] | None,
    beta: dict[str, str] | None,
    cross_modal_control: dict[str, str] | None,
) -> SentimentContagionControlRow:
    belief_sources = _belief_sources(
        attention=attention,
        event=event,
        hedge=hedge,
        cross_modal_control=cross_modal_control,
    )
    belief_score = sum(score for _, score in belief_sources)
    support_score, price_evidence = _return_support_score(
        news=news,
        attention_label=attention_label,
        alignment=alignment,
        beta=beta,
    )
    gap = max(belief_score - support_score, 0.0)
    strongest_source = max(belief_sources, key=lambda item: item[1], default=("none", 0.0))[0]
    status, reason = _status_reason(
        symbol=symbol,
        gap=gap,
        attention=attention,
        news=news,
        alignment=alignment,
        beta=beta,
        cross_modal_control=cross_modal_control,
    )
    return SentimentContagionControlRow(
        symbol=symbol,
        status=status,
        belief_proxy_score=belief_score,
        return_support_score=support_score,
        control_gap=gap,
        strongest_belief_source=strongest_source,
        price_evidence=price_evidence,
        control_reason=reason,
        missing_data=(
            "social graph, source influence, non-price belief outcome, duplicate-source control, "
            "beta attribution, and leakage-safe return labels"
        ),
        next_probe=_next_probe(symbol=symbol, status=status),
    )


def _belief_sources(
    *,
    attention: dict[str, str] | None,
    event: dict[str, str] | None,
    hedge: dict[str, str] | None,
    cross_modal_control: dict[str, str] | None,
) -> tuple[tuple[str, float], ...]:
    sources: list[tuple[str, float]] = []
    if attention:
        sources.append((f"attention:{attention.get('decision', '')}", min(_float(attention.get("priority")) * 0.45, 80.0)))
    if event:
        sources.append((f"event_pressure:{event.get('status', '')}", min(_float(event.get("score")) * 0.70, 80.0)))
    if hedge:
        sources.append((f"event_probability:{hedge.get('status', '')}", min(_float(hedge.get("score")) * 0.60, 80.0)))
    if cross_modal_control:
        sources.append(
            (
                f"cross_modal_control:{cross_modal_control.get('source', '')}",
                min(_float(cross_modal_control.get("priority_score")) * 0.40, 60.0),
            )
        )
    return tuple(sources)


def _return_support_score(
    *,
    news: dict[str, str] | None,
    attention_label: dict[str, str] | None,
    alignment: dict[str, str] | None,
    beta: dict[str, str] | None,
) -> tuple[float, str]:
    score = 0.0
    evidence: list[str] = []
    if news:
        supported = _float(news.get("supported_count"))
        rejected = _float(news.get("rejected_count"))
        source_count = _float(news.get("source_count"))
        news_score = max(supported - rejected, 0.0) * 15.0 + min(source_count * 5.0, 20.0)
        score += news_score
        evidence.append(
            f"news={news.get('decision', '')}; support/reject={news.get('supported_count', '')}/{news.get('rejected_count', '')}"
        )
    if attention_label:
        label_status = attention_label.get("label_status", "")
        if label_status.startswith("labeled") or label_status.startswith("direction_supported"):
            label_score = min(abs(_float(attention_label.get("directional_return_1h"))) * 500.0, 30.0)
            score += label_score
        evidence.append(f"attention_label={label_status}")
    if alignment:
        align_status = alignment.get("alignment_status", "")
        if align_status == "event_probability_flat_crypto_moved":
            score *= 0.5
            evidence.append("event_probability_flat_crypto_moved")
        else:
            score += min(abs(_float(alignment.get("same_asset_control_gap_bps"))) * 0.10, 25.0)
            evidence.append(f"event_alignment={align_status}")
    if beta:
        beta_status = beta.get("attribution_status", "")
        if beta_status == "event_crypto_beta_move_supported":
            score *= 0.65
            evidence.append("beta_move_supported")
        else:
            evidence.append(f"beta={beta_status}")
    return score, " || ".join(evidence) if evidence else "no return-support evidence"


def _status_reason(
    *,
    symbol: str,
    gap: float,
    attention: dict[str, str] | None,
    news: dict[str, str] | None,
    alignment: dict[str, str] | None,
    beta: dict[str, str] | None,
    cross_modal_control: dict[str, str] | None,
) -> tuple[str, str]:
    if alignment and alignment.get("alignment_status") == "event_probability_flat_crypto_moved":
        return (
            "belief_price_decoupling_control_required",
            "crypto moved while the event-probability ticket was flat; do not treat the belief market as causal yet",
        )
    if beta and beta.get("attribution_status") == "event_crypto_beta_move_supported":
        return (
            "beta_move_negative_control_required",
            "the reaction is explained by crypto beta before event-specific belief alpha is proven",
        )
    if cross_modal_control and cross_modal_control.get("paper_action") == "label_conflict_or_negative_control":
        return (
            "conflicting_social_source_control",
            "cross-modal source split already marks this source as a conflict or negative control",
        )
    if attention and attention.get("decision") == "dedupe_news_before_attention_label":
        return (
            "attention_news_dedupe_control_required",
            "attention and news/event pressure are duplicated before a standalone attention alpha is proven",
        )
    if news and _float(news.get("source_count")) <= 1 and _float(news.get("supported_count")) > 0:
        return (
            "single_source_story_control_required",
            "return support exists but comes from one story source, so source contagion is not ruled out",
        )
    if gap >= 80.0:
        return (
            "belief_proxy_dominates_return_support",
            f"{symbol} has stronger belief/attention evidence than clean return evidence",
        )
    return "control_watch", "keep as social contagion control until stronger independent return evidence appears"


def _next_probe(*, symbol: str, status: str) -> str:
    if status == "belief_price_decoupling_control_required":
        return f"for {symbol}, require event-probability movement or stronger timestamp evidence before event hedge promotion"
    if status == "beta_move_negative_control_required":
        return f"for {symbol}, beta-adjust the event/attention label before calling it alpha"
    if status == "attention_news_dedupe_control_required":
        return f"for {symbol}, dedupe attention/news sources and add a non-price belief outcome"
    if status == "single_source_story_control_required":
        return f"for {symbol}, require independent source repeat before social/event alpha promotion"
    return f"keep {symbol} as a negative control in the social/event alpha lane"


def _cross_modal_control_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        if row.get("paper_action") != "label_conflict_or_negative_control":
            continue
        symbol = row.get("symbol", "").upper()
        if not symbol:
            continue
        if symbol not in rows or _float(row.get("priority_score")) > _float(rows[symbol].get("priority_score")):
            rows[symbol] = row
    return rows


def _best_rows_by_symbol(path: Path, *, symbol_key: str, score_key: str) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        symbol = row.get(symbol_key, "").upper()
        if not symbol:
            continue
        if symbol not in best or abs(_float(row.get(score_key))) > abs(_float(best[symbol].get(score_key))):
            best[symbol] = row
    return best


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_sentiment_contagion_negative_control.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_sentiment_contagion_negative_control.md",
    )
    args = parser.parse_args()

    rows = build_sentiment_contagion_control_rows()
    write_sentiment_contagion_control_csv(rows, output_path=args.output_path)
    write_sentiment_contagion_control_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(row.status, row.symbol, f"gap={row.control_gap:.4f}", row.strongest_belief_source)


if __name__ == "__main__":
    main()
