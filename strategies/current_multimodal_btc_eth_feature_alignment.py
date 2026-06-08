from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://www.sciencedirect.com/science/article/pii/S0169207025000147"


@dataclass(frozen=True)
class MultimodalFeatureRow:
    symbol: str
    status: str
    feature_count: int
    alignment_score: float
    nlp_event_score: float
    ticker_attention_score: float
    stablecoin_flow_score: float
    wallet_flow_score: float
    funding_market_score: float
    equity_factor_score: float
    feature_summary: str
    timestamp_boundary: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_multimodal_feature_rows(
    *,
    event_pressure_path: Path = ROOT / "news_social" / "current_event_pressure_cluster.csv",
    ticker_attention_path: Path = ROOT / "news_social" / "current_ticker_attention_source_split.csv",
    stablecoin_proxy_path: Path = ROOT / "stablecoin_liquidity" / "current_stablecoin_exchange_inflow_proxy.csv",
    wallet_flow_path: Path = ROOT / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv",
    funding_path: Path = ROOT / "cross_exchange_funding" / "current_okx_hl_funding_spread.csv",
    equity_factor_path: Path = ROOT / "crypto_equity_proxy" / "current_crypto_equity_factor_split.csv",
) -> tuple[MultimodalFeatureRow, ...]:
    event_by_symbol = _best_rows_by_symbol(event_pressure_path, symbol_key="symbol", score_key="score")
    attention_by_symbol = _best_rows_by_symbol(ticker_attention_path, symbol_key="symbol", score_key="priority")
    stablecoin_by_symbol = _best_rows_by_symbol(stablecoin_proxy_path, symbol_key="token_symbol", score_key="priority")
    wallet_by_symbol = _best_rows_by_symbol(wallet_flow_path, symbol_key="execution_asset", score_key="score")
    funding_by_symbol = _best_rows_by_symbol(funding_path, symbol_key="asset", score_key="net_24h_proxy")
    equity_rows = _read_rows(equity_factor_path)
    rows = tuple(
        _build_row(
            symbol=symbol,
            event=event_by_symbol.get(symbol),
            attention=attention_by_symbol.get(symbol),
            stablecoin=stablecoin_by_symbol.get(symbol),
            wallet=wallet_by_symbol.get(symbol),
            funding=funding_by_symbol.get(symbol),
            equity=_equity_factor_for_symbol(symbol, equity_rows),
        )
        for symbol in ("BTC", "ETH")
    )
    return tuple(sorted(rows, key=lambda row: row.alignment_score, reverse=True))


def write_multimodal_feature_csv(rows: tuple[MultimodalFeatureRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "feature_count",
                "alignment_score",
                "nlp_event_score",
                "ticker_attention_score",
                "stablecoin_flow_score",
                "wallet_flow_score",
                "funding_market_score",
                "equity_factor_score",
                "feature_summary",
                "timestamp_boundary",
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
                    row.feature_count,
                    f"{row.alignment_score:.8f}",
                    f"{row.nlp_event_score:.8f}",
                    f"{row.ticker_attention_score:.8f}",
                    f"{row.stablecoin_flow_score:.8f}",
                    f"{row.wallet_flow_score:.8f}",
                    f"{row.funding_market_score:.8f}",
                    f"{row.equity_factor_score:.8f}",
                    row.feature_summary,
                    row.timestamp_boundary,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_multimodal_feature_md(
    rows: tuple[MultimodalFeatureRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Multimodal BTC/ETH Feature Alignment\n\n")
        handle.write(
            "This aligns NLP/news, ticker attention, stablecoin/on-chain proxy, wallet flow, "
            "funding market, and crypto-equity factor features for BTC and ETH. It is a feature "
            "alignment table, not a model or trade instruction.\n\n"
        )
        handle.write(
            "| symbol | status | features | alignment | nlp | attention | stablecoin | wallet | funding | equity | boundary | next probe |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.feature_count} | {row.alignment_score:.4f} | "
                f"{row.nlp_event_score:.4f} | {row.ticker_attention_score:.4f} | "
                f"{row.stablecoin_flow_score:.4f} | {row.wallet_flow_score:.4f} | "
                f"{row.funding_market_score:.4f} | {row.equity_factor_score:.4f} | "
                f"{_escape(row.timestamp_boundary)} | {_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A high alignment score means many feature families are present for the asset. It does not "
            "mean the features are causal or tradable. The next step is a leakage-safe timestamp table "
            "with feature ablation and beta-adjusted labels.\n"
        )
    return output_path


def _build_row(
    *,
    symbol: str,
    event: dict[str, str] | None,
    attention: dict[str, str] | None,
    stablecoin: dict[str, str] | None,
    wallet: dict[str, str] | None,
    funding: dict[str, str] | None,
    equity: dict[str, str] | None,
) -> MultimodalFeatureRow:
    nlp_score = min(_float((event or {}).get("score")), 100.0)
    attention_score = _attention_score(attention)
    stablecoin_score = _stablecoin_score(stablecoin)
    wallet_score = _wallet_score(wallet)
    funding_score = _funding_score(funding)
    equity_score = min(_float((equity or {}).get("score")), 100.0)
    component_scores = (nlp_score, attention_score, stablecoin_score, wallet_score, funding_score, equity_score)
    feature_count = sum(score > 0.0 for score in component_scores)
    timestamp_boundary = _timestamp_boundary(event=event, attention=attention, stablecoin=stablecoin, funding=funding, equity=equity)
    boundary_penalty = 25.0 if timestamp_boundary != "aligned_timestamp_required" else 0.0
    alignment_score = sum(component_scores) + feature_count * 12.0 - boundary_penalty
    status = _status(feature_count=feature_count, timestamp_boundary=timestamp_boundary, alignment_score=alignment_score)
    return MultimodalFeatureRow(
        symbol=symbol,
        status=status,
        feature_count=feature_count,
        alignment_score=alignment_score,
        nlp_event_score=nlp_score,
        ticker_attention_score=attention_score,
        stablecoin_flow_score=stablecoin_score,
        wallet_flow_score=wallet_score,
        funding_market_score=funding_score,
        equity_factor_score=equity_score,
        feature_summary=_feature_summary(
            event=event,
            attention=attention,
            stablecoin=stablecoin,
            wallet=wallet,
            funding=funding,
            equity=equity,
        ),
        timestamp_boundary=timestamp_boundary,
        missing_data=(
            "source timestamps, leakage-safe feature table, feature ablation, beta-adjusted target, "
            "funding PnL, spread/depth, and train/test split"
        ),
        next_probe=_next_probe(symbol=symbol, status=status),
    )


def _attention_score(row: dict[str, str] | None) -> float:
    if not row:
        return 0.0
    decision = row.get("decision", "")
    priority = _float(row.get("priority"))
    if decision == "ticker_specific_attention_alpha_candidate":
        return min(priority * 0.45, 80.0)
    if decision == "dedupe_news_before_attention_label":
        return min(priority * 0.25, 50.0)
    return min(priority * 0.10, 20.0)


def _stablecoin_score(row: dict[str, str] | None) -> float:
    if not row:
        return 0.0
    status = row.get("status", "")
    priority = _float(row.get("priority"))
    if status == "needs_exchange_wallet_map_before_exchange_inflow_alpha":
        return min(priority * 0.45, 55.0)
    if status == "chain_liquidity_proxy_alpha_candidate":
        return min(priority * 0.65, 70.0)
    return min(priority * 0.30, 30.0)


def _wallet_score(row: dict[str, str] | None) -> float:
    if not row:
        return 0.0
    status = row.get("status", "")
    if "reject" in status or "blocked" in status:
        return 0.0
    return min(_float(row.get("score")), 80.0)


def _funding_score(row: dict[str, str] | None) -> float:
    if not row:
        return 0.0
    net_24h = max(_float(row.get("net_24h_proxy")), 0.0)
    annualized = abs(_float(row.get("annualized_spread")))
    return min(net_24h * 5000.0 + annualized * 10.0, 70.0)


def _timestamp_boundary(
    *,
    event: dict[str, str] | None,
    attention: dict[str, str] | None,
    stablecoin: dict[str, str] | None,
    funding: dict[str, str] | None,
    equity: dict[str, str] | None,
) -> str:
    reasons: list[str] = []
    if event and _float(event.get("newest_age_hours")) > 1.0:
        reasons.append("stale_event")
    if attention and attention.get("decision") == "dedupe_news_before_attention_label":
        reasons.append("attention_news_dedupe")
    if stablecoin and stablecoin.get("status") == "needs_exchange_wallet_map_before_exchange_inflow_alpha":
        reasons.append("exchange_wallet_map_missing")
    if equity and equity.get("status") == "timestamp_boundary_required":
        reasons.append("equity_market_hours_gap")
    if funding and _float(funding.get("breakeven_hold_hours")) > 8.0:
        reasons.append("funding_hold_time")
    return ",".join(reasons) if reasons else "aligned_timestamp_required"


def _status(*, feature_count: int, timestamp_boundary: str, alignment_score: float) -> str:
    if timestamp_boundary != "aligned_timestamp_required":
        return "multimodal_timestamp_boundary_required"
    if feature_count >= 4 and alignment_score >= 180.0:
        return "multimodal_feature_label_priority"
    if feature_count >= 3:
        return "multimodal_feature_watchlist"
    return "insufficient_multimodal_coverage"


def _feature_summary(
    *,
    event: dict[str, str] | None,
    attention: dict[str, str] | None,
    stablecoin: dict[str, str] | None,
    wallet: dict[str, str] | None,
    funding: dict[str, str] | None,
    equity: dict[str, str] | None,
) -> str:
    parts: list[str] = []
    if event:
        parts.append(f"event={event.get('status', '')}/{event.get('score', '')}")
    if attention:
        parts.append(f"attention={attention.get('decision', '')}/{attention.get('priority', '')}")
    if stablecoin:
        parts.append(f"stablecoin={stablecoin.get('status', '')}/{stablecoin.get('priority', '')}")
    if wallet:
        parts.append(f"wallet={wallet.get('status', '')}/{wallet.get('score', '')}")
    if funding:
        parts.append(f"funding_net24={funding.get('net_24h_proxy', '')}")
    if equity:
        parts.append(f"equity={equity.get('status', '')}/{equity.get('score', '')}")
    return " || ".join(parts) if parts else "no current multimodal feature"


def _next_probe(*, symbol: str, status: str) -> str:
    if status == "multimodal_timestamp_boundary_required":
        return f"build {symbol} timestamp-aligned feature row before any multimodal label"
    if status == "multimodal_feature_label_priority":
        return f"label {symbol} multimodal row with feature ablation and beta-adjusted target"
    if status == "multimodal_feature_watchlist":
        return f"collect one more independent {symbol} feature family before labeling"
    return f"keep {symbol} multimodal row as coverage context until more sources appear"


def _equity_factor_for_symbol(symbol: str, rows: tuple[dict[str, str], ...]) -> dict[str, str] | None:
    direct = tuple(row for row in rows if row.get("target_asset") == symbol)
    shared = tuple(row for row in rows if row.get("target_asset") == "BTC_ETH")
    candidates = direct or shared
    if not candidates:
        return None
    return max(candidates, key=lambda row: _float(row.get("score")))


def _best_rows_by_symbol(path: Path, *, symbol_key: str, score_key: str) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        symbol = row.get(symbol_key, "").upper()
        if not symbol:
            continue
        if symbol not in best or _float(row.get(score_key)) > _float(best[symbol].get(score_key)):
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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_multimodal_btc_eth_feature_alignment.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_multimodal_btc_eth_feature_alignment.md",
    )
    args = parser.parse_args()

    rows = build_multimodal_feature_rows()
    write_multimodal_feature_csv(rows, output_path=args.output_path)
    write_multimodal_feature_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(row.status, row.symbol, f"features={row.feature_count}", f"score={row.alignment_score:.4f}")


if __name__ == "__main__":
    main()
