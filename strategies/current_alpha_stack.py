from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaStackRow:
    opportunity: str
    status: str
    side: str
    priority_score: float
    sources: str
    evidence: str
    conflict: str
    next_step: str


def build_alpha_stack(root: Path = ROOT) -> tuple[AlphaStackRow, ...]:
    rows = [
        _btc_risk_off_short_stack(root),
        _mstr_btc_relative_value_stack(root),
        *_volatility_actionability_stacks(root),
        *_options_volatility_stacks(root),
        *_event_probability_actionability_stacks(root),
        _prediction_market_event_model_stack(root),
        _cross_market_stress_anomaly_stack(root),
        *_peg_anomaly_tradeability_stacks(root),
        *_futures_basis_stacks(root),
        *_derivatives_positioning_stacks(root),
        *_binance_derivatives_feature_prior_stacks(root),
        *_binance_derivatives_regime_feature_stacks(root),
        *_binance_derivatives_intraday_live_gate_stacks(root),
        *_binance_derivatives_intraday_paper_stacks(root),
        *_binance_derivatives_intraday_repeat_stacks(root),
        *_binance_derivatives_intraday_feature_stacks(root),
        *_cross_exchange_funding_stacks(root),
        *_perp_crowding_stacks(root),
        *_hyperliquid_dislocation_actionability_stacks(root),
        *_hyperliquid_dislocation_stacks(root),
        *_hyperliquid_oi_shift_stacks(root),
        *_protocol_fundamental_stacks(root),
        *_protocol_fee_valuation_stacks(root),
        *_protocol_fee_actionability_stacks(root),
        *_protocol_fee_price_context_stacks(root),
        *_yield_peg_risk_stacks(root),
        *_defi_yield_stacks(root),
        *_lending_stress_actionability_stacks(root),
        *_defi_lending_stacks(root),
        *_dex_pool_flow_stacks(root),
        *_news_event_stacks(root),
        *_attention_funding_stacks(root),
        *_attention_price_context_stacks(root),
        *_market_breadth_stacks(root),
        *_protocol_activity_stacks(root),
        *_on_chain_flow_stacks(root),
        *_chain_stablecoin_migration_stacks(root),
        *_stablecoin_peg_stress_stacks(root),
        *_token_unlock_actionability_stacks(root),
        *_token_unlock_stacks(root),
        *_liquidation_flow_stacks(root),
        *_l2_imbalance_stacks(root),
    ]
    return tuple(
        sorted((row for row in rows if row is not None), key=lambda row: row.priority_score, reverse=True)
    )


def write_alpha_stack_csv(rows: tuple[AlphaStackRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("opportunity", "status", "side", "priority_score", "sources", "evidence", "conflict", "next_step")
        )
        for row in rows:
            writer.writerow(
                (
                    row.opportunity,
                    row.status,
                    row.side,
                    f"{row.priority_score:.8f}",
                    row.sources,
                    row.evidence,
                    row.conflict,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_stack_md(rows: tuple[AlphaStackRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Stack\n\n")
        handle.write(
            "This stack joins current paper tickets and watches across lanes. "
            "It is a candidate-generation view, not an approval list or trade instruction.\n\n"
        )
        handle.write("| opportunity | status | side | priority score | sources | evidence | conflict | next step |\n")
        handle.write("| --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.opportunity} | {row.status} | {row.side} | {row.priority_score:.4f} | "
                f"{row.sources} | {_escape(row.evidence)} | {_escape(row.conflict)} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _btc_risk_off_short_stack(root: Path) -> AlphaStackRow | None:
    institutional = _best_by_score(
        root / "institutional_flow" / "current_btc_etf_funding_paper_ticket.csv",
        score_key="score",
        status_values={"paper_venue_candidate"},
    )
    macro = _row_by_name(root / "macro_regime" / "current_macro_crypto_paper_tickets.csv", "crypto_risk_off_lagged_short")
    speculative = _row_by_name(
        root / "speculative_beta" / "current_speculative_beta_paper_tickets.csv",
        "vix_high_beta_air_pocket",
    )
    if not institutional:
        return None
    raw_score = _float(institutional.get("score")) + _abs_float(macro.get("score") if macro else "") + _abs_float(
        speculative.get("score") if speculative else ""
    )
    evidence_parts = [
        f"{institutional.get('venue', '')}/{institutional.get('instrument', '')} {institutional.get('side', '')} funding ticket score={institutional.get('score', '')}",
    ]
    if macro:
        evidence_parts.append(macro.get("reason", ""))
    if speculative:
        evidence_parts.append(speculative.get("reason", ""))
    return AlphaStackRow(
        opportunity="btc_risk_off_short_stack",
        status="paper_watch",
        side="short_btc_perp",
        priority_score=_priority_score("paper_watch", source_count=3, raw_score=raw_score),
        sources="institutional_flow + macro_regime + speculative_beta",
        evidence="; ".join(part for part in evidence_parts if part),
        conflict="BTC and ETH may already have repriced lower; Deribit put-skew screen points to rich downside vol rather than clean directional short",
        next_step="label 4h/12h/24h BTC outcomes when ETF/funding short watch overlaps macro and speculative-beta risk-off pressure",
    )


def _mstr_btc_relative_value_stack(root: Path) -> AlphaStackRow | None:
    mstr = _row_by_name(
        root / "crypto_equity_proxy" / "current_crypto_equity_proxy_paper_tickets.csv",
        "mstr_btc_dislocation",
    )
    prediction = _first_matching(
        root / "prediction_markets" / "current_prediction_market_paper_tickets.csv",
        lambda row: "Microstrategy" in row.get("question", "") and row.get("status") == "paper_event_model_candidate",
    )
    if not mstr:
        return None
    raw_score = _abs_float(mstr.get("score")) * 100.0
    if prediction:
        raw_score += min(_float(prediction.get("score")), 100.0) / 10.0
    evidence = mstr.get("reason", "")
    if prediction:
        evidence = f"{evidence}; prediction market event model candidate: {prediction.get('question', '')} {prediction.get('outcome', '')}"
    return AlphaStackRow(
        opportunity="mstr_btc_relative_value",
        status=mstr.get("status", "paper_relative_value_watch"),
        side=mstr.get("side", "long_mstr_short_btc"),
        priority_score=_priority_score(mstr.get("status", ""), source_count=2, raw_score=raw_score),
        sources="crypto_equity_proxy + prediction_markets",
        evidence=evidence,
        conflict="requires equity borrow/liquidity, corporate-news check, and BTC hedge mapping; prediction market odds are not a direct equity fair-value model",
        next_step="label MSTR/BTC relative returns around BTC-purchase news and compare against borrow, spread, and hedge slippage",
    )


def _options_volatility_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "options_volatility" / "current_volatility_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "options_volatility" / "current_options_volatility_paper_tickets.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_short_put_spread_candidate",
                "paper_long_vol_candidate",
                "paper_long_vol_quote_candidate",
                "paper_calendar_spread_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:4]:
        currency = ticket.get("currency", "")
        structure = ticket.get("structure", "")
        expiry = ticket.get("expiry", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{currency.lower()}_{structure}_{expiry.replace('-', '')}",
                status=ticket.get("status", ""),
                side=f"{currency}_{structure}",
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="options_volatility",
                evidence=(
                    f"{currency} {expiry}: "
                    f"iv_premium_24h={ticket.get('iv_premium_24h', '')}, "
                    f"skew={ticket.get('skew_iv', '')}, "
                    f"term={ticket.get('term_iv_spread_to_next', '')}, "
                    f"max_loss_pct={ticket.get('max_loss_pct', '')}, "
                    f"realized_move_pct={ticket.get('realized_move_pct', '')}, "
                    f"premium_to_realized_move={ticket.get('premium_to_realized_move', '')}, "
                    f"top_ask_depth_usd={ticket.get('top_ask_premium_depth_usd', '')}"
                ),
                conflict=_options_volatility_conflict(ticket),
                next_step=_options_volatility_next_step(ticket),
            )
        )
    return tuple(output)


def _volatility_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "options_volatility" / "current_volatility_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "volatility_candidate_needs_sweep_hedge",
                "volatility_quote_mechanics_watch",
                "volatility_short_expiry_hedge_watch",
                "volatility_structure_mechanics_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:4]:
        currency = ticket.get("currency", "")
        structure = ticket.get("structure", "")
        expiry = ticket.get("expiry", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{currency.lower()}_{structure}_{expiry.replace('-', '')}_volatility_actionability",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="options_volatility",
                evidence=(
                    f"{currency} {expiry}: "
                    f"iv_premium_24h={ticket.get('iv_premium_24h', '')}, "
                    f"quote_spread={ticket.get('quote_spread_pct', '')}, "
                    f"max_loss_pct={ticket.get('max_loss_pct', '')}, "
                    f"premium_to_realized_move={ticket.get('premium_to_realized_move', '')}, "
                    f"top_ask_depth_usd={ticket.get('top_ask_premium_depth_usd', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "options candidate still needs multi-level depth, hedge, margin, and exit checks",
                ),
                next_step=ticket.get(
                    "next_step",
                    "paper-check option sweep depth, delta hedge plan, max loss, margin, and exit bid",
                ),
            )
        )
    return tuple(output)


def _options_volatility_conflict(ticket: dict[str, str]) -> str:
    if ticket.get("status") == "paper_short_put_spread_candidate":
        return "macro/speculative-beta risk-off pressure can turn rich put premium into real tail loss"
    if ticket.get("status") == "paper_long_vol_candidate":
        return "cheap IV can stay cheap or realized volatility can collapse; needs actual option quotes, premium-at-risk, and delta-hedge plan"
    if ticket.get("status") == "paper_long_vol_quote_candidate":
        return "cheap IV can stay cheap or realized volatility can collapse; top ask depth still excludes sweep, hedge, and exit checks"
    return "calendar spread depends on expiry curve, event timing, bid/ask, margin, and hedge PnL rather than direction alone"


def _options_volatility_next_step(ticket: dict[str, str]) -> str:
    if ticket.get("status") == "paper_short_put_spread_candidate":
        return "paper-check bid/ask spread, margin, max loss, delta hedge cost, and behavior during the current risk-off shock"
    if ticket.get("status") == "paper_long_vol_candidate":
        return "paper-check long-vol spread quotes, max premium loss, delta hedge plan, and realized-vol persistence"
    if ticket.get("status") == "paper_long_vol_quote_candidate":
        return "paper-check ATM straddle multi-level depth, max premium loss, delta hedge plan, and realized-vol persistence"
    return "paper-check calendar spread quotes, event timing, vega/theta exposure, margin, and delta hedge cost"


def _prediction_market_event_model_stack(root: Path) -> AlphaStackRow | None:
    if (root / "prediction_markets" / "current_event_probability_actionability.csv").exists():
        return None
    paper_refresh = _best_by_score(
        root / "prediction_markets" / "current_event_probability_paper_outcome_refresh.csv",
        score_key="score",
        status_values={"paper_outcome_survived_refresh", "paper_outcome_weak_refresh"},
    )
    paper_outcome = _best_by_score(
        root / "prediction_markets" / "current_event_probability_paper_outcome.csv",
        score_key="score",
        status_values={"paper_outcome_active_watch", "paper_outcome_edge_watch"},
    )
    paper_ticket = _best_by_score(
        root / "prediction_markets" / "current_event_probability_paper_tickets.csv",
        score_key="score",
        status_values={"paper_event_probability_ticket", "event_probability_watch"},
    )
    gap = _best_by_score(
        root / "prediction_markets" / "current_event_probability_gap.csv",
        score_key="score",
        status_values={"paper_probability_gap_candidate", "probability_gap_watch"},
    )
    ticket = _best_by_score(
        root / "prediction_markets" / "current_prediction_market_paper_tickets.csv",
        score_key="score",
        status_values={"paper_event_model_candidate", "paper_event_model_watch"},
    )
    if not any((paper_refresh, paper_outcome, paper_ticket, gap, ticket)):
        return None
    if paper_refresh:
        return AlphaStackRow(
            opportunity="prediction_market_event_model",
            status=paper_refresh.get("status", "paper_outcome_survived_refresh"),
            side=f"{paper_refresh.get('suggested_side', '')}: {paper_refresh.get('question', '')}",
            priority_score=_priority_score(
                paper_refresh.get("status", ""),
                source_count=6,
                raw_score=_float(paper_refresh.get("score")),
            ),
            sources="prediction_markets + external_news + probability_gap + clob_depth + source_quality + refresh",
            evidence=(
                f"entry_ask={paper_refresh.get('previous_entry_ask', '')}, "
                f"bid={paper_refresh.get('current_bid', '')}, "
                f"ask={paper_refresh.get('current_ask', '')}, "
                f"bid_pnl={paper_refresh.get('mark_to_bid_pnl', '')}, "
                f"edge_now={paper_refresh.get('current_edge_after_ask', '')}, "
                f"edge_change={paper_refresh.get('edge_change', '')}, "
                f"source_quality={paper_refresh.get('source_quality_status', '')}"
            ),
            conflict="survived refresh still uses public quotes and a rough headline probability; fill, queue, fees, resolution risk, and adverse selection remain unresolved",
            next_step="repeat the refresh and only promote if edge survives another quote/news update with executable depth",
        )
    if paper_outcome:
        return AlphaStackRow(
            opportunity="prediction_market_event_model",
            status=paper_outcome.get("status", "paper_outcome_active_watch"),
            side=f"{paper_outcome.get('suggested_side', '')}: {paper_outcome.get('question', '')}",
            priority_score=_priority_score(
                paper_outcome.get("status", ""),
                source_count=5,
                raw_score=_float(paper_outcome.get("score")),
            ),
            sources="prediction_markets + external_news + probability_gap + clob_depth + source_quality",
            evidence=(
                f"entry_ask={paper_outcome.get('entry_ask', '')}, "
                f"bid={paper_outcome.get('current_bid', '')}, "
                f"ask={paper_outcome.get('current_ask', '')}, "
                f"bid_pnl={paper_outcome.get('mark_to_bid_pnl', '')}, "
                f"mid_pnl={paper_outcome.get('mark_to_mid_pnl', '')}, "
                f"edge_after_ask={paper_outcome.get('current_edge_after_ask', '')}, "
                f"source_quality={paper_outcome.get('source_quality_status', '')}"
            ),
            conflict="paper outcome is still based on public quotes and rough headline probability; fill, queue, fees, and adverse selection are unresolved",
            next_step="refresh market/news snapshots and require the edge to survive quote movement before any live action",
        )
    if paper_ticket:
        source_quality = _row_by_market_id(
            root / "prediction_markets" / "current_event_source_quality.csv",
            paper_ticket.get("market_id", ""),
        )
        source_count = 5 if source_quality else 4
        raw_score = _float(paper_ticket.get("score")) + _float(source_quality.get("quality_score") if source_quality else "")
        quality_evidence = ""
        if source_quality:
            quality_evidence = (
                f", source_quality={source_quality.get('status', '')}, "
                f"sources={source_quality.get('source_count_72h', '')}, "
                f"recent_articles={source_quality.get('article_count_24h', '')}"
            )
        return AlphaStackRow(
            opportunity="prediction_market_event_model",
            status=paper_ticket.get("status", "paper_event_probability_ticket"),
            side=f"{paper_ticket.get('suggested_side', '')}: {paper_ticket.get('question', '')}",
            priority_score=_priority_score(
                paper_ticket.get("status", ""),
                source_count=source_count,
                raw_score=raw_score,
            ),
            sources=(
                "prediction_markets + external_news + probability_gap + clob_depth"
                if not source_quality
                else "prediction_markets + external_news + probability_gap + clob_depth + source_quality"
            ),
            evidence=(
                f"ask={paper_ticket.get('entry_ask', '')}, "
                f"estimated_payout={paper_ticket.get('estimated_payout_probability', '')}, "
                f"edge_after_ask={paper_ticket.get('edge_after_ask', '')}, "
                f"max_loss={paper_ticket.get('max_loss_per_share', '')}, "
                f"ask_depth_5c={paper_ticket.get('ask_depth_to_5c', '')}"
                f"{quality_evidence}"
            ),
            conflict="paper ticket still depends on rough headline probability, source timing, fill quality, fees, and adverse-selection behavior",
            next_step="paper-check source freshness, duplicate headlines, queue/fill assumptions, and outcome movement before any live action",
        )
    if gap:
        return AlphaStackRow(
            opportunity="prediction_market_event_model",
            status=gap.get("status", "paper_probability_gap_candidate"),
            side=f"{gap.get('suggested_side', '')}: {gap.get('question', '')}",
            priority_score=_priority_score(
                gap.get("status", ""),
                source_count=3,
                raw_score=_float(gap.get("score")),
            ),
            sources="prediction_markets + external_news + probability_gap",
            evidence=(
                f"market_yes={gap.get('market_yes_probability', '')}, "
                f"estimated_yes={gap.get('estimated_yes_probability', '')}, "
                f"gap={gap.get('probability_gap', '')}, "
                f"confidence={gap.get('confidence_score', '')}, "
                f"evidence={gap.get('evidence_terms', '')}"
            ),
            conflict="headline-derived probability is rough and uncalibrated; needs source verification, timing checks, costs, and adverse-selection analysis",
            next_step="paper-check the probability gap with source-level verification, stale-news filtering, and CLOB execution assumptions",
        )
    if not ticket:
        return None
    news = _row_by_market_id(
        root / "prediction_markets" / "current_event_news_pressure.csv",
        ticket.get("market_id", ""),
    )
    evidence_parts = [
        (
            f"{ticket.get('category', '')}: spread={ticket.get('spread', '')}, "
            f"depth={ticket.get('visible_depth_score', '')}, vol24={ticket.get('volume_24h', '')}"
        )
    ]
    source_count = 1
    raw_score = _float(ticket.get("score"))
    if news:
        source_count += 1
        raw_score += _float(news.get("score"))
        evidence_parts.append(
            f"news={news.get('status', '')}, articles24={news.get('article_count_24h', '')}, "
            f"sources={news.get('source_count_72h', '')}, newest_h={news.get('newest_age_hours', '')}"
        )
    return AlphaStackRow(
        opportunity="prediction_market_event_model",
        status=ticket.get("status", "paper_event_model_candidate"),
        side=f"{ticket.get('question', '')} {ticket.get('outcome', '')}",
        priority_score=_priority_score(ticket.get("status", ""), source_count=source_count, raw_score=raw_score),
        sources="prediction_markets" if not news else "prediction_markets + external_news",
        evidence="; ".join(evidence_parts),
        conflict="market depth is not edge; needs independent true-probability model and latency/adverse-selection checks",
        next_step=(
            "compare external news-flow evidence against market-implied odds before any paper event-market action"
            if news
            else "build an external event-probability model before any paper event-market action"
        ),
    )


def _cross_market_stress_anomaly_stack(root: Path) -> AlphaStackRow | None:
    status_values = {
        "cross_market_peg_stress_anomaly",
        "cross_market_lending_stress_anomaly",
        "cross_market_yield_peg_anomaly",
        "cross_market_volatility_mispricing_watch",
        "cross_market_event_probability_anomaly",
        "cross_market_execution_spread_anomaly",
    }
    if (root / "anomaly_stress" / "current_peg_anomaly_tradeability.csv").exists():
        status_values = status_values - {"cross_market_peg_stress_anomaly"}
    if (root / "defi_lending" / "current_lending_stress_actionability.csv").exists():
        status_values = status_values - {"cross_market_lending_stress_anomaly"}
    if (root / "defi_yield" / "current_yield_peg_risk_join.csv").exists():
        status_values = status_values - {"cross_market_yield_peg_anomaly"}
    if (root / "options_volatility" / "current_volatility_actionability.csv").exists():
        status_values = status_values - {"cross_market_volatility_mispricing_watch"}
    if (root / "prediction_markets" / "current_event_probability_paper_outcome.csv").exists():
        status_values = status_values - {"cross_market_event_probability_anomaly"}
    if (root / "cross_exchange_funding" / "current_dislocation_execution_check.csv").exists():
        status_values = status_values - {"cross_market_execution_spread_anomaly"}
    anomaly = _best_by_score(
        root / "anomaly_stress" / "current_cross_market_stress_anomaly.csv",
        score_key="score",
        status_values=status_values,
    )
    if not anomaly:
        return None
    return AlphaStackRow(
        opportunity="cross_market_stress_anomaly",
        status=anomaly.get("status", "cross_market_stress_anomaly"),
        side=f"{anomaly.get('side', '')}: {anomaly.get('subject', '')}",
        priority_score=_priority_score(
            anomaly.get("status", ""),
            source_count=4,
            raw_score=_float(anomaly.get("score")),
        ),
        sources=f"anomaly_stress + {anomaly.get('source_lane', '')}",
        evidence=(
            f"score={anomaly.get('score', '')}, "
            f"severity={anomaly.get('severity', '')}, "
            f"evidence={anomaly.get('evidence', '')}"
        ),
        conflict=anomaly.get(
            "failure_mode",
            "anomaly can be stale, untradable, or explained by risk rather than alpha",
        ),
        next_step=anomaly.get("next_step", "run a specific falsification test for the top anomaly"),
    )


def _peg_anomaly_tradeability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "anomaly_stress" / "current_peg_anomaly_tradeability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "peg_anomaly_tradeability_candidate",
                "peg_anomaly_mechanics_watch",
                "peg_anomaly_stale_or_unrouted",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_peg_anomaly_tradeability",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=2 if _intish(ticket.get("yield_conflict_count")) > 0 else 1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="anomaly_stress + stablecoin_liquidity",
                evidence=(
                    f"{symbol}: price={ticket.get('price', '')}, "
                    f"peg_deviation={ticket.get('peg_deviation', '')}, "
                    f"pool_matches={ticket.get('dex_pool_match_count', '')}, "
                    f"best_pool={ticket.get('best_pool', '')}, "
                    f"pool_reserve={ticket.get('best_pool_reserve_usd', '')}, "
                    f"yield_conflicts={ticket.get('yield_conflict_count', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "peg anomaly needs a route, quote freshness, redemption path, and executable depth",
                ),
                next_step=ticket.get(
                    "next_step",
                    f"check {symbol} route, quote freshness, redemption path, and executable depth",
                ),
            )
        )
    return tuple(output)


def _futures_basis_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "basis_term_structure" / "current_deribit_futures_basis.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_short_basis_watch",
                "paper_long_basis_watch",
            }
            and _abs_float(row.get("basis")) > _float(row.get("bid_ask_spread_pct"))
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen_currency_sides: set[tuple[str, str]] = set()
    for ticket in tickets:
        currency_side = (ticket.get("currency", ""), ticket.get("side", ""))
        if currency_side in seen_currency_sides:
            continue
        seen_currency_sides.add(currency_side)
        instrument = ticket.get("instrument_name", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{instrument.lower()}_basis",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="basis_term_structure",
                evidence=(
                    f"{instrument}: ann_basis={ticket.get('annualized_basis', '')}, "
                    f"basis={ticket.get('basis', '')}, dte={ticket.get('days_to_expiry', '')}, "
                    f"volume_usd={ticket.get('volume_usd', '')}, spread={ticket.get('bid_ask_spread_pct', '')}"
                ),
                conflict="basis trade still needs spot/perp hedge route, funding, margin, fees, and order-book depth checks",
                next_step=ticket.get(
                    "next_step",
                    f"check {instrument} hedge route, fees, margin, funding, and depth",
                ),
            )
        )
        if len(output) >= 6:
            break
    return tuple(output)


def _derivatives_positioning_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "derivatives_positioning" / "current_coingecko_derivatives_positioning.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_oi_funding_crowding_watch",
                "paper_basis_funding_dislocation_watch",
                "paper_derivatives_momentum_risk_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen: set[str] = set()
    for ticket in tickets:
        market = ticket.get("market", "")
        symbol = ticket.get("symbol", "")
        key = f"{market}:{symbol}"
        if key in seen:
            continue
        seen.add(key)
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_{market.lower().replace(' ', '_')}_positioning",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="derivatives_positioning",
                evidence=(
                    f"{market} {symbol}: oi={ticket.get('open_interest', '')}, "
                    f"vol24={ticket.get('volume_24h', '')}, oi_vol={ticket.get('oi_volume_ratio', '')}, "
                    f"funding={ticket.get('funding_rate', '')}, basis={ticket.get('basis', '')}"
                ),
                conflict="aggregated derivatives data still needs venue-specific depth, funding timing, fees, margin, and forward labels",
                next_step=ticket.get(
                    "next_step",
                    f"label {market} {symbol} forward returns, funding PnL, depth, fees, and margin constraints",
                ),
            )
        )
        if len(output) >= 6:
            break
    return tuple(output)


def _binance_derivatives_feature_prior_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        _read_rows(root / "p0_parallel" / "binance_derivatives_signal_summary.csv"),
        key=_binance_derivatives_feature_priority,
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for row in rows[:5]:
        feature = row.get("feature", "")
        direction = _binance_feature_direction(row)
        output.append(
            AlphaStackRow(
                opportunity=f"binance_um_{feature}_feature_prior",
                status="historical_derivatives_feature_prior",
                side=direction,
                priority_score=_priority_score(
                    "historical_derivatives_feature_prior",
                    source_count=1,
                    raw_score=_binance_derivatives_feature_priority(row),
                ),
                sources="p0_parallel + binance_derivatives_history",
                evidence=(
                    f"{feature}: obs={row.get('observations', '')}, "
                    f"corr={row.get('correlation_to_next_return', '')}, "
                    f"low_mean={row.get('low_bucket_mean_next_return', '')}, "
                    f"low_hit={row.get('low_bucket_hit_rate', '')}, "
                    f"high_mean={row.get('high_bucket_mean_next_return', '')}, "
                    f"high_hit={row.get('high_bucket_hit_rate', '')}"
                ),
                conflict=(
                    "this is a 2024Q1 Binance USD-M historical feature prior, "
                    "not a current trade; it can be regime-specific and excludes execution costs"
                ),
                next_step=(
                    f"rerun {feature} on recent windows, split by symbol/regime, "
                    "and only then join to current execution gates"
                ),
            )
        )
    return tuple(output)


def _binance_derivatives_feature_priority(row: dict[str, str]) -> float:
    high_mean = _float(row.get("high_bucket_mean_next_return"))
    low_mean = _float(row.get("low_bucket_mean_next_return"))
    high_hit = _float(row.get("high_bucket_hit_rate"))
    low_hit = _float(row.get("low_bucket_hit_rate"))
    correlation = _abs_float(row.get("correlation_to_next_return"))
    return abs(high_mean - low_mean) * 10_000.0 + abs(high_hit - low_hit) * 20.0 + correlation * 25.0


def _binance_feature_direction(row: dict[str, str]) -> str:
    feature = row.get("feature", "")
    high_mean = _float(row.get("high_bucket_mean_next_return"))
    low_mean = _float(row.get("low_bucket_mean_next_return"))
    high_hit = _float(row.get("high_bucket_hit_rate"))
    low_hit = _float(row.get("low_bucket_hit_rate"))
    if high_mean > low_mean and high_hit >= low_hit:
        return f"long_high_{feature}"
    if low_mean > high_mean and low_hit >= high_hit:
        return f"long_low_{feature}"
    if high_mean > low_mean:
        return f"mean_prefers_high_{feature}"
    return f"mean_prefers_low_{feature}"


def _binance_derivatives_regime_feature_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        (
            row
            for row in _read_rows(root / "p0_parallel" / "binance_derivatives_feature_regime_compare.csv")
            if row.get("status")
            in {
                "persistent_symbol_feature",
                "recent_symbol_feature_priority",
                "bucket_regime_shift",
            }
        ),
        key=lambda row: _float(row.get("combined_score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for row in rows[:8]:
        status = _binance_regime_feature_status(row.get("status", ""))
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{_symbol_slug(symbol)}_{_slug(feature)}_binance_derivatives_symbol_feature",
                status=status,
                side=f"{row.get('recent_bucket', '')}_{feature}",
                priority_score=_priority_score(
                    status,
                    source_count=2,
                    raw_score=_float(row.get("combined_score")),
                ),
                sources="p0_parallel + binance_derivatives_history + recent_window",
                evidence=(
                    f"{symbol}: feature={feature}, "
                    f"historical_bucket={row.get('historical_bucket', '')}, "
                    f"recent_bucket={row.get('recent_bucket', '')}, "
                    f"historical_score={row.get('historical_score', '')}, "
                    f"recent_score={row.get('recent_score', '')}, "
                    f"combined={row.get('combined_score', '')}"
                ),
                conflict=(
                    "symbol-feature prior is historical/recent research evidence, not a current trade; "
                    "it still needs recent intraday labels, costs, and execution gates"
                ),
                next_step=row.get(
                    "next_step",
                    f"rerun {symbol} {feature} with recent intraday labels and execution costs",
                ),
            )
        )
    return tuple(output)


def _binance_derivatives_intraday_feature_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        (
            row
            for row in _read_rows(root / "p0_parallel" / "binance_derivatives_intraday_feature_candidates.csv")
            if row.get("status") in {"intraday_feature_priority", "intraday_feature_watch"}
        ),
        key=lambda row: _float(row.get("edge_score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for row in rows[:8]:
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        status = _binance_intraday_feature_status(row.get("status", ""))
        output.append(
            AlphaStackRow(
                opportunity=f"{_symbol_slug(symbol)}_{_slug(feature)}_intraday_derivatives_feature",
                status=status,
                side=f"{row.get('preferred_bucket', '')}_{feature}",
                priority_score=_priority_score(
                    status,
                    source_count=2,
                    raw_score=_float(row.get("edge_score")),
                ),
                sources="p0_parallel + binance_derivatives_intraday_labels",
                evidence=(
                    f"{symbol}: feature={feature}, obs={row.get('observations', '')}, "
                    f"bucket={row.get('preferred_bucket', '')}, "
                    f"low_1h={row.get('low_bucket_mean_next_1h_return', '')}, "
                    f"high_1h={row.get('high_bucket_mean_next_1h_return', '')}, "
                    f"low_hit={row.get('low_bucket_hit_rate', '')}, "
                    f"high_hit={row.get('high_bucket_hit_rate', '')}, "
                    f"score={row.get('edge_score', '')}"
                ),
                conflict=(
                    "intraday feature label excludes fees, spread, fill probability, "
                    "funding PnL, stop behavior, and fresh-window repeat checks"
                ),
                next_step=row.get(
                    "next_step",
                    f"repeat {symbol} {feature} 5m-to-1h label on a fresh window",
                ),
            )
        )
    return tuple(output)


def _binance_derivatives_intraday_repeat_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        (
            row
            for row in _read_rows(root / "p0_parallel" / "binance_derivatives_intraday_repeat_compare.csv")
            if row.get("status") in {"intraday_repeat_priority", "intraday_repeat_watch"}
        ),
        key=lambda row: _float(row.get("combined_score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for row in rows[:8]:
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        status = _binance_intraday_repeat_status(row.get("status", ""))
        output.append(
            AlphaStackRow(
                opportunity=f"{_symbol_slug(symbol)}_{_slug(feature)}_repeat_intraday_derivatives_feature",
                status=status,
                side=f"{row.get('recent_bucket', '')}_repeat_{feature}",
                priority_score=_priority_score(
                    status,
                    source_count=3,
                    raw_score=_float(row.get("combined_score")),
                ),
                sources="p0_parallel + binance_derivatives_intraday_repeat",
                evidence=(
                    f"{symbol}: feature={feature}, "
                    f"prior_bucket={row.get('prior_bucket', '')}, "
                    f"recent_bucket={row.get('recent_bucket', '')}, "
                    f"prior_score={row.get('prior_score', '')}, "
                    f"recent_score={row.get('recent_score', '')}, "
                    f"combined={row.get('combined_score', '')}"
                ),
                conflict=(
                    "non-overlapping intraday label repeat still excludes fees, spread, "
                    "fill probability, funding PnL, stop behavior, and sizing assumptions"
                ),
                next_step=row.get(
                    "next_step",
                    f"run {symbol} {feature} intraday paper label with costs and fills",
                ),
            )
        )
    return tuple(output)


def _binance_derivatives_intraday_paper_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = [
        *_read_rows(root / "p0_parallel" / "binance_derivatives_intraday_paper_labels.csv"),
        *_read_rows(root / "p0_parallel" / "binance_derivatives_intraday_paper_labels_2bps.csv"),
    ]
    rows = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_intraday_cost_supported",
                "paper_intraday_recent_only",
                "paper_intraday_positive_mean_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row.get("symbol", ""),
            row.get("feature", ""),
            row.get("action", ""),
            row.get("round_trip_cost_bps", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        if len(output) >= 8:
            break
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        action = row.get("action", "")
        cost_bps = row.get("round_trip_cost_bps", "")
        status = _binance_intraday_paper_status(row.get("status", ""), cost_bps=cost_bps)
        output.append(
            AlphaStackRow(
                opportunity=f"{_symbol_slug(symbol)}_{_slug(feature)}_{_slug(action)}_intraday_paper_label",
                status=status,
                side=action,
                priority_score=_priority_score(
                    status,
                    source_count=3,
                    raw_score=_float(row.get("score")),
                ),
                sources="p0_parallel + binance_derivatives_intraday_paper_labels",
                evidence=(
                    f"{symbol}: feature={feature}, action={action}, cost_bps={cost_bps}, "
                    f"prior_net1h={row.get('prior_net_mean_1h', '')}, "
                    f"recent_net1h={row.get('recent_net_mean_1h', '')}, "
                    f"combined_net1h={row.get('combined_net_mean_1h', '')}, "
                    f"combined_hit={row.get('combined_hit_rate', '')}, "
                    f"trades={row.get('combined_trades', '')}, score={row.get('score', '')}"
                ),
                conflict=(
                    "intraday paper label uses rough cost only; it still needs live spread, "
                    "funding timestamp, fill delay, stop rules, and sizing assumptions"
                ),
                next_step=row.get(
                    "next_step",
                    f"paper-check {symbol} {feature} {action} with live spread and funding timing",
                ),
            )
        )
    return tuple(output)


def _binance_derivatives_intraday_live_gate_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        _read_rows(root / "p0_parallel" / "binance_derivatives_intraday_live_execution_gate.csv"),
        key=lambda row: _float(row.get("estimated_low_fee_net_1h_bps")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for row in rows[:4]:
        status = _binance_intraday_live_gate_status(row.get("gate_action", ""))
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        action = row.get("action", "")
        size = row.get("candidate_size_usd", "")
        size_slug = size.replace(".", "_")
        output.append(
            AlphaStackRow(
                opportunity=f"{_symbol_slug(symbol)}_{_slug(feature)}_{_slug(action)}_{size_slug}usd_intraday_live_gate",
                status=status,
                side=action,
                priority_score=_priority_score(
                    status,
                    source_count=2,
                    raw_score=_float(row.get("estimated_low_fee_net_1h_bps")),
                ),
                sources="p0_parallel + binance_intraday_live_execution_gate + okx_book",
                evidence=(
                    f"{symbol}: feature={feature}, action={action}, size={row.get('candidate_size_usd', '')}, "
                    f"source={row.get('source_status', '')}, condition={row.get('live_condition', '')}, "
                    f"spread_bps={row.get('spread_bps', '')}, depth5={row.get('side_depth_5bps_notional', '')}, "
                    f"funding1h_bps={row.get('funding_return_1h_bps', '')}, "
                    f"low_fee_net_bps={row.get('estimated_low_fee_net_1h_bps', '')}, "
                    f"taker_net_bps={row.get('estimated_taker_net_1h_bps', '')}"
                ),
                conflict=(
                    "live execution gate uses OKX book/funding because Binance live feature endpoints are unavailable; "
                    "maker fill probability, queue position, and stop behavior remain unmodeled"
                ),
                next_step=row.get(
                    "reason",
                    "obtain live feature source and repeat live spread/funding/fill checks",
                ),
            )
        )
    return tuple(output)


def _binance_intraday_live_gate_status(gate_action: str) -> str:
    if gate_action == "taker_paper_probe":
        return "live_taker_intraday_probe"
    if gate_action == "low_fee_paper_probe":
        return "live_low_fee_intraday_probe"
    if gate_action == "feature_source_blocked":
        return "intraday_live_feature_source_blocked"
    return "intraday_live_execution_blocked"


def _binance_intraday_paper_status(status: str, *, cost_bps: str) -> str:
    cost = _float(cost_bps)
    if status == "paper_intraday_cost_supported" and cost <= 2.0:
        return "low_cost_intraday_paper_supported"
    if status == "paper_intraday_cost_supported":
        return "intraday_paper_supported"
    if status == "paper_intraday_recent_only" and cost <= 2.0:
        return "low_cost_intraday_paper_recent_only"
    if status == "paper_intraday_positive_mean_watch":
        return "intraday_paper_positive_mean_watch"
    return "intraday_paper_watch"


def _binance_intraday_repeat_status(status: str) -> str:
    if status == "intraday_repeat_priority":
        return "repeat_intraday_derivatives_feature_priority"
    return "repeat_intraday_derivatives_feature_watch"


def _binance_intraday_feature_status(status: str) -> str:
    if status == "intraday_feature_priority":
        return "intraday_derivatives_feature_priority"
    return "intraday_derivatives_feature_watch"


def _binance_regime_feature_status(status: str) -> str:
    if status == "persistent_symbol_feature":
        return "persistent_derivatives_symbol_feature_prior"
    if status == "bucket_regime_shift":
        return "derivatives_symbol_feature_regime_shift"
    return "recent_derivatives_symbol_feature_prior"


def _symbol_slug(symbol: str) -> str:
    value = symbol.lower()
    for suffix in ("usdt", "usd"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _protocol_fundamental_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_screen.csv")
    unlocks = {row.get("symbol", ""): row for row in _read_rows(root / "token_unlocks" / "current_token_unlock_paper_tickets.csv")}
    reviews = {
        row.get("token_symbol", ""): row
        for row in _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_candidate_review.csv")
    }
    tickets = sorted(
        (row for row in rows if row.get("status") in {"paper_long_context", "funding_crowded_watch"}),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        token = ticket.get("token_symbol", "")
        unlock = unlocks.get(token)
        review = reviews.get(token, {})
        conflict = (
            "protocol fees are not a direct token valuation model; fee growth can be lagging, crowded, or disconnected from token value"
        )
        if review.get("review_status"):
            conflict = f"{conflict}; review={review.get('review_status', '')}"
        if unlock:
            conflict = (
                f"{conflict}; token unlock lane has {unlock.get('status', '')} "
                f"with side={unlock.get('side', '')}"
            )
        evidence = (
            f"{token}/{ticket.get('name', '')}: "
            f"fees7d={ticket.get('total_7d', '')}, "
            f"fees30d={ticket.get('total_30d', '')}, "
            f"growth7d={ticket.get('change_7d_over_7d', '')}, "
            f"funding={ticket.get('funding', '')}"
        )
        if review.get("evidence"):
            evidence = f"{evidence}; {review.get('evidence', '')}"
        output.append(
            AlphaStackRow(
                opportunity=f"{token.lower()}_protocol_fee_growth",
                status=ticket.get("status", "paper_long_context"),
                side=ticket.get("side", "long_token_or_relative_value"),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="protocol_fundamentals",
                evidence=evidence,
                conflict=conflict,
                next_step=review.get("next_step") or ticket.get(
                    "next_step",
                    f"label {token} returns after protocol fee-growth snapshots",
                ),
            )
        )
    return tuple(output)


def _cross_exchange_funding_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "cross_exchange_funding" / "current_dislocation_watchlist.csv")
    tickets = sorted(
        (row for row in rows if row.get("action") != "blocked_by_cost_or_capacity"),
        key=lambda row: _float(row.get("annualized_edge")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:5]:
        asset = ticket.get("asset", "")
        action = ticket.get("action", "")
        long_venue = ticket.get("long_venue", "")
        short_venue = ticket.get("short_venue", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_{long_venue.lower()}_{short_venue.lower()}_funding",
                status="paper_funding_dislocation_watch",
                side=f"{long_venue}_vs_{short_venue}",
                priority_score=_priority_score(
                    "paper_funding_dislocation_watch",
                    source_count=1,
                    raw_score=_float(ticket.get("annualized_edge")) * 10.0,
                ),
                sources="cross_exchange_funding",
                evidence=(
                    f"{asset}: action={action}, annualized_edge={ticket.get('annualized_edge', '')}, "
                    f"liquidity={ticket.get('liquidity_proxy', '')}, friction={ticket.get('friction_proxy', '')}"
                ),
                conflict="funding dislocation can disappear before execution; borrow, margin, transfer, and venue failure risks are still unvalidated",
                next_step=f"paper-check {asset} cross-venue funding persistence with real fee, margin, and venue constraints",
            )
        )
    return tuple(output)


def _perp_crowding_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    validated_rows = _read_rows(root / "perp_market_map" / "current_crowding_reversion_validated_candidates.csv")
    if validated_rows:
        execution_rows = _read_rows(root / "perp_market_map" / "current_crowding_reversion_execution_check.csv")
        outcome_rows = _read_rows(root / "perp_market_map" / "current_crowding_reversion_paper_outcome.csv")
        return _perp_crowding_validated_stacks(validated_rows, execution_rows, outcome_rows)
    rows = _read_rows(root / "perp_market_map" / "current_crowding_reversion_screen.csv")
    tickets = sorted(rows, key=lambda row: _float(row.get("carry_reversion_score")), reverse=True)
    output: list[AlphaStackRow] = []
    for ticket in tickets[:5]:
        asset = ticket.get("asset", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_perp_crowding_reversion",
                status="paper_crowding_reversion_watch",
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    "paper_crowding_reversion_watch",
                    source_count=1,
                    raw_score=_float(ticket.get("carry_reversion_score")),
                ),
                sources="perp_market_map",
                evidence=(
                    f"{asset}: funding={ticket.get('annualized_funding', '')}, "
                    f"mark_oracle_diff={ticket.get('mark_oracle_diff', '')}, "
                    f"oi_volume_ratio={ticket.get('oi_volume_ratio', '')}, "
                    f"impact_spread={ticket.get('impact_spread', '')}"
                ),
                conflict="crowding can persist or squeeze; this needs forward labels, funding PnL, and stop logic before paper promotion",
                next_step=f"label {asset} crowding-reversion returns against funding decay, OI/volume, and spread costs",
            )
        )
    return tuple(output)


def _hyperliquid_dislocation_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "perp_market_map" / "current_hyperliquid_dislocation_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_candidates.csv")
    labels = _best_hyperliquid_dislocation_labels(
        _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_forward_labels.csv")
    )
    monitor_rows = _best_hyperliquid_dislocation_monitor_rows(
        _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_monitor_summary.csv")
    )
    execution_checks = _best_hyperliquid_dislocation_execution_checks(
        _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_execution_check.csv")
    )
    tickets = _selected_hyperliquid_dislocation_tickets(rows=rows, labels=labels)
    output: list[AlphaStackRow] = []
    for ticket in tickets:
        asset = ticket.get("asset", "")
        status = ticket.get("status", "paper_hyperliquid_dislocation_candidate")
        label = labels.get((asset, status, ticket.get("side", "")), {})
        monitor = monitor_rows.get((asset, status, ticket.get("side", "")), {})
        execution = execution_checks.get((asset, status, ticket.get("side", "")), {})
        stack_status = _hyperliquid_dislocation_stack_status(status, label, monitor, execution)
        monitor_note = ""
        if monitor:
            monitor_note = (
                f", monitor_obs={monitor.get('observations', '')}, "
                f"monitor_mean_score={monitor.get('mean_score', '')}"
            )
        label_note = ""
        if label:
            label_note = (
                f", out15={label.get('outcome_15m', '')}, "
                f"net15_bps={label.get('net_15m_bps', '')}, "
                f"out1h={label.get('outcome_1h', '')}, "
                f"net1h_bps={label.get('net_1h_bps', '')}"
            )
        execution_note = ""
        if execution:
            execution_note = (
                f", gate={execution.get('gate_action', '')}, "
                f"size={execution.get('candidate_size_usd', '')}, "
                f"conservative_net15_bps={execution.get('conservative_net_15m_bps', '')}, "
                f"spread_bps={execution.get('spread_bps', '')}, "
                f"depth_usage={execution.get('visible_depth_usage_10bps', '')}"
            )
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_{_slug(status)}_{_slug(ticket.get('side', ''))}",
                status=stack_status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    stack_status,
                    source_count=(
                        1
                        + (1 if monitor else 0)
                        + (1 if label else 0)
                        + (1 if execution else 0)
                    ),
                    raw_score=_float(ticket.get("score")) * 10.0,
                )
                + (_float(ticket.get("score")) / 100.0)
                + min(_float(monitor.get("observations")) / 2.0, 2.0)
                + _hyperliquid_dislocation_label_bonus(label),
                sources="perp_market_map",
                evidence=(
                    f"{asset}: ret24={ticket.get('return_24h', '')}, "
                    f"funding={ticket.get('annualized_funding', '')}, "
                    f"mark_oracle={ticket.get('mark_oracle_diff', '')}, "
                    f"premium={ticket.get('premium', '')}, "
                    f"oi_vol={ticket.get('oi_volume_ratio', '')}, "
                    f"impact={ticket.get('impact_spread', '')}, "
                    f"reason={ticket.get('reason', '')}"
                    f"{monitor_note}"
                    f"{label_note}"
                    f"{execution_note}"
                ),
                conflict=(
                    "current dislocation screen can produce both continuation and reversal hypotheses "
                    "for the same move; 15m labels are not enough to establish persistence"
                ),
                next_step=_hyperliquid_dislocation_next_step(
                    asset=asset,
                    ticket=ticket,
                    label=label,
                    execution=execution,
                ),
            )
        )
    return tuple(output)


def _slug(value: str) -> str:
    return value.removeprefix("paper_").removesuffix("_candidate")


def _best_hyperliquid_dislocation_labels(
    rows: tuple[dict[str, str], ...],
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    sorted_rows = sorted(rows, key=lambda row: row.get("timestamp", ""), reverse=True)
    for row in sorted_rows:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in output:
            continue
        output[key] = row
    return output


def _best_hyperliquid_dislocation_monitor_rows(
    rows: tuple[dict[str, str], ...],
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("monitor_action") == "repeat_label_priority",
            _float(row.get("observations")),
            _float(row.get("mean_score")),
        ),
        reverse=True,
    )
    for row in sorted_rows:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in output:
            continue
        output[key] = row
    return output


def _best_hyperliquid_dislocation_execution_checks(
    rows: tuple[dict[str, str], ...],
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    sorted_rows = sorted(rows, key=_hyperliquid_dislocation_execution_sort_key, reverse=True)
    for row in sorted_rows:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in output:
            continue
        output[key] = row
    return output


def _hyperliquid_dislocation_execution_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    gate_rank = {
        "paper_execution_probe": 4,
        "wide_spread_watch": 3,
        "no_edge_after_rough_cost": 2,
        "too_large_for_visible_depth": 1,
        "no_visible_depth": 0,
    }.get(row.get("gate_action", ""), 0)
    return (
        gate_rank,
        _float(row.get("conservative_net_15m_bps")),
        -_float(row.get("candidate_size_usd")),
    )


def _selected_hyperliquid_dislocation_tickets(
    *,
    rows: tuple[dict[str, str], ...],
    labels: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[dict[str, str], ...]:
    candidates_by_key = {
        (row.get("asset", ""), row.get("status", ""), row.get("side", "")): row
        for row in rows
    }
    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def add(row: dict[str, str]) -> None:
        key = (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not key[0] or not key[1] or not key[2] or key in selected_keys:
            return
        selected.append(row)
        selected_keys.add(key)

    for row in sorted(rows, key=lambda row: _float(row.get("score")), reverse=True)[:10]:
        add(row)

    supported_labels = sorted(
        (
            label
            for label in labels.values()
            if label.get("outcome_15m") == "paper_15m_win" or label.get("outcome_1h") == "paper_1h_win"
        ),
        key=lambda row: (
            row.get("outcome_1h") == "paper_1h_win",
            _float(row.get("net_15m_bps")),
            _float(row.get("score")),
        ),
        reverse=True,
    )
    for label in supported_labels[:10]:
        row = candidates_by_key.get((label.get("asset", ""), label.get("status", ""), label.get("side", "")))
        if row:
            add(row)

    return tuple(selected[:18])


def _hyperliquid_dislocation_stack_status(
    status: str,
    label: dict[str, str],
    monitor: dict[str, str],
    execution: dict[str, str],
) -> str:
    if execution.get("gate_action") == "paper_execution_probe":
        return "paper_dislocation_executable_probe"
    if label.get("outcome_1h") == "paper_1h_win":
        return "paper_dislocation_1h_supported_candidate"
    if label.get("outcome_15m") == "paper_15m_win":
        return "paper_dislocation_15m_supported_candidate"
    if label.get("outcome_15m") == "paper_15m_loss":
        return "paper_dislocation_15m_failed_candidate"
    if monitor.get("monitor_action") == "repeat_label_priority":
        return "paper_dislocation_repeat_monitor_candidate"
    return status


def _hyperliquid_dislocation_label_bonus(label: dict[str, str]) -> float:
    net_1h = label.get("net_1h_bps")
    if net_1h:
        return max(min(_float(net_1h) / 100.0, 5.0), -15.0)
    net_15m = label.get("net_15m_bps")
    if net_15m:
        return max(min(_float(net_15m) / 150.0, 3.0), -10.0)
    return 0.0


def _hyperliquid_dislocation_next_step(
    *,
    asset: str,
    ticket: dict[str, str],
    label: dict[str, str],
    execution: dict[str, str],
) -> str:
    if execution.get("gate_action") == "paper_execution_probe":
        return f"paper probe {asset} dislocation at the gated size and wait for 1h/4h confirmation"
    if label.get("outcome_1h") == "paper_1h_win":
        return f"repeat {asset} dislocation labels on fresh snapshots and add depth-gated paper probes"
    if label.get("outcome_15m") == "paper_15m_win":
        return f"wait for {asset} 1h/4h dislocation labels before promotion"
    if label.get("outcome_15m") == "paper_15m_loss":
        return f"deprioritize {asset} until a fresh dislocation snapshot appears"
    return ticket.get(
        "next_step",
        f"label {asset} dislocation candidate over 15m/1h/4h with costs",
    )


def _hyperliquid_dislocation_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "perp_market_map" / "current_hyperliquid_dislocation_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "dislocation_repeat_execution_candidate",
                "dislocation_repeat_needs_execution_check",
                "dislocation_single_snapshot_1h_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    selected_tickets: list[dict[str, str]] = []
    seen_asset_sides: set[tuple[str, str]] = set()
    for ticket in tickets:
        asset_side = (ticket.get("asset", ""), ticket.get("side", ""))
        if asset_side in seen_asset_sides:
            continue
        seen_asset_sides.add(asset_side)
        selected_tickets.append(ticket)
        if len(selected_tickets) >= 8:
            break
    output: list[AlphaStackRow] = []
    for ticket in selected_tickets:
        asset = ticket.get("asset", "")
        status = ticket.get("status", "")
        source_count = 3 if status == "dislocation_repeat_execution_candidate" else 1
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_{_slug(ticket.get('source_status', ''))}_actionability",
                status=status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    status,
                    source_count=source_count,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="perp_market_map",
                evidence=(
                    f"{asset}: source={ticket.get('source_status', '')}, "
                    f"monitor_obs={ticket.get('monitor_observations', '')}, "
                    f"current15={ticket.get('current_outcome_15m', '')} {ticket.get('current_net_15m_bps', '')}, "
                    f"current1h={ticket.get('current_outcome_1h', '')} {ticket.get('current_net_1h_bps', '')}, "
                    f"gate={ticket.get('execution_gate', '')}, "
                    f"cons_net1h={ticket.get('conservative_net_1h_bps', '')}, "
                    f"history={ticket.get('history_action', '')}, "
                    f"hist_win1h={ticket.get('history_win_1h', '')}, "
                    f"hist_mean1h={ticket.get('history_mean_net_1h_bps', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "dislocation candidate still needs repeated paper probes, stop behavior, and adverse-selection checks",
                ),
                next_step=ticket.get(
                    "next_step",
                    f"repeat {asset} dislocation paper probe with current execution evidence",
                ),
            )
        )
    return tuple(output)


def _hyperliquid_oi_shift_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "perp_market_map" / "current_hyperliquid_oi_shift_candidates.csv")
    tickets = sorted(rows, key=lambda row: _float(row.get("score")), reverse=True)
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        asset = ticket.get("asset", "")
        status = ticket.get("status", "paper_oi_funding_crowding_watch")
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_{_slug(status)}",
                status=status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    status,
                    source_count=1,
                    raw_score=_float(ticket.get("score")) * 2.5,
                ),
                sources="perp_market_map",
                evidence=(
                    f"{asset}: obs={ticket.get('observations', '')}, "
                    f"oi_change={ticket.get('open_interest_notional_change_pct', '')}, "
                    f"ret24={ticket.get('return_24h', '')}, "
                    f"funding={ticket.get('annualized_funding', '')}, "
                    f"oi_vol={ticket.get('oi_volume_ratio', '')}, "
                    f"impact={ticket.get('impact_spread', '')}, "
                    f"reason={ticket.get('reason', '')}"
                ),
                conflict=(
                    "short-window OI notional can reflect mark-price movement, "
                    "position changes, or stale monitor sampling; it needs forward labels "
                    "and cross-venue OI before promotion"
                ),
                next_step=ticket.get(
                    "next_step",
                    f"label {asset} OI-shift candidate over 15m/1h/4h with costs",
                ),
            )
        )
    return tuple(output)


def _perp_crowding_validated_stacks(
    rows: tuple[dict[str, str], ...],
    execution_rows: tuple[dict[str, str], ...],
    outcome_rows: tuple[dict[str, str], ...],
) -> tuple[AlphaStackRow, ...]:
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_validated_carry_reversion_candidate",
                "paper_delayed_carry_reversion_watch",
                "paper_carry_reversion_needs_more_labels",
            }
        ),
        key=lambda row: _float(row.get("validation_score")),
        reverse=True,
    )
    best_execution = _best_crowding_execution_by_candidate(execution_rows)
    best_outcome = _best_crowding_outcome_by_candidate(outcome_rows)
    output: list[AlphaStackRow] = []
    for ticket in tickets[:8]:
        asset = ticket.get("asset", "")
        status = ticket.get("status", "")
        execution = best_execution.get((asset, ticket.get("action", "")), {})
        outcome = best_outcome.get((asset, ticket.get("action", "")), {})
        stack_status = _perp_crowding_stack_status(status, execution, outcome)
        if stack_status == "paper_outcome_failed_carry_reversion_probe":
            continue
        execution_note = ""
        if execution:
            execution_note = (
                f", gate={execution.get('gate_action', '')}, "
                f"size={execution.get('candidate_size_usd', '')}, "
                f"conservative_net1h_bps={execution.get('conservative_net_1h_bps', '')}, "
                f"spread_bps={execution.get('spread_bps', '')}, "
                f"depth_usage={execution.get('visible_depth_usage_10bps', '')}"
            )
        outcome_note = ""
        if outcome:
            outcome_note = (
                f", outcome15={outcome.get('outcome_15m', '')}, "
                f"net15_bps={outcome.get('net_15m_bps', '')}, "
                f"outcome1h={outcome.get('outcome_1h', '')}, "
                f"net1h_bps={outcome.get('net_1h_bps', '')}"
            )
        outcome_bonus = _perp_crowding_outcome_bonus(outcome)
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_validated_perp_crowding_reversion",
                status=stack_status,
                side=ticket.get("action", ""),
                priority_score=(
                    _priority_score(
                        stack_status,
                        source_count=3 if execution else 2,
                        raw_score=_float(ticket.get("validation_score")),
                    )
                    + (_float(execution.get("conservative_net_1h_bps")) / 1_000.0 if execution else 0.0)
                    + outcome_bonus
                ),
                sources="perp_market_map + candidate_validation",
                evidence=(
                    f"{asset}: action={ticket.get('action', '')}, "
                    f"monitor_obs={ticket.get('monitor_observations', '')}, "
                    f"label_obs={ticket.get('label_observations', '')}, "
                    f"dir15={ticket.get('mean_directional_return_15m', '')}, "
                    f"dir1h={ticket.get('mean_directional_return_1h', '')}, "
                    f"net1h_proxy={ticket.get('net_directional_return_1h_proxy', '')}, "
                    f"hit1h={ticket.get('positive_directional_1h_rate', '')}, "
                    f"funding={ticket.get('mean_annualized_funding', '')}, "
                    f"impact={ticket.get('mean_impact_spread', '')}"
                    f"{execution_note}"
                    f"{outcome_note}"
                ),
                conflict=(
                    "validated label sample is still tiny; public-book gate excludes queue position, "
                    "repeated adverse selection, stop behavior, and live fill evidence"
                ),
                next_step=_perp_crowding_next_step(
                    asset=asset,
                    ticket=ticket,
                    execution=execution,
                    outcome=outcome,
                ),
            )
        )
    return tuple(output)


def _perp_crowding_stack_status(
    status: str,
    execution: dict[str, str],
    outcome: dict[str, str],
) -> str:
    if outcome.get("outcome_1h") == "paper_1h_win":
        if outcome.get("outcome_15m") == "paper_15m_loss":
            return "paper_delayed_carry_reversion_probe"
        return "paper_outcome_supported_carry_reversion_probe"
    if outcome.get("outcome_1h") == "paper_1h_loss":
        return "paper_outcome_failed_carry_reversion_probe"
    if outcome.get("outcome_15m") == "paper_15m_win":
        return "paper_short_horizon_supported_carry_reversion_probe"
    if outcome.get("outcome_15m") == "paper_15m_loss":
        return "paper_outcome_failed_carry_reversion_probe"
    gate_action = execution.get("gate_action")
    if status == "paper_validated_carry_reversion_candidate" and gate_action == "paper_execution_probe":
        return "paper_executable_carry_reversion_probe"
    if gate_action in {"wide_spread_watch", "too_large_for_visible_depth", "no_edge_after_rough_cost"}:
        return gate_action
    return status


def _best_crowding_execution_by_candidate(
    rows: tuple[dict[str, str], ...],
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    sorted_rows = sorted(rows, key=_crowding_execution_sort_key, reverse=True)
    for row in sorted_rows:
        key = (row.get("asset", ""), row.get("action", ""))
        if not key[0] or not key[1] or key in output:
            continue
        output[key] = row
    return output


def _crowding_execution_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    gate_rank = {
        "paper_execution_probe": 4,
        "wide_spread_watch": 3,
        "no_edge_after_rough_cost": 2,
        "too_large_for_visible_depth": 1,
        "no_visible_depth": 0,
    }.get(row.get("gate_action", ""), 0)
    return (
        gate_rank,
        _float(row.get("conservative_net_1h_bps")),
        -_float(row.get("candidate_size_usd")),
    )


def _best_crowding_outcome_by_candidate(
    rows: tuple[dict[str, str], ...],
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    sorted_rows = sorted(rows, key=lambda row: row.get("entry_timestamp", ""), reverse=True)
    for row in sorted_rows:
        key = (row.get("asset", ""), row.get("action", ""))
        if not key[0] or not key[1] or key in output:
            continue
        output[key] = row
    return output


def _perp_crowding_outcome_bonus(outcome: dict[str, str]) -> float:
    net_1h = outcome.get("net_1h_bps")
    if net_1h:
        return max(min(_float(net_1h) / 100.0, 5.0), -15.0)
    net_15m = outcome.get("net_15m_bps")
    if net_15m:
        return max(min(_float(net_15m) / 150.0, 3.0), -10.0)
    return 0.0


def _perp_crowding_next_step(
    *,
    asset: str,
    ticket: dict[str, str],
    execution: dict[str, str],
    outcome: dict[str, str],
) -> str:
    if outcome.get("outcome_1h") == "paper_1h_loss":
        return f"do not promote {asset}; repeat only if a fresh crowding snapshot passes the execution gate"
    if outcome.get("outcome_1h") == "paper_1h_win" or outcome.get("outcome_15m") == "paper_15m_win":
        return f"repeat {asset} gated paper probes on fresh snapshots and record live fill evidence"
    if outcome.get("outcome_15m") == "pending_15m" or outcome.get("outcome_1h") == "pending_1h":
        return f"wait for {asset} gated paper-probe horizons, then refresh the outcome tracker"
    if outcome.get("outcome_15m") == "paper_15m_loss":
        return f"do not promote {asset}; repeat only if a fresh crowding snapshot passes the execution gate"
    if execution.get("gate_action") == "paper_execution_probe":
        return f"start {asset} paper outcome tracking at the gated size"
    return ticket.get(
        "next_step",
        f"repeat {asset} carry-reversion labels and add execution costs",
    )


def _protocol_fee_valuation_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "protocol_fundamentals" / "current_protocol_fee_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_valuation.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status") in {"paper_value_growth_candidate", "paper_value_watch"}
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:5]:
        token = ticket.get("token_symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{token.lower()}_fee_yield_valuation",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="protocol_fundamentals",
                evidence=(
                    f"{token}/{ticket.get('protocol', '')}: "
                    f"fee_to_mcap={ticket.get('fee_to_market_cap', '')}, "
                    f"fee_to_fdv={ticket.get('fee_to_fdv', '')}, "
                    f"growth7d={ticket.get('change_7d_over_7d', '')}, "
                    f"funding={ticket.get('funding', '')}"
                ),
                conflict="DeFiLlama fees are not strict token-holder revenue; token capture, emissions, FDV, and unlocks can break the valuation link",
                next_step=ticket.get(
                    "next_step",
                    f"label {token} fee-yield valuation snapshots against forward returns",
                ),
            )
        )
    return tuple(output)


def _protocol_fee_price_context_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "protocol_fundamentals" / "current_protocol_fee_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_price_context.csv")
    execution_by_token = {
        row.get("token_symbol", ""): row
        for row in _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_execution_context.csv")
        if row.get("token_symbol")
    }
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "fee_growth_price_lag_candidate",
                "fee_growth_price_confirmation",
                "fee_growth_price_chase_risk",
                "fee_decay_price_weakness_context",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        token = ticket.get("token_symbol", "")
        execution = execution_by_token.get(token, {})
        execution_bonus = _protocol_fee_execution_bonus(execution)
        output.append(
            AlphaStackRow(
                opportunity=f"{token.lower()}_fee_growth_price_context",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=3 if execution.get("action") == "paper_observation_ready" else 2,
                    raw_score=_float(ticket.get("score")) + execution_bonus,
                ),
                sources="protocol_fundamentals + market_price_context",
                evidence=(
                    f"{token}/{ticket.get('protocol', '')}: "
                    f"fee_to_mcap={ticket.get('fee_to_market_cap', '')}, "
                    f"fee_to_fdv={ticket.get('fee_to_fdv', '')}, "
                    f"growth7d={ticket.get('fee_growth_7d', '')}, "
                    f"price7d={ticket.get('price_change_7d', '')}, "
                    f"price30d={ticket.get('price_change_30d', '')}"
                    f"{_protocol_fee_execution_evidence(execution)}"
                ),
                conflict=_protocol_fee_price_context_conflict(execution),
                next_step=execution.get("next_step")
                or ticket.get(
                    "next_step",
                    f"paper-label {token} fee-growth price context over multiple horizons",
                ),
            )
        )
    return tuple(output)


def _protocol_fee_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "protocol_fundamentals" / "current_protocol_fee_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "protocol_fee_repeat_execution_candidate",
                "protocol_fee_label_supported_watch",
                "protocol_fee_pending_forward_label",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:8]:
        token = ticket.get("token_symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{token.lower()}_protocol_fee_actionability",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=4 if ticket.get("status") == "protocol_fee_repeat_execution_candidate" else 3,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="protocol_fundamentals + market_price_context + forward_label + execution_context",
                evidence=(
                    f"{token}/{ticket.get('protocol', '')}: "
                    f"thesis={ticket.get('thesis_status', '')} {ticket.get('thesis_score', '')}, "
                    f"fee_to_mcap={ticket.get('fee_to_market_cap', '')}, "
                    f"growth7d={ticket.get('fee_growth_7d', '')}, "
                    f"price7d={ticket.get('price_change_7d', '')}, "
                    f"exec={ticket.get('execution_action', '')}, "
                    f"labels={ticket.get('label_observations', '')}, "
                    f"labeled4h={ticket.get('labeled_4h', '')}, "
                    f"wins4h={ticket.get('wins_4h', '')}, "
                    f"mean4h={ticket.get('mean_directional_4h', '')}, "
                    f"latest_label={ticket.get('latest_label_status', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "protocol fees are not strict token-holder revenue and the forward-label evidence may be immature",
                ),
                next_step=ticket.get(
                    "next_step",
                    f"wait for {token} forward label before promotion",
                ),
            )
        )
    return tuple(output)


def _protocol_fee_execution_bonus(execution: dict[str, str]) -> float:
    action = execution.get("action", "")
    if action == "paper_observation_ready":
        return 18.0
    if action == "non_hyperliquid_route_check":
        return 6.0
    if action in {"wide_spread_watch", "thin_depth_watch", "thin_volume_watch"}:
        return -8.0
    if action == "venue_gap":
        return -20.0
    return 0.0


def _protocol_fee_execution_evidence(execution: dict[str, str]) -> str:
    if not execution:
        return ""
    return (
        f", exec_action={execution.get('action', '')}, "
        f"venues={execution.get('venue_count', '')}, "
        f"hl_funding={execution.get('hl_annualized_funding', '')}, "
        f"hl_spread_bps={execution.get('hl_spread_bps', '')}, "
        f"hl_depth_10bps={execution.get('hl_near_depth_10bps_notional', '')}"
    )


def _protocol_fee_price_context_conflict(execution: dict[str, str]) -> str:
    base = (
        "fee growth can be lagging, already chased, or disconnected from token value; "
        "CoinGecko price context is current movement, not a forward label"
    )
    if not execution:
        return base
    return f"{base}; execution gate: {execution.get('reason', '')}"


def _defi_yield_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "defi_yield" / "current_yield_quality_screen.csv")
    conflict_symbols = {
        row.get("symbol", "")
        for row in _read_rows(root / "defi_yield" / "current_yield_peg_risk_join.csv")
        if row.get("status")
        in {
            "paper_yield_depeg_conflict_watch",
            "paper_yield_premium_conflict_watch",
            "yield_supply_stress_watch",
        }
    }
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status") in {"paper_base_yield_watch", "paper_incentive_yield_watch"}
            and row.get("symbol", "") not in conflict_symbols
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:5]:
        chain = ticket.get("chain", "")
        project = ticket.get("project", "")
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower().replace(' ', '_')}_{project}_{symbol.lower()}_yield",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="defi_yield",
                evidence=(
                    f"{chain}/{project} {symbol}: apy={ticket.get('apy', '')}, "
                    f"base={ticket.get('apy_base', '')}, reward={ticket.get('apy_reward', '')}, "
                    f"tvl={ticket.get('tvl_usd', '')}, dev30={ticket.get('apy_deviation_30d', '')}"
                ),
                conflict="DeFi yield requires custody, withdrawal, smart-contract, issuer, APY-decay, and exit-liquidity checks",
                next_step=ticket.get(
                    "next_step",
                    f"check {chain}/{project} custody, APY source, capacity, and exit liquidity",
                ),
            )
        )
    return tuple(output)


def _yield_peg_risk_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "defi_yield" / "current_yield_peg_risk_join.csv")
    covered_peg_symbols = _covered_peg_anomaly_symbols(root)
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_yield_depeg_conflict_watch",
                "paper_yield_premium_conflict_watch",
                "yield_supply_stress_watch",
                "paper_yield_without_peg_stress_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen_peg_symbols: set[str] = set()
    for ticket in tickets:
        chain = ticket.get("chain", "")
        project = ticket.get("project", "")
        symbol = ticket.get("symbol", "")
        peg_symbol = ticket.get("peg_symbol", "")
        normalized_peg_symbol = _normalize_symbol(peg_symbol or symbol)
        if normalized_peg_symbol in covered_peg_symbols:
            continue
        if normalized_peg_symbol in seen_peg_symbols:
            continue
        seen_peg_symbols.add(normalized_peg_symbol)
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower().replace(' ', '_')}_{project}_{symbol.lower()}_yield_peg",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=2 if peg_symbol else 1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="defi_yield + stablecoin_liquidity",
                evidence=(
                    f"{chain}/{project} {symbol}: apy={ticket.get('apy', '')}, "
                    f"base={ticket.get('apy_base', '')}, tvl={ticket.get('tvl_usd', '')}, "
                    f"peg={peg_symbol or 'unmatched'}, price={ticket.get('peg_price', '')}, "
                    f"peg_deviation={ticket.get('peg_deviation', '')}"
                ),
                conflict=(
                    "yield can be real carry, but peg, redemption, issuer, custody, "
                    "and exit-liquidity risk can fully explain the APY"
                ),
                next_step=ticket.get(
                    "next_step",
                    f"check {symbol} peg source, redemption path, custody, and exit liquidity",
                ),
            )
        )
        if len(output) >= 6:
            break
    return tuple(output)


def _dex_pool_flow_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "dex_pool_flow" / "current_geckoterminal_pool_flow.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_dex_pool_momentum_watch",
                "paper_dex_reversal_risk_watch",
                "dex_liquidity_stress_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen: set[str] = set()
    for ticket in tickets:
        name = ticket.get("name", "")
        pool_key = f"{ticket.get('network', '')}:{name.lower()}"
        if pool_key in seen:
            continue
        seen.add(pool_key)
        output.append(
            AlphaStackRow(
                opportunity=f"{ticket.get('network', '').lower()}_{name.lower().replace(' / ', '_')}_dex_pool",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="dex_pool_flow",
                evidence=(
                    f"{ticket.get('network', '')}/{ticket.get('dex', '')} {name}: "
                    f"reserve={ticket.get('reserve_usd', '')}, vol1h={ticket.get('volume_h1_usd', '')}, "
                    f"vol_reserve={ticket.get('volume_reserve_ratio_h1', '')}, "
                    f"chg1h={ticket.get('price_change_h1', '')}, chg24h={ticket.get('price_change_h24', '')}"
                ),
                conflict="DEX pool flow can be thin, manipulated, or unexecutable; route depth, slippage, gas, MEV, and token restrictions are unvalidated",
                next_step=ticket.get(
                    "next_step",
                    f"check {name} route depth, slippage, gas, MEV, and repeated pool flow",
                ),
            )
        )
        if len(output) >= 5:
            break
    return tuple(output)


def _defi_lending_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "defi_lending" / "current_lending_stress_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "defi_lending" / "current_morpho_lending_rates.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_borrow_liquidity_stress_watch",
                "paper_stable_lending_yield_watch",
                "borrow_demand_context_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen: set[tuple[str, str, str]] = set()
    for ticket in tickets:
        chain = ticket.get("chain", "")
        loan = ticket.get("loan_asset", "")
        collateral = ticket.get("collateral_asset", "")
        key = (chain, loan, collateral)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower()}_{loan.lower()}_{collateral.lower()}_lending_pressure",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="defi_lending",
                evidence=(
                    f"{chain} {loan}/{collateral}: util={ticket.get('utilization', '')}, "
                    f"supply={ticket.get('supply_usd', '')}, borrow={ticket.get('borrow_usd', '')}, "
                    f"liquidity={ticket.get('liquidity_usd', '')}, "
                    f"avg_supply_apy={ticket.get('avg_net_supply_apy', '')}, "
                    f"avg_borrow_apy={ticket.get('avg_net_borrow_apy', '')}"
                ),
                conflict="lending-rate pressure can be protocol, oracle, collateral, or withdrawal risk rather than clean alpha",
                next_step=ticket.get(
                    "next_step",
                    f"check {chain} {loan}/{collateral} rate persistence and liquidation risk",
                ),
            )
        )
        if len(output) >= 6:
            break
    return tuple(output)


def _event_probability_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "prediction_markets" / "current_event_probability_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "event_probability_candidate_after_refresh_check",
                "event_probability_candidate_after_current_quote_check",
                "event_probability_restart_after_failed_refresh",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:4]:
        status = ticket.get("status", "")
        source_count = 6
        if status == "event_probability_candidate_after_current_quote_check":
            source_count = 5
        if status == "event_probability_restart_after_failed_refresh":
            source_count = 3
        output.append(
            AlphaStackRow(
                opportunity="event_probability_actionability",
                status=status,
                side=f"{ticket.get('suggested_side', '')}: {ticket.get('question', '')}",
                priority_score=_priority_score(
                    status,
                    source_count=source_count,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="prediction_markets + external_news + probability_gap + clob_depth + source_quality + refresh",
                evidence=(
                    f"bid={ticket.get('current_bid', '')}, ask={ticket.get('current_ask', '')}, "
                    f"spread={ticket.get('spread', '')}, edge_after_ask={ticket.get('current_edge_after_ask', '')}, "
                    f"bid_pnl={ticket.get('mark_to_bid_pnl', '')}, depth_5c={ticket.get('ask_depth_to_5c', '')}, "
                    f"source_quality={ticket.get('source_quality_status', '')}, refresh={ticket.get('refresh_status', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "event-probability candidate still needs fill, fee, queue, resolution-risk, and adverse-selection checks",
                ),
                next_step=ticket.get(
                    "next_step",
                    "paper-check event probability candidate under explicit execution and resolution assumptions",
                ),
            )
        )
    return tuple(output)


def _lending_stress_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "defi_lending" / "current_lending_stress_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "lending_rate_candidate_after_risk_check",
                "lending_stress_mechanics_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        chain = ticket.get("chain", "")
        loan = ticket.get("loan_asset", "")
        collateral = ticket.get("collateral_asset", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower()}_{loan.lower()}_{collateral.lower()}_lending_actionability",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="defi_lending",
                evidence=(
                    f"{chain} {loan}/{collateral}: util={ticket.get('utilization', '')}, "
                    f"supply={ticket.get('supply_usd', '')}, borrow={ticket.get('borrow_usd', '')}, "
                    f"liquidity={ticket.get('liquidity_usd', '')}, "
                    f"avg_supply_apy={ticket.get('avg_net_supply_apy', '')}, "
                    f"avg_borrow_apy={ticket.get('avg_net_borrow_apy', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "lending stress needs capacity, exit liquidity, collateral, oracle, and withdrawal checks",
                ),
                next_step=ticket.get(
                    "next_step",
                    f"check {chain} {loan}/{collateral} lending capacity and risk mechanics",
                ),
            )
        )
    return tuple(output)


def _attention_funding_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "news_social" / "current_attention_market_join.csv")
    tickets = sorted(rows, key=lambda row: _float(row.get("score")), reverse=True)
    output: list[AlphaStackRow] = []
    for ticket in tickets[:3]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_attention_funding_overlap",
                status="paper_attention_funding_watch",
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    "paper_attention_funding_watch",
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="news_social",
                evidence=(
                    f"{symbol}/{ticket.get('name', '')}: attention_rank={ticket.get('attention_rank', '')}, "
                    f"attention_change={ticket.get('attention_24h_change', '')}, "
                    f"funding={ticket.get('annualized_funding', '')}, "
                    f"impact_spread={ticket.get('impact_spread', '')}"
                ),
                conflict="attention is not causal edge; social/narrative signals need repeat labels and latency checks",
                next_step=f"collect repeated {symbol} attention/funding overlaps and compare 15m/1h decay after costs",
            )
        )
    return tuple(output)


def _attention_price_context_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "news_social" / "current_attention_price_context.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "attention_price_lag_candidate",
                "attention_breakout_continuation_watch",
                "attention_capitulation_reversal_watch",
                "attention_chase_risk",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_attention_price_context",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="news_social + market_price_context",
                evidence=(
                    f"{symbol}: "
                    f"name={ticket.get('name', '')}, "
                    f"rank={ticket.get('attention_rank', '')}, "
                    f"price24h={ticket.get('price_change_24h', '')}, "
                    f"price7d={ticket.get('price_change_7d', '')}, "
                    f"price30d={ticket.get('price_change_30d', '')}, "
                    f"vol_mcap={ticket.get('volume_to_market_cap', '')}"
                ),
                conflict=(
                    "attention can be non-causal, late, bot-driven, or already crowded; "
                    "needs leakage-safe forward labels and execution checks"
                ),
                next_step=ticket.get(
                    "next_step",
                    f"paper-label {symbol} attention price context over short horizons",
                ),
            )
        )
    return tuple(output)


def _market_breadth_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "market_breadth" / "current_volume_price_dislocation.csv")
    labels = _read_rows(root / "market_breadth" / "current_volume_price_dislocation_labels.csv")
    execution_rows = _read_rows(root / "market_breadth" / "current_volume_price_dislocation_execution_gate.csv")
    candidates_by_symbol = {row.get("symbol", ""): row for row in rows}
    execution_by_symbol = {row.get("symbol", ""): row for row in execution_rows}
    tickets = (
        _ranked_market_breadth_labels(labels, execution_by_symbol=execution_by_symbol)
        if labels
        else sorted(
            (
                row
                for row in rows
                if row.get("status")
                in {
                    "volume_reversal_candidate",
                    "capitulation_reversal_watch",
                    "breakout_continuation_watch",
                    "chase_risk",
                }
            ),
            key=lambda row: _float(row.get("score")),
            reverse=True,
        )
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:8]:
        symbol = ticket.get("symbol", "")
        candidate = candidates_by_symbol.get(symbol, ticket)
        execution = execution_by_symbol.get(symbol, {})
        status = _market_breadth_status(ticket, execution=execution)
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_volume_price_dislocation",
                status=status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    status,
                    source_count=3 if execution else 2 if ticket.get("label_status") else 1,
                    raw_score=_market_breadth_raw_score(ticket=ticket, execution=execution),
                ),
                sources="market_breadth + market_price_context",
                evidence=(
                    f"{symbol}: "
                    f"name={ticket.get('name', '')}, "
                    f"rank={candidate.get('market_cap_rank', '')}, "
                    f"vol_mcap={candidate.get('volume_to_market_cap', '')}, "
                    f"price24h={candidate.get('price_change_24h', '')}, "
                    f"price7d={candidate.get('price_change_7d', '')}, "
                    f"price30d={candidate.get('price_change_30d', '')}"
                    f"{_market_breadth_label_evidence(ticket)}"
                    f"{_market_breadth_execution_evidence(execution)}"
                ),
                conflict=_market_breadth_conflict(ticket, execution=execution),
                next_step=_market_breadth_next_step(ticket, execution=execution),
            )
        )
    return tuple(output)


def _ranked_market_breadth_labels(
    rows: tuple[dict[str, str], ...],
    *,
    execution_by_symbol: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _market_breadth_execution_rank(execution_by_symbol.get(row.get("symbol", ""), {})),
            _float(execution_by_symbol.get(row.get("symbol", ""), {}).get("conservative_net_4h_bps")),
            _market_breadth_label_rank(row),
            _float(row.get("directional_return_4h")),
            _float(row.get("directional_return_1h")),
            _float(row.get("score")),
        ),
        reverse=True,
    )


def _market_breadth_execution_rank(execution: dict[str, str]) -> int:
    return {
        "paper_execution_probe": 6,
        "thin_volume_watch": 4,
        "wide_spread_watch": 3,
        "too_large_for_visible_depth": 3,
        "no_edge_after_rough_cost": 2,
        "label_contradicted": 1,
        "not_hyperliquid": 0,
        "missing_l2_context": 0,
    }.get(execution.get("action", ""), 0)


def _market_breadth_raw_score(*, ticket: dict[str, str], execution: dict[str, str]) -> float:
    net_4h_bps = _float(execution.get("conservative_net_4h_bps"))
    return _float(ticket.get("score")) + min(max(net_4h_bps, 0.0) / 10.0, 50.0)


def _market_breadth_label_rank(row: dict[str, str]) -> int:
    dir_4h = _float(row.get("directional_return_4h"))
    dir_1h = _float(row.get("directional_return_1h"))
    if dir_4h > 0.0 and dir_1h > 0.0:
        return 4
    if dir_4h > 0.0:
        return 3
    if dir_1h > 0.0:
        return 2
    if row.get("directional_return_1h", "") != "":
        return 1
    return 0


def _market_breadth_status(ticket: dict[str, str], *, execution: dict[str, str]) -> str:
    action = execution.get("action", "")
    if action == "paper_execution_probe":
        return "volume_dislocation_execution_probe"
    if action == "thin_volume_watch":
        return "volume_dislocation_thin_volume_watch"
    if action == "wide_spread_watch":
        return "volume_dislocation_wide_spread_watch"
    if action == "too_large_for_visible_depth":
        return "volume_dislocation_too_large_for_visible_depth"
    if action == "no_edge_after_rough_cost":
        return "volume_dislocation_no_edge_after_rough_cost"
    if action == "label_contradicted":
        return "volume_dislocation_4h_contradicted_after_cost_check"
    if action == "not_hyperliquid":
        return "volume_dislocation_no_hyperliquid_venue"
    if action == "missing_l2_context":
        return "volume_dislocation_missing_l2_context"
    if not ticket.get("label_status"):
        return ticket.get("status", "")
    dir_4h = _float(ticket.get("directional_return_4h"))
    dir_1h = _float(ticket.get("directional_return_1h"))
    if dir_4h > 0.0 and dir_1h > 0.0:
        return "volume_dislocation_4h_supported_pending_12h"
    if dir_4h > 0.0:
        return "volume_dislocation_delayed_4h_support"
    if dir_1h > 0.0:
        return "volume_dislocation_1h_only_watch"
    if ticket.get("directional_return_1h", "") != "":
        return "volume_dislocation_4h_contradicted_pending_12h"
    return ticket.get("status", "")


def _market_breadth_label_evidence(label: dict[str, str]) -> str:
    if not label.get("label_status"):
        return ""
    return (
        f"; label={label.get('label_status', '')}, "
        f"dir1h={label.get('directional_return_1h', '')}, "
        f"dir4h={label.get('directional_return_4h', '')}, "
        f"source={label.get('price_source', '')}"
    )


def _market_breadth_execution_evidence(execution: dict[str, str]) -> str:
    if not execution:
        return ""
    return (
        f"; exec={execution.get('action', '')}, "
        f"net4h_bps={execution.get('conservative_net_4h_bps', '')}, "
        f"spread_bps={execution.get('spread_bps', '')}, "
        f"depth_usage_250={execution.get('visible_depth_usage_250', '')}"
    )


def _market_breadth_conflict(label: dict[str, str], *, execution: dict[str, str]) -> str:
    base = (
        "volume/price dislocation can be a liquidation bounce, news reaction, or crowded trap; "
        "venue depth, fees, funding, stop behavior, and repeat labels are still required"
    )
    action = execution.get("action", "")
    if action == "paper_execution_probe":
        return f"{base}; rough public-book gate passes but realized fills, stops, and repeat labels are unproven"
    if action == "no_edge_after_rough_cost":
        return f"{base}; current rough cost model erases the 4h label"
    if action == "label_contradicted":
        return f"{base}; current 4h label contradicts the direction"
    if action in {"thin_volume_watch", "wide_spread_watch", "too_large_for_visible_depth"}:
        return f"{base}; current venue context is too weak for a small paper probe"
    status = _market_breadth_status(label, execution=execution)
    if status == "volume_dislocation_4h_supported_pending_12h":
        return f"{base}; current 1h and 4h labels support the direction, but 12h confirmation is pending"
    if status == "volume_dislocation_delayed_4h_support":
        return f"{base}; current 4h label supports the direction after a weak or negative 1h mark"
    if status == "volume_dislocation_1h_only_watch":
        return f"{base}; current 1h label is positive but 4h is weak, negative, or pending"
    if status == "volume_dislocation_4h_contradicted_pending_12h":
        return f"{base}; current short-horizon labels contradict the direction"
    return base


def _market_breadth_next_step(label: dict[str, str], *, execution: dict[str, str]) -> str:
    if execution.get("next_step"):
        return execution["next_step"]
    symbol = label.get("symbol", "")
    status = _market_breadth_status(label, execution=execution)
    if status == "volume_dislocation_4h_supported_pending_12h":
        return f"repeat {symbol} volume-dislocation label and add execution cost, funding, depth, and stop checks"
    if status == "volume_dislocation_delayed_4h_support":
        return f"repeat {symbol} delayed 4h volume-dislocation label and check stop behavior"
    if status == "volume_dislocation_1h_only_watch":
        return f"wait for stronger {symbol} 4h/12h confirmation before promotion"
    if status == "volume_dislocation_4h_contradicted_pending_12h":
        return f"do not promote {symbol} without a fresh non-overlapping positive label"
    return label.get("next_step", f"paper-label {symbol} market-breadth dislocation")


def _news_event_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "news_social" / "current_news_event_screen.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_news_event_reaction_watch",
                "paper_news_security_risk_watch",
                "paper_news_regulatory_risk_watch",
                "paper_news_macro_crypto_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen: set[tuple[str, str]] = set()
    for ticket in tickets:
        symbol = ticket.get("symbol", "")
        event_kind = ticket.get("event_kind", "")
        key = (symbol, event_kind)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_{event_kind}_news_event",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="news_social",
                evidence=(
                    f"{ticket.get('source', '')} {symbol}: kind={event_kind}, "
                    f"age_h={ticket.get('age_hours', '')}, funding={ticket.get('annualized_funding', '')}, "
                    f"perp={ticket.get('perp_action', '')}, title={ticket.get('title', '')}"
                ),
                conflict="news headlines can be stale, duplicated, already priced, or non-causal; timestamp and source-leakage checks are required",
                next_step=ticket.get(
                    "next_step",
                    f"label {symbol} returns after current {event_kind} headlines",
                ),
            )
        )
        if len(output) >= 5:
            break
    return tuple(output)


def _protocol_activity_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "protocol_activity" / "current_protocol_activity_market_join.csv")
    tickets = sorted(rows, key=lambda row: _float(row.get("score")), reverse=True)
    output: list[AlphaStackRow] = []
    for ticket in tickets[:4]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_protocol_activity_context",
                status="paper_protocol_activity_watch",
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    "paper_protocol_activity_watch",
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="protocol_activity",
                evidence=(
                    f"{symbol}/{ticket.get('name', '')}: commits4w={ticket.get('commit_count_4_weeks', '')}, "
                    f"telegram={ticket.get('telegram_users', '')}, "
                    f"funding={ticket.get('annualized_funding', '')}, "
                    f"spread={ticket.get('impact_spread', '')}"
                ),
                conflict="developer and community activity are slow context, not immediate execution edge",
                next_step=f"label {symbol} protocol-activity context over longer horizons and join to funding/events",
            )
        )
    return tuple(output)


def _on_chain_flow_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "on_chain_flow" / "chain_tvl_flow_market_context_summary.csv")
    tickets = sorted(
        (row for row in rows if row.get("group_type") in {"token", "venue_token"}),
        key=lambda row: _float(row.get("mean_context_score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:4]:
        group = ticket.get("group_key", "")
        token = group.split("/")[-1]
        output.append(
            AlphaStackRow(
                opportunity=f"{group.lower().replace('/', '_')}_chain_flow_context",
                status="paper_chain_flow_watch",
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    "paper_chain_flow_watch",
                    source_count=1,
                    raw_score=_float(ticket.get("mean_context_score")) * 100.0,
                ),
                sources="on_chain_flow",
                evidence=(
                    f"{group}: observations={ticket.get('observations', '')}, "
                    f"hit15={ticket.get('hit_rate_15m', '')}, "
                    f"mean_dir15={ticket.get('mean_dir15', '')}, "
                    f"funding_support={ticket.get('mean_funding_support', '')}"
                ),
                conflict="sample is tiny and chain TVL flow may be slow or indirect relative to tradable venue returns",
                next_step=f"repeat {token} chain-flow labels and isolate venue-specific execution costs",
            )
        )
    return tuple(output)


def _stablecoin_peg_stress_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "stablecoin_liquidity" / "current_peg_stress_screen.csv")
    tradeability_symbols = {
        row.get("symbol", "").lower()
        for row in _read_rows(root / "anomaly_stress" / "current_peg_anomaly_tradeability.csv")
        if row.get("symbol")
    }
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_depeg_repeg_watch",
                "paper_premium_mean_reversion_watch",
                "peg_supply_stress_watch",
            }
            and row.get("symbol", "").lower() not in tradeability_symbols
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_stablecoin_peg_stress",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="stablecoin_liquidity",
                evidence=(
                    f"{symbol}/{ticket.get('name', '')}: price={ticket.get('price', '')}, "
                    f"peg_deviation={ticket.get('peg_deviation', '')}, "
                    f"supply={ticket.get('current_supply_usd', '')}, "
                    f"week_change={ticket.get('week_change_usd', '')}"
                ),
                conflict="stablecoin price can be stale or untradable; redemption path, venue depth, custody, and issuer risk must be checked first",
                next_step=ticket.get(
                    "next_step",
                    f"check {symbol} redemption route, exchange depth, and repeated peg snapshots",
                ),
            )
        )
    return tuple(output)


def _chain_stablecoin_migration_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "stablecoin_liquidity" / "current_chain_stablecoin_migration.csv")
    labels_by_chain = {
        row.get("chain", ""): row
        for row in _read_rows(root / "stablecoin_liquidity" / "current_chain_stablecoin_migration_forward_labels.csv")
        if row.get("chain")
    }
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_chain_stablecoin_inflow_watch",
                "paper_chain_stablecoin_outflow_watch",
                "chain_stablecoin_flow_reversal_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        chain = ticket.get("chain", "")
        token = ticket.get("token_symbol", "") or chain
        display_token = ticket.get("token_symbol", "") or "-"
        label = labels_by_chain.get(chain, {})
        status = _chain_stablecoin_migration_status(ticket.get("status", ""), label)
        source_count = 2 if label else 1
        label_evidence = _chain_stablecoin_migration_label_evidence(label)
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower().replace(' ', '_')}_stablecoin_migration",
                status=status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    status,
                    source_count=source_count,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="stablecoin_liquidity",
                evidence=(
                    f"{chain}/{display_token}: supply={ticket.get('current_supply_usd', '')}, "
                    f"day_change={ticket.get('day_change_usd', '')}, "
                    f"week_change={ticket.get('week_change_usd', '')}, "
                    f"week_pct={ticket.get('week_change_pct', '')}, "
                    f"top_asset={ticket.get('top_asset', '')}"
                    f"{label_evidence}"
                ),
                conflict=_chain_stablecoin_migration_conflict(label),
                next_step=label.get("next_step") or ticket.get(
                    "next_step",
                    f"label {token} returns after {chain} stablecoin migration",
                ),
            )
        )
    return tuple(output)


def _chain_stablecoin_migration_status(status: str, label: dict[str, str]) -> str:
    label_status = label.get("label_status", "")
    if label_status == "chain_migration_direction_supported":
        return "chain_stablecoin_label_supported_watch"
    if label_status == "chain_migration_direction_contradicted":
        return "chain_stablecoin_label_contradicted"
    if label_status == "mixed_chain_migration_direction":
        return "chain_stablecoin_mixed_label_watch"
    if label_status == "labeled_4h_pending_12h":
        directional_4h = _float(label.get("directional_return_4h"))
        if directional_4h > 0.0:
            return "chain_stablecoin_4h_supported_pending_12h"
        if directional_4h < 0.0:
            return "chain_stablecoin_4h_contradicted_pending_12h"
    return status


def _chain_stablecoin_migration_label_evidence(label: dict[str, str]) -> str:
    if not label:
        return ""
    return (
        f"; label={label.get('label_status', '')}, "
        f"dir1h={label.get('directional_return_1h', '')}, "
        f"dir4h={label.get('directional_return_4h', '')}, "
        f"dir12h={label.get('directional_return_12h', '')}"
    )


def _chain_stablecoin_migration_conflict(label: dict[str, str]) -> str:
    base = "stablecoin migration is a capital-flow proxy, not a bridge-fill; chain-token mapping, venues, and execution costs are still required"
    label_status = label.get("label_status", "")
    if label_status == "labeled_4h_pending_12h":
        return f"{base}; 4h label is available but 12h confirmation is still pending"
    if label_status == "chain_migration_direction_contradicted":
        return f"{base}; current forward labels contradict the migration direction"
    if label_status == "mixed_chain_migration_direction":
        return f"{base}; current forward labels are mixed"
    return base


def _token_unlock_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    if (root / "token_unlocks" / "current_token_unlock_actionability.csv").exists():
        return ()
    rows = _read_rows(root / "token_unlocks" / "current_token_unlock_paper_tickets.csv")
    tickets = sorted(
        (row for row in rows if row.get("status") in {"paper_short_candidate", "crowded_short_risk"}),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:3]:
        symbol = ticket.get("symbol", "")
        status = ticket.get("status", "")
        side = ticket.get("side", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_unlock_event",
                status=status,
                side=side,
                priority_score=_priority_score(status, source_count=1, raw_score=_float(ticket.get("score"))),
                sources="token_unlocks",
                evidence=(
                    f"{symbol}: unlock_value={ticket.get('unlock_value_usd', '')}, "
                    f"supply={ticket.get('percent_supply', '')}%, funding={ticket.get('annualized_funding', '')}, "
                    f"impact={ticket.get('impact_spread', '')}"
                ),
                conflict="unlock event can be crowded or already priced; negative funding can turn short into squeeze risk",
                next_step=f"label {symbol} pre/post unlock returns, funding persistence, depth decay, and stop behavior",
            )
        )
    return tuple(output)


def _token_unlock_actionability_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "token_unlocks" / "current_token_unlock_actionability.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "unlock_event_label_pending",
                "unlock_event_crowded_squeeze_watch",
                "unlock_event_execution_blocked",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
        symbol = ticket.get("symbol", "")
        status = ticket.get("status", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_unlock_actionability",
                status=status,
                side=ticket.get("side", ""),
                priority_score=_priority_score(status, source_count=2, raw_score=_float(ticket.get("score"))),
                sources="token_unlocks + perp_market_context",
                evidence=(
                    f"{symbol}: ticket={ticket.get('ticket_status', '')} {ticket.get('ticket_score', '')}, "
                    f"days_until={ticket.get('days_until', '')}, "
                    f"unlock_value={ticket.get('unlock_value_usd', '')}, "
                    f"supply={ticket.get('percent_supply', '')}%, "
                    f"funding={ticket.get('annualized_funding', '')}, "
                    f"volume={ticket.get('day_notional_volume', '')}, "
                    f"impact={ticket.get('impact_spread', '')}, "
                    f"market_action={ticket.get('market_action', '')}"
                ),
                conflict=ticket.get(
                    "reason",
                    "unlock event has no event-window label and can be already priced or crowded",
                ),
                next_step=ticket.get(
                    "next_step",
                    f"label {symbol} unlock event window before promotion",
                ),
            )
        )
    return tuple(output)


def _liquidation_flow_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    gate_rows = sorted(
        (
            row
            for row in _read_rows(root / "liquidation_flow" / "current_okx_liquidation_paper_gate.csv")
            if row.get("gate_action") == "small_paper_probe"
        ),
        key=lambda row: _float(row.get("conservative_net_bps")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen_assets: set[str] = set()
    for ticket in gate_rows:
        asset = ticket.get("asset", "")
        if not asset or asset in seen_assets:
            continue
        seen_assets.add(asset)
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_okx_liquidation_continuation",
                status=ticket.get("gate_action", "small_paper_probe"),
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    ticket.get("gate_action", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("conservative_net_bps")),
                ),
                sources="liquidation_flow",
                evidence=(
                    f"{asset}: net15={ticket.get('conservative_net_bps', '')}bps, "
                    f"size={ticket.get('candidate_size_usd', '')}, "
                    f"depth_usage={ticket.get('visible_depth_usage', '')}, "
                    f"gross_continuation={ticket.get('gross_continuation_bps', '')}bps"
                ),
                conflict="retrospective paper outcome can overstate edge; needs fresh-event repeats and live depth/fill checks",
                next_step=f"repeat {asset} liquidation event on fresh observations with fees, spread, fill, and funding included",
            )
        )
        if len(output) >= 8:
            break
    review_rows = sorted(
        _read_rows(root / "liquidation_flow" / "current_okx_liquidation_actionability_review.csv"),
        key=lambda row: _float(row.get("actionability_score")),
        reverse=True,
    )
    for ticket in review_rows:
        asset = ticket.get("asset", "")
        if not asset or asset in seen_assets:
            continue
        note = ticket.get("note", "")
        if note == "first checks support follow-up":
            status = "liquidation_followup_watch"
        elif note == "waiting for matching forward label":
            status = "liquidation_label_needed_watch"
        else:
            continue
        seen_assets.add(asset)
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_okx_liquidation_followup",
                status=status,
                side=ticket.get("action", ""),
                priority_score=_priority_score(
                    status,
                    source_count=2 if ticket.get("near_touch_depth_5bps") else 1,
                    raw_score=_float(ticket.get("actionability_score")),
                ),
                sources="liquidation_flow",
                evidence=(
                    f"{asset}: obs={ticket.get('monitor_observations', '')}, "
                    f"monitor_score={ticket.get('monitor_mean_score', '')}, "
                    f"cont15={ticket.get('continuation_return_15m', '')}, "
                    f"spread_bps={ticket.get('spread_bps', '')}, "
                    f"near_depth_5bps={ticket.get('near_touch_depth_5bps', '')}"
                ),
                conflict=(
                    "liquidation-flow signal is not yet a gated paper trade; "
                    "fees, spread, label coverage, venue depth, and fresh-event repeats can kill the edge"
                ),
                next_step=f"repeat {asset} liquidation-flow observation and require positive label plus paper-gate net after costs",
            )
        )
        if len(output) >= 8:
            break
    return tuple(output)


def _l2_imbalance_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
        (
            row
            for row in _read_rows(root / "market_making" / "current_l2_imbalance_paper_gate.csv")
            if row.get("gate_action") == "small_paper_probe"
        ),
        key=lambda row: _float(row.get("net_15m_bps")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    seen_assets: set[str] = set()
    for ticket in rows:
        asset = ticket.get("asset", "")
        if not asset or asset in seen_assets:
            continue
        seen_assets.add(asset)
        status = _l2_imbalance_status(ticket)
        output.append(
            AlphaStackRow(
                opportunity=f"{asset.lower()}_l2_imbalance_probe",
                status=status,
                side="directional_l2_probe",
                priority_score=_priority_score(
                    status,
                    source_count=1,
                    raw_score=_float(ticket.get("net_15m_bps")),
                ),
                sources="market_making",
                evidence=(
                    f"{asset}: size={ticket.get('candidate_size_usd', '')}, "
                    f"net15={ticket.get('net_15m_bps', '')}bps, "
                    f"net1h={ticket.get('net_1h_bps', '')}bps, "
                    f"imbalance_10bps={ticket.get('imbalance_10_bps', '')}, "
                    f"depth_usage={ticket.get('visible_depth_usage', '')}"
                ),
                conflict=(
                    "directional L2 probe is not maker edge; queue position, fill probability, "
                    "rebates, and adverse selection are still missing"
                ),
                next_step=f"repeat {asset} L2 imbalance on fresh snapshots and log trade prints plus fill-side next return",
            )
        )
        if len(output) >= 8:
            break
    return tuple(output)


def _l2_imbalance_status(ticket: dict[str, str]) -> str:
    if _float(ticket.get("net_1h_bps")) > 0.0:
        return "l2_imbalance_15m_1h_supported_probe"
    return "l2_imbalance_15m_only_probe"


def _best_by_score(
    path: Path,
    *,
    score_key: str,
    status_values: set[str],
    status_key: str = "status",
) -> dict[str, str] | None:
    rows = tuple(row for row in _read_rows(path) if row.get(status_key) in status_values)
    if not rows:
        return None
    return max(rows, key=lambda row: _float(row.get(score_key)))


def _row_by_name(path: Path, name: str) -> dict[str, str] | None:
    return _first_matching(path, lambda row: row.get("name") == name)


def _row_by_market_id(path: Path, market_id: str) -> dict[str, str] | None:
    if not market_id:
        return None
    return _first_matching(path, lambda row: row.get("market_id") == market_id)


def _first_matching(path: Path, predicate: object) -> dict[str, str] | None:
    for row in _read_rows(path):
        if predicate(row):
            return row
    return None


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _covered_peg_anomaly_symbols(root: Path) -> set[str]:
    return {
        _normalize_symbol(row.get("symbol", ""))
        for row in _read_rows(root / "anomaly_stress" / "current_peg_anomaly_tradeability.csv")
        if row.get("status") == "peg_anomaly_mechanics_watch"
    }


def _normalize_symbol(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _intish(value: str | None) -> int:
    try:
        return int(float(value)) if value else 0
    except ValueError:
        return 0


def _abs_float(value: str | None) -> float:
    return abs(_float(value))


def _priority_score(status: str, *, source_count: int, raw_score: float) -> float:
    status_score = {
        "paper_event_model_candidate": 75.0,
        "paper_probability_gap_candidate": 78.0,
        "probability_gap_watch": 62.0,
        "paper_event_probability_ticket": 82.0,
        "event_probability_watch": 65.0,
        "paper_outcome_survived_refresh": 88.0,
        "paper_outcome_weak_refresh": 70.0,
        "paper_outcome_active_watch": 84.0,
        "paper_outcome_edge_watch": 68.0,
        "event_probability_candidate_after_refresh_check": 66.0,
        "event_probability_candidate_after_current_quote_check": 63.0,
        "event_probability_restart_after_failed_refresh": 42.0,
        "event_probability_edge_watch": 52.0,
        "event_probability_quote_mechanics_watch": 44.0,
        "event_probability_source_quality_blocked": 38.0,
        "event_probability_quote_blocked": 30.0,
        "event_probability_deprioritize": 28.0,
        "cross_market_peg_stress_anomaly": 86.0,
        "cross_market_lending_stress_anomaly": 82.0,
        "cross_market_yield_peg_anomaly": 74.0,
        "cross_market_volatility_mispricing_watch": 76.0,
        "cross_market_event_probability_anomaly": 78.0,
        "cross_market_execution_spread_anomaly": 72.0,
        "peg_anomaly_tradeability_candidate": 72.0,
        "peg_anomaly_mechanics_watch": 40.0,
        "peg_anomaly_stale_or_unrouted": 42.0,
        "peg_anomaly_deprioritize": 35.0,
        "paper_short_candidate": 72.0,
        "paper_long_candidate": 72.0,
        "unlock_event_label_pending": 36.0,
        "unlock_event_crowded_squeeze_watch": 32.0,
        "unlock_event_execution_blocked": 24.0,
        "unlock_event_not_tradeable": 12.0,
        "paper_short_put_spread_candidate": 68.0,
        "paper_long_vol_candidate": 66.0,
        "paper_long_vol_quote_candidate": 70.0,
        "paper_calendar_spread_watch": 58.0,
        "volatility_candidate_needs_sweep_hedge": 52.0,
        "volatility_quote_mechanics_watch": 56.0,
        "volatility_short_expiry_hedge_watch": 52.0,
        "volatility_structure_mechanics_watch": 50.0,
        "volatility_premium_or_depth_blocked": 38.0,
        "volatility_quote_blocked": 34.0,
        "paper_relative_value_watch": 64.0,
        "small_paper_probe": 60.0,
        "l2_imbalance_15m_1h_supported_probe": 68.0,
        "l2_imbalance_15m_only_probe": 56.0,
        "paper_funding_dislocation_watch": 63.0,
        "paper_outcome_supported_carry_reversion_probe": 74.0,
        "paper_short_horizon_supported_carry_reversion_probe": 72.0,
        "paper_executable_carry_reversion_probe": 70.0,
        "paper_delayed_carry_reversion_probe": 60.0,
        "paper_outcome_failed_carry_reversion_probe": 43.0,
        "paper_validated_carry_reversion_candidate": 66.0,
        "wide_spread_watch": 55.0,
        "too_large_for_visible_depth": 52.0,
        "no_edge_after_rough_cost": 45.0,
        "paper_delayed_carry_reversion_watch": 58.0,
        "paper_carry_reversion_needs_more_labels": 48.0,
        "paper_crowding_reversion_watch": 59.0,
        "paper_extreme_funding_carry_candidate": 61.0,
        "paper_crowded_momentum_continuation_candidate": 60.0,
        "paper_crowded_momentum_reversal_candidate": 57.0,
        "paper_mark_oracle_reversion_candidate": 56.0,
        "paper_dislocation_executable_probe": 67.0,
        "paper_dislocation_1h_supported_candidate": 69.0,
        "paper_dislocation_15m_supported_candidate": 64.0,
        "paper_dislocation_15m_failed_candidate": 42.0,
        "dislocation_repeat_execution_candidate": 61.0,
        "dislocation_repeat_needs_execution_check": 40.0,
        "dislocation_single_snapshot_1h_watch": 45.0,
        "dislocation_15m_only_watch": 38.0,
        "dislocation_monitor_conflict_relabel": 34.0,
        "dislocation_failed_1h_confirmation": 32.0,
        "dislocation_history_deprioritize": 28.0,
        "dislocation_deprioritize": 24.0,
        "paper_attention_funding_watch": 57.0,
        "attention_price_lag_candidate": 61.0,
        "attention_breakout_continuation_watch": 58.0,
        "attention_capitulation_reversal_watch": 56.0,
        "attention_chase_risk": 50.0,
        "volume_dislocation_execution_probe": 72.0,
        "volume_dislocation_4h_supported_pending_12h": 66.0,
        "volume_dislocation_delayed_4h_support": 52.0,
        "volume_dislocation_thin_volume_watch": 42.0,
        "volume_dislocation_wide_spread_watch": 40.0,
        "volume_dislocation_too_large_for_visible_depth": 38.0,
        "volume_dislocation_1h_only_watch": 44.0,
        "volume_dislocation_no_edge_after_rough_cost": 28.0,
        "volume_dislocation_4h_contradicted_after_cost_check": 24.0,
        "volume_dislocation_4h_contradicted_pending_12h": 26.0,
        "volume_dislocation_no_hyperliquid_venue": 20.0,
        "volume_dislocation_missing_l2_context": 18.0,
        "volume_reversal_candidate": 60.0,
        "capitulation_reversal_watch": 55.0,
        "breakout_continuation_watch": 58.0,
        "chase_risk": 48.0,
        "paper_news_event_reaction_watch": 58.0,
        "paper_news_security_risk_watch": 56.0,
        "paper_news_regulatory_risk_watch": 55.0,
        "paper_news_macro_crypto_watch": 54.0,
        "paper_protocol_activity_watch": 47.0,
        "paper_chain_flow_watch": 55.0,
        "paper_depeg_repeg_watch": 62.0,
        "paper_premium_mean_reversion_watch": 62.0,
        "peg_supply_stress_watch": 50.0,
        "paper_chain_stablecoin_inflow_watch": 58.0,
        "paper_chain_stablecoin_outflow_watch": 56.0,
        "chain_stablecoin_flow_reversal_watch": 50.0,
        "chain_stablecoin_label_supported_watch": 64.0,
        "chain_stablecoin_4h_supported_pending_12h": 54.0,
        "chain_stablecoin_mixed_label_watch": 42.0,
        "chain_stablecoin_4h_contradicted_pending_12h": 24.0,
        "chain_stablecoin_label_contradicted": 18.0,
        "paper_short_basis_watch": 62.0,
        "paper_long_basis_watch": 62.0,
        "basis_term_structure_watch": 52.0,
        "paper_oi_funding_crowding_watch": 61.0,
        "paper_oi_unwind_watch": 59.0,
        "paper_basis_funding_dislocation_watch": 59.0,
        "paper_derivatives_momentum_risk_watch": 56.0,
        "historical_derivatives_feature_prior": 44.0,
        "persistent_derivatives_symbol_feature_prior": 48.0,
        "recent_derivatives_symbol_feature_prior": 44.0,
        "derivatives_symbol_feature_regime_shift": 38.0,
        "live_taker_intraday_probe": 66.0,
        "live_low_fee_intraday_probe": 62.0,
        "intraday_live_feature_source_blocked": 32.0,
        "intraday_live_execution_blocked": 28.0,
        "low_cost_intraday_paper_supported": 62.0,
        "intraday_paper_supported": 60.0,
        "low_cost_intraday_paper_recent_only": 48.0,
        "intraday_paper_positive_mean_watch": 36.0,
        "intraday_paper_watch": 34.0,
        "repeat_intraday_derivatives_feature_priority": 58.0,
        "repeat_intraday_derivatives_feature_watch": 50.0,
        "intraday_derivatives_feature_priority": 54.0,
        "intraday_derivatives_feature_watch": 46.0,
        "paper_yield_depeg_conflict_watch": 42.0,
        "paper_yield_premium_conflict_watch": 40.0,
        "yield_supply_stress_watch": 55.0,
        "paper_yield_without_peg_stress_watch": 54.0,
        "paper_base_yield_watch": 60.0,
        "paper_incentive_yield_watch": 52.0,
        "lending_rate_candidate_after_risk_check": 62.0,
        "lending_stress_mechanics_watch": 50.0,
        "lending_stress_no_liquidity_risk": 38.0,
        "lending_stress_deprioritize": 32.0,
        "paper_borrow_liquidity_stress_watch": 60.0,
        "paper_stable_lending_yield_watch": 58.0,
        "borrow_demand_context_watch": 52.0,
        "paper_dex_pool_momentum_watch": 58.0,
        "paper_dex_reversal_risk_watch": 56.0,
        "dex_liquidity_stress_watch": 50.0,
        "paper_watch": 52.0,
        "paper_long_context": 50.0,
        "paper_value_growth_candidate": 67.0,
        "paper_value_watch": 54.0,
        "fee_growth_price_lag_candidate": 70.0,
        "fee_growth_price_confirmation": 64.0,
        "fee_growth_price_chase_risk": 52.0,
        "fee_decay_price_weakness_context": 55.0,
        "protocol_fee_repeat_execution_candidate": 66.0,
        "protocol_fee_label_supported_watch": 56.0,
        "protocol_fee_pending_forward_label": 34.0,
        "protocol_fee_unlabeled_watch": 28.0,
        "protocol_fee_label_failed": 18.0,
        "liquidation_followup_watch": 47.0,
        "liquidation_label_needed_watch": 40.0,
        "funding_crowded_watch": 46.0,
        "crowded_short_risk": 48.0,
        "paper_risk_context": 45.0,
        "watch": 35.0,
    }.get(status, 30.0)
    return status_score + min(source_count * 7.5, 25.0) + min(abs(raw_score) / 10.0, 15.0)


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_alpha_stack.md")
    args = parser.parse_args()

    rows = build_alpha_stack()
    write_alpha_stack_csv(rows, output_path=args.output_path)
    write_alpha_stack_md(rows, output_path=args.markdown_output_path)
    for row in rows[:10]:
        print(row.status, row.side, f"priority={row.priority_score:.4f}", row.opportunity)


if __name__ == "__main__":
    main()
