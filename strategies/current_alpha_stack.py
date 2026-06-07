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
        _btc_options_volatility_stack(root),
        _prediction_market_event_model_stack(root),
        *_futures_basis_stacks(root),
        *_derivatives_positioning_stacks(root),
        *_cross_exchange_funding_stacks(root),
        *_perp_crowding_stacks(root),
        *_protocol_fundamental_stacks(root),
        *_protocol_fee_valuation_stacks(root),
        *_defi_yield_stacks(root),
        *_dex_pool_flow_stacks(root),
        *_attention_funding_stacks(root),
        *_protocol_activity_stacks(root),
        *_on_chain_flow_stacks(root),
        *_stablecoin_peg_stress_stacks(root),
        *_token_unlock_stacks(root),
        _liquidation_flow_stack(root),
        _l2_imbalance_stack(root),
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


def _btc_options_volatility_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "options_volatility" / "current_options_volatility_paper_tickets.csv",
        score_key="score",
        status_values={"paper_short_put_spread_candidate"},
    )
    if not ticket:
        return None
    return AlphaStackRow(
        opportunity="btc_options_short_put_spread",
        status=ticket.get("status", "paper_short_put_spread_candidate"),
        side=f"{ticket.get('currency', 'BTC')}_{ticket.get('structure', 'short_put_spread')}",
        priority_score=_priority_score(ticket.get("status", ""), source_count=1, raw_score=_float(ticket.get("score"))),
        sources="options_volatility",
        evidence=(
            f"{ticket.get('currency', '')} {ticket.get('expiry', '')}: "
            f"iv_premium_24h={ticket.get('iv_premium_24h', '')}, "
            f"skew={ticket.get('skew_iv', '')}, volume_usd={ticket.get('volume_usd', '')}"
        ),
        conflict="macro/speculative-beta risk-off pressure can turn rich put premium into real tail loss",
        next_step="paper-check bid/ask spread, margin, max loss, delta hedge cost, and behavior during the current VIX shock",
    )


def _prediction_market_event_model_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "prediction_markets" / "current_prediction_market_paper_tickets.csv",
        score_key="score",
        status_values={"paper_event_model_candidate", "paper_event_model_watch"},
    )
    if not ticket:
        return None
    return AlphaStackRow(
        opportunity="prediction_market_event_model",
        status=ticket.get("status", "paper_event_model_candidate"),
        side=f"{ticket.get('question', '')} {ticket.get('outcome', '')}",
        priority_score=_priority_score(ticket.get("status", ""), source_count=1, raw_score=_float(ticket.get("score"))),
        sources="prediction_markets",
        evidence=(
            f"{ticket.get('category', '')}: spread={ticket.get('spread', '')}, "
            f"depth={ticket.get('visible_depth_score', '')}, vol24={ticket.get('volume_24h', '')}"
        ),
        conflict="market depth is not edge; needs independent true-probability model and latency/adverse-selection checks",
        next_step="build external news/filing probability model before any paper event-market action",
    )


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
                "basis_term_structure_watch",
            }
        ),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:6]:
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
                opportunity=f"{market.lower().replace(' ', '_')}_{symbol.lower()}_positioning",
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


def _protocol_fee_valuation_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
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


def _defi_yield_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "defi_yield" / "current_yield_quality_screen.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status") in {"paper_base_yield_watch", "paper_incentive_yield_watch"}
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


def _token_unlock_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
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


def _liquidation_flow_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "liquidation_flow" / "current_okx_liquidation_paper_gate.csv",
        score_key="conservative_net_bps",
        status_values={"small_paper_probe"},
        status_key="gate_action",
    )
    if not ticket:
        return None
    asset = ticket.get("asset", "")
    return AlphaStackRow(
        opportunity=f"{asset.lower()}_liquidation_continuation",
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
            f"size={ticket.get('candidate_size_usd', '')}, depth_usage={ticket.get('visible_depth_usage', '')}"
        ),
        conflict="retrospective paper outcome can overstate edge; needs fresh-event repeats and live depth/fill checks",
        next_step=f"repeat {asset} liquidation event on fresh observations with fees, spread, fill, and funding included",
    )


def _l2_imbalance_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "market_making" / "current_l2_imbalance_paper_gate.csv",
        score_key="net_15m_bps",
        status_values={"small_paper_probe"},
        status_key="gate_action",
    )
    if not ticket:
        return None
    asset = ticket.get("asset", "")
    return AlphaStackRow(
        opportunity=f"{asset.lower()}_l2_imbalance_probe",
        status=ticket.get("gate_action", "small_paper_probe"),
        side="directional_l2_probe",
        priority_score=_priority_score(
            ticket.get("gate_action", ""),
            source_count=1,
            raw_score=_float(ticket.get("net_15m_bps")),
        ),
        sources="market_making",
        evidence=(
            f"{asset}: net15={ticket.get('net_15m_bps', '')}bps, "
            f"imbalance_10bps={ticket.get('imbalance_10_bps', '')}, "
            f"depth_usage={ticket.get('visible_depth_usage', '')}"
        ),
        conflict="directional L2 probe is not maker edge; queue position, fill probability, and adverse selection are still missing",
        next_step=f"collect repeated {asset} L2 snapshots with trade prints and estimate fill-side next return",
    )


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


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _abs_float(value: str | None) -> float:
    return abs(_float(value))


def _priority_score(status: str, *, source_count: int, raw_score: float) -> float:
    status_score = {
        "paper_event_model_candidate": 75.0,
        "paper_short_candidate": 72.0,
        "paper_long_candidate": 72.0,
        "paper_short_put_spread_candidate": 68.0,
        "paper_relative_value_watch": 64.0,
        "small_paper_probe": 60.0,
        "paper_funding_dislocation_watch": 63.0,
        "paper_crowding_reversion_watch": 59.0,
        "paper_attention_funding_watch": 57.0,
        "paper_protocol_activity_watch": 47.0,
        "paper_chain_flow_watch": 55.0,
        "paper_depeg_repeg_watch": 62.0,
        "paper_premium_mean_reversion_watch": 62.0,
        "peg_supply_stress_watch": 50.0,
        "paper_short_basis_watch": 62.0,
        "paper_long_basis_watch": 62.0,
        "basis_term_structure_watch": 52.0,
        "paper_oi_funding_crowding_watch": 61.0,
        "paper_basis_funding_dislocation_watch": 59.0,
        "paper_derivatives_momentum_risk_watch": 56.0,
        "paper_base_yield_watch": 60.0,
        "paper_incentive_yield_watch": 52.0,
        "paper_dex_pool_momentum_watch": 58.0,
        "paper_dex_reversal_risk_watch": 56.0,
        "dex_liquidity_stress_watch": 50.0,
        "paper_watch": 52.0,
        "paper_long_context": 50.0,
        "paper_value_growth_candidate": 67.0,
        "paper_value_watch": 54.0,
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
