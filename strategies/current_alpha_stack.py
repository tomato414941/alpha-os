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
        *_options_volatility_stacks(root),
        _prediction_market_event_model_stack(root),
        *_futures_basis_stacks(root),
        *_derivatives_positioning_stacks(root),
        *_cross_exchange_funding_stacks(root),
        *_perp_crowding_stacks(root),
        *_hyperliquid_dislocation_stacks(root),
        *_hyperliquid_oi_shift_stacks(root),
        *_protocol_fundamental_stacks(root),
        *_protocol_fee_valuation_stacks(root),
        *_protocol_fee_price_context_stacks(root),
        *_yield_peg_risk_stacks(root),
        *_defi_yield_stacks(root),
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
        *_token_unlock_stacks(root),
        *_liquidation_flow_stacks(root),
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


def _options_volatility_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "options_volatility" / "current_options_volatility_paper_tickets.csv")
    tickets = sorted(
        (
            row
            for row in rows
            if row.get("status")
            in {
                "paper_short_put_spread_candidate",
                "paper_long_vol_candidate",
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
                    f"volume_usd={ticket.get('volume_usd', '')}"
                ),
                conflict=_options_volatility_conflict(ticket),
                next_step=_options_volatility_next_step(ticket),
            )
        )
    return tuple(output)


def _options_volatility_conflict(ticket: dict[str, str]) -> str:
    if ticket.get("status") == "paper_short_put_spread_candidate":
        return "macro/speculative-beta risk-off pressure can turn rich put premium into real tail loss"
    if ticket.get("status") == "paper_long_vol_candidate":
        return "cheap IV can stay cheap or realized volatility can collapse; needs actual option quotes, premium-at-risk, and delta-hedge plan"
    return "calendar spread depends on expiry curve, event timing, bid/ask, margin, and hedge PnL rather than direction alone"


def _options_volatility_next_step(ticket: dict[str, str]) -> str:
    if ticket.get("status") == "paper_short_put_spread_candidate":
        return "paper-check bid/ask spread, margin, max loss, delta hedge cost, and behavior during the current risk-off shock"
    if ticket.get("status") == "paper_long_vol_candidate":
        return "paper-check long-vol spread quotes, max premium loss, delta hedge plan, and realized-vol persistence"
    return "paper-check calendar spread quotes, event timing, vega/theta exposure, margin, and delta hedge cost"


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
        return "paper_outcome_supported_carry_reversion_probe"
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
    for ticket in tickets[:6]:
        chain = ticket.get("chain", "")
        project = ticket.get("project", "")
        symbol = ticket.get("symbol", "")
        peg_symbol = ticket.get("peg_symbol", "")
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
    tickets = sorted(
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
    output: list[AlphaStackRow] = []
    for ticket in tickets[:8]:
        symbol = ticket.get("symbol", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_volume_price_dislocation",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="market_breadth + market_price_context",
                evidence=(
                    f"{symbol}: "
                    f"name={ticket.get('name', '')}, "
                    f"rank={ticket.get('market_cap_rank', '')}, "
                    f"vol_mcap={ticket.get('volume_to_market_cap', '')}, "
                    f"price24h={ticket.get('price_change_24h', '')}, "
                    f"price7d={ticket.get('price_change_7d', '')}, "
                    f"price30d={ticket.get('price_change_30d', '')}"
                ),
                conflict=(
                    "volume/price dislocation can be a liquidation bounce, news reaction, or crowded trap; "
                    "needs forward labels, venue depth, and execution cost checks"
                ),
                next_step=ticket.get(
                    "next_step",
                    f"paper-label {symbol} market-breadth dislocation",
                ),
            )
        )
    return tuple(output)


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


def _chain_stablecoin_migration_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "stablecoin_liquidity" / "current_chain_stablecoin_migration.csv")
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
        output.append(
            AlphaStackRow(
                opportunity=f"{chain.lower().replace(' ', '_')}_stablecoin_migration",
                status=ticket.get("status", ""),
                side=ticket.get("side", ""),
                priority_score=_priority_score(
                    ticket.get("status", ""),
                    source_count=1,
                    raw_score=_float(ticket.get("score")),
                ),
                sources="stablecoin_liquidity",
                evidence=(
                    f"{chain}/{display_token}: supply={ticket.get('current_supply_usd', '')}, "
                    f"day_change={ticket.get('day_change_usd', '')}, "
                    f"week_change={ticket.get('week_change_usd', '')}, "
                    f"week_pct={ticket.get('week_change_pct', '')}, "
                    f"top_asset={ticket.get('top_asset', '')}"
                ),
                conflict="stablecoin migration is a capital-flow proxy, not a bridge-fill; chain-token mapping, venues, and forward labels are still required",
                next_step=ticket.get(
                    "next_step",
                    f"label {token} returns after {chain} stablecoin migration",
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


def _liquidation_flow_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = sorted(
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
    for ticket in rows:
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
    return tuple(output)


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
        "paper_long_vol_candidate": 66.0,
        "paper_calendar_spread_watch": 58.0,
        "paper_relative_value_watch": 64.0,
        "small_paper_probe": 60.0,
        "paper_funding_dislocation_watch": 63.0,
        "paper_outcome_supported_carry_reversion_probe": 74.0,
        "paper_short_horizon_supported_carry_reversion_probe": 72.0,
        "paper_executable_carry_reversion_probe": 70.0,
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
        "paper_attention_funding_watch": 57.0,
        "attention_price_lag_candidate": 61.0,
        "attention_breakout_continuation_watch": 58.0,
        "attention_capitulation_reversal_watch": 56.0,
        "attention_chase_risk": 50.0,
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
        "paper_short_basis_watch": 62.0,
        "paper_long_basis_watch": 62.0,
        "basis_term_structure_watch": 52.0,
        "paper_oi_funding_crowding_watch": 61.0,
        "paper_oi_unwind_watch": 59.0,
        "paper_basis_funding_dislocation_watch": 59.0,
        "paper_derivatives_momentum_risk_watch": 56.0,
        "paper_yield_depeg_conflict_watch": 60.0,
        "paper_yield_premium_conflict_watch": 58.0,
        "yield_supply_stress_watch": 55.0,
        "paper_yield_without_peg_stress_watch": 54.0,
        "paper_base_yield_watch": 60.0,
        "paper_incentive_yield_watch": 52.0,
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
