from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BroadAlphaExecutionQueueItem:
    queue_id: str
    lane: str
    subject: str
    side: str
    action: str
    status: str
    score: str
    size_or_risk: str
    evidence: str
    checkpoints: str
    source_path: str
    required_record: str
    next_step: str


def build_broad_alpha_execution_queue(*, root: Path = ROOT) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    rows: list[BroadAlphaExecutionQueueItem] = []
    rows.extend(_prediction_market_queue(root))
    rows.extend(_liquidation_intensity_queue(root))
    rows.extend(_l2_imbalance_queue(root))
    rows.extend(_options_volatility_queue(root))
    rows.extend(_stablecoin_flow_queue(root))
    rows.extend(_protocol_fee_queue(root))
    rows.extend(_token_unlock_queue(root))
    rows.extend(_defi_lending_queue(root))
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_broad_alpha_execution_queue_csv(
    rows: tuple[BroadAlphaExecutionQueueItem, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(BroadAlphaExecutionQueueItem.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_broad_alpha_execution_queue_md(
    rows: tuple[BroadAlphaExecutionQueueItem, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Broad Alpha Execution Queue\n\n")
        handle.write(
            "This is a cross-lane work queue for current alpha probes. It only reads existing "
            "candidate, gate, and ticket files; it is not a strategy abstraction or a live order list.\n\n"
        )
        handle.write(
            "| queue | lane | subject | side | action | status | score | size/risk | evidence | checkpoints | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.queue_id} | {row.lane} | {_escape(row.subject)} | {row.side} | "
                f"{row.action} | {row.status} | {row.score} | {_escape(row.size_or_risk)} | "
                f"{_escape(row.evidence)} | {row.checkpoints} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The queue deliberately mixes execution probes, label-only probes, and risk checks. "
            "That keeps alpha discovery broad while preventing unresolved labels from being promoted "
            "as realized reward.\n"
        )
    return output_path


def _prediction_market_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "prediction_markets/current_event_probability_execution_queue.csv"
    rows = []
    for row in _read_rows(path):
        rows.append(
            BroadAlphaExecutionQueueItem(
                queue_id=row.get("queue_id", ""),
                lane="prediction_market_probability",
                subject=row.get("question", ""),
                side=row.get("outcome_to_buy", ""),
                action=row.get("queue_action", ""),
                status="quote_survival_probe",
                score=f"{_float(row.get('current_edge_after_ask')) * 100.0:.4f}",
                size_or_risk=f"ask={row.get('current_ask', '')}; depth5c={row.get('ask_depth_to_5c', '')}",
                evidence=f"edge_after_ask={row.get('current_edge_after_ask', '')}",
                checkpoints=row.get("checkpoints", ""),
                source_path=_relative(path),
                required_record=row.get("required_record", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _liquidation_intensity_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "liquidation_flow/current_okx_liquidation_intensity_paper_gate.csv"
    selected = _best_by_subject(
        (
            row
            for row in _read_rows(path)
            if row.get("gate_action") in {"small_paper_probe", "small_paper_probe_pending_1h"}
        ),
        subject_key="asset",
        score_key="conservative_net_bps",
        size_key="candidate_size_usd",
    )
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"liq-intensity-{_slug(row.get('asset', ''))}-{_slug(row.get('trade_direction', ''))}",
            lane="liquidation_intensity",
            subject=row.get("asset", ""),
            side=row.get("trade_direction", ""),
            action=row.get("gate_action", ""),
            status=row.get("label_status", ""),
            score=_fmt(row.get("conservative_net_bps")),
            size_or_risk=f"size_usd={row.get('candidate_size_usd', '')}; depth10={row.get('depth_10bps_notional', '')}",
            evidence=f"label15m={row.get('label_bps_15m', '')}; cost={row.get('conservative_cost_bps', '')}",
            checkpoints="1h,repeat_event",
            source_path=_relative(path),
            required_record="fill, funding, stop behavior, 1h label, and repeat-event evidence",
            next_step=row.get("next_step", ""),
        )
        for row in selected
    )


def _l2_imbalance_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "market_making/current_l2_imbalance_paper_gate.csv"
    selected = _best_by_subject(
        (row for row in _read_rows(path) if row.get("gate_action") == "small_paper_probe"),
        subject_key="asset",
        score_key="net_15m_bps",
        size_key="candidate_size_usd",
    )
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"l2-imbalance-{_slug(row.get('asset', ''))}",
            lane="l2_imbalance_microstructure",
            subject=row.get("asset", ""),
            side=_direction_to_side(row.get("imbalance_10_bps", "")),
            action=row.get("gate_action", ""),
            status="15m_supported_1h_unproven",
            score=_fmt(row.get("net_15m_bps")),
            size_or_risk=f"size_usd={row.get('candidate_size_usd', '')}; depth10={row.get('near_depth_10bps_notional', '')}",
            evidence=f"net15m={row.get('net_15m_bps', '')}; net1h={row.get('net_1h_bps', '')}",
            checkpoints="1h,repeat_snapshot",
            source_path=_relative(path),
            required_record="fresh book snapshot, 1h label, fill/cost check, and repeat-snapshot behavior",
            next_step=f"open or refresh a minimal {row.get('asset', '')} L2 imbalance paper probe only after fresh-state confirmation",
        )
        for row in selected
    )


def _options_volatility_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "options_volatility/current_volatility_actionability.csv"
    selected = tuple(
        row
        for row in _read_rows(path)
        if row.get("side") in {"paper_long_vol_sweep_hedge_check", "paper_quote_check", "paper_hedge_check"}
    )[:8]
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"long-vol-{_slug(row.get('currency', ''))}-{_slug(row.get('expiry', ''))}",
            lane="options_volatility",
            subject=f"{row.get('currency', '')} {row.get('expiry', '')} {row.get('structure', '')}",
            side=row.get("side", ""),
            action=row.get("side", ""),
            status=row.get("status", ""),
            score=_fmt(row.get("score")),
            size_or_risk=f"max_loss_pct={row.get('max_loss_pct', '')}; depth_usd={row.get('top_ask_premium_depth_usd', '')}",
            evidence=f"atm_iv={row.get('atm_iv', '')}; realized24h={row.get('realized_vol_24h', '')}; iv_premium={row.get('iv_premium_24h', '')}",
            checkpoints="quote_sweep,hedge_path,exit_bid",
            source_path=_relative(path),
            required_record="multi-level sweep quote, delta-hedge path, max premium loss, margin, and exit bid",
            next_step=row.get("next_step", ""),
        )
        for row in selected
    )


def _stablecoin_flow_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "stablecoin_liquidity/current_stablecoin_flow_proxy_tickets.csv"
    outcome_path = root / "stablecoin_liquidity/current_stablecoin_flow_proxy_outcomes.csv"
    risk_path = root / "stablecoin_liquidity/current_stablecoin_flow_proxy_fill_risk_check.csv"
    outcomes = {row.get("ticket_id", ""): row for row in _read_rows(outcome_path)}
    risks = {row.get("ticket_id", ""): row for row in _read_rows(risk_path)}
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=row.get("ticket_id", ""),
            lane="stablecoin_chain_liquidity_proxy",
            subject=row.get("asset", "") or row.get("opportunity", ""),
            side=row.get("side", ""),
            action=_stablecoin_action(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            status=_stablecoin_status(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            score=_stablecoin_score(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            size_or_risk=_stablecoin_size_or_risk(
                row=row,
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            evidence=_stablecoin_evidence(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            checkpoints=row.get("checkpoints", ""),
            source_path=_relative(risk_path if risks.get(row.get("ticket_id", "")) else path),
            required_record=_stablecoin_required_record(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
            next_step=_stablecoin_next_step(
                row=row,
                outcome=outcomes.get(row.get("ticket_id", ""), {}),
                risk=risks.get(row.get("ticket_id", ""), {}),
            ),
        )
        for row in _read_rows(path)
    )


def _protocol_fee_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "protocol_fundamentals/current_protocol_fee_actionability.csv"
    risk_path = root / "protocol_fundamentals/current_protocol_fee_repeat_risk_check.csv"
    ticket_path = root / "protocol_fundamentals/current_protocol_fee_repeat_tickets.csv"
    outcome_path = root / "protocol_fundamentals/current_protocol_fee_repeat_outcomes.csv"
    risks = {
        (row.get("token_symbol", ""), row.get("protocol", "")): row
        for row in _read_rows(risk_path)
    }
    tickets = {
        _protocol_fee_ticket_key(row): row
        for row in _read_rows(ticket_path)
    }
    outcomes = {
        _protocol_fee_outcome_key(row): row
        for row in _read_rows(outcome_path)
    }
    selected = tuple(
        row
        for row in _read_rows(path)
        if row.get("action") in {"repeat_paper_probe", "refresh_execution_gate", "wait_for_forward_label"}
        and row.get("execution_action") in {"paper_observation_ready", "thin_volume_watch"}
    )[:6]
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"protocol-fee-{_slug(row.get('token_symbol', ''))}-{_slug(row.get('protocol', ''))}",
            lane="protocol_fee_growth_lag",
            subject=f"{row.get('token_symbol', '')}/{row.get('protocol', '')}",
            side=row.get("side", ""),
            action=_protocol_fee_action(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                outcome=outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
            status=_protocol_fee_status(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                outcome=outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
            score=_protocol_fee_score(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                outcome=outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
            size_or_risk=_protocol_fee_size_or_risk(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                ticket=tickets.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
            evidence=_protocol_fee_evidence(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                outcome=outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
            checkpoints="4h,12h,24h",
            source_path=_relative(
                outcome_path
                if outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")))
                else risk_path
                if risks.get((row.get("token_symbol", ""), row.get("protocol", "")))
                else path
            ),
            required_record="fresh forward labels, funding, spread, depth, and stale-ticket rejection",
            next_step=_protocol_fee_next_step(
                row=row,
                risk=risks.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
                outcome=outcomes.get((row.get("token_symbol", ""), row.get("protocol", "")), {}),
            ),
        )
        for row in selected
    )


def _token_unlock_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "token_unlocks/current_token_unlock_actionability.csv"
    ticket_path = root / "token_unlocks/current_token_unlock_event_window_tickets.csv"
    outcome_path = root / "token_unlocks/current_token_unlock_event_window_outcomes.csv"
    risk_path = root / "token_unlocks/current_token_unlock_event_window_risk_check.csv"
    tickets = {row.get("asset", ""): row for row in _read_rows(ticket_path)}
    outcomes = {row.get("asset", ""): row for row in _read_rows(outcome_path)}
    risks = {row.get("asset", ""): row for row in _read_rows(risk_path)}
    selected = tuple(
        row
        for row in _read_rows(path)
        if row.get("action") in {"create_event_window_label", "label_before_short"}
    )[:6]
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"token-unlock-{_slug(row.get('symbol', ''))}",
            lane="token_unlock_event_window",
            subject=f"{row.get('symbol', '')}/{row.get('name', '')}",
            side=row.get("side", ""),
            action=_token_unlock_action(
                row=row,
                outcome=outcomes.get(row.get("symbol", ""), {}),
                risk=risks.get(row.get("symbol", ""), {}),
            ),
            status=_token_unlock_status(
                row=row,
                outcome=outcomes.get(row.get("symbol", ""), {}),
                risk=risks.get(row.get("symbol", ""), {}),
            ),
            score=_token_unlock_score(
                row=row,
                outcome=outcomes.get(row.get("symbol", ""), {}),
                risk=risks.get(row.get("symbol", ""), {}),
            ),
            size_or_risk=_token_unlock_size_or_risk(
                row=row,
                ticket=tickets.get(row.get("symbol", ""), {}),
            ),
            evidence=_token_unlock_evidence(
                row=row,
                outcome=outcomes.get(row.get("symbol", ""), {}),
                risk=risks.get(row.get("symbol", ""), {}),
            ),
            checkpoints="15m,1h,4h,pre_event,event_window,post_event",
            source_path=_relative(
                risk_path
                if risks.get(row.get("symbol", ""))
                else outcome_path
                if outcomes.get(row.get("symbol", ""))
                else path
            ),
            required_record="event-window label, funding, crowding, liquidity, and squeeze-vs-pressure split",
            next_step=_token_unlock_next_step(
                row=row,
                outcome=outcomes.get(row.get("symbol", ""), {}),
                risk=risks.get(row.get("symbol", ""), {}),
            ),
        )
        for row in selected
    )


def _defi_lending_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "defi_lending/current_lending_stress_actionability.csv"
    risk_path = root / "defi_lending/current_lending_yield_risk_check.csv"
    risks = {
        (row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")): row
        for row in _read_rows(risk_path)
    }
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"defi-lending-{_slug(row.get('chain', ''))}-{_slug(row.get('loan_asset', ''))}-{_slug(row.get('collateral_asset', ''))}",
            lane="defi_lending_yield",
            subject=f"{row.get('chain', '')} {row.get('loan_asset', '')}/{row.get('collateral_asset', '')}",
            side=row.get("side", ""),
            action=_defi_lending_action(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
            status=_defi_lending_status(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
            score=_defi_lending_score(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
            size_or_risk=_defi_lending_size_or_risk(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
            evidence=_defi_lending_evidence(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
            checkpoints="withdrawal_path,rate_persistence,risk_check",
            source_path=_relative(risk_path if risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", ""))) else path),
            required_record="withdrawal path, rate persistence, gas, oracle, liquidation, smart-contract risk, and position sizing",
            next_step=_defi_lending_next_step(row=row, risk=risks.get((row.get("chain", ""), row.get("loan_asset", ""), row.get("collateral_asset", "")), {})),
        )
        for row in _read_rows(path)
        if row.get("side") == "paper_lend_after_risk_check"
    )


def _best_by_subject(
    rows: object,
    *,
    subject_key: str,
    score_key: str,
    size_key: str,
) -> tuple[dict[str, str], ...]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        subject = row.get(subject_key, "")
        current = best.get(subject)
        if current is None:
            best[subject] = row
            continue
        row_score = _float(row.get(score_key))
        current_score = _float(current.get(score_key))
        if row_score > current_score:
            best[subject] = row
            continue
        if row_score == current_score and _float(row.get(size_key)) < _float(current.get(size_key)):
            best[subject] = row
    return tuple(best.values())


def _stablecoin_action(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return risk.get("risk_action", "")
    if outcome.get("checkpoint_status") == "ready":
        return outcome.get("outcome", "")
    return row.get("decision", "")


def _stablecoin_status(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return f"{row.get('status', '')}:{outcome.get('outcome', '')}:{risk.get('risk_action', '')}"
    if outcome.get("checkpoint_status") == "ready":
        return f"{row.get('status', '')}:{outcome.get('outcome', '')}"
    return row.get("status", "")


def _stablecoin_score(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return _fmt(risk.get("estimated_net_bps"))
    if outcome.get("checkpoint_status") == "ready":
        return _fmt(outcome.get("directional_return_bps"))
    return str(max(0.0, 100.0 - _float(row.get("rank"))))


def _stablecoin_size_or_risk(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return (
            f"notional={risk.get('notional_usd', '')}; "
            f"depth10={risk.get('near_depth_10bps_notional', '')}; "
            f"usage={risk.get('visible_depth_usage', '')}; "
            f"stop50={risk.get('stop_50bps_survived', '')}"
        )
    return f"entry_mark={row.get('entry_mark', '')}; venue={row.get('venue', '')}"


def _stablecoin_evidence(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return (
            f"{row.get('opportunity', '')}; "
            f"dir_bps={risk.get('directional_return_bps', '')}; "
            f"net_bps={risk.get('estimated_net_bps', '')}; "
            f"MAE={risk.get('max_adverse_excursion_bps', '')}; "
            f"reason={risk.get('reason', '')}"
        )
    if outcome.get("checkpoint_status") == "ready":
        return (
            f"{row.get('opportunity', '')}; "
            f"mark_bps={outcome.get('directional_return_bps', '')}; "
            f"current={outcome.get('current_mark', '')}"
        )
    return row.get("opportunity", "")


def _stablecoin_required_record(
    *,
    row: dict[str, str],
    outcome: dict[str, str],
    risk: dict[str, str],
) -> str:
    if risk:
        return "fresh chain-flow evidence, updated execution context, and repeated cost-adjusted outcome"
    return outcome.get("missing_evidence") or row.get("required_record", "")


def _stablecoin_next_step(
    *,
    row: dict[str, str],
    outcome: dict[str, str],
    risk: dict[str, str],
) -> str:
    if risk:
        return risk.get("next_step", "")
    return outcome.get("next_step") or row.get("next_step", "")


def _defi_lending_action(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return risk.get("risk_action", "")
    return row.get("side", "")


def _defi_lending_status(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return f"{row.get('status', '')}:{risk.get('risk_action', '')}"
    return row.get("status", "")


def _defi_lending_score(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return _fmt(risk.get("risk_score"))
    return _fmt(row.get("score"))


def _defi_lending_size_or_risk(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return (
            f"notional={risk.get('paper_notional_usd', '')}; "
            f"liquidity={risk.get('liquidity_usd', '')}; "
            f"usage={risk.get('capacity_usage', '')}; "
            f"util={risk.get('utilization', '')}; "
            f"lltv={risk.get('lltv', '')}"
        )
    return f"liquidity_usd={row.get('liquidity_usd', '')}; utilization={row.get('utilization', '')}"


def _defi_lending_evidence(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return (
            f"apy={risk.get('supply_apy', '')}; "
            f"avg_apy={risk.get('avg_net_supply_apy', '')}; "
            f"spike={risk.get('supply_apy_spike_ratio', '')}; "
            f"collateral={risk.get('collateral_category', '')}; "
            f"reason={risk.get('reason', '')}"
        )
    return f"supply_apy={row.get('avg_net_supply_apy', '')}; borrow_apy={row.get('avg_net_borrow_apy', '')}"


def _defi_lending_next_step(*, row: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return risk.get("next_step", "")
    return row.get("next_step", "")


def _token_unlock_action(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return risk.get("risk_action", "")
    if outcome.get("checkpoint_status") == "ready":
        return outcome.get("outcome", "")
    if outcome:
        return "event_window_label_opened"
    return row.get("action", "")


def _token_unlock_status(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return f"{row.get('status', '')}:{outcome.get('outcome', '')}:{risk.get('risk_action', '')}"
    if outcome:
        return f"{row.get('status', '')}:{outcome.get('checkpoint_status', '')}:{outcome.get('outcome', '')}"
    return row.get("status", "")


def _token_unlock_score(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return _fmt(risk.get("net_directional_bps"))
    if outcome.get("checkpoint_status") == "ready":
        return _fmt(outcome.get("directional_return_bps"))
    return _fmt(row.get("score"))


def _token_unlock_size_or_risk(*, row: dict[str, str], ticket: dict[str, str]) -> str:
    if ticket:
        return (
            f"days={ticket.get('days_until', '')}; "
            f"unlock_usd={ticket.get('unlock_value_usd', '')}; "
            f"pct_supply={ticket.get('percent_supply', '')}; "
            f"entry={ticket.get('entry_mark', '')}"
        )
    return (
        f"days={row.get('days_until', '')}; "
        f"unlock_usd={row.get('unlock_value_usd', '')}; "
        f"pct_supply={row.get('percent_supply', '')}"
    )


def _token_unlock_evidence(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return (
            f"{row.get('reason', '')}; "
            f"dir_bps={risk.get('directional_return_bps', '')}; "
            f"net_bps={risk.get('net_directional_bps', '')}; "
            f"cost={risk.get('round_trip_cost_bps', '')}; "
            f"reason={risk.get('reason', '')}"
        )
    if outcome:
        return (
            f"{row.get('reason', '')}; "
            f"entry={outcome.get('entry_mark', '')}; "
            f"current={outcome.get('current_mark', '')}; "
            f"dir_bps={outcome.get('directional_return_bps', '')}; "
            f"outcome={outcome.get('outcome', '')}"
        )
    return row.get("reason", "")


def _token_unlock_next_step(*, row: dict[str, str], outcome: dict[str, str], risk: dict[str, str]) -> str:
    if risk:
        return risk.get("next_step", "")
    if outcome:
        return outcome.get("next_step", "")
    return row.get("next_step", "")


def _protocol_fee_action(*, row: dict[str, str], risk: dict[str, str], outcome: dict[str, str]) -> str:
    if outcome.get("checkpoint_status") == "ready":
        return outcome.get("outcome", "")
    if outcome:
        return "repeat_probe_opened"
    if risk:
        return risk.get("risk_action", "")
    return row.get("action", "")


def _protocol_fee_status(*, row: dict[str, str], risk: dict[str, str], outcome: dict[str, str]) -> str:
    if outcome:
        return f"{row.get('status', '')}:{risk.get('risk_action', '')}:{outcome.get('checkpoint_status', '')}:{outcome.get('outcome', '')}"
    if risk:
        return f"{row.get('status', '')}:{risk.get('risk_action', '')}"
    return row.get("status", "")


def _protocol_fee_score(*, row: dict[str, str], risk: dict[str, str], outcome: dict[str, str]) -> str:
    if outcome.get("checkpoint_status") == "ready":
        return _fmt(outcome.get("directional_return_bps"))
    if risk:
        return _fmt(risk.get("net_mean_directional_4h_bps"))
    return _fmt(row.get("score"))


def _protocol_fee_size_or_risk(*, row: dict[str, str], risk: dict[str, str], ticket: dict[str, str]) -> str:
    if ticket:
        return (
            f"notional={ticket.get('candidate_size_usd', '')}; "
            f"entry={ticket.get('entry_mark', '')}; "
            f"depth10={risk.get('depth_10bps', '')}; "
            f"usage={risk.get('visible_depth_usage', '')}; "
            f"cost={risk.get('round_trip_cost_bps', '')}"
        )
    if risk:
        return (
            f"notional={risk.get('paper_notional_usd', '')}; "
            f"depth10={risk.get('depth_10bps', '')}; "
            f"usage={risk.get('visible_depth_usage', '')}; "
            f"cost={risk.get('round_trip_cost_bps', '')}"
        )
    return f"venues={row.get('venue_count', '')}; depth10={row.get('depth_10bps', '')}; spread={row.get('spread_bps', '')}"


def _protocol_fee_evidence(*, row: dict[str, str], risk: dict[str, str], outcome: dict[str, str]) -> str:
    if outcome:
        return (
            f"fee_growth_7d={row.get('fee_growth_7d', '')}; "
            f"net4h_bps={risk.get('net_mean_directional_4h_bps', '')}; "
            f"entry={outcome.get('entry_mark', '')}; "
            f"current={outcome.get('current_mark', '')}; "
            f"dir_bps={outcome.get('directional_return_bps', '')}; "
            f"outcome={outcome.get('outcome', '')}"
        )
    if risk:
        return (
            f"fee_growth_7d={row.get('fee_growth_7d', '')}; "
            f"mean4h_bps={risk.get('mean_directional_4h_bps', '')}; "
            f"net4h_bps={risk.get('net_mean_directional_4h_bps', '')}; "
            f"labels={risk.get('wins_4h', '')}/{risk.get('labeled_4h', '')}; "
            f"reason={risk.get('reason', '')}"
        )
    return f"fee_growth_7d={row.get('fee_growth_7d', '')}; price_change_7d={row.get('price_change_7d', '')}"


def _protocol_fee_next_step(*, row: dict[str, str], risk: dict[str, str], outcome: dict[str, str]) -> str:
    if outcome:
        return outcome.get("next_step", "")
    if risk:
        return risk.get("next_step", "")
    return row.get("next_step", "")


def _protocol_fee_ticket_key(row: dict[str, str]) -> tuple[str, str]:
    opportunity = row.get("opportunity", "").removeprefix("protocol_fee_repeat:")
    token, _, protocol = opportunity.partition("/")
    return token, protocol


def _protocol_fee_outcome_key(row: dict[str, str]) -> tuple[str, str]:
    opportunity = row.get("opportunity", "").removeprefix("protocol_fee_repeat:")
    token, _, protocol = opportunity.partition("/")
    return token, protocol


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _sort_key(row: BroadAlphaExecutionQueueItem) -> tuple[float, float]:
    lane_bonus = {
        "prediction_market_probability": 35.0,
        "liquidation_intensity": 30.0,
        "l2_imbalance_microstructure": 25.0,
        "options_volatility": 20.0,
        "defi_lending_yield": 15.0,
        "stablecoin_chain_liquidity_proxy": 10.0,
        "protocol_fee_growth_lag": 8.0,
        "token_unlock_event_window": 5.0,
    }.get(row.lane, 0.0)
    action_bonus = {
        "cost_adjusted_repeat_probe": 80.0,
        "repeat_probe_opened": 75.0,
        "repeat_paper_probe": 70.0,
        "paper_check_pure_probability": 35.0,
        "cost_adjusted_event_window_probe": 30.0,
        "thin_event_window_support": 5.0,
        "event_window_label_opened": 12.0,
        "refresh_execution_gate": 10.0,
        "refresh_before_repeat": -40.0,
        "wait_for_forward_label": 0.0,
        "event_window_label_not_supported": -80.0,
        "cost_adjusted_event_window_failed": -120.0,
        "collateral_review_required": -50.0,
        "exit_liquidity_watch": -50.0,
        "depth_too_thin_for_1k_probe": -250.0,
        "stop_risk_blocks_probe": -250.0,
        "cost_adjusted_edge_failed": -300.0,
    }.get(row.action, 0.0)
    return (lane_bonus + action_bonus + _float(row.score), _float(row.score))


def _direction_to_side(value: str) -> str:
    return "long" if _float(value) >= 0.0 else "short"


def _fmt(value: str | None) -> str:
    number = _float(value)
    return f"{number:.4f}"


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part) or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_execution_queue.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_broad_alpha_execution_queue.md",
    )
    args = parser.parse_args()

    rows = build_broad_alpha_execution_queue()
    write_broad_alpha_execution_queue_csv(rows, output_path=args.output_path)
    write_broad_alpha_execution_queue_md(rows, output_path=args.md_output_path)
    for row in rows[:20]:
        print(row.queue_id, row.lane, row.subject, row.action, row.score)


if __name__ == "__main__":
    main()
