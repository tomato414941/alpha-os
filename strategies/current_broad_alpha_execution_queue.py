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
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=row.get("ticket_id", ""),
            lane="stablecoin_chain_liquidity_proxy",
            subject=row.get("asset", "") or row.get("opportunity", ""),
            side=row.get("side", ""),
            action=row.get("decision", ""),
            status=row.get("status", ""),
            score=str(max(0.0, 100.0 - _float(row.get("rank")))),
            size_or_risk=f"entry_mark={row.get('entry_mark', '')}; venue={row.get('venue', '')}",
            evidence=row.get("opportunity", ""),
            checkpoints=row.get("checkpoints", ""),
            source_path=_relative(path),
            required_record=row.get("required_record", ""),
            next_step=row.get("next_step", ""),
        )
        for row in _read_rows(path)
    )


def _protocol_fee_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "protocol_fundamentals/current_protocol_fee_actionability.csv"
    selected = tuple(
        row
        for row in _read_rows(path)
        if row.get("action") == "wait_for_forward_label"
        and row.get("execution_action") in {"paper_observation_ready", "thin_volume_watch"}
    )[:6]
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"protocol-fee-{_slug(row.get('token_symbol', ''))}-{_slug(row.get('protocol', ''))}",
            lane="protocol_fee_growth_lag",
            subject=f"{row.get('token_symbol', '')}/{row.get('protocol', '')}",
            side=row.get("side", ""),
            action=row.get("action", ""),
            status=row.get("status", ""),
            score=_fmt(row.get("score")),
            size_or_risk=f"venues={row.get('venue_count', '')}; depth10={row.get('depth_10bps', '')}; spread={row.get('spread_bps', '')}",
            evidence=f"fee_growth_7d={row.get('fee_growth_7d', '')}; price_change_7d={row.get('price_change_7d', '')}",
            checkpoints="4h,12h,24h",
            source_path=_relative(path),
            required_record="fresh forward labels, funding, spread, depth, and stale-ticket rejection",
            next_step=row.get("next_step", ""),
        )
        for row in selected
    )


def _token_unlock_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "token_unlocks/current_token_unlock_actionability.csv"
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
            action=row.get("action", ""),
            status=row.get("status", ""),
            score=_fmt(row.get("score")),
            size_or_risk=f"days={row.get('days_until', '')}; unlock_usd={row.get('unlock_value_usd', '')}; pct_supply={row.get('percent_supply', '')}",
            evidence=row.get("reason", ""),
            checkpoints="pre_event,event_window,post_event",
            source_path=_relative(path),
            required_record="event-window label, funding, crowding, liquidity, and squeeze-vs-pressure split",
            next_step=row.get("next_step", ""),
        )
        for row in selected
    )


def _defi_lending_queue(root: Path) -> tuple[BroadAlphaExecutionQueueItem, ...]:
    path = root / "defi_lending/current_lending_stress_actionability.csv"
    return tuple(
        BroadAlphaExecutionQueueItem(
            queue_id=f"defi-lending-{_slug(row.get('chain', ''))}-{_slug(row.get('loan_asset', ''))}-{_slug(row.get('collateral_asset', ''))}",
            lane="defi_lending_yield",
            subject=f"{row.get('chain', '')} {row.get('loan_asset', '')}/{row.get('collateral_asset', '')}",
            side=row.get("side", ""),
            action=row.get("side", ""),
            status=row.get("status", ""),
            score=_fmt(row.get("score")),
            size_or_risk=f"liquidity_usd={row.get('liquidity_usd', '')}; utilization={row.get('utilization', '')}",
            evidence=f"supply_apy={row.get('avg_net_supply_apy', '')}; borrow_apy={row.get('avg_net_borrow_apy', '')}",
            checkpoints="withdrawal_path,rate_persistence,risk_check",
            source_path=_relative(path),
            required_record="withdrawal path, rate persistence, gas, oracle, liquidation, and smart-contract risk",
            next_step=row.get("next_step", ""),
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
    return (lane_bonus + _float(row.score), _float(row.score))


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
