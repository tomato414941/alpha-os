from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TemplateValidationRoute:
    template_id: str
    lane: str
    status: str
    priority_score: float
    validation_route: str
    current_artifact: str
    current_status: str
    best_evidence: str
    next_step: str


@dataclass(frozen=True)
class RouteRule:
    template_id: str
    validation_route: str
    artifact_path: Path
    status_columns: tuple[str, ...]
    score_columns: tuple[str, ...]
    evidence_columns: tuple[str, ...]
    next_step: str


ROUTE_RULES = {
    "ofi_liquidity_state_factor": RouteRule(
        template_id="ofi_liquidity_state_factor",
        validation_route="market_making L2 imbalance gate plus event_flow book-depth walk-forward",
        artifact_path=ROOT / "market_making" / "current_l2_imbalance_paper_gate.csv",
        status_columns=("gate_action",),
        score_columns=("net_15m_bps", "net_1h_bps"),
        evidence_columns=("asset", "candidate_size_usd", "net_15m_bps", "net_1h_bps", "visible_depth_usage"),
        next_step="repeat L2 imbalance labels on fresh snapshots and join with historical bookDepth walk-forward",
    ),
    "crowded_positioning_unwind_factor": RouteRule(
        template_id="crowded_positioning_unwind_factor",
        validation_route="perp positioning repeat labels plus cross-venue OI/funding context",
        artifact_path=ROOT / "perp_market_map" / "current_crowding_cross_venue_confirmation.csv",
        status_columns=("decision", "status"),
        score_columns=("score", "net_directional_return_1h_proxy", "max_derivatives_score"),
        evidence_columns=("asset", "action", "venue_count", "actionable_venue_count", "net_directional_return_1h_proxy"),
        next_step="split crowded continuation and unwind labels, then require cross-venue OI confirmation",
    ),
    "source_diverse_news_shock_factor": RouteRule(
        template_id="source_diverse_news_shock_factor",
        validation_route="news event quality gate",
        artifact_path=ROOT / "news_social" / "current_news_event_quality_gate.csv",
        status_columns=("decision",),
        score_columns=("score", "mean_directional_4h_bps", "mean_directional_1h_bps"),
        evidence_columns=("symbol", "event_kind", "side", "source_count", "supported_count", "rejected_count"),
        next_step="wait for fresh archives, dedupe sources, and rerun 15m/1h/4h labels after costs",
    ),
    "prediction_market_crypto_beta_factor": RouteRule(
        template_id="prediction_market_crypto_beta_factor",
        validation_route="prediction-market refresh plus crypto hedge labels",
        artifact_path=ROOT / "prediction_markets" / "current_event_crypto_hedge_candidates.csv",
        status_columns=("status",),
        score_columns=("score", "current_edge_after_ask", "actionability_score"),
        evidence_columns=("asset", "market_id", "hedge_action", "current_edge_after_ask", "question"),
        next_step="label refreshed event-probability crypto hedges with beta attribution and resolution risk",
    ),
    "wallet_entity_follow_or_fade_factor": RouteRule(
        template_id="wallet_entity_follow_or_fade_factor",
        validation_route="public wallet/entity flow actionability",
        artifact_path=ROOT / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv",
        status_columns=("status", "action"),
        score_columns=("score", "net_closed_pnl_after_fees", "net_buy_notional"),
        evidence_columns=("execution_asset", "wallet_label", "source_coin", "net_buy_notional", "net_closed_pnl_after_fees"),
        next_step="collect independent wallet quality and forward labels before following wallet flow",
    ),
    "chain_liquidity_migration_factor": RouteRule(
        template_id="chain_liquidity_migration_factor",
        validation_route="stablecoin migration and DEX/liquidity mapping",
        artifact_path=ROOT / "stablecoin_liquidity" / "current_chain_stablecoin_migration_forward_labels.csv",
        status_columns=("label_status", "migration_status"),
        score_columns=("directional_return_12h", "directional_return_4h", "directional_return_1h"),
        evidence_columns=("chain", "token_symbol", "side", "week_change_usd", "directional_return_4h"),
        next_step="map chain flow to liquid instruments and label against beta-adjusted returns",
    ),
    "vol_surface_dislocation_factor": RouteRule(
        template_id="vol_surface_dislocation_factor",
        validation_route="Deribit option actionability and delta-hedge paper checks",
        artifact_path=ROOT / "options_volatility" / "current_volatility_actionability.csv",
        status_columns=("status",),
        score_columns=("score", "premium_to_realized_move", "iv_premium_24h"),
        evidence_columns=("currency", "expiry", "structure", "premium_to_realized_move", "quote_spread_pct"),
        next_step="simulate fills, hedge marks, exit bid, max loss, and margin before promotion",
    ),
    "cross_venue_funding_basis_factor": RouteRule(
        template_id="cross_venue_funding_basis_factor",
        validation_route="cross-exchange funding persistence and basis route",
        artifact_path=ROOT / "cross_exchange_funding" / "current_dislocation_execution_check.csv",
        status_columns=("gate_action", "action"),
        score_columns=("conservative_taker_net_24h", "mean_net_24h_proxy", "fee_only_net_24h"),
        evidence_columns=("asset", "mean_net_24h_proxy", "conservative_taker_net_24h", "combined_taker_slippage_bps"),
        next_step="record real funding persistence, hedge drift, fees, margin, and venue constraints",
    ),
    "liquidation_cascade_state_factor": RouteRule(
        template_id="liquidation_cascade_state_factor",
        validation_route="OKX forced-liquidation intensity labels and paper gate",
        artifact_path=ROOT / "liquidation_flow" / "current_okx_liquidation_intensity_paper_gate.csv",
        status_columns=("gate_action", "status"),
        score_columns=("net_bps", "label15_bps", "score"),
        evidence_columns=("symbol", "action", "side", "net_bps", "depth10", "usage"),
        next_step="repeat liquidation events with entry delay, stop, funding, and adverse-excursion logs",
    ),
    "protocol_fee_repricing_factor": RouteRule(
        template_id="protocol_fee_repricing_factor",
        validation_route="protocol fee growth to sector-adjusted token return labels",
        artifact_path=ROOT / "protocol_fundamentals" / "current_protocol_fee_price_context.csv",
        status_columns=("status", "action"),
        score_columns=("score", "fee_growth", "price_change_24h"),
        evidence_columns=("symbol", "protocol", "fee_growth", "price_change_24h", "sector"),
        next_step="label fee-growth candidates against sector-adjusted token returns",
    ),
    "multi_source_event_graph_factor": RouteRule(
        template_id="multi_source_event_graph_factor",
        validation_route="cross-modal event graph join",
        artifact_path=ROOT / "current_alpha_source_gaps.csv",
        status_columns=("status",),
        score_columns=("priority",),
        evidence_columns=("gap_id", "coverage", "current_coverage", "missing_work"),
        next_step="join one news event source with one on-chain/liquidity source and label per asset",
    ),
    "rl_state_action_reward_dataset_factor": RouteRule(
        template_id="rl_state_action_reward_dataset_factor",
        validation_route="policy-learning observation/action/reward sample audit",
        artifact_path=ROOT / "policy_learning" / "current_policy_learning_samples.csv",
        status_columns=("outcome", "status"),
        score_columns=("reward_bps", "net_bps", "score"),
        evidence_columns=("sample_id", "lane", "symbol", "action", "reward_bps"),
        next_step="audit observation/action/reward leakage and cost completeness before model training",
    ),
}


def build_template_validation_queue(root: Path = ROOT) -> tuple[TemplateValidationRoute, ...]:
    templates = _read_rows(root / "llm_factor_generation" / "current_factor_hypothesis_templates.csv")
    rows = []
    for template in templates:
        rule = ROUTE_RULES.get(template.get("template_id", ""))
        if rule is None:
            continue
        rows.append(_build_route(template=template, rule=rule))
    return tuple(sorted(rows, key=lambda row: row.priority_score, reverse=True))


def write_queue_csv(rows: tuple[TemplateValidationRoute, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "template_id",
                "lane",
                "status",
                "priority_score",
                "validation_route",
                "current_artifact",
                "current_status",
                "best_evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.template_id,
                    row.lane,
                    row.status,
                    f"{row.priority_score:.8f}",
                    row.validation_route,
                    row.current_artifact,
                    row.current_status,
                    row.best_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_queue_md(rows: tuple[TemplateValidationRoute, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Factor Template Validation Queue\n\n")
        handle.write(
            "This routes generated factor templates to concrete validation artifacts. "
            "It is a routing queue, not a strategy list or trade instruction.\n\n"
        )
        handle.write("| template | route | artifact status | score | evidence | next step |\n")
        handle.write("| --- | --- | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.template_id} | "
                f"{_escape(row.validation_route)} | "
                f"{_escape(row.current_status)} | "
                f"{row.priority_score:.4f} | "
                f"{_escape(row.best_evidence)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _build_route(*, template: dict[str, str], rule: RouteRule) -> TemplateValidationRoute:
    artifact_rows = _read_rows(rule.artifact_path)
    best = _best_artifact_row(artifact_rows, score_columns=rule.score_columns)
    artifact_status = _artifact_status(best, status_columns=rule.status_columns)
    evidence = _artifact_evidence(best, evidence_columns=rule.evidence_columns)
    score = _float(template.get("priority_score"))
    if not artifact_rows:
        artifact_status = "missing_artifact"
        evidence = "no current validation artifact found"
        score -= 5.0
    elif artifact_status in {"", "unknown"}:
        artifact_status = "artifact_present"
    return TemplateValidationRoute(
        template_id=template.get("template_id", ""),
        lane=template.get("lane", ""),
        status=template.get("status", ""),
        priority_score=score,
        validation_route=rule.validation_route,
        current_artifact=str(rule.artifact_path.relative_to(ROOT)),
        current_status=artifact_status,
        best_evidence=evidence,
        next_step=rule.next_step,
    )


def _best_artifact_row(
    rows: tuple[dict[str, str], ...],
    *,
    score_columns: tuple[str, ...],
) -> dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: max((_float(row.get(column)) for column in score_columns), default=0.0))


def _artifact_status(row: dict[str, str], *, status_columns: tuple[str, ...]) -> str:
    for column in status_columns:
        value = row.get(column)
        if value:
            return value
    return "unknown"


def _artifact_evidence(row: dict[str, str], *, evidence_columns: tuple[str, ...]) -> str:
    parts = [f"{column}={row.get(column, '')}" for column in evidence_columns if row.get(column)]
    return ", ".join(parts) if parts else "no evidence row"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = build_template_validation_queue(args.root)
    write_queue_csv(
        rows,
        output_path=args.root / "llm_factor_generation" / "current_factor_template_validation_queue.csv",
    )
    write_queue_md(
        rows,
        output_path=args.root / "llm_factor_generation" / "current_factor_template_validation_queue.md",
    )
    for row in rows[:10]:
        print(row.template_id, row.current_status, f"{row.priority_score:.4f}")


if __name__ == "__main__":
    main()
