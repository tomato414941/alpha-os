from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaPromotionFrontierRow:
    frontier_id: str
    lane: str
    source_artifact: str
    candidate_id: str
    asset: str
    action: str
    status: str
    frontier_score: float
    edge_bps: str
    support_count: str
    blocker: str
    evidence: str
    next_step: str


def build_alpha_promotion_frontier(root: Path = ROOT) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    rows.extend(_cost_survival_rows(root))
    rows.extend(_cost_candidate_rows(root))
    rows.extend(_options_rows(root))
    rows.extend(_news_rows(root))
    rows.extend(_lob_maker_rows(root))
    rows.extend(_event_hedge_rows(root))
    rows.extend(_crowded_positioning_rows(root))
    rows.extend(_stablecoin_exchange_rows(root))
    return tuple(sorted(rows, key=lambda row: row.frontier_score, reverse=True))


def write_alpha_promotion_frontier_csv(
    rows: tuple[AlphaPromotionFrontierRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "frontier_id",
                "lane",
                "source_artifact",
                "candidate_id",
                "asset",
                "action",
                "status",
                "frontier_score",
                "edge_bps",
                "support_count",
                "blocker",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.frontier_id,
                    row.lane,
                    row.source_artifact,
                    row.candidate_id,
                    row.asset,
                    row.action,
                    row.status,
                    f"{row.frontier_score:.8f}",
                    row.edge_bps,
                    row.support_count,
                    row.blocker,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_promotion_frontier_md(
    rows: tuple[AlphaPromotionFrontierRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Promotion Frontier\n\n")
        handle.write(
            "This separates paper alpha candidates from the blockers that must be cleared before promotion. "
            "It does not mark any row as live-tradable.\n\n"
        )
        handle.write("## Status Counts\n\n")
        handle.write("| status | rows |\n")
        handle.write("| --- | ---: |\n")
        for status, count in sorted(status_counts.items(), key=lambda item: item[1], reverse=True):
            handle.write(f"| {status} | {count} |\n")
        handle.write("\n## Top Frontier Rows\n\n")
        handle.write(
            "| frontier | lane | asset | action | status | score | edge bps | support | blocker | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:60]:
            handle.write(
                "| "
                f"{row.frontier_id} | "
                f"{row.lane} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.status} | "
                f"{row.frontier_score:.4f} | "
                f"{row.edge_bps} | "
                f"{row.support_count} | "
                f"{_escape(row.blocker)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _cost_survival_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "current_cost_survival_cross_section.csv"
    for row in _read_rows(root / artifact)[:25]:
        raw_status = row.get("status", "")
        status = {
            "cost_surviving_watchlist": "paper_cost_survival_watchlist",
            "duplicate_pressure_control_required": "duplicate_dedupe_required",
            "repeat_outcome_conflicted": "repeat_conflict_split_required",
            "capacity_blocks_cost_survival": "capacity_blocked",
        }.get(raw_status, raw_status or "cost_survival_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"cost_cluster:{row.get('cluster_id', '')}",
                lane="cost_survival",
                source_artifact=artifact,
                candidate_id=row.get("cluster_id", ""),
                asset=row.get("asset", ""),
                action=row.get("decision", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("mean_net_after_cost_bps", ""),
                support_count=row.get("repeat_win_count", ""),
                blocker=row.get("missing_work", ""),
                evidence=(
                    f"best_net={row.get('best_net_after_cost_bps', '')}; "
                    f"mean_net={row.get('mean_net_after_cost_bps', '')}; "
                    f"candidates={row.get('candidate_count', '')}; "
                    f"repeat_wins={row.get('repeat_win_count', '')}; "
                    f"repeat_losses={row.get('repeat_loss_count', '')}; "
                    f"dup={row.get('duplicate_pressure', '')}; "
                    f"usage={row.get('max_visible_depth_usage', '')}"
                ),
                next_step=row.get("next_probe", ""),
            )
        )
    return tuple(rows)


def _cost_candidate_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "current_cost_adjusted_alpha_candidates.csv"
    for row in _read_rows(root / artifact)[:20]:
        raw_status = row.get("status", "")
        status = {
            "repeat_supported_cost_adjusted_alpha": "repeat_supported_paper_alpha",
            "first_repeat_cost_adjusted_alpha": "first_repeat_paper_alpha",
            "cost_adjusted_alpha_candidate": "paper_alpha_needs_repeat",
            "capacity_gated_alpha_candidate": "capacity_blocked",
        }.get(raw_status, raw_status or "paper_alpha_review")
        score = _float(row.get("priority_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"cost_candidate:{row.get('candidate_id', '')}",
                lane="cost_adjusted_candidate",
                source_artifact=artifact,
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("asset", ""),
                action=row.get("decision", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("estimated_net_after_cost_bps", ""),
                support_count="1",
                blocker=row.get("missing_work", ""),
                evidence=row.get("evidence", ""),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _options_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "options_volatility/current_options_volatility_survival.csv"
    for row in _read_rows(root / artifact)[:10]:
        raw_status = row.get("status", "")
        status = {
            "long_vol_hedge_path_required": "options_hedge_path_required",
            "quote_mechanics_required": "options_quote_mechanics_required",
            "short_expiry_gamma_timing_required": "options_gamma_timing_required",
            "premium_size_blocks_survival": "options_premium_blocked",
            "top_depth_blocks_survival": "options_depth_blocked",
        }.get(raw_status, raw_status or "options_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"options:{row.get('candidate_id', '')}",
                lane="options_volatility",
                source_artifact=artifact,
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("currency", ""),
                action=row.get("structure", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("iv_premium_24h", ""),
                support_count=row.get("top_ask_premium_depth_usd", ""),
                blocker=row.get("missing_work", ""),
                evidence=row.get("evidence", ""),
                next_step=row.get("next_probe", ""),
            )
        )
    return tuple(rows)


def _news_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "news_social/current_news_event_survival.csv"
    for row in _read_rows(root / artifact)[:12]:
        raw_status = row.get("survival_status", "")
        status = {
            "news_event_pending_archive": "news_pending_forward_archive",
            "news_event_single_source_blocked": "news_single_source_blocked",
            "news_event_rejected": "news_rejected",
        }.get(raw_status, raw_status or "news_event_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"news:{row.get('candidate_id', '')}",
                lane="news_event",
                source_artifact=artifact,
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("symbol", ""),
                action=row.get("side", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("mean_directional_4h_bps", ""),
                support_count=row.get("source_count", ""),
                blocker=row.get("reason", ""),
                evidence=(
                    f"event={row.get('event_kind', '')}; "
                    f"sources={row.get('sources', '')}; "
                    f"labels={row.get('label_count', '')}; "
                    f"stories={row.get('unique_story_count', '')}; "
                    f"dominant_share={row.get('dominant_story_share', '')}"
                ),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _lob_maker_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "event_flow/current_lob_maker_fill_survival.csv"
    for row in _read_rows(root / artifact)[:8]:
        raw_status = row.get("survival_status", "")
        status = {
            "maker_adverse_selection_blocked": "maker_adverse_selection_blocked",
        }.get(raw_status, raw_status or "maker_fill_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"lob_maker:{row.get('candidate_id', '')}",
                lane="lob_maker_fill",
                source_artifact=artifact,
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("state_family", ""),
                action=row.get("signal_action", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("filled_mark_reward_bps", ""),
                support_count=row.get("fill_count", ""),
                blocker=row.get("reason", ""),
                evidence=(
                    f"source={row.get('source_probe', '')}; "
                    f"feature={row.get('feature', '')}; "
                    f"fill_rate={row.get('fill_rate', '')}; "
                    f"adverse={row.get('adverse_fill_rate', '')}; "
                    f"optimistic_net={row.get('optimistic_net_bps', '')}"
                ),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _event_hedge_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "prediction_markets/current_event_crypto_hedge_survival.csv"
    for row in _read_rows(root / artifact)[:10]:
        raw_status = row.get("survival_status", "")
        status = {
            "event_crypto_hedge_pending_mark": "event_hedge_pending_mark",
            "event_crypto_hedge_candidate_unproven": "event_hedge_unproven",
            "event_crypto_hedge_rejected_event_flat": "event_hedge_event_alignment_blocked",
        }.get(raw_status, raw_status or "event_hedge_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"event_hedge:{row.get('candidate_id', '')}",
                lane="prediction_market_event_hedge",
                source_artifact=artifact,
                candidate_id=row.get("candidate_id", ""),
                asset=row.get("asset", ""),
                action=row.get("hedge_action", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("residual_vs_basket_bps", ""),
                support_count=row.get("market_id", ""),
                blocker=row.get("reason", ""),
                evidence=(
                    f"question={row.get('question', '')}; "
                    f"event_bias={row.get('event_bias', '')}; "
                    f"event_mark_return={row.get('event_mark_return_bps', '')}; "
                    f"asset_return={row.get('asset_directional_return_bps', '')}; "
                    f"alignment={row.get('alignment_status', '')}"
                ),
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _crowded_positioning_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "perp_market_map/current_crowded_positioning_survival.csv"
    for row in _read_rows(root / artifact)[:10]:
        raw_status = row.get("status", "")
        status = {
            "needs_forward_unwind_label": "crowding_needs_forward_unwind_label",
            "crowded_context_without_unwind_label": "crowding_without_unwind_label",
        }.get(raw_status, raw_status or "crowding_review")
        score = _float(row.get("survival_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"crowding:{row.get('asset', '')}:{row.get('action', '')}",
                lane="crowded_positioning",
                source_artifact=artifact,
                candidate_id=row.get("asset", ""),
                asset=row.get("asset", ""),
                action=row.get("action", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("net_directional_return_1h_proxy", ""),
                support_count=row.get("venue_count", ""),
                blocker=row.get("missing_work", ""),
                evidence=row.get("evidence", ""),
                next_step=row.get("next_probe", ""),
            )
        )
    return tuple(rows)


def _stablecoin_exchange_rows(root: Path) -> tuple[AlphaPromotionFrontierRow, ...]:
    rows: list[AlphaPromotionFrontierRow] = []
    artifact = "stablecoin_liquidity/current_exchange_stablecoin_inflow_readiness.csv"
    for row in _read_rows(root / artifact)[:10]:
        raw_status = row.get("status", "")
        status = {
            "direct_exchange_inflow_data_required": "direct_exchange_inflow_data_required",
            "proxy_label_candidate_not_exchange_inflow": "chain_proxy_alpha_needs_label",
            "chain_proxy_watch_not_exchange_inflow": "chain_proxy_context_only",
            "unmapped_chain_context_not_alpha": "unmapped_chain_context_not_alpha",
        }.get(raw_status, raw_status or "stablecoin_flow_review")
        score = _float(row.get("readiness_score"))
        rows.append(
            AlphaPromotionFrontierRow(
                frontier_id=f"stablecoin_flow:{row.get('subject', '')}",
                lane="stablecoin_exchange_inflow",
                source_artifact=artifact,
                candidate_id=row.get("subject", ""),
                asset=row.get("token_symbol", ""),
                action=row.get("flow_direction", ""),
                status=status,
                frontier_score=_rank_score(status, score),
                edge_bps=row.get("directional_return_4h", ""),
                support_count=row.get("week_change_usd", ""),
                blocker=row.get("missing_data", ""),
                evidence=row.get("current_proxy_evidence", ""),
                next_step=row.get("next_probe", ""),
            )
        )
    return tuple(rows)


def _rank_score(status: str, score: float) -> float:
    base = 0.0
    if status in {
        "repeat_supported_paper_alpha",
        "first_repeat_paper_alpha",
        "paper_alpha_needs_repeat",
        "paper_cost_survival_watchlist",
    }:
        base = 1000.0
    elif status in {
        "duplicate_dedupe_required",
        "repeat_conflict_split_required",
        "options_hedge_path_required",
        "options_quote_mechanics_required",
        "options_gamma_timing_required",
    }:
        base = 700.0
    elif status in {
        "news_pending_forward_archive",
        "event_hedge_pending_mark",
        "event_hedge_unproven",
        "crowding_needs_forward_unwind_label",
        "crowding_without_unwind_label",
        "direct_exchange_inflow_data_required",
        "chain_proxy_alpha_needs_label",
    }:
        base = 500.0
    elif status in {
        "capacity_blocked",
        "maker_adverse_selection_blocked",
        "news_single_source_blocked",
        "event_hedge_event_alignment_blocked",
        "options_premium_blocked",
        "options_depth_blocked",
    }:
        base = 200.0
    else:
        base = 100.0
    return base + score


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_promotion_frontier.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_promotion_frontier.md")
    args = parser.parse_args()

    rows = build_alpha_promotion_frontier()
    write_alpha_promotion_frontier_csv(rows, output_path=args.output_path)
    write_alpha_promotion_frontier_md(rows, output_path=args.md_output_path)
    for row in rows[:12]:
        print(row.status, row.frontier_id, row.asset, f"{row.frontier_score:.4f}")


if __name__ == "__main__":
    main()
