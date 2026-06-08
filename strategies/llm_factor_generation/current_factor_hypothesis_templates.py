from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FactorHypothesisTemplate:
    template_id: str
    lane: str
    status: str
    priority_score: float
    input_features: str
    transformation: str
    target_label: str
    action_mapping: str
    seed_opportunities: str
    validation_check: str
    failure_mode: str
    research_reference: str
    next_step: str


@dataclass(frozen=True)
class TemplateRule:
    template_id: str
    lane: str
    source_terms: tuple[str, ...]
    status: str
    base_score: float
    input_features: str
    transformation: str
    target_label: str
    action_mapping: str
    validation_check: str
    failure_mode: str
    research_reference: str
    next_step: str


TEMPLATE_RULES = (
    TemplateRule(
        template_id="ofi_liquidity_state_factor",
        lane="LOB / order-flow representation learning",
        source_terms=("book_depth", "l2_imbalance", "microstructure_flow", "market_making"),
        status="needs_feature_history",
        base_score=96.0,
        input_features="book depth imbalance, taker buy/sell ratio, depth slope, spread, premium, funding",
        transformation="rank cross-section by OFI shock conditioned on liquidity state; test continuation and fade buckets separately",
        target_label="5m/15m/1h directional return after fees, spread, funding, and adverse excursion",
        action_mapping="long continuation only when OFI shock and depth support agree; short/fade when shock is crowded and depth thins",
        validation_check="purged walk-forward by day and symbol with explicit execution-cost stress",
        failure_mode="gross-only microstructure edge disappears after spread, queue, and adverse-selection costs",
        research_reference="https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full",
        next_step="extend Binance bookDepth walk-forward with OFI and queue/adverse-selection labels",
    ),
    TemplateRule(
        template_id="crowded_positioning_unwind_factor",
        lane="derivatives positioning / crowding",
        source_terms=("oi_", "crowding", "derivatives_positioning", "perp_market_map", "long_short"),
        status="needs_cross_venue_labels",
        base_score=94.0,
        input_features="open-interest change, funding, long/short ratios, premium, volume shock, impact spread",
        transformation="separate crowded continuation from crowded unwind by OI direction and funding sign",
        target_label="15m/1h/4h return plus funding PnL and fill-adjusted execution cost",
        action_mapping="follow crowded continuation only before squeeze signs; fade when OI unwinds into stretched funding",
        validation_check="repeat across venues and require a neutral baseline against same-symbol momentum",
        failure_mode="OI notional reflects mark-price changes rather than fresh position changes",
        research_reference="https://www.nature.com/articles/s41598-026-46271-w",
        next_step="join Binance metrics, CoinGecko derivatives, and Hyperliquid OI labels into one repeat table",
    ),
    TemplateRule(
        template_id="source_diverse_news_shock_factor",
        lane="news / social contagion",
        source_terms=("news_social", "event_pressure", "rss:", "forward_label", "quality_gate"),
        status="needs_duplicate_source_review",
        base_score=93.0,
        input_features="timestamped event kind, source count, source independence, headline freshness, attention rank",
        transformation="score only non-duplicate multi-source shocks; compare immediate continuation versus delayed fade",
        target_label="15m/1h/4h directional return after spread and funding",
        action_mapping="paper long/short only when independent sources repeat and pending archive labels confirm",
        validation_check="dedupe headlines, enforce timestamp control, and test non-overlapping event windows",
        failure_mode="headline is stale, duplicated, already priced, or sentiment is non-causal",
        research_reference="https://www.dallasfed.org/research/papers/2026/wp2605",
        next_step="repeat BTC/ZEC gated news labels after fresh archives and add duplicate-source review",
    ),
    TemplateRule(
        template_id="prediction_market_crypto_beta_factor",
        lane="prediction markets / event beta",
        source_terms=("prediction_markets", "event_market", "probability", "crypto_hedge"),
        status="needs_event_beta_attribution",
        base_score=92.0,
        input_features="prediction-market probability gap, bid/ask depth, event topic, crypto beta bucket, macro/news context",
        transformation="map event probability change to crypto risk-on/risk-off exposure with beta attribution",
        target_label="4h/12h/24h crypto basket return around probability refresh and event resolution",
        action_mapping="hedge BTC/ETH/SOL only when probability edge survives quote refresh and depth checks",
        validation_check="separate market PnL, crypto hedge PnL, resolution risk, and quote slippage",
        failure_mode="prediction market is stale, manipulated, illiquid, or unrelated to crypto beta",
        research_reference="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6416881",
        next_step="label refreshed event-probability crypto hedges with beta attribution and resolution-risk notes",
    ),
    TemplateRule(
        template_id="wallet_entity_follow_or_fade_factor",
        lane="wallet / entity flow",
        source_terms=("wallet_entity_flow", "public_hypertracker", "wallet_flow"),
        status="needs_entity_quality",
        base_score=90.0,
        input_features="public wallet fills, realized PnL, current position notional, symbol liquidity, copy-crowding proxy",
        transformation="classify entity flow as informed follow, forced unwind, or copy-crowded late signal",
        target_label="15m/1h/4h return after observed fill timestamp with funding and spread",
        action_mapping="follow high-quality fresh accumulation; fade stale crowded copy-flow after adverse move",
        validation_check="hold out wallets, require repeated timestamps, and compare against same-symbol momentum",
        failure_mode="survivorship bias, fake smart-wallet quality, or copied flow already exhausted",
        research_reference="https://link.springer.com/article/10.1007/s10614-025-10940-1",
        next_step="collect independent wallet/entity quality labels before treating public wallet flow as alpha",
    ),
    TemplateRule(
        template_id="chain_liquidity_migration_factor",
        lane="on-chain liquidity / stablecoin migration",
        source_terms=("stablecoin_liquidity", "chain_stablecoin", "defillama", "dex_pool_flow"),
        status="needs_asset_mapping",
        base_score=89.0,
        input_features="stablecoin supply change, chain migration, DEX pool turnover, protocol TVL/fees, tradable token mapping",
        transformation="separate capital migration from price-chasing flow and map chain-level flow to liquid instruments",
        target_label="4h/12h/24h return of mapped tokens and chain beta basket after costs",
        action_mapping="long mapped assets only when liquidity inflow is fresh and tradable venue depth is adequate",
        validation_check="time-control supply snapshots and compare mapped assets to broad crypto beta",
        failure_mode="chain flow has no direct tradable asset, is stale, or reflects stablecoin rotation only",
        research_reference="https://link.springer.com/article/10.1007/s44257-025-00046-1",
        next_step="build one chain-flow-to-asset map and label against liquid perps instead of generic chain names",
    ),
    TemplateRule(
        template_id="vol_surface_dislocation_factor",
        lane="options volatility",
        source_terms=("options_volatility", "delta_hedge", "iv_", "skew"),
        status="needs_hedged_pnl",
        base_score=88.0,
        input_features="IV premium to realized vol, skew, term structure, option depth, spread, delta hedge interval",
        transformation="identify cheap convexity or rich downside skew conditioned on realized-vol regime",
        target_label="option PnL plus delta-hedge PnL, fees, spread, and max loss",
        action_mapping="buy convexity only when premium-to-RV and exit depth pass; otherwise treat as hedge context",
        validation_check="simulate multi-leg fills, delta-hedge marks, margin, and exit-bid behavior",
        failure_mode="vol edge is a mark artifact or option spread/hedge cost consumes it",
        research_reference="https://doi.org/10.1007/s00500-025-10980-7",
        next_step="paper-check short-dated BTC/ETH delta-hedge candidates with realized hedge PnL",
    ),
    TemplateRule(
        template_id="cross_venue_funding_basis_factor",
        lane="cross-exchange funding / basis",
        source_terms=("cross_exchange_funding", "basis_term_structure", "funding_dislocation", "basis"),
        status="needs_real_funding_persistence",
        base_score=87.0,
        input_features="predicted funding, venue pair, basis, hedge route, fee ceiling, depth, margin requirements",
        transformation="score net carry only when persistence, borrow/margin route, and hedge liquidity agree",
        target_label="8h/24h net funding plus hedge PnL and execution slippage",
        action_mapping="open carry pair only when fee ceiling and venue constraints leave durable positive edge",
        validation_check="monitor real funding payment, hedge drift, and transfer/margin constraints",
        failure_mode="funding disappears before entry or venue/borrow/margin constraints dominate",
        research_reference="https://link.springer.com/article/10.1007/s11704-025-41061-5",
        next_step="convert funding dislocation candidates into real funding-persistence paper records",
    ),
    TemplateRule(
        template_id="liquidation_cascade_state_factor",
        lane="liquidation flow",
        source_terms=("liquidation_flow", "liquidation", "cascade", "squeeze"),
        status="needs_event_replay",
        base_score=86.0,
        input_features="forced-liquidation notional, OI ratio, imbalance, depth, funding, local return, venue",
        transformation="classify cascade continuation versus forced-liquidation rebound by intensity and depth recovery",
        target_label="5m/15m/1h return with stop/adverse-excursion and funding",
        action_mapping="trade only after cascade direction and rebound/follow-through labels repeat",
        validation_check="event replay with entry-delay, fill, stop, and adverse-excursion logging",
        failure_mode="liquidation print is the end of the move, not the start of exploitable continuation",
        research_reference="https://arxiv.org/abs/2604.24590",
        next_step="repeat OKX liquidation intensity labels with event replay and stop behavior",
    ),
    TemplateRule(
        template_id="protocol_fee_repricing_factor",
        lane="protocol fundamentals",
        source_terms=("protocol_fundamentals", "protocol_fee", "fee_growth", "valuation"),
        status="needs_repricing_window",
        base_score=84.0,
        input_features="protocol fees, fee growth, valuation multiple, token liquidity, sector beta, unlock context",
        transformation="score fresh fee-growth surprise relative to valuation and sector beta",
        target_label="1d/3d/7d token return adjusted for sector and crypto beta",
        action_mapping="long only when fee-growth surprise is fresh, liquid, and not explained by sector beta",
        validation_check="use lagged DeFiLlama snapshots and hold out protocol categories",
        failure_mode="fundamental data is delayed, already priced, or token does not capture protocol economics",
        research_reference="https://alphabench.cc/",
        next_step="label protocol fee-growth candidates against sector-adjusted token returns",
    ),
    TemplateRule(
        template_id="rl_state_action_reward_dataset_factor",
        lane="RL-shaped policy dataset",
        source_terms=("policy_learning", "paper_ticket", "action_queue", "reward"),
        status="needs_dataset_contract",
        base_score=83.0,
        input_features="observation snapshot, action, fill/funding/cost context, realized mark outcome, failure flags",
        transformation="turn paper tickets into state/action/reward rows without inventing a strategy class",
        target_label="realized reward after costs and risk constraints",
        action_mapping="learn action preferences only after observations, actions, and rewards are explicitly separated",
        validation_check="chronological split, no leakage from outcome fields into observation, cost-aware reward",
        failure_mode="paper-ticket bookkeeping leaks outcomes into features or optimizes non-tradable marks",
        research_reference="https://arxiv.org/abs/2307.01599",
        next_step="audit policy learning samples for observation/action/reward leakage and cost completeness",
    ),
    TemplateRule(
        template_id="multi_source_event_graph_factor",
        lane="cross-modal event graph",
        source_terms=("event_pressure", "stablecoin_liquidity", "wallet_entity_flow", "dex_pool_flow", "news_social"),
        status="needs_graph_join",
        base_score=82.0,
        input_features="news event, wallet/entity flow, stablecoin migration, DEX pool turnover, liquid perp venue",
        transformation="join independent event modalities by timestamp and asset; score agreement and conflict",
        target_label="15m/1h/4h/24h return by asset after costs and beta control",
        action_mapping="act only when independent modalities align and each has a tradable asset mapping",
        validation_check="non-overlapping event windows and modality-ablation tests",
        failure_mode="modalities are correlated copies of the same price move or do not map to a tradable asset",
        research_reference="https://link.springer.com/article/10.1007/s41109-026-00778-3",
        next_step="join one news event source with one on-chain/liquidity source and label per asset",
    ),
)


def build_factor_hypothesis_templates(root: Path = ROOT) -> tuple[FactorHypothesisTemplate, ...]:
    stack_rows = _read_rows(root / "current_alpha_stack.csv")
    output = tuple(_build_template(rule, stack_rows=stack_rows) for rule in TEMPLATE_RULES)
    return tuple(sorted(output, key=lambda row: row.priority_score, reverse=True))


def write_templates_csv(rows: tuple[FactorHypothesisTemplate, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "template_id",
                "lane",
                "status",
                "priority_score",
                "input_features",
                "transformation",
                "target_label",
                "action_mapping",
                "seed_opportunities",
                "validation_check",
                "failure_mode",
                "research_reference",
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
                    row.input_features,
                    row.transformation,
                    row.target_label,
                    row.action_mapping,
                    row.seed_opportunities,
                    row.validation_check,
                    row.failure_mode,
                    row.research_reference,
                    row.next_step,
                )
            )
    return output_path


def write_templates_md(rows: tuple[FactorHypothesisTemplate, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Factor Hypothesis Templates\n\n")
        handle.write(
            "These are LLM-assisted factor hypotheses generated from the current alpha stack. "
            "They are validation templates, not trade instructions and not library abstractions.\n\n"
        )
        handle.write(
            "| template | lane | status | score | inputs | transform | target | seeds | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.template_id} | "
                f"{_escape(row.lane)} | "
                f"{row.status} | "
                f"{row.priority_score:.4f} | "
                f"{_escape(row.input_features)} | "
                f"{_escape(row.transformation)} | "
                f"{_escape(row.target_label)} | "
                f"{_escape(row.seed_opportunities)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _build_template(
    rule: TemplateRule,
    *,
    stack_rows: tuple[dict[str, str], ...],
) -> FactorHypothesisTemplate:
    matching = _matching_stack_rows(rule, stack_rows)
    seed_opportunities = ";".join(row.get("opportunity", "") for row in matching[:5])
    best_stack = max((_float(row.get("priority_score")) for row in matching), default=0.0)
    source_bonus = min(len(matching), 8) * 0.75
    score = rule.base_score + source_bonus + best_stack * 0.04
    if not matching:
        seed_opportunities = "no_current_seed"
        score = rule.base_score
    return FactorHypothesisTemplate(
        template_id=rule.template_id,
        lane=rule.lane,
        status=rule.status,
        priority_score=score,
        input_features=rule.input_features,
        transformation=rule.transformation,
        target_label=rule.target_label,
        action_mapping=rule.action_mapping,
        seed_opportunities=seed_opportunities,
        validation_check=rule.validation_check,
        failure_mode=rule.failure_mode,
        research_reference=rule.research_reference,
        next_step=rule.next_step,
    )


def _matching_stack_rows(rule: TemplateRule, stack_rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    terms = tuple(term.lower() for term in rule.source_terms)
    matches = []
    for row in stack_rows:
        text = " ".join(
            (
                row.get("opportunity", ""),
                row.get("status", ""),
                row.get("side", ""),
                row.get("sources", ""),
                row.get("evidence", ""),
            )
        ).lower()
        if any(term in text for term in terms):
            matches.append(row)
    return tuple(sorted(matches, key=lambda row: _float(row.get("priority_score")), reverse=True))


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
    rows = build_factor_hypothesis_templates(args.root)
    write_templates_csv(
        rows,
        output_path=args.root / "llm_factor_generation" / "current_factor_hypothesis_templates.csv",
    )
    write_templates_md(
        rows,
        output_path=args.root / "llm_factor_generation" / "current_factor_hypothesis_templates.md",
    )
    for row in rows[:10]:
        print(row.template_id, row.status, f"{row.priority_score:.4f}")


if __name__ == "__main__":
    main()
