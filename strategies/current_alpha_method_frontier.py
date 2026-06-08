from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MethodRule:
    method_id: str
    family: str
    ml_rl_analogy: str
    related_gaps: tuple[str, ...]
    alpha_lane_tokens: tuple[str, ...]
    data_categories: tuple[str, ...]
    missing_link: str
    first_probe: str
    research_reference: str
    base_priority: float


@dataclass(frozen=True)
class AlphaMethodFrontierRow:
    method_id: str
    family: str
    ml_rl_analogy: str
    decision: str
    score: float
    source_gap_evidence: str
    alpha_lane_evidence: str
    data_evidence: str
    missing_link: str
    first_probe: str
    research_reference: str


METHOD_RULES = (
    MethodRule(
        method_id="rl_lob_execution_world",
        family="execution learning",
        ml_rl_analogy="RL environment over book state; actions are market, limit, hold, or cancel",
        related_gaps=("rl_execution_policy", "lob_ofi_hierarchical_model"),
        alpha_lane_tokens=("execution", "microstructure", "flow"),
        data_categories=("lob", "event_flow", "perp_dex", "cross_exchange"),
        missing_link="real queue position, cancel-before-cross rule, partial fills, and post-fill adverse-selection horizon",
        first_probe="replace optimistic maker full-fill rows with queue/cancel labels before any LOB policy promotion",
        research_reference="https://arxiv.org/abs/2507.06345",
        base_priority=118.0,
    ),
    MethodRule(
        method_id="group_aware_lob_policy_gradient",
        family="execution learning",
        ml_rl_analogy="policy-gradient learner over order-flow state with downside-aware reward shaping",
        related_gaps=("rl_execution_policy", "lob_ofi_hierarchical_model", "rl_observation_action_reward_dataset"),
        alpha_lane_tokens=("execution", "microstructure", "flow", "policy"),
        data_categories=("lob", "event_flow", "perp_dex", "cross_exchange"),
        missing_link="grouped reward baseline, downside-aware reward, maker/taker action labels, and leakage-safe LOB split",
        first_probe="convert OFI short-horizon tickets into grouped OAR rows before trying PPO or GRPO-style policy updates",
        research_reference="https://arxiv.org/abs/2605.25527",
        base_priority=119.0,
    ),
    MethodRule(
        method_id="ofi_sequence_model",
        family="order-flow representation",
        ml_rl_analogy="supervised sequence model over OFI, depth, spread, and trade imbalance",
        related_gaps=("lob_ofi_hierarchical_model",),
        alpha_lane_tokens=("microstructure", "execution", "event_flow"),
        data_categories=("lob", "event_flow"),
        missing_link="purged walk-forward labels, fee-aware target, and maker/taker execution split",
        first_probe="turn bookDepth/trades into OFI sequences and compare 15m/1h labels against current L2 imbalance rows",
        research_reference="https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full",
        base_priority=112.0,
    ),
    MethodRule(
        method_id="portable_crypto_microstructure_library",
        family="cross-asset microstructure",
        ml_rl_analogy="shared feature library over order book and trade state before per-asset policy fitting",
        related_gaps=("lob_ofi_hierarchical_model", "rl_execution_policy"),
        alpha_lane_tokens=("microstructure", "execution", "flow", "crowded"),
        data_categories=("lob", "event_flow", "perp_dex"),
        missing_link="cross-asset feature stability test, SHAP or feature-importance audit, taker/maker split, and crash-regime stress",
        first_probe="build one shared BTC/ETH/SOL/HYPE LOB feature table and test whether the same features rank across assets",
        research_reference="https://arxiv.org/abs/2602.00776",
        base_priority=116.0,
    ),
    MethodRule(
        method_id="crypto_pair_execution_overlay",
        family="pair trading execution",
        ml_rl_analogy="RL execution overlay inside deterministic pair-selection and risk boundaries",
        related_gaps=("rl_observation_action_reward_dataset", "cross_modal_event_graph"),
        alpha_lane_tokens=("relative", "pair", "beta", "hedge", "policy", "execution"),
        data_categories=("event_flow", "cross_exchange", "perp_dex", "crypto_equity_proxy"),
        missing_link="pair universe, hedge ratio, divergence stop, funding-aware reward, and deterministic risk shield",
        first_probe="build fresh BTC/ETH/SOL/HYPE pair-spread labels and keep the RL action space limited to execution timing and sizing",
        research_reference="https://arxiv.org/abs/2606.04574",
        base_priority=115.0,
    ),
    MethodRule(
        method_id="on_chain_graph_pressure",
        family="on-chain graph",
        ml_rl_analogy="graph/subgraph representation over wallets, pools, bridges, and exchange-like flow",
        related_gaps=("on_chain_transaction_graph", "on_chain_transaction_pressure"),
        alpha_lane_tokens=("wallet", "on_chain", "stablecoin", "dex", "defi"),
        data_categories=("wallet_entity_flow", "stablecoin_liquidity", "dex_pool_flow", "defi"),
        missing_link="entity quality, exchange-deposit labels, transfer timestamps, and tradable asset mapping",
        first_probe="create one timestamped wallet/pool/chain-flow graph sample and label token pressure separately from beta",
        research_reference="https://www.sciencedirect.com/science/article/pii/S0261560625001433",
        base_priority=110.0,
    ),
    MethodRule(
        method_id="coordinated_manipulation_graph",
        family="market manipulation graph",
        ml_rl_analogy="spatio-temporal graph model over coordinated token moves and repeated flow patterns",
        related_gaps=("on_chain_transaction_graph", "cross_modal_event_graph", "social_sentiment_contagion"),
        alpha_lane_tokens=("anomaly", "stress", "wallet", "social", "event", "flow"),
        data_categories=("wallet_entity_flow", "news", "event_flow", "stablecoin_liquidity"),
        missing_link="token relation graph, hourly synchronized labels, manipulation-negative controls, and tradable exit window",
        first_probe="join pump-like event clusters with wallet/stablecoin/attention features and label fade-vs-follow separately",
        research_reference="https://arxiv.org/abs/2604.24590",
        base_priority=108.0,
    ),
    MethodRule(
        method_id="exchange_flow_supply_pressure",
        family="on-chain exchange pressure",
        ml_rl_analogy="exogenous world-flow feature for token supply pressure, separate from strategy internals",
        related_gaps=("on_chain_transaction_pressure", "on_chain_transaction_graph"),
        alpha_lane_tokens=("stablecoin", "wallet", "on_chain", "flow", "exchange"),
        data_categories=("stablecoin_liquidity", "wallet_entity_flow", "on_chain_flow"),
        missing_link="exchange wallet/entity map, deposit-vs-withdrawal direction, transfer timestamp, and tradable token mapping",
        first_probe="label one exchange-supply-pressure sample separately from chain-level liquidity migration and same-asset beta",
        research_reference="https://www.sciencedirect.com/science/article/pii/S0261560625001433",
        base_priority=111.0,
    ),
    MethodRule(
        method_id="stablecoin_exchange_flow_timing",
        family="stablecoin flow timing",
        ml_rl_analogy="market-environment liquidity state that can condition strategy actions",
        related_gaps=("on_chain_transaction_graph", "on_chain_transaction_pressure", "cross_modal_event_graph"),
        alpha_lane_tokens=("stablecoin", "liquidity", "flow", "exchange", "macro"),
        data_categories=("stablecoin_liquidity", "on_chain_flow", "cross_exchange"),
        missing_link="exchange-tagged stablecoin inflow/outflow, asset mapping, 1h labels, and chain-vs-exchange separation",
        first_probe="build one exchange-tagged stablecoin flow proxy for BTC/ETH/SOL and compare it against chain supply migration",
        research_reference="https://www.elibrary.imf.org/view/journals/001/2026/056/article-A001-en.xml",
        base_priority=109.0,
    ),
    MethodRule(
        method_id="llm_event_agent_triage",
        family="agentic event research",
        ml_rl_analogy="LLM agent as hypothesis router and source-quality judge, not an autonomous trader",
        related_gaps=("llm_factor_generation", "social_sentiment_contagion", "cross_modal_event_graph"),
        alpha_lane_tokens=("news", "attention", "prediction", "social"),
        data_categories=("news",),
        missing_link="duplicate-source controls, timestamp quality, beta attribution, and rejection of stale narratives",
        first_probe="route current RSS/news/event-pressure clusters through a source-quality and beta-attribution checklist",
        research_reference="https://arxiv.org/abs/2510.11695",
        base_priority=106.0,
    ),
    MethodRule(
        method_id="prediction_market_order_flow_skill",
        family="prediction-market microstructure",
        ml_rl_analogy="market-specific skill/leakage detector before using event odds as strategy state",
        related_gaps=("prediction_market_order_flow_skill", "cross_modal_event_graph"),
        alpha_lane_tokens=("prediction", "event_crypto_hedge", "event_probability"),
        data_categories=("prediction_market", "news"),
        missing_link="account-level order-flow history, per-market leakage score, event timestamp, and crypto beta attribution",
        first_probe="split prediction-market quote outcomes from crypto hedge outcomes and require event-alignment evidence before promotion",
        research_reference="https://arxiv.org/abs/2605.02287",
        base_priority=107.0,
    ),
    MethodRule(
        method_id="options_surface_vol_policy",
        family="volatility trading",
        ml_rl_analogy="forecast or policy over IV surface, realized vol, hedge path, and max-loss constraint",
        related_gaps=("options_iv_surface",),
        alpha_lane_tokens=("options", "volatility", "skew"),
        data_categories=("options_volatility",),
        missing_link="surface history, quote freshness, delta-hedge PnL, exit bid, and margin/max-loss handling",
        first_probe="build BTC/ETH IV-vs-realized-vol labels with explicit hedge marks and exit spread",
        research_reference="https://arxiv.org/search/?query=cryptocurrency+options+volatility+surface+trading&searchtype=all",
        base_priority=102.0,
    ),
    MethodRule(
        method_id="oar_policy_learning_baseline",
        family="policy learning",
        ml_rl_analogy="offline observation/action/reward dataset before any policy training",
        related_gaps=("rl_observation_action_reward_dataset",),
        alpha_lane_tokens=("directional", "policy", "rl", "execution"),
        data_categories=("event_flow", "lob", "perp_dex", "cross_exchange"),
        missing_link="state fields, action constraints, reward definition, cost/fill model, and repeat split",
        first_probe="train no model yet; first audit OAR rows for leakage, missing state, and repeat-split coverage",
        research_reference="https://arxiv.org/abs/2307.01599",
        base_priority=101.0,
    ),
    MethodRule(
        method_id="cross_modal_event_graph",
        family="cross-modal event graph",
        ml_rl_analogy="event graph joining news, attention, on-chain flow, liquidity, and tradable actions",
        related_gaps=("cross_modal_event_graph", "social_sentiment_contagion"),
        alpha_lane_tokens=("news", "attention", "stablecoin", "wallet", "dex"),
        data_categories=("news", "stablecoin_liquidity", "dex_pool_flow", "wallet_entity_flow"),
        missing_link="timestamped joins, source independence, non-causal headline rejection, and per-asset labels",
        first_probe="join one current event cluster to stablecoin/dex/wallet context and label per asset instead of per headline",
        research_reference="https://arxiv.org/search/?query=financial+event+graph+trading+large+language+model&searchtype=all",
        base_priority=100.0,
    ),
    MethodRule(
        method_id="crowding_funding_sequence_model",
        family="derivatives positioning",
        ml_rl_analogy="sequence model over OI, funding, premium, liquidations, and returns",
        related_gaps=("rl_observation_action_reward_dataset",),
        alpha_lane_tokens=("derivatives", "funding", "crowding", "liquidation"),
        data_categories=("derivatives_positioning", "event_flow", "cross_exchange", "liquidation_flow"),
        missing_link="multi-venue OI history, funding timestamp, liquidation context, spread/depth, and stop behavior",
        first_probe="extend NEAR/ARB/OP intraday derivatives labels with funding PnL and liquidation intensity context",
        research_reference="https://arxiv.org/search/?query=cryptocurrency+open+interest+funding+rate+prediction&searchtype=all",
        base_priority=99.0,
    ),
)


def build_alpha_method_frontier(
    *,
    source_gaps_path: Path = ROOT / "current_alpha_source_gaps.csv",
    alpha_frontier_path: Path = ROOT / "current_alpha_frontier.csv",
    data_source_probe_path: Path = ROOT / "data_source_probe.csv",
) -> tuple[AlphaMethodFrontierRow, ...]:
    source_gaps = _read_rows(source_gaps_path)
    alpha_lanes = _read_rows(alpha_frontier_path)
    data_sources = _read_rows(data_source_probe_path)
    rows = tuple(
        _build_method_row(
            rule,
            source_gaps=source_gaps,
            alpha_lanes=alpha_lanes,
            data_sources=data_sources,
        )
        for rule in METHOD_RULES
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_alpha_method_frontier_csv(rows: tuple[AlphaMethodFrontierRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "method_id",
                "family",
                "ml_rl_analogy",
                "decision",
                "score",
                "source_gap_evidence",
                "alpha_lane_evidence",
                "data_evidence",
                "missing_link",
                "first_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.method_id,
                    row.family,
                    row.ml_rl_analogy,
                    row.decision,
                    f"{row.score:.8f}",
                    row.source_gap_evidence,
                    row.alpha_lane_evidence,
                    row.data_evidence,
                    row.missing_link,
                    row.first_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_alpha_method_frontier_md(rows: tuple[AlphaMethodFrontierRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Method Frontier\n\n")
        handle.write(
            "This converts modern ML/RL/agentic research directions into concrete alpha-os probes. "
            "It is a method-expansion queue, not a strategy abstraction or a trade list.\n\n"
        )
        handle.write(
            "| method | family | analogy | decision | score | source gap | alpha support | data | missing link | first probe | reference |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.method_id} | "
                f"{row.family} | "
                f"{_escape(row.ml_rl_analogy)} | "
                f"{row.decision} | "
                f"{row.score:.4f} | "
                f"{_escape(row.source_gap_evidence)} | "
                f"{_escape(row.alpha_lane_evidence)} | "
                f"{_escape(row.data_evidence)} | "
                f"{_escape(row.missing_link)} | "
                f"{_escape(row.first_probe)} | "
                f"{row.research_reference} |\n"
            )
    return output_path


def _build_method_row(
    rule: MethodRule,
    *,
    source_gaps: tuple[dict[str, str], ...],
    alpha_lanes: tuple[dict[str, str], ...],
    data_sources: tuple[dict[str, str], ...],
) -> AlphaMethodFrontierRow:
    matching_gaps = tuple(row for row in source_gaps if row.get("gap_id") in rule.related_gaps)
    matching_lanes = tuple(row for row in alpha_lanes if _lane_matches(row, tokens=rule.alpha_lane_tokens))
    matching_sources = tuple(
        row
        for row in data_sources
        if row.get("category") in rule.data_categories and row.get("available") == "True"
    )
    best_gap = _best_numeric(matching_gaps, key="priority")
    best_lane = _best_numeric(matching_lanes, key="frontier_score")
    gap_score = _float(best_gap.get("priority")) if best_gap else 0.0
    lane_score = _float(best_lane.get("frontier_score")) if best_lane else 0.0
    data_score = min(len(matching_sources) * 4.0, 24.0)
    score = rule.base_priority + gap_score * 0.35 + lane_score * 0.25 + data_score
    decision = _decision(score=score, has_data=bool(matching_sources), has_gap=bool(best_gap), has_lane=bool(best_lane))
    return AlphaMethodFrontierRow(
        method_id=rule.method_id,
        family=rule.family,
        ml_rl_analogy=rule.ml_rl_analogy,
        decision=decision,
        score=score,
        source_gap_evidence=_gap_evidence(best_gap),
        alpha_lane_evidence=_lane_evidence(best_lane),
        data_evidence=_data_evidence(matching_sources),
        missing_link=rule.missing_link,
        first_probe=rule.first_probe,
        research_reference=rule.research_reference,
    )


def _decision(*, score: float, has_data: bool, has_gap: bool, has_lane: bool) -> str:
    if has_data and has_gap and has_lane and score >= 170.0:
        return "build_probe_now"
    if has_data and score >= 150.0:
        return "connect_existing_probe"
    if not has_data:
        return "unblock_data_source"
    return "keep_method_backlog"


def _lane_matches(row: dict[str, str], *, tokens: tuple[str, ...]) -> bool:
    text = " ".join(
        (
            row.get("lane", ""),
            row.get("current_status", ""),
            row.get("best_opportunity", ""),
            row.get("evidence_sources", ""),
            row.get("missing_work", ""),
            row.get("next_probe", ""),
        )
    ).lower()
    return any(token.lower() in text for token in tokens)


def _gap_evidence(row: dict[str, str] | None) -> str:
    if not row:
        return "no source-gap row yet"
    return f"{row.get('gap_id', '')}: {row.get('status', '')}; priority={row.get('priority', '')}"


def _lane_evidence(row: dict[str, str] | None) -> str:
    if not row:
        return "no alpha-frontier lane yet"
    return (
        f"{row.get('lane', '')}: status={row.get('current_status', '')}; "
        f"active={row.get('active_candidates', '')}; score={row.get('frontier_score', '')}"
    )


def _data_evidence(rows: tuple[dict[str, str], ...]) -> str:
    if not rows:
        return "no available data source"
    return ", ".join(row.get("name", "") for row in rows[:5])


def _best_numeric(rows: tuple[dict[str, str], ...], *, key: str) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _float(row.get(key)))


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gaps-path", type=Path, default=ROOT / "current_alpha_source_gaps.csv")
    parser.add_argument("--alpha-frontier-path", type=Path, default=ROOT / "current_alpha_frontier.csv")
    parser.add_argument("--data-source-probe-path", type=Path, default=ROOT / "data_source_probe.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_method_frontier.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_method_frontier.md")
    args = parser.parse_args()

    rows = build_alpha_method_frontier(
        source_gaps_path=args.source_gaps_path,
        alpha_frontier_path=args.alpha_frontier_path,
        data_source_probe_path=args.data_source_probe_path,
    )
    write_alpha_method_frontier_csv(rows, output_path=args.output_path)
    write_alpha_method_frontier_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.method_id, row.decision, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
