from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaSourceGap:
    gap_id: str
    lane: str
    status: str
    priority: float
    current_coverage: str
    missing_work: str
    next_probe: str
    research_reference: str


@dataclass(frozen=True)
class GapRule:
    gap_id: str
    lane: str
    probe_categories: tuple[str, ...]
    required_probe_names: tuple[str, ...]
    missing_work: str
    next_probe: str
    research_reference: str
    base_priority: float


GAP_RULES = (
    GapRule(
        gap_id="lob_ofi_hierarchical_model",
        lane="LOB / order-flow representation learning",
        probe_categories=("lob",),
        required_probe_names=("binance_um_book_depth_daily_probe",),
        missing_work="limit-order-book or OFI history, queue/adverse-selection labels, purged walk-forward split",
        next_probe="download one daily bookDepth sample, then build OFI and liquidity-state labels before any model",
        research_reference="https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full",
        base_priority=100.0,
    ),
    GapRule(
        gap_id="rl_execution_policy",
        lane="RL execution policy",
        probe_categories=("lob", "event_flow", "perp_dex", "cross_exchange"),
        required_probe_names=("binance_um_book_depth_daily_probe", "hyperliquid_meta"),
        missing_work="state, action, reward, queue/fill simulator, market-vs-limit choice, and live-cost stress",
        next_probe="define a tiny execution world from book depth plus fills, then compare market/limit/hold actions before training",
        research_reference="https://arxiv.org/abs/2507.06345",
        base_priority=98.0,
    ),
    GapRule(
        gap_id="on_chain_transaction_graph",
        lane="on-chain transaction graph",
        probe_categories=("stablecoin_liquidity", "wallet_entity_flow", "dex_pool_flow", "defi"),
        required_probe_names=("defillama_stablecoins", "geckoterminal_trending_pools"),
        missing_work="entity graph, exchange-deposit labels, transfer timing, address quality, and tradable-asset mapping",
        next_probe="build one timestamped transfer/entity graph sample and label exchange-supply pressure against token returns",
        research_reference="https://arxiv.org/abs/2411.10325",
        base_priority=97.0,
    ),
    GapRule(
        gap_id="options_iv_surface",
        lane="options IV surface and skew",
        probe_categories=("options_volatility",),
        required_probe_names=("deribit_btc_options_summary",),
        missing_work="surface construction, quote freshness, spread/depth, delta hedge path, realized-vol label, and exit bid",
        next_probe="turn Deribit option summaries into BTC/ETH IV surface snapshots and label IV-vs-realized-vol plus hedge PnL",
        research_reference="https://arxiv.org/abs/2407.21138",
        base_priority=95.0,
    ),
    GapRule(
        gap_id="social_sentiment_contagion",
        lane="social / sentiment contagion",
        probe_categories=("news",),
        required_probe_names=(),
        missing_work="source-account graph, timestamped sentiment, duplicate-source filtering, market-impact labels",
        next_probe="turn RSS/news into timestamped event labels, then add explicit social account sources if accessible",
        research_reference="https://www.dallasfed.org/research/papers/2026/wp2605",
        base_priority=96.0,
    ),
    GapRule(
        gap_id="llm_factor_generation",
        lane="LLM-assisted factor generation",
        probe_categories=(),
        required_probe_names=(),
        missing_work="hypothesis journal, generated factor templates, leakage-safe validation, failure-regime log",
        next_probe="generate candidate factor templates from current lanes and route only validated ones into paper labels",
        research_reference="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6461691",
        base_priority=94.0,
    ),
    GapRule(
        gap_id="on_chain_transaction_pressure",
        lane="on-chain transaction pressure",
        probe_categories=("stablecoin_liquidity", "wallet_entity_flow", "defi", "dex_pool_flow"),
        required_probe_names=("defillama_stablecoins", "geckoterminal_trending_pools"),
        missing_work="exchange-supply pressure, usage-vs-trading intent, entity quality, tradable venue mapping",
        next_probe="separate chain/user activity from exchange-supply pressure and label each against token returns",
        research_reference="https://www.sciencedirect.com/science/article/abs/pii/S0261560625001433",
        base_priority=92.0,
    ),
    GapRule(
        gap_id="rl_observation_action_reward_dataset",
        lane="RL-shaped policy dataset",
        probe_categories=(),
        required_probe_names=(),
        missing_work="world state, action space, reward definition, cost/fill model, train/test split",
        next_probe="convert paper samples into an observation/action/reward table before training any policy",
        research_reference="https://arxiv.org/abs/2307.01599",
        base_priority=90.0,
    ),
    GapRule(
        gap_id="cross_modal_event_graph",
        lane="multi-source event graph",
        probe_categories=("news", "wallet_entity_flow", "dex_pool_flow", "stablecoin_liquidity"),
        required_probe_names=(),
        missing_work="event graph linking news, on-chain flow, liquidity, and tradable assets with timestamp controls",
        next_probe="join one news/event source with one on-chain/liquidity source and label per asset, not per headline",
        research_reference="https://link.springer.com/article/10.1007/s41109-026-00778-3",
        base_priority=88.0,
    ),
)


def build_alpha_source_gaps(
    *,
    data_source_probe_path: Path = ROOT / "data_source_probe.csv",
    policy_samples_path: Path = ROOT / "policy_learning" / "current_policy_learning_samples.csv",
    policy_oar_dataset_path: Path = (
        ROOT / "policy_learning" / "current_observation_action_reward_dataset.csv"
    ),
    book_depth_screen_path: Path = ROOT / "event_flow" / "book_depth_imbalance_screen.csv",
    book_depth_walk_forward_path: Path = ROOT / "event_flow" / "book_depth_walk_forward_check.csv",
    book_depth_execution_cost_sweep_path: Path = ROOT / "event_flow" / "book_depth_execution_cost_sweep.csv",
    lob_maker_fill_survival_path: Path = ROOT / "event_flow" / "current_lob_maker_fill_survival.csv",
    news_event_forward_labels_path: Path = ROOT / "news_social" / "current_news_event_forward_labels.csv",
    news_event_quality_gate_path: Path = ROOT / "news_social" / "current_news_event_quality_gate.csv",
    news_event_source_independence_path: Path = ROOT / "news_social" / "current_news_event_source_independence.csv",
    factor_hypothesis_templates_path: Path = (
        ROOT / "llm_factor_generation" / "current_factor_hypothesis_templates.csv"
    ),
    factor_template_validation_queue_path: Path = (
        ROOT / "llm_factor_generation" / "current_factor_template_validation_queue.csv"
    ),
) -> tuple[AlphaSourceGap, ...]:
    probe_rows = _read_rows(data_source_probe_path)
    policy_sample_count = len(_read_rows(policy_samples_path))
    policy_oar_count = len(_read_rows(policy_oar_dataset_path))
    book_depth_screen_rows = _read_rows(book_depth_screen_path)
    book_depth_walk_forward_rows = _read_rows(book_depth_walk_forward_path)
    book_depth_execution_cost_sweep_rows = _read_rows(book_depth_execution_cost_sweep_path)
    lob_maker_fill_survival_rows = _read_rows(lob_maker_fill_survival_path)
    news_event_forward_label_rows = _read_rows(news_event_forward_labels_path)
    news_event_quality_gate_rows = _read_rows(news_event_quality_gate_path)
    news_event_source_independence_rows = _read_rows(news_event_source_independence_path)
    factor_hypothesis_template_rows = _read_rows(factor_hypothesis_templates_path)
    factor_template_validation_queue_rows = _read_rows(factor_template_validation_queue_path)
    rows = tuple(
        _build_gap(
            rule,
            probe_rows=probe_rows,
            policy_sample_count=policy_sample_count,
            policy_oar_count=policy_oar_count,
            book_depth_screen_rows=book_depth_screen_rows,
            book_depth_walk_forward_rows=book_depth_walk_forward_rows,
            book_depth_execution_cost_sweep_rows=book_depth_execution_cost_sweep_rows,
            lob_maker_fill_survival_rows=lob_maker_fill_survival_rows,
            news_event_forward_label_rows=news_event_forward_label_rows,
            news_event_quality_gate_rows=news_event_quality_gate_rows,
            news_event_source_independence_rows=news_event_source_independence_rows,
            factor_hypothesis_template_rows=factor_hypothesis_template_rows,
            factor_template_validation_queue_rows=factor_template_validation_queue_rows,
        )
        for rule in GAP_RULES
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_alpha_source_gaps_csv(rows: tuple[AlphaSourceGap, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "gap_id",
                "lane",
                "status",
                "priority",
                "current_coverage",
                "missing_work",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.gap_id,
                    row.lane,
                    row.status,
                    f"{row.priority:.8f}",
                    row.current_coverage,
                    row.missing_work,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_alpha_source_gaps_md(rows: tuple[AlphaSourceGap, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Source Gaps\n\n")
        handle.write(
            "This board tracks broad external alpha-source gaps. "
            "It is a research queue, not a trade list or a strategy abstraction.\n\n"
        )
        handle.write(
            "| gap | lane | status | priority | coverage | missing work | next probe | reference |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.gap_id} | "
                f"{row.lane} | "
                f"{row.status} | "
                f"{row.priority:.4f} | "
                f"{_escape(row.current_coverage)} | "
                f"{_escape(row.missing_work)} | "
                f"{_escape(row.next_probe)} | "
                f"{row.research_reference} |\n"
            )
    return output_path


def _build_gap(
    rule: GapRule,
    *,
    probe_rows: tuple[dict[str, str], ...],
    policy_sample_count: int,
    policy_oar_count: int,
    book_depth_screen_rows: tuple[dict[str, str], ...],
    book_depth_walk_forward_rows: tuple[dict[str, str], ...],
    book_depth_execution_cost_sweep_rows: tuple[dict[str, str], ...],
    lob_maker_fill_survival_rows: tuple[dict[str, str], ...],
    news_event_forward_label_rows: tuple[dict[str, str], ...],
    news_event_quality_gate_rows: tuple[dict[str, str], ...],
    news_event_source_independence_rows: tuple[dict[str, str], ...],
    factor_hypothesis_template_rows: tuple[dict[str, str], ...],
    factor_template_validation_queue_rows: tuple[dict[str, str], ...],
) -> AlphaSourceGap:
    available = _available_probe_rows(probe_rows, rule)
    missing_required = tuple(name for name in rule.required_probe_names if not _probe_available(probe_rows, name=name))
    if rule.gap_id == "lob_ofi_hierarchical_model" and lob_maker_fill_survival_rows:
        best = max(lob_maker_fill_survival_rows, key=lambda row: _float(row.get("survival_score")))
        status = best.get("survival_status", "maker_fill_checked")
        coverage = (
            f"{best.get('source_probe', '')}/{best.get('feature', '')}/{best.get('bucket', '')}/"
            f"{best.get('signal_action', '')} fill_rate={best.get('fill_rate', '')} "
            f"filled_bps={best.get('filled_mark_reward_bps', '')} "
            f"all_bps={best.get('all_state_reward_bps', '')} "
            f"adverse={best.get('adverse_fill_rate', '')}"
        )
        priority = rule.base_priority + 10.0
        next_probe = best.get(
            "next_step",
            "replace optimistic maker full-fill rows with queue/cancel labels before promotion",
        )
    elif rule.gap_id == "rl_execution_policy" and lob_maker_fill_survival_rows:
        best = max(lob_maker_fill_survival_rows, key=lambda row: _float(row.get("survival_score")))
        status = "maker_fill_proxy_blocks_policy"
        coverage = (
            f"{best.get('source_probe', '')}/{best.get('feature', '')}/{best.get('bucket', '')}/"
            f"{best.get('signal_action', '')} fill_rate={best.get('fill_rate', '')} "
            f"filled_bps={best.get('filled_mark_reward_bps', '')} "
            f"adverse={best.get('adverse_fill_rate', '')}"
        )
        priority = rule.base_priority + 8.0
        next_probe = "add real queue position, cancel-before-cross, partial-fill, and post-fill adverse-selection labels"
    elif rule.gap_id == "lob_ofi_hierarchical_model" and book_depth_execution_cost_sweep_rows:
        best = max(book_depth_execution_cost_sweep_rows, key=lambda row: _float(row.get("viability_score")))
        status = best.get("viability_status", "execution_cost_swept")
        coverage = (
            f"{best.get('feature', '')}/{best.get('bucket', '')}/{best.get('action', '')}/"
            f"{best.get('execution_mode', '')} gross={best.get('test_gross_bps', '')} "
            f"net={best.get('test_net_bps', '')}"
        )
        priority = rule.base_priority + 16.0
        next_probe = "test maker/low-fee execution, queue position, adverse selection, and longer OOS windows"
    elif rule.gap_id == "lob_ofi_hierarchical_model" and book_depth_walk_forward_rows:
        best = max(book_depth_walk_forward_rows, key=lambda row: _float(row.get("test_net_bps")))
        status = best.get("decision", "walk_forward_checked")
        coverage = (
            f"{best.get('feature', '')}/{best.get('bucket', '')}/{best.get('action', '')} "
            f"gross={best.get('test_gross_bps', '')} net={best.get('test_net_bps', '')}"
        )
        priority = rule.base_priority + 16.0
        next_probe = "extend LOB/basis/positioning walk-forward and add liquidation/event timestamps"
    elif rule.gap_id == "social_sentiment_contagion" and news_event_source_independence_rows:
        actionable = tuple(
            row
            for row in news_event_source_independence_rows
            if row.get("independence_status")
            in {
                "independent_multi_source_story",
                "same_story_multi_source_repeat",
                "pending_archive_before_independence",
                "single_source_supported_story",
            }
        )
        best_pool = actionable or news_event_source_independence_rows
        best = max(best_pool, key=lambda row: _float(row.get("score")))
        status = best.get("independence_status", "source_independence_ready")
        coverage = (
            f"{best.get('symbol', '')}/{best.get('event_kind', '')}/{best.get('side', '')} "
            f"sources={best.get('source_count', '')} stories={best.get('unique_story_count', '')} "
            f"dominant={best.get('dominant_story_terms', '')}"
        )
        priority = rule.base_priority + 14.0
        next_probe = "repeat source-independent news labels with execution-cost, beta, and duplicate-story controls"
    elif rule.gap_id == "social_sentiment_contagion" and news_event_quality_gate_rows:
        actionable = tuple(
            row
            for row in news_event_quality_gate_rows
            if row.get("decision")
            in {
                "repeat_supported_multi_source_label",
                "repeat_after_pending_archive",
                "repeat_single_source_label",
                "watch_1h_only_news_label",
            }
        )
        best_pool = actionable or news_event_quality_gate_rows
        best = max(best_pool, key=lambda row: _float(row.get("score")))
        status = best.get("decision", "quality_gate_ready")
        coverage = (
            f"{best.get('symbol', '')}/{best.get('event_kind', '')}/{best.get('side', '')} "
            f"sources={best.get('source_count', '')} support={best.get('supported_count', '')} "
            f"reject={best.get('rejected_count', '')}"
        )
        priority = rule.base_priority + 14.0
        next_probe = "repeat the strongest gated news-event label with execution-cost and duplicate-source review"
    elif rule.gap_id == "social_sentiment_contagion" and news_event_forward_label_rows:
        supported = tuple(
            row
            for row in news_event_forward_label_rows
            if row.get("label_status") in {"direction_supported_1h_4h", "direction_supported_1h_only"}
        )
        best_pool = supported or news_event_forward_label_rows
        best = max(best_pool, key=lambda row: _float(row.get("directional_4h_bps") or row.get("directional_1h_bps")))
        status = best.get("label_status", "forward_label_ready")
        coverage = (
            f"{best.get('symbol', '')}/{best.get('event_kind', '')}/{best.get('side', '')} "
            f"dir1h={best.get('directional_1h_bps', '')} dir4h={best.get('directional_4h_bps', '')}"
        )
        priority = rule.base_priority + 12.0
        next_probe = "repeat news-event labels with duplicate-source, stale-headline, and execution-cost checks"
    elif rule.gap_id == "llm_factor_generation" and factor_template_validation_queue_rows:
        best = max(factor_template_validation_queue_rows, key=lambda row: _float(row.get("priority_score")))
        status = "templates_routed_to_validation"
        coverage = (
            f"routes={len(factor_template_validation_queue_rows)} "
            f"top={best.get('template_id', '')} artifact_status={best.get('current_status', '')}"
        )
        priority = rule.base_priority + 15.0
        next_probe = "execute the top validation routes and reject templates that only restate existing weak artifacts"
    elif rule.gap_id == "llm_factor_generation" and factor_hypothesis_template_rows:
        best = max(factor_hypothesis_template_rows, key=lambda row: _float(row.get("priority_score")))
        status = "templates_generated"
        coverage = (
            f"templates={len(factor_hypothesis_template_rows)} "
            f"top={best.get('template_id', '')} score={best.get('priority_score', '')}"
        )
        priority = rule.base_priority + 13.0
        next_probe = "route the top generated templates into concrete data labels and reject duplicate formula variants"
    elif rule.gap_id == "lob_ofi_hierarchical_model" and book_depth_screen_rows:
        best = max(book_depth_screen_rows, key=lambda row: _float(row.get("mean_next_return")))
        status = "feature_screen_ready"
        coverage = (
            f"{best.get('feature', '')}/{best.get('bucket', '')} "
            f"mean={best.get('mean_next_return', '')} hit={best.get('hit_rate', '')}"
        )
        priority = rule.base_priority + 14.0
        next_probe = "run purged walk-forward on LOB/basis/positioning features with explicit costs"
    elif rule.gap_id == "rl_observation_action_reward_dataset" and policy_oar_count > 0:
        status = "observation_action_reward_dataset_ready"
        coverage = f"oar_records={policy_oar_count}; policy_samples={policy_sample_count}"
        priority = rule.base_priority + min(policy_oar_count * 0.05, 12.0)
        next_probe = "audit observation state, action constraints, cost/fill fields, and repeat split before training"
    elif rule.gap_id == "rl_observation_action_reward_dataset" and policy_sample_count > 0:
        status = "sample_records_exist"
        coverage = f"policy_samples={policy_sample_count}"
        priority = rule.base_priority + min(policy_sample_count * 0.1, 10.0)
        next_probe = rule.next_probe
    elif missing_required:
        status = "data_path_missing"
        coverage = "missing_required=" + ",".join(missing_required)
        priority = rule.base_priority + 12.0
        next_probe = rule.next_probe
    elif available:
        status = "data_path_available"
        coverage = ", ".join(row.get("name", "") for row in available[:4])
        priority = rule.base_priority + min(len(available) * 1.5, 8.0)
        next_probe = rule.next_probe
    else:
        status = "not_started"
        coverage = "no current probe"
        priority = rule.base_priority + 6.0
        next_probe = rule.next_probe
    return AlphaSourceGap(
        gap_id=rule.gap_id,
        lane=rule.lane,
        status=status,
        priority=priority,
        current_coverage=coverage,
        missing_work=rule.missing_work,
        next_probe=next_probe,
        research_reference=rule.research_reference,
    )


def _available_probe_rows(probe_rows: tuple[dict[str, str], ...], rule: GapRule) -> tuple[dict[str, str], ...]:
    categories = set(rule.probe_categories)
    return tuple(row for row in probe_rows if row.get("category") in categories and row.get("available") == "True")


def _probe_available(probe_rows: tuple[dict[str, str], ...], *, name: str) -> bool:
    return any(row.get("name") == name and row.get("available") == "True" for row in probe_rows)


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source-probe-path", type=Path, default=ROOT / "data_source_probe.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_source_gaps.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_source_gaps.md")
    args = parser.parse_args()

    rows = build_alpha_source_gaps(data_source_probe_path=args.data_source_probe_path)
    write_alpha_source_gaps_csv(rows, output_path=args.output_path)
    write_alpha_source_gaps_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.gap_id, row.status, f"{row.priority:.4f}", row.next_probe)


if __name__ == "__main__":
    main()
