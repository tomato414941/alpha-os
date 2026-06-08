from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ResearchBackedAlphaExpansion:
    expansion_id: str
    family: str
    external_evidence: str
    current_coverage: str
    status: str
    priority: float
    target_assets: str
    missing_data: str
    first_probe: str
    current_files: str
    research_reference: str


@dataclass(frozen=True)
class ExpansionRule:
    expansion_id: str
    family: str
    external_evidence: str
    coverage_tokens: tuple[str, ...]
    target_assets: tuple[str, ...]
    missing_data: str
    first_probe: str
    research_reference: str
    base_priority: float


EXPANSION_RULES = (
    ExpansionRule(
        expansion_id="exchange_stablecoin_inflow_1h",
        family="on-chain exchange-flow timing",
        external_evidence="USDT net exchange inflow has reported 1h return-forecasting power for BTC/ETH",
        coverage_tokens=("stablecoin", "chain", "flow", "migration"),
        target_assets=("BTC", "ETH", "SOL"),
        missing_data="exchange-deposit tagged stablecoin flow, exchange wallet map, and 1h labels by asset",
        first_probe="separate current stablecoin migration into exchange-inflow proxy vs chain-level liquidity proxy",
        research_reference="https://arxiv.org/abs/2411.06327",
        base_priority=116.0,
    ),
    ExpansionRule(
        expansion_id="ticker_attention_development_activity",
        family="social attention source quality",
        external_evidence="ticker-level social attention can differ from broad crypto-channel chatter and may proxy future activity",
        coverage_tokens=("attention", "event", "news", "social"),
        target_assets=("HYPE", "SOL", "ZEC", "ETH"),
        missing_data="source-account identity, ticker-vs-channel split, dev-activity follow-up, and duplicate-source controls",
        first_probe="split current attention candidates into ticker-specific vs general-channel sources before paper labels",
        research_reference="https://www.sciencedirect.com/science/article/abs/pii/S0378426625001384",
        base_priority=112.0,
    ),
    ExpansionRule(
        expansion_id="fundamental_sentiment_cross_section_sort",
        family="cross-sectional crypto sorting",
        external_evidence="fundamental and sentiment indicators are reported to improve cryptocurrency sorting predictability",
        coverage_tokens=("protocol", "fee", "sentiment", "attention", "sector"),
        target_assets=("HYPE", "UNI", "AAVE", "CRV", "ZEC"),
        missing_data="cross-sectional feature table, neutral universe, leakage-safe rebalance date, and transaction-cost model",
        first_probe="build one cross-sectional rank table from fee growth, sector rotation, attention, and funding support",
        research_reference="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5079335",
        base_priority=109.0,
    ),
    ExpansionRule(
        expansion_id="hidden_factor_crypto_equity_link",
        family="latent factor and equity linkage",
        external_evidence="recent work reports crypto expected returns loading on crypto-specific and selected equity-industry factors",
        coverage_tokens=("equity", "proxy", "macro", "btc", "eth"),
        target_assets=("BTC", "ETH", "SOL"),
        missing_data="equity factor time series, crypto beta attribution, market-hours gap handling, and residual label",
        first_probe="separate crypto-equity proxy into beta hedge, residual reversal, and market-hours gap labels",
        research_reference="https://arxiv.org/abs/2601.07664",
        base_priority=105.0,
    ),
    ExpansionRule(
        expansion_id="cross_asset_tail_connectedness_regime",
        family="tail regime and connectedness",
        external_evidence="crypto tail connectedness and systemic dependence can vary across asset groups and regimes",
        coverage_tokens=("stress", "anomaly", "sector", "regime", "beta"),
        target_assets=("BTC", "ETH", "SOL", "HYPE", "ZEC"),
        missing_data="rolling tail dependence, regime labels, cross-asset network edges, and stress-conditioned action labels",
        first_probe="turn current anomaly/stress and sector rows into a tail-regime label before directional paper action",
        research_reference="https://link.springer.com/article/10.1186/s40854-025-00831-7",
        base_priority=102.0,
    ),
    ExpansionRule(
        expansion_id="multimodal_nlp_onchain_market_fusion",
        family="multimodal forecasting",
        external_evidence="NLP/news, social, blockchain, and market features are commonly fused for BTC/ETH forecasting",
        coverage_tokens=("news", "attention", "on_chain", "wallet", "stablecoin", "market"),
        target_assets=("BTC", "ETH"),
        missing_data="aligned feature timestamps, source freshness, feature ablation, and beta-adjusted target",
        first_probe="create one BTC/ETH aligned feature row joining news, attention, stablecoin, wallet, funding, and return label",
        research_reference="https://www.sciencedirect.com/science/article/pii/S0169207025000147",
        base_priority=100.0,
    ),
    ExpansionRule(
        expansion_id="sentiment_contagion_negative_control",
        family="social contagion control",
        external_evidence="peer sentiment can move beliefs even when it does not predict future Bitcoin prices",
        coverage_tokens=("attention", "news", "event", "social"),
        target_assets=("BTC", "ETH", "HYPE"),
        missing_data="social graph, source influence, negative-control labels, and belief-vs-return separation",
        first_probe="add a negative-control outcome to attention/event lanes so social contagion is not mistaken for alpha",
        research_reference="https://www.dallasfed.org/research/papers/2026/wp2605",
        base_priority=98.0,
    ),
    ExpansionRule(
        expansion_id="ml_cross_section_cost_survival",
        family="cost-aware ML cross-section",
        external_evidence="machine-learning crypto strategies can remain profitable after costs but rely on cost-aware features",
        coverage_tokens=("policy", "reward", "cost", "fill", "cross_section"),
        target_assets=("HYPE", "SOL", "ETH", "ZEC", "SUI"),
        missing_data="broad cross-section panel, turnover model, cost survival label, and purged train/test split",
        first_probe="convert current cost-adjusted clusters into a cost-survival cross-section table by asset and lane",
        research_reference="https://doi.org/10.1016/j.irfa.2024.103244",
        base_priority=96.0,
    ),
)


def build_research_backed_alpha_expansion_plan(
    *,
    alpha_stack_path: Path = ROOT / "current_alpha_stack.csv",
    alpha_method_frontier_path: Path = ROOT / "current_alpha_method_frontier.csv",
    alpha_source_gaps_path: Path = ROOT / "current_alpha_source_gaps.csv",
    split_lane_plan_path: Path = ROOT / "current_split_first_cluster_lane_plan.csv",
) -> tuple[ResearchBackedAlphaExpansion, ...]:
    context_rows = (
        _read_rows(alpha_stack_path)
        + _read_rows(alpha_method_frontier_path)
        + _read_rows(alpha_source_gaps_path)
        + _read_rows(split_lane_plan_path)
    )
    rows = tuple(_build_row(rule, context_rows=context_rows) for rule in EXPANSION_RULES)
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_research_backed_alpha_expansion_plan_csv(
    rows: tuple[ResearchBackedAlphaExpansion, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "expansion_id",
                "family",
                "external_evidence",
                "current_coverage",
                "status",
                "priority",
                "target_assets",
                "missing_data",
                "first_probe",
                "current_files",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.expansion_id,
                    row.family,
                    row.external_evidence,
                    row.current_coverage,
                    row.status,
                    f"{row.priority:.8f}",
                    row.target_assets,
                    row.missing_data,
                    row.first_probe,
                    row.current_files,
                    row.research_reference,
                )
            )
    return output_path


def write_research_backed_alpha_expansion_plan_md(
    rows: tuple[ResearchBackedAlphaExpansion, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Research Backed Alpha Expansion Plan\n\n")
        handle.write(
            "This maps recent external crypto-alpha research directions into concrete alpha-os probes. "
            "It is not a claim that the papers' results already hold in the current data.\n\n"
        )
        handle.write("| expansion | family | status | priority | targets | current coverage | first probe | reference |\n")
        handle.write("| --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.expansion_id} | "
                f"{row.family} | "
                f"{row.status} | "
                f"{row.priority:.4f} | "
                f"{row.target_assets} | "
                f"{_escape(row.current_coverage)} | "
                f"{_escape(row.first_probe)} | "
                f"{row.research_reference} |\n"
            )
    return output_path


def _build_row(
    rule: ExpansionRule,
    *,
    context_rows: tuple[dict[str, str], ...],
) -> ResearchBackedAlphaExpansion:
    matching_rows = tuple(row for row in context_rows if _row_matches(row, tokens=rule.coverage_tokens))
    coverage_score = min(len(matching_rows) * 3.0, 36.0)
    executable_bonus = 18.0 if matching_rows else 0.0
    priority = rule.base_priority + coverage_score + executable_bonus
    status = _status(matching_rows)
    return ResearchBackedAlphaExpansion(
        expansion_id=rule.expansion_id,
        family=rule.family,
        external_evidence=rule.external_evidence,
        current_coverage=_coverage_text(matching_rows),
        status=status,
        priority=priority,
        target_assets=", ".join(rule.target_assets),
        missing_data=rule.missing_data,
        first_probe=rule.first_probe,
        current_files=_current_files(matching_rows),
        research_reference=rule.research_reference,
    )


def _status(rows: tuple[dict[str, str], ...]) -> str:
    if len(rows) >= 8:
        return "build_probe_from_existing_coverage"
    if rows:
        return "connect_existing_partial_coverage"
    return "new_data_source_required"


def _coverage_text(rows: tuple[dict[str, str], ...]) -> str:
    if not rows:
        return "no direct current coverage"
    labels = []
    for row in rows[:5]:
        labels.append(
            row.get("opportunity")
            or row.get("method_id")
            or row.get("gap_id")
            or row.get("lane_opportunity")
            or row.get("lane")
            or row.get("expansion_id")
            or "covered_row"
        )
    return f"{len(rows)} matching rows: {', '.join(labels)}"


def _current_files(rows: tuple[dict[str, str], ...]) -> str:
    if not rows:
        return ""
    # The row dictionaries do not retain source paths; this lists the generated sources used by this plan.
    return (
        "current_alpha_stack.csv, current_alpha_method_frontier.csv, "
        "current_alpha_source_gaps.csv, current_split_first_cluster_lane_plan.csv"
    )


def _row_matches(row: dict[str, str], *, tokens: tuple[str, ...]) -> bool:
    haystack = " ".join(str(value).lower() for value in row.values())
    return any(token.lower() in haystack for token in tokens)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_research_backed_alpha_expansion_plan.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_research_backed_alpha_expansion_plan.md")
    args = parser.parse_args()

    rows = build_research_backed_alpha_expansion_plan()
    write_research_backed_alpha_expansion_plan_csv(rows, output_path=args.output_path)
    write_research_backed_alpha_expansion_plan_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.status, row.expansion_id, f"{row.priority:.4f}")


if __name__ == "__main__":
    main()
