from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FrontierLane:
    lane: str
    current_status: str
    frontier_score: float
    active_candidates: int
    best_score: float
    best_opportunity: str
    evidence_sources: str
    missing_work: str
    next_probe: str


@dataclass(frozen=True)
class LaneRule:
    lane: str
    tokens: tuple[str, ...]
    missing_work: str
    next_probe: str
    base_priority: float


LANE_RULES = (
    LaneRule(
        lane="repeat-surviving microstructure / flow",
        tokens=("microstructure", "l2_imbalance", "on_chain_flow", "liquidation"),
        missing_work="repeat survival, fill model, queue/adverse-selection evidence, and funding PnL",
        next_probe="expand NEAR/SOL/SUI repeat winners across fresh snapshots, venues, and failure regimes",
        base_priority=100.0,
    ),
    LaneRule(
        lane="derivatives positioning and crowding",
        tokens=("derivatives", "funding", "open_interest", "basis", "crowding", "perp", "positioning", "premium"),
        missing_work="venue-specific depth, funding timestamp, margin, and repeated forward labels",
        next_probe="separate crowding continuation from squeeze/reversal labels and require repeat survival",
        base_priority=98.0,
    ),
    LaneRule(
        lane="liquidation cascade",
        tokens=("liquidation",),
        missing_work="true liquidation events, OI context, book depth, stop behavior, and post-event labels",
        next_probe="label liquidation intensity candidates by venue and direction, then retest after costs",
        base_priority=97.0,
    ),
    LaneRule(
        lane="protocol economics",
        tokens=("protocol_fee", "revenue", "valuation", "protocol_fundamentals"),
        missing_work="token value capture, lag structure, crowding, and tradable venue mapping",
        next_probe="label fee/revenue acceleration against token returns and relative-value alternatives",
        base_priority=95.0,
    ),
    LaneRule(
        lane="DeFi yield / borrow / carry",
        tokens=("yield", "lending", "borrow", "defi"),
        missing_work="withdrawal path, protocol risk, gas, peg risk, capacity, and unwind mechanics",
        next_probe="separate real carry from protocol/redemption risk and test small executable routes",
        base_priority=94.0,
    ),
    LaneRule(
        lane="stablecoin and bridge liquidity migration",
        tokens=("stablecoin", "peg", "migration", "bridge"),
        missing_work="chain-token mapping, bridge flow, peg liquidity, redemption path, and beta attribution",
        next_probe="label chain stablecoin flow changes against chain-token beta and peg stress outcomes",
        base_priority=93.0,
    ),
    LaneRule(
        lane="token unlock / supply event",
        tokens=("unlock", "emission"),
        missing_work="float, borrow/perp access, hedge demand, event-window labels, and crowding",
        next_probe="turn unlock watches into dated event-window labels, not generic supply-pressure shorts",
        base_priority=92.0,
    ),
    LaneRule(
        lane="news / attention / social impulse",
        tokens=("news", "attention", "social", "catalyst"),
        missing_work="timestamp quality, duplicate-source filtering, causality, and stale-headline rejection",
        next_probe="label attention/news impulses over 1h/4h/24h with duplicate-source and crowding checks",
        base_priority=91.0,
    ),
    LaneRule(
        lane="options volatility",
        tokens=("options", "volatility", "straddle", "skew"),
        missing_work="quote freshness, spread/depth, hedging path, premium-at-risk, and realized-vol labels",
        next_probe="promote only quoteable structures with explicit max loss, hedge schedule, and exit bid",
        base_priority=90.0,
    ),
    LaneRule(
        lane="prediction-market signal",
        tokens=("prediction_market", "probability", "polymarket", "kalshi"),
        missing_work="resolution risk, fill/queue risk, market manipulation, and event-to-trade mapping",
        next_probe="separate pure prediction-market trades from crypto hedges informed by event odds",
        base_priority=89.0,
    ),
    LaneRule(
        lane="macro / institutional flow",
        tokens=("macro", "etf", "institutional", "risk_off", "credit", "vix"),
        missing_work="timely flow data, regime labels, hedge mapping, and crypto beta decomposition",
        next_probe="join ETF/funding/macro risk-off windows to BTC/ETH and high-beta token labels",
        base_priority=88.0,
    ),
    LaneRule(
        lane="directional ML / RL policy learning",
        tokens=("ml_policy", "rl_policy", "model_policy"),
        missing_work="dataset/world definition, reward, costs, simulator limits, and out-of-sample protocol",
        next_probe="start with a tiny supervised/RL-shaped dataset over existing candidate states and actions",
        base_priority=87.0,
    ),
    LaneRule(
        lane="execution edge",
        tokens=("execution_edge", "maker", "taker", "queue", "routing", "latency"),
        missing_work="order path, maker/taker fill probability, queue position, latency, and fee tier",
        next_probe="measure whether maker/taker choice or order slicing changes realized edge on repeat winners",
        base_priority=86.0,
    ),
    LaneRule(
        lane="wallet / entity on-chain flow",
        tokens=("wallet", "entity_flow", "exchange_inflow", "exchange_outflow"),
        missing_work="wallet/entity data source, exchange mapping, time alignment, and false-positive filtering",
        next_probe="add one concrete wallet/entity-flow source or mark this lane as blocked by data access",
        base_priority=85.0,
    ),
)


def build_alpha_frontier(
    *,
    alpha_stack_path: Path = ROOT / "current_alpha_stack.csv",
) -> tuple[FrontierLane, ...]:
    alpha_rows = _read_rows(alpha_stack_path)
    lanes = tuple(_build_lane(rule, alpha_rows=alpha_rows) for rule in LANE_RULES)
    return tuple(sorted(lanes, key=lambda row: row.frontier_score, reverse=True))


def write_alpha_frontier_csv(rows: tuple[FrontierLane, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "lane",
                "current_status",
                "frontier_score",
                "active_candidates",
                "best_score",
                "best_opportunity",
                "evidence_sources",
                "missing_work",
                "next_probe",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.lane,
                    row.current_status,
                    f"{row.frontier_score:.8f}",
                    row.active_candidates,
                    f"{row.best_score:.8f}",
                    row.best_opportunity,
                    row.evidence_sources,
                    row.missing_work,
                    row.next_probe,
                )
            )
    return output_path


def write_alpha_frontier_md(rows: tuple[FrontierLane, ...], *, output_path: Path, top: int = 40) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Frontier\n\n")
        handle.write(
            "This frontier keeps alpha discovery broad. It shows which profit-source lanes already "
            "have active candidates and which lanes are still missing concrete probes. "
            "It is not a trade list.\n\n"
        )
        handle.write(
            "| lane | status | frontier score | active | best score | best opportunity | sources | missing work | next probe |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.lane} | "
                f"{row.current_status} | "
                f"{row.frontier_score:.4f} | "
                f"{row.active_candidates} | "
                f"{row.best_score:.4f} | "
                f"{_escape(row.best_opportunity)} | "
                f"{_escape(row.evidence_sources)} | "
                f"{_escape(row.missing_work)} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _build_lane(rule: LaneRule, *, alpha_rows: tuple[dict[str, str], ...]) -> FrontierLane:
    if rule.lane == "wallet / entity on-chain flow":
        access_rows = _read_rows(ROOT / "wallet_entity_flow" / "current_wallet_entity_flow_access.csv")
        flow_rows = _read_rows(ROOT / "wallet_entity_flow" / "current_hyperliquid_seed_wallet_flow.csv")
        if flow_rows:
            best = max(flow_rows, key=lambda row: _float(row.get("score")))
            active_candidates = len(flow_rows)
            return FrontierLane(
                lane=rule.lane,
                current_status="seed_flow_probe_ready",
                frontier_score=rule.base_priority + 10.0 + min(active_candidates * 0.4, 10.0),
                active_candidates=active_candidates,
                best_score=_float(best.get("score")),
                best_opportunity=best.get("coin", ""),
                evidence_sources="wallet_entity_flow/current_hyperliquid_seed_wallet_flow",
                missing_work=rule.missing_work,
                next_probe=rule.next_probe,
            )
        if access_rows:
            access_ok = tuple(row for row in access_rows if row.get("status") in {"access_ok", "implemented_proxy"})
            best = access_ok[0] if access_ok else access_rows[0]
            active_candidates = len(access_rows)
            status = "access_probe_ready" if access_ok else "needs_data_access"
            return FrontierLane(
                lane=rule.lane,
                current_status=status,
                frontier_score=rule.base_priority + 8.0 + min(len(access_ok) * 2.0, 8.0),
                active_candidates=active_candidates,
                best_score=float(len(access_ok)),
                best_opportunity=best.get("source", ""),
                evidence_sources="wallet_entity_flow/current_wallet_entity_flow_access",
                missing_work=rule.missing_work,
                next_probe=rule.next_probe,
            )
    if rule.lane == "directional ML / RL policy learning":
        oos_rows = _read_rows(ROOT / "policy_learning" / "current_action_preference_oos_check.csv")
        if oos_rows:
            supported = tuple(
                row
                for row in oos_rows
                if row.get("decision")
                in {"oos_supported_action_preference", "mixed_oos_action_preference"}
            )
            best_pool = supported or oos_rows
            best = max(best_pool, key=lambda row: _float(row.get("oos_score")))
            best_score = _float(best.get("oos_score"))
            active_candidates = len(supported) or len(oos_rows)
            return FrontierLane(
                lane=rule.lane,
                current_status=best.get("decision", "action_preference_oos_ready"),
                frontier_score=rule.base_priority
                + min(active_candidates * 1.0, 12.0)
                + min(best_score / 10.0, 12.0),
                active_candidates=active_candidates,
                best_score=best_score,
                best_opportunity=best.get("candidate_id", ""),
                evidence_sources="policy_learning/current_action_preference_oos_check",
                missing_work=rule.missing_work,
                next_probe="paper-check OOS-supported action preferences with explicit fill and stop rules",
            )
        preference_rows = _read_rows(ROOT / "policy_learning" / "current_action_preference_candidates.csv")
        if preference_rows:
            best = max(preference_rows, key=lambda row: _float(row.get("score")))
            active = tuple(row for row in preference_rows if row.get("decision") != "collect_more_labels")
            active_candidates = len(active) or len(preference_rows)
            best_score = _float(best.get("score"))
            return FrontierLane(
                lane=rule.lane,
                current_status=best.get("decision", "action_preferences_ready"),
                frontier_score=rule.base_priority + min(active_candidates * 0.5, 12.0) + min(best_score / 10.0, 12.0),
                active_candidates=active_candidates,
                best_score=best_score,
                best_opportunity=best.get("candidate_id", ""),
                evidence_sources="policy_learning/current_action_preference_candidates",
                missing_work=rule.missing_work,
                next_probe="turn the strongest action preference into a leakage-safe split before training",
            )
        policy_rows = _read_rows(ROOT / "policy_learning" / "current_policy_learning_samples.csv")
        if policy_rows:
            best = max(policy_rows, key=_policy_sample_sort_key)
            best_score = _policy_sample_score(best)
            active_candidates = len(policy_rows)
            return FrontierLane(
                lane=rule.lane,
                current_status="sample_dataset_ready",
                frontier_score=rule.base_priority + min(active_candidates * 0.2, 12.0) + min(best_score / 10.0, 12.0),
                active_candidates=active_candidates,
                best_score=best_score,
                best_opportunity=best.get("opportunity", ""),
                evidence_sources="policy_learning/current_policy_learning_samples",
                missing_work=rule.missing_work,
                next_probe=rule.next_probe,
            )
    matches = tuple(row for row in alpha_rows if _matches_rule(row, rule))
    best = max(matches, key=lambda row: _float(row.get("priority_score")), default={})
    best_score = _float(best.get("priority_score"))
    active_candidates = len(matches)
    current_status = _current_status(active_candidates)
    frontier_score = rule.base_priority + min(active_candidates * 1.5, 12.0) + min(best_score / 10.0, 12.0)
    if active_candidates == 0:
        frontier_score += 10.0
    return FrontierLane(
        lane=rule.lane,
        current_status=current_status,
        frontier_score=frontier_score,
        active_candidates=active_candidates,
        best_score=best_score,
        best_opportunity=best.get("opportunity", ""),
        evidence_sources=best.get("sources", ""),
        missing_work=rule.missing_work,
        next_probe=rule.next_probe,
    )


def _matches_rule(row: dict[str, str], rule: LaneRule) -> bool:
    haystack = " ".join(
        (
            row.get("opportunity", ""),
            row.get("status", ""),
            row.get("side", ""),
            row.get("sources", ""),
        )
    ).lower()
    return any(token.lower() in haystack for token in rule.tokens)


def _current_status(active_candidates: int) -> str:
    if active_candidates == 0:
        return "missing_concrete_probe"
    if active_candidates < 3:
        return "thin_probe"
    if active_candidates < 10:
        return "active_probe"
    return "crowded_probe"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _policy_sample_score(row: dict[str, str]) -> float:
    return _float(row.get("cost_adjusted_reward_bps") or row.get("reward_bps"))


def _policy_sample_sort_key(row: dict[str, str]) -> tuple[int, float]:
    status_rank = {
        "cost_adjusted_win": 5,
        "mark_win_without_cost": 4,
        "cost_adjusted_edge_failed": 3,
        "depth_too_thin_for_probe": 2,
        "mark_loss": 1,
    }.get(row.get("reward_status", ""), 0)
    return (status_rank, _policy_sample_score(row))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_frontier.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_frontier.md")
    args = parser.parse_args()
    rows = build_alpha_frontier(alpha_stack_path=args.alpha_stack_path)
    write_alpha_frontier_csv(rows, output_path=args.output_path)
    write_alpha_frontier_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(
            row.lane,
            row.current_status,
            f"active={row.active_candidates}",
            f"best={row.best_score:.4f}",
            row.next_probe,
        )


if __name__ == "__main__":
    main()
