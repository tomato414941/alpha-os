from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MIN_DIVERSE_PRIORITY_SCORE = 50.0
TARGET_CANDIDATES_PER_PROBE_TYPE = 4
DEFAULT_TOP = 80
POLICY_EXPANSION_STATUSES = {
    "expand_supported_preference_now",
    "collect_expansion_labels",
    "repeat_seed_before_expansion",
    "split_failure_before_expansion",
}


@dataclass(frozen=True)
class PaperProbePlanRow:
    rank: int
    opportunity: str
    probe_type: str
    status: str
    side: str
    priority_score: float
    asset: str
    venue: str
    candidate_size_usd: str
    observation_horizon: str
    evidence: str
    missing_evidence: str
    next_step: str


def build_paper_probe_plan(
    *,
    stack_path: Path = ROOT / "current_alpha_stack.csv",
    policy_expansion_path: Path = ROOT / "policy_learning" / "current_policy_expansion_targets.csv",
    context_frontier_path: Path = ROOT / "policy_learning" / "current_policy_context_frontier.csv",
    top: int = 50,
) -> tuple[PaperProbePlanRow, ...]:
    suppressed_contexts = _suppressed_contexts(context_frontier_path)
    candidates = tuple(
        row
        for row in _read_rows(stack_path)
        if _probe_type(row) != ""
    ) + _policy_expansion_candidates(policy_expansion_path)
    candidates = tuple(row for row in candidates if _context_for_row(row) not in suppressed_contexts)
    sorted_candidates = sorted(candidates, key=lambda row: _float(row.get("priority_score")), reverse=True)
    selected_candidates = _select_diverse_candidates(sorted_candidates, top=top)
    return tuple(
        _build_plan_row(rank=index + 1, row=row)
        for index, row in enumerate(selected_candidates)
    )


def write_paper_probe_plan_csv(
    rows: tuple[PaperProbePlanRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "rank",
                "opportunity",
                "probe_type",
                "status",
                "side",
                "priority_score",
                "asset",
                "venue",
                "candidate_size_usd",
                "observation_horizon",
                "evidence",
                "missing_evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.rank,
                    row.opportunity,
                    row.probe_type,
                    row.status,
                    row.side,
                    f"{row.priority_score:.8f}",
                    row.asset,
                    row.venue,
                    row.candidate_size_usd,
                    row.observation_horizon,
                    row.evidence,
                    row.missing_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_paper_probe_plan_md(
    rows: tuple[PaperProbePlanRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Probe Plan\n\n")
        handle.write(
            "This is the current cross-lane queue for small paper observations. "
            "It is not a trade instruction, not a live execution system, and not "
            "a deployable strategy list.\n\n"
        )
        handle.write(
            "| rank | opportunity | probe type | side | priority | asset | venue | size USD | horizon | missing evidence | next step |\n"
        )
        handle.write("| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.rank} | "
                f"{row.opportunity} | "
                f"{row.probe_type} | "
                f"{row.side} | "
                f"{row.priority_score:.4f} | "
                f"{row.asset} | "
                f"{row.venue} | "
                f"{row.candidate_size_usd} | "
                f"{row.observation_horizon} | "
                f"{_escape(row.missing_evidence)} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The queue promotes candidates only when another screen has already "
            "found a label or rough execution gate. The remaining work is to record "
            "fresh paper observations with fill, funding, stop, and adverse-selection "
            "evidence where the venue supports it.\n"
        )
    return output_path


def _build_plan_row(*, rank: int, row: dict[str, str]) -> PaperProbePlanRow:
    evidence = row.get("evidence", "")
    return PaperProbePlanRow(
        rank=rank,
        opportunity=row.get("opportunity", ""),
        probe_type=_probe_type(row),
        status=row.get("status", ""),
        side=row.get("side", ""),
        priority_score=_float(row.get("priority_score")),
        asset=_asset(evidence=evidence, opportunity=row.get("opportunity", "")),
        venue=_venue(evidence=evidence, sources=row.get("sources", "")),
        candidate_size_usd=_candidate_size(row),
        observation_horizon=_observation_horizon(row),
        evidence=evidence,
        missing_evidence=_missing_evidence(row.get("conflict", "")),
        next_step=row.get("next_step", ""),
    )


def _probe_type(row: dict[str, str]) -> str:
    status = row.get("status", "")
    text = " ".join((status, row.get("opportunity", ""), row.get("side", ""), row.get("next_step", ""))).lower()
    if status == "small_repeat_paper_check":
        return "repeat_execution_probe"
    if status == "microstructure_small_paper_probe":
        return "microstructure_flow_probe"
    if status == "volume_dislocation_execution_probe":
        return "volume_dislocation_probe"
    if status in {
        "paper_outcome_supported_carry_reversion_probe",
        "paper_short_horizon_supported_carry_reversion_probe",
        "paper_executable_carry_reversion_probe",
        "paper_delayed_carry_reversion_probe",
    }:
        return "crowding_reversion_probe"
    if status == "small_paper_probe":
        return "liquidation_intensity_probe"
    if status == "low_cost_intraday_paper_supported":
        return "intraday_derivatives_probe"
    if status == "dislocation_repeat_execution_candidate":
        return "dislocation_repeat_probe"
    if status in {"attention_price_lag_candidate", "attention_chase_risk", "paper_attention_funding_watch"}:
        return "attention_event_probe"
    if status in {"sector_perp_repeat_candidate", "sector_rotation_keep_sampling", "sector_rotation_label_pending"}:
        return "sector_rotation_probe"
    if status in {"multi_source_event_pressure", "two_source_event_pressure", "repeated_event_context"}:
        return "event_pressure_probe"
    if status.startswith("paper_news_"):
        return "news_event_probe"
    if status in {
        "protocol_fee_pending_forward_label",
        "paper_long_context",
        "fee_growth_unconfirmed",
        "fee_growth_unlock_conflict",
    }:
        return "protocol_fee_probe"
    if status in {
        "paper_chain_stablecoin_inflow_watch",
        "paper_chain_stablecoin_outflow_watch",
        "chain_stablecoin_flow_reversal_watch",
    }:
        return "stablecoin_migration_probe"
    if status in {"peg_anomaly_mechanics_watch", "paper_premium_mean_reversion_watch", "paper_depeg_repeg_watch"}:
        return "stablecoin_peg_probe"
    if status in {"lending_rate_candidate_after_risk_check"}:
        return "defi_lending_probe"
    if status in {
        "paper_yield_without_peg_stress_watch",
        "paper_base_yield_watch",
        "paper_incentive_yield_watch",
        "yield_supply_stress_watch",
    }:
        return "defi_yield_probe"
    if status in {
        "volatility_candidate_needs_sweep_hedge",
        "volatility_quote_mechanics_watch",
        "volatility_short_expiry_hedge_watch",
        "paper_delta_hedge_candidate",
        "expiry_gamma_hedge_watch",
        "quote_only_hedge_watch",
    }:
        return "options_volatility_probe"
    if status in {"unlock_event_label_pending", "unlock_event_crowded_squeeze_watch"}:
        return "token_unlock_probe"
    if status == "paper_protocol_activity_watch":
        return "protocol_activity_probe"
    if status in {
        "seed_wallet_flow_watch",
        "wallet_position_follow_candidate",
        "wallet_recent_flow_candidate",
        "wallet_flow_watch",
    }:
        return "wallet_entity_flow_probe"
    if status.startswith("execution_"):
        return "execution_edge_probe"
    if status in {
        "event_crypto_hedge_after_refresh_candidate",
        "event_crypto_hedge_current_quote_candidate",
        "event_crypto_hedge_news_gap_candidate",
        "event_crypto_hedge_watch",
    }:
        return "event_crypto_hedge_probe"
    if status in {
        "paper_oi_funding_crowding_watch",
        "paper_oi_unwind_watch",
        "paper_funding_dislocation_watch",
        "paper_basis_funding_dislocation_watch",
    }:
        return "derivatives_positioning_probe"
    if status in {"paper_short_basis_watch", "paper_long_basis_watch", "basis_term_structure_watch"}:
        return "basis_term_structure_probe"
    if status in POLICY_EXPANSION_STATUSES:
        return "policy_expansion_probe"
    if status in {
        "paper_dex_pool_momentum_watch",
        "paper_dex_reversal_risk_watch",
        "dex_liquidity_stress_watch",
    }:
        return "dex_pool_flow_probe"
    if row.get("sources") == "crypto_equity_proxy":
        if row.get("status") == "eth_treasury_proxy_watch":
            return "eth_treasury_proxy_probe"
        return "crypto_equity_proxy_probe"
    if row.get("sources") == "institutional_flow + public_treasury":
        return "public_treasury_probe"
    if "paper-check" in text and "candidate_after_refresh_check" in status:
        return "event_probability_probe"
    return ""


def _select_diverse_candidates(candidates: list[dict[str, str]], *, top: int) -> tuple[dict[str, str], ...]:
    selected: list[dict[str, str]] = []
    selected_ids: set[tuple[str, str, str]] = set()
    type_counts: dict[str, int] = {}
    best_by_type: dict[str, dict[str, str]] = {}
    for row in candidates:
        probe_type = _probe_type(row)
        if not probe_type or _float(row.get("priority_score")) < MIN_DIVERSE_PRIORITY_SCORE:
            continue
        best_by_type.setdefault(probe_type, row)

    def add(row: dict[str, str]) -> None:
        selected.append(row)
        selected_ids.add(_candidate_id(row))
        probe_type = _probe_type(row)
        type_counts[probe_type] = type_counts.get(probe_type, 0) + 1

    for row in sorted(best_by_type.values(), key=lambda item: _float(item.get("priority_score")), reverse=True):
        if len(selected) >= top:
            break
        if _candidate_id(row) not in selected_ids:
            add(row)

    for row in candidates:
        if len(selected) >= top:
            break
        candidate_id = _candidate_id(row)
        probe_type = _probe_type(row)
        if candidate_id in selected_ids or type_counts.get(probe_type, 0) >= TARGET_CANDIDATES_PER_PROBE_TYPE:
            continue
        add(row)

    for row in candidates:
        if len(selected) >= top:
            break
        if _candidate_id(row) not in selected_ids:
            add(row)
    return tuple(sorted(selected, key=lambda row: _float(row.get("priority_score")), reverse=True))


def _candidate_id(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("opportunity", ""), row.get("status", ""), row.get("side", ""))


def _asset(*, evidence: str, opportunity: str) -> str:
    coin_match = re.search(r"\bcoin=([^,\s]+)", evidence)
    if coin_match:
        coin = coin_match.group(1)
        if ":" in coin:
            coin = coin.rsplit(":", 1)[-1]
        return re.sub(r"[^A-Za-z0-9]", "", coin).upper()
    if ":" in evidence:
        subject = evidence.split(":", 1)[0].strip()
        if "/" in subject:
            parts = [part.strip() for part in subject.split("/")]
            if len(parts) >= 3 and parts[0].islower():
                pool_base = re.sub(r"[^A-Za-z0-9]", "", parts[1].split()[-1])
                if pool_base:
                    return pool_base.upper()
            left, right = subject.split("/", 1)
            right_symbol = re.sub(r"[^A-Za-z0-9]", "", right.split()[0])
            if right_symbol.isupper() and 2 <= len(right_symbol) <= 12 and right_symbol not in {"AI"}:
                return right_symbol
            left_symbol = re.sub(r"[^A-Za-z0-9]", "", left.split()[-1])
            if left_symbol:
                return left_symbol.upper()
        basis_match = re.match(r"^(BTC|ETH|SOL)-[A-Z0-9]+$", subject)
        if basis_match:
            return basis_match.group(1)
        derivative_symbol = _derivative_base_symbol(subject)
        if derivative_symbol:
            return derivative_symbol
        option_match = re.match(r"^((?:BTC|ETH))\s+\d{4}-\d{2}-\d{2}\b", subject)
        if option_match:
            return option_match.group(1)
        source_symbol_match = re.search(r"\b([A-Z0-9]{2,12})(?:/|$)", subject)
        if source_symbol_match:
            return source_symbol_match.group(1)
        return subject
    return opportunity.split("_", 1)[0].upper()


def _derivative_base_symbol(value: str) -> str:
    match = re.search(r"\b([A-Z0-9]{2,12})-(?:USDT|USD|USDC)\b", value)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z0-9]{2,12})(?:USDTM|USDT|USD|USDC)\b", value)
    if match:
        return match.group(1)
    return ""


def _venue(*, evidence: str, sources: str) -> str:
    match = re.search(r"\bvenue=([^,\s]+)", evidence)
    if match:
        return match.group(1)
    if "source=hyperliquid" in evidence.lower() or "hyperliquid" in sources.lower():
        return "HL"
    if "OKX" in evidence or "okx" in sources.lower():
        return "OKX"
    if "prediction_markets" in sources:
        return "prediction_market"
    return ""


def _candidate_size(row: dict[str, str]) -> str:
    evidence = row.get("evidence", "")
    match = re.search(r"\bsize=([0-9]+(?:\.[0-9]+)?)", evidence)
    if match:
        return match.group(1)
    match = re.search(r"\bdepth_usage_([0-9]+)=", evidence)
    if match:
        return match.group(1)
    if row.get("status") == "small_repeat_paper_check":
        return "1000"
    if row.get("status") in POLICY_EXPANSION_STATUSES:
        return "100"
    return ""


def _observation_horizon(row: dict[str, str]) -> str:
    if row.get("status", "").startswith("event_crypto_hedge_"):
        return "15m/1h/4h"
    if row.get("status") in POLICY_EXPANSION_STATUSES:
        return "15m/1h"
    text = " ".join((row.get("evidence", ""), row.get("next_step", ""))).lower()
    horizons = tuple(horizon for horizon in ("15m", "1h", "4h", "12h", "24h") if horizon in text)
    if horizons:
        return "/".join(horizons)
    if _has_directional_side(row.get("side", "")):
        return "15m/1h"
    return "fresh"


def _has_directional_side(side: str) -> bool:
    value = side.lower()
    return (
        value.startswith("long")
        or value.startswith("short")
        or value.startswith("paper_long")
        or value.startswith("paper_short")
        or "buy_yes" in value
    )


def _missing_evidence(conflict: str) -> str:
    parts = [part.strip() for part in re.split(r";|,", conflict) if part.strip()]
    return "; ".join(parts[:4]) if parts else conflict


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _policy_expansion_candidates(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for row in _read_rows(path):
        decision = row.get("decision", "")
        if decision not in POLICY_EXPANSION_STATUSES:
            continue
        rows.append(
            {
                "opportunity": row.get("target_id", ""),
                "status": decision,
                "side": row.get("action", ""),
                "priority_score": row.get("expansion_score", ""),
                "sources": "policy_learning/current_policy_expansion_targets",
                "evidence": (
                    f"{row.get('target_asset', '')}: seed={row.get('seed_id', '')}, "
                    f"context={row.get('context', '')}, "
                    f"target={row.get('target_opportunity', '')}, "
                    f"support={row.get('support_state', '')}"
                ),
                "conflict": row.get("reason", ""),
                "next_step": row.get("next_step", ""),
            }
        )
    return tuple(rows)


def _suppressed_contexts(path: Path) -> frozenset[str]:
    return frozenset(
        row.get("context", "")
        for row in _read_rows(path)
        if row.get("decision", "") == "shrink_or_rework_context"
    )


def _context_for_row(row: dict[str, str]) -> str:
    if row.get("context", ""):
        return row.get("context", "")
    return {
        "execution_edge_probe": "execution_edge",
        "intraday_derivatives_probe": "intraday_derivatives",
        "options_volatility_probe": "options_volatility",
        "protocol_fee_probe": "protocol_fee",
        "token_unlock_probe": "token_unlock",
    }.get(_probe_type(row), "")


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument(
        "--policy-expansion-path",
        type=Path,
        default=ROOT / "policy_learning" / "current_policy_expansion_targets.csv",
    )
    parser.add_argument(
        "--context-frontier-path",
        type=Path,
        default=ROOT / "policy_learning" / "current_policy_context_frontier.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_probe_plan.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_probe_plan.md")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = parser.parse_args()

    rows = build_paper_probe_plan(
        stack_path=args.stack_path,
        policy_expansion_path=args.policy_expansion_path,
        context_frontier_path=args.context_frontier_path,
        top=args.top,
    )
    write_paper_probe_plan_csv(rows, output_path=args.output_path)
    write_paper_probe_plan_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.rank, row.opportunity, row.probe_type, row.side, f"priority={row.priority_score:.4f}")


if __name__ == "__main__":
    main()
