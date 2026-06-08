from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUTS: tuple[tuple[str, Path], ...] = (
    ("paper_ticket", ROOT / "current_paper_ticket_fill_risk_check.csv"),
    ("promoted_repeat", ROOT / "current_promoted_ticket_repeat_fill_risk_check.csv"),
    ("second_promoted_repeat", ROOT / "current_second_promoted_ticket_repeat_fill_risk_check.csv"),
    ("symbol_lane_paper", ROOT / "current_symbol_lane_paper_fill_risk_check.csv"),
    ("symbol_lane_promoted_repeat", ROOT / "current_symbol_lane_promoted_repeat_fill_risk_check.csv"),
)


@dataclass(frozen=True)
class CostAdjustedAlphaCandidate:
    candidate_id: str
    source_lane: str
    ticket_id: str
    asset: str
    opportunity: str
    decision: str
    candidate_size_usd: str
    directional_return_bps: float
    estimated_net_after_cost_bps: float
    spread_bps: str
    near_depth_10bps_notional: str
    visible_depth_usage: str
    annualized_funding: str
    estimated_funding_1h_bps: str
    risk_action: str
    status: str
    priority_score: float
    evidence: str
    missing_work: str
    next_step: str


def build_cost_adjusted_alpha_candidates(
    inputs: tuple[tuple[str, Path], ...] = INPUTS,
) -> tuple[CostAdjustedAlphaCandidate, ...]:
    candidates: list[CostAdjustedAlphaCandidate] = []
    for source_lane, path in inputs:
        for row in _read_rows(path):
            risk_action = row.get("risk_action", "")
            net_bps = _float(row.get("estimated_net_after_cost_bps"))
            if net_bps <= 0.0 or risk_action not in {"cost_adjusted_paper_probe", "depth_too_thin_for_probe"}:
                continue
            candidates.append(
                _candidate_from_row(
                    row=row,
                    source_lane=source_lane,
                    net_bps=net_bps,
                    risk_action=risk_action,
                )
            )
    return tuple(sorted(candidates, key=lambda row: row.priority_score, reverse=True))


def write_cost_adjusted_alpha_candidates_csv(
    rows: tuple[CostAdjustedAlphaCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "source_lane",
                "ticket_id",
                "asset",
                "opportunity",
                "decision",
                "candidate_size_usd",
                "directional_return_bps",
                "estimated_net_after_cost_bps",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage",
                "annualized_funding",
                "estimated_funding_1h_bps",
                "risk_action",
                "status",
                "priority_score",
                "evidence",
                "missing_work",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.source_lane,
                    row.ticket_id,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.candidate_size_usd,
                    f"{row.directional_return_bps:.8f}",
                    f"{row.estimated_net_after_cost_bps:.8f}",
                    row.spread_bps,
                    row.near_depth_10bps_notional,
                    row.visible_depth_usage,
                    row.annualized_funding,
                    row.estimated_funding_1h_bps,
                    row.risk_action,
                    row.status,
                    f"{row.priority_score:.8f}",
                    row.evidence,
                    row.missing_work,
                    row.next_step,
                )
            )
    return output_path


def write_cost_adjusted_alpha_candidates_md(
    rows: tuple[CostAdjustedAlphaCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cost Adjusted Alpha Candidates\n\n")
        handle.write(
            "This consolidates candidates that still have positive paper edge after rough spread, "
            "taker-fee, funding, and visible-depth checks across the current lanes.\n\n"
        )
        handle.write(
            "| candidate | lane | asset | decision | size USD | dir bps | net bps | usage | status | priority | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:40]:
            handle.write(
                "| "
                f"{row.candidate_id} | "
                f"{row.source_lane} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.candidate_size_usd} | "
                f"{row.directional_return_bps:.4f} | "
                f"{row.estimated_net_after_cost_bps:.4f} | "
                f"{row.visible_depth_usage} | "
                f"{row.status} | "
                f"{row.priority_score:.4f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _candidate_from_row(
    *,
    row: dict[str, str],
    source_lane: str,
    net_bps: float,
    risk_action: str,
) -> CostAdjustedAlphaCandidate:
    directional_bps = _float(row.get("directional_return_bps"))
    status = _status(source_lane=source_lane, risk_action=risk_action)
    priority_score = _priority_score(
        source_lane=source_lane,
        risk_action=risk_action,
        directional_bps=directional_bps,
        net_bps=net_bps,
        visible_depth_usage=_float(row.get("visible_depth_usage")),
    )
    candidate_id = f"{source_lane}:{row.get('ticket_id', '')}"
    evidence = (
        f"dir_bps={directional_bps:.4f}; "
        f"net_bps={net_bps:.4f}; "
        f"spread_bps={row.get('spread_bps', '')}; "
        f"depth_10bps={row.get('near_depth_10bps_notional', '')}; "
        f"usage={row.get('visible_depth_usage', '')}; "
        f"funding_1h_bps={row.get('estimated_funding_1h_bps', '')}; "
        f"risk={risk_action}"
    )
    return CostAdjustedAlphaCandidate(
        candidate_id=candidate_id,
        source_lane=source_lane,
        ticket_id=row.get("ticket_id", ""),
        asset=row.get("asset", ""),
        opportunity=row.get("opportunity", ""),
        decision=row.get("decision", ""),
        candidate_size_usd=row.get("candidate_size_usd", ""),
        directional_return_bps=directional_bps,
        estimated_net_after_cost_bps=net_bps,
        spread_bps=row.get("spread_bps", ""),
        near_depth_10bps_notional=row.get("near_depth_10bps_notional", ""),
        visible_depth_usage=row.get("visible_depth_usage", ""),
        annualized_funding=row.get("annualized_funding", ""),
        estimated_funding_1h_bps=row.get("estimated_funding_1h_bps", ""),
        risk_action=risk_action,
        status=status,
        priority_score=priority_score,
        evidence=evidence,
        missing_work=_missing_work(risk_action),
        next_step=_next_step(risk_action),
    )


def _status(*, source_lane: str, risk_action: str) -> str:
    if risk_action == "depth_too_thin_for_probe":
        return "capacity_gated_alpha_candidate"
    if source_lane in {"second_promoted_repeat", "symbol_lane_promoted_repeat"}:
        return "repeat_supported_cost_adjusted_alpha"
    if source_lane == "promoted_repeat":
        return "first_repeat_cost_adjusted_alpha"
    return "cost_adjusted_alpha_candidate"


def _priority_score(
    *,
    source_lane: str,
    risk_action: str,
    directional_bps: float,
    net_bps: float,
    visible_depth_usage: float,
) -> float:
    repeat_bonus = {
        "second_promoted_repeat": 80.0,
        "symbol_lane_promoted_repeat": 70.0,
        "promoted_repeat": 50.0,
        "symbol_lane_paper": 20.0,
        "paper_ticket": 0.0,
    }.get(source_lane, 0.0)
    capacity_penalty = 180.0 if risk_action == "depth_too_thin_for_probe" else 0.0
    usage_penalty = max(0.0, visible_depth_usage - 0.02) * 100.0
    return net_bps + repeat_bonus + min(abs(directional_bps) * 0.05, 40.0) - capacity_penalty - usage_penalty


def _missing_work(risk_action: str) -> str:
    if risk_action == "depth_too_thin_for_probe":
        return "paper edge is positive but current size consumes too much visible depth"
    return "paper edge is cost-adjusted only; still needs repeat outcome, adverse excursion, and fill notes"


def _next_step(risk_action: str) -> str:
    if risk_action == "depth_too_thin_for_probe":
        return "reduce size or wait for better depth before repeating this candidate"
    return "repeat the candidate with stop/adverse-excursion and realized fill notes"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cost_adjusted_alpha_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cost_adjusted_alpha_candidates.md")
    args = parser.parse_args()

    rows = build_cost_adjusted_alpha_candidates()
    write_cost_adjusted_alpha_candidates_csv(rows, output_path=args.output_path)
    write_cost_adjusted_alpha_candidates_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.candidate_id, row.asset, f"{row.priority_score:.4f}")


if __name__ == "__main__":
    main()
