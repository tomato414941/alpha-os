from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent

FILL_RISK_PATHS = (
    STRATEGIES_ROOT / "current_paper_ticket_fill_risk_check.csv",
    STRATEGIES_ROOT / "current_promoted_ticket_repeat_fill_risk_check.csv",
    STRATEGIES_ROOT / "current_second_promoted_ticket_repeat_fill_risk_check.csv",
    STRATEGIES_ROOT / "current_symbol_lane_paper_fill_risk_check.csv",
    STRATEGIES_ROOT / "current_symbol_lane_promoted_repeat_fill_risk_check.csv",
)


@dataclass(frozen=True)
class ExecutionModeCandidate:
    ticket_id: str
    asset: str
    opportunity: str
    decision: str
    execution_mode: str
    action: str
    score: float
    directional_return_bps: float
    current_net_bps: float
    estimated_mode_net_bps: float
    spread_bps: float
    visible_depth_usage: float
    suggested_size_usd: str
    source_file: str
    reason: str
    next_step: str


def build_execution_mode_candidates(paths: tuple[Path, ...] = FILL_RISK_PATHS) -> tuple[ExecutionModeCandidate, ...]:
    output: list[ExecutionModeCandidate] = []
    for path in paths:
        for row in _read_rows(path):
            output.extend(_candidates_for_row(row=row, source_file=path.name))
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_execution_mode_candidates_csv(
    rows: tuple[ExecutionModeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "asset",
                "opportunity",
                "decision",
                "execution_mode",
                "action",
                "score",
                "directional_return_bps",
                "current_net_bps",
                "estimated_mode_net_bps",
                "spread_bps",
                "visible_depth_usage",
                "suggested_size_usd",
                "source_file",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.asset,
                    row.opportunity,
                    row.decision,
                    row.execution_mode,
                    row.action,
                    f"{row.score:.8f}",
                    f"{row.directional_return_bps:.8f}",
                    f"{row.current_net_bps:.8f}",
                    f"{row.estimated_mode_net_bps:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    row.suggested_size_usd,
                    row.source_file,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_execution_mode_candidates_md(
    rows: tuple[ExecutionModeCandidate, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Execution Mode Candidates\n\n")
        handle.write(
            "This turns paper-ticket fill-risk checks into execution-mode candidates. "
            "It is not a live order router and does not assume maker fills.\n\n"
        )
        handle.write(
            "| ticket | asset | mode | action | score | current net | mode net | spread | usage | suggested size | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.ticket_id} | {row.asset} | {row.execution_mode} | {row.action} | "
                f"{row.score:.4f} | {row.current_net_bps:.4f} | {row.estimated_mode_net_bps:.4f} | "
                f"{row.spread_bps:.4f} | {row.visible_depth_usage:.4f} | "
                f"{row.suggested_size_usd} | {_escape(row.reason)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Execution can be an alpha source only when order mode, size, fee tier, visible depth, "
            "fill probability, queue position, and adverse selection are made explicit. "
            "Rows here identify where a paper edge survives or fails under simple execution choices.\n"
        )
    return output_path


def _candidates_for_row(*, row: dict[str, str], source_file: str) -> tuple[ExecutionModeCandidate, ...]:
    if not row.get("ticket_id"):
        return ()
    action = row.get("risk_action", "")
    directional_bps = _float(row.get("directional_return_bps"))
    current_net = _float(row.get("estimated_net_after_cost_bps"))
    spread_bps = _float(row.get("spread_bps"))
    depth_usage = _float(row.get("visible_depth_usage"))
    output: list[ExecutionModeCandidate] = []
    if action == "cost_adjusted_paper_probe":
        output.append(
            _build_candidate(
                row=row,
                source_file=source_file,
                execution_mode="taker_small",
                action="repeat_taker_probe",
                estimated_mode_net_bps=current_net,
                suggested_size_usd=row.get("candidate_size_usd", ""),
                reason="paper edge survives rough taker spread, fee, funding, and depth checks",
            )
        )
        output.append(
            _build_candidate(
                row=row,
                source_file=source_file,
                execution_mode="maker_or_low_fee_small",
                action="compare_taker_vs_low_fee",
                estimated_mode_net_bps=current_net + 6.0 + min(spread_bps / 2.0, 3.0),
                suggested_size_usd=row.get("candidate_size_usd", ""),
                reason="low-fee or maker-like execution may improve the already surviving paper edge",
            )
        )
    elif action == "depth_too_thin_for_probe":
        suggested_size = _reduced_size(row.get("candidate_size_usd", ""), depth_usage=depth_usage)
        if suggested_size:
            output.append(
                _build_candidate(
                    row=row,
                    source_file=source_file,
                    execution_mode="reduced_size_taker",
                    action="retry_with_depth_capped_size",
                    estimated_mode_net_bps=current_net,
                    suggested_size_usd=suggested_size,
                    reason="paper edge survives directionally but current size consumes too much visible depth",
                )
            )
    elif action == "cost_adjusted_edge_failed":
        low_fee_net = current_net + 6.0 + min(spread_bps / 2.0, 3.0)
        output.append(
            _build_candidate(
                row=row,
                source_file=source_file,
                execution_mode="low_fee_rescue_check",
                action="only_retest_if_low_fee_net_positive" if low_fee_net > 0.0 else "deprioritize_execution",
                estimated_mode_net_bps=low_fee_net,
                suggested_size_usd=row.get("candidate_size_usd", ""),
                reason="taker execution fails; only a lower-fee or maker-like path could rescue it",
            )
        )
    elif action == "missing_execution_context":
        output.append(
            _build_candidate(
                row=row,
                source_file=source_file,
                execution_mode="context_required",
                action="refresh_execution_context",
                estimated_mode_net_bps=directional_bps,
                suggested_size_usd=row.get("candidate_size_usd", ""),
                reason="paper edge cannot be evaluated because spread, funding, or visible depth is missing",
            )
        )
    return tuple(output)


def _build_candidate(
    *,
    row: dict[str, str],
    source_file: str,
    execution_mode: str,
    action: str,
    estimated_mode_net_bps: float,
    suggested_size_usd: str,
    reason: str,
) -> ExecutionModeCandidate:
    directional_bps = _float(row.get("directional_return_bps"))
    current_net = _float(row.get("estimated_net_after_cost_bps"))
    spread_bps = _float(row.get("spread_bps"))
    depth_usage = _float(row.get("visible_depth_usage"))
    return ExecutionModeCandidate(
        ticket_id=row.get("ticket_id", ""),
        asset=row.get("asset", ""),
        opportunity=row.get("opportunity", ""),
        decision=row.get("decision", ""),
        execution_mode=execution_mode,
        action=action,
        score=_score(
            estimated_mode_net_bps=estimated_mode_net_bps,
            directional_return_bps=directional_bps,
            spread_bps=spread_bps,
            visible_depth_usage=depth_usage,
            action=action,
        ),
        directional_return_bps=directional_bps,
        current_net_bps=current_net,
        estimated_mode_net_bps=estimated_mode_net_bps,
        spread_bps=spread_bps,
        visible_depth_usage=depth_usage,
        suggested_size_usd=suggested_size_usd,
        source_file=source_file,
        reason=reason,
        next_step=_next_step(action=action, asset=row.get("asset", ""), mode=execution_mode),
    )


def _score(
    *,
    estimated_mode_net_bps: float,
    directional_return_bps: float,
    spread_bps: float,
    visible_depth_usage: float,
    action: str,
) -> float:
    action_bonus = {
        "repeat_taker_probe": 16.0,
        "compare_taker_vs_low_fee": 14.0,
        "retry_with_depth_capped_size": 10.0,
        "only_retest_if_low_fee_net_positive": 6.0,
        "refresh_execution_context": 4.0,
        "deprioritize_execution": -10.0,
    }.get(action, 0.0)
    spread_penalty = max(spread_bps - 2.0, 0.0) * 0.8
    depth_penalty = max(visible_depth_usage - 0.05, 0.0) * 40.0
    return estimated_mode_net_bps + min(abs(directional_return_bps) / 10.0, 20.0) + action_bonus - spread_penalty - depth_penalty


def _reduced_size(candidate_size: str, *, depth_usage: float) -> str:
    size = _float(candidate_size)
    if size <= 0.0 or depth_usage <= 0.0:
        return ""
    return f"{max(size * 0.05 / depth_usage, 1.0):.2f}"


def _next_step(*, action: str, asset: str, mode: str) -> str:
    if action == "refresh_execution_context":
        return f"refresh {asset} spread, funding, and depth before any execution-mode comparison"
    if action == "deprioritize_execution":
        return f"do not repeat {asset} unless a stronger signal or cheaper execution path appears"
    return f"paper-repeat {asset} with {mode}, explicit fill assumption, funding, stop, and adverse-selection notes"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {"", None} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_execution_mode_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_execution_mode_candidates.md")
    args = parser.parse_args()
    rows = build_execution_mode_candidates()
    write_execution_mode_candidates_csv(rows, output_path=args.output_path)
    write_execution_mode_candidates_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.asset, row.execution_mode, row.action, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
