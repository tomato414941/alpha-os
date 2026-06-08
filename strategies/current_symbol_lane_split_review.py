from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.current_symbol_opportunity_map import _symbols_for_stack_row


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SymbolLaneSplitReviewRow:
    symbol: str
    symbol_queue_action: str
    lane_action: str
    lane_bias: str
    opportunity: str
    status: str
    side: str
    priority_score: float
    sources: str
    support_state: str
    conflict_role: str
    evidence: str
    next_step: str


def build_symbol_lane_split_review(
    *,
    stack_path: Path = ROOT / "current_alpha_stack.csv",
    queue_path: Path = ROOT / "current_symbol_cluster_label_queue.csv",
    top_symbols: int = 12,
) -> tuple[SymbolLaneSplitReviewRow, ...]:
    stack_rows = _read_rows(stack_path)
    queue_rows = _read_rows(queue_path)[:top_symbols]
    output: list[SymbolLaneSplitReviewRow] = []
    for queue_row in queue_rows:
        symbol = queue_row.get("symbol", "")
        symbol_stack_rows = [
            row
            for row in stack_rows
            if symbol in _symbols_for_stack_row(row)
        ]
        for row in sorted(symbol_stack_rows, key=lambda item: _float(item.get("priority_score")), reverse=True):
            output.append(_build_review_row(symbol=symbol, queue_row=queue_row, stack_row=row))
    return tuple(output)


def write_symbol_lane_split_review_csv(
    rows: tuple[SymbolLaneSplitReviewRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "symbol_queue_action",
                "lane_action",
                "lane_bias",
                "opportunity",
                "status",
                "side",
                "priority_score",
                "sources",
                "support_state",
                "conflict_role",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.symbol_queue_action,
                    row.lane_action,
                    row.lane_bias,
                    row.opportunity,
                    row.status,
                    row.side,
                    f"{row.priority_score:.8f}",
                    row.sources,
                    row.support_state,
                    row.conflict_role,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_symbol_lane_split_review_md(
    rows: tuple[SymbolLaneSplitReviewRow, ...],
    *,
    output_path: Path,
    top: int = 80,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Symbol Lane Split Review\n\n")
        handle.write(
            "This expands the top symbol-label queue into lane-level work. "
            "The point is to avoid collapsing conflicting alpha ideas into one trade.\n\n"
        )
        handle.write(
            "| symbol | symbol action | lane action | bias | opportunity | status | side | priority | support | role | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | "
                f"{row.symbol_queue_action} | "
                f"{row.lane_action} | "
                f"{row.lane_bias} | "
                f"{row.opportunity} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.priority_score:.4f} | "
                f"{row.support_state} | "
                f"{row.conflict_role} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "For mixed symbols, every row should be treated as a separate label task. "
            "A positive perp label, a yield premium warning, and a liquidation continuation "
            "are different hypotheses even when they share a symbol.\n"
        )
    return output_path


def _build_review_row(
    *,
    symbol: str,
    queue_row: dict[str, str],
    stack_row: dict[str, str],
) -> SymbolLaneSplitReviewRow:
    lane_bias = _lane_bias(stack_row)
    support_state = _support_state(stack_row)
    return SymbolLaneSplitReviewRow(
        symbol=symbol,
        symbol_queue_action=queue_row.get("queue_action", ""),
        lane_action=_lane_action(queue_row=queue_row, stack_row=stack_row, lane_bias=lane_bias),
        lane_bias=lane_bias,
        opportunity=stack_row.get("opportunity", ""),
        status=stack_row.get("status", ""),
        side=stack_row.get("side", ""),
        priority_score=_float(stack_row.get("priority_score")),
        sources=stack_row.get("sources", ""),
        support_state=support_state,
        conflict_role=_conflict_role(queue_row=queue_row, lane_bias=lane_bias),
        evidence=stack_row.get("evidence", ""),
        next_step=_lane_next_step(symbol=symbol, stack_row=stack_row, support_state=support_state),
    )


def _lane_action(
    *,
    queue_row: dict[str, str],
    stack_row: dict[str, str],
    lane_bias: str,
) -> str:
    symbol_action = queue_row.get("queue_action", "")
    if symbol_action == "split_lane_forward_label":
        return "label_this_lane_separately"
    if symbol_action == "confirmed_direction_forward_label":
        dominant = queue_row.get("dominant_bias", "")
        return "label_confirming_lane" if lane_bias == dominant else "check_contradicting_lane"
    if "yield" in lane_bias or "risk" in lane_bias:
        return "validate_mechanics_before_return_label"
    if stack_row.get("status") in {"paper_dislocation_executable_probe", "small_paper_probe"}:
        return "repeat_paper_probe"
    return "collect_more_lane_observations"


def _lane_bias(row: dict[str, str]) -> str:
    side = row.get("side", "").lower()
    text = " ".join((side, row.get("status", ""), row.get("opportunity", ""), row.get("evidence", ""))).lower()
    if "avoid" in side or "premium_reversion" in side or "peg_stress" in text:
        return "risk_or_avoid"
    if "yield" in side or "yield" in text or "apy" in text:
        return "yield"
    if any(token in side for token in ("straddle", "long_vol", "calendar_spread")):
        return "relative_value"
    if any(token in side for token in ("long_mstr_short", "short_future_long", "long_future_short")):
        return "relative_value"
    if side.startswith("short") or " short_" in text or "_short" in text:
        return "short"
    if side.startswith("long") or " long_" in text or "_long" in text:
        return "long"
    return "neutral"


def _support_state(row: dict[str, str]) -> str:
    text = " ".join((row.get("status", ""), row.get("evidence", ""), row.get("next_step", ""))).lower()
    if "out1h=paper_1h_win" in text or "paper_1h_win" in text:
        return "paper_1h_supported"
    if "protocol_fee_label_supported_watch" in text or "chain_stablecoin_4h_supported_pending_12h" in text:
        return "paper_4h_supported"
    if "volume_dislocation_4h_supported_pending_12h" in text:
        return "paper_4h_supported"
    if "volume_dislocation_delayed_4h_support" in text:
        return "paper_4h_supported"
    if "volume_dislocation_1h_only_watch" in text:
        return "paper_1h_supported"
    if "low_cost_intraday_paper_supported" in text or "paper_intraday_cost_supported" in text:
        return "paper_cost_supported"
    if "low_cost_intraday_paper_recent_only" in text or "paper_intraday_recent_only" in text:
        return "paper_recent_only"
    if "intraday_live_feature_source_blocked" in text or "feature_source_blocked" in text:
        return "feature_source_blocked"
    if "out15=paper_15m_win" in text or "paper_15m_win" in text:
        return "paper_15m_supported"
    if "volume_dislocation_execution_probe" in text:
        return "paper_execution_gated"
    if "paper_execution_probe" in text or "small_paper_probe" in text:
        return "paper_execution_gated"
    if "protocol_fee_label_failed" in text or "chain_stablecoin_4h_contradicted_pending_12h" in text:
        return "failed_label"
    if "volume_dislocation_4h_contradicted_pending_12h" in text:
        return "failed_label"
    if "volume_dislocation_4h_contradicted_after_cost_check" in text:
        return "failed_label"
    if "volume_dislocation_no_edge_after_rough_cost" in text:
        return "failed_label"
    if "pending" in text:
        return "pending_label"
    if "premium" in text or "redemption" in text or "custody" in text:
        return "mechanics_unvalidated"
    return "unlabeled"


def _conflict_role(*, queue_row: dict[str, str], lane_bias: str) -> str:
    dominant = queue_row.get("dominant_bias", "")
    if queue_row.get("queue_action") != "split_lane_forward_label":
        return "not_split_symbol"
    if lane_bias == dominant:
        return "dominant_lane"
    if lane_bias in {"yield", "risk_or_avoid", "relative_value"}:
        return "structure_conflict_lane"
    return "direction_conflict_lane"


def _lane_next_step(*, symbol: str, stack_row: dict[str, str], support_state: str) -> str:
    if support_state == "paper_1h_supported":
        return f"rerun {symbol} lane on a fresh window and add execution/fill evidence"
    if support_state == "paper_4h_supported":
        return f"repeat {symbol} lane on another 4h window and refresh execution evidence"
    if support_state == "paper_15m_supported":
        return f"wait for {symbol} 1h/4h label or repeat this lane on a fresh snapshot"
    if support_state == "paper_execution_gated":
        return f"paper-probe {symbol} lane at the gated size and log outcome"
    if support_state == "paper_cost_supported":
        return f"check {symbol} live spread, funding timing, fill delay, stop rules, and realized cost"
    if support_state == "paper_recent_only":
        return f"extend {symbol} lane to another non-overlapping window before promotion"
    if support_state == "feature_source_blocked":
        return f"obtain a live feature source for {symbol} before treating this lane as active"
    if support_state == "mechanics_unvalidated":
        return f"validate {symbol} mechanics, venue access, unwind path, and stale-price risk"
    if support_state == "failed_label":
        return f"deprioritize {symbol} lane until a fresh independent snapshot appears"
    return stack_row.get("next_step", f"collect more {symbol} lane observations")


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--queue-path", type=Path, default=ROOT / "current_symbol_cluster_label_queue.csv")
    parser.add_argument("--top-symbols", type=int, default=12)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_symbol_lane_split_review.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_symbol_lane_split_review.md")
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    rows = build_symbol_lane_split_review(
        stack_path=args.stack_path,
        queue_path=args.queue_path,
        top_symbols=args.top_symbols,
    )
    write_symbol_lane_split_review_csv(rows, output_path=args.output_path)
    write_symbol_lane_split_review_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.lane_action,
            row.lane_bias,
            row.opportunity,
            f"support={row.support_state}",
        )


if __name__ == "__main__":
    main()
