from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTIER_PATH = ROOT / "current_alpha_promotion_frontier.csv"

STATUS_LIMITS = {
    "paper_cost_survival_watchlist": 5,
    "duplicate_dedupe_required": 3,
    "repeat_conflict_split_required": 6,
    "options_hedge_path_required": 3,
    "options_quote_mechanics_required": 3,
    "options_gamma_timing_required": 2,
    "direct_exchange_inflow_data_required": 2,
    "chain_proxy_alpha_needs_label": 3,
    "crowding_needs_forward_unwind_label": 3,
    "crowding_without_unwind_label": 5,
    "news_pending_forward_archive": 3,
    "news_single_source_blocked": 3,
    "event_hedge_pending_mark": 2,
    "event_hedge_unproven": 3,
    "event_hedge_event_alignment_blocked": 3,
    "maker_adverse_selection_blocked": 3,
    "capacity_blocked": 4,
}


@dataclass(frozen=True)
class AlphaPromotionWorkItem:
    work_id: str
    work_kind: str
    priority: float
    source_frontier_id: str
    lane: str
    asset: str
    action: str
    status: str
    why_now: str
    required_evidence: str
    next_step: str


def build_alpha_promotion_worklist(
    frontier_path: Path = FRONTIER_PATH,
) -> tuple[AlphaPromotionWorkItem, ...]:
    rows = _read_rows(frontier_path)
    selected: list[AlphaPromotionWorkItem] = []
    per_status_count: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        status = row.get("status", "")
        limit = STATUS_LIMITS.get(status)
        if limit is None or per_status_count.get(status, 0) >= limit:
            continue
        key = (status, row.get("asset", ""), row.get("action", ""))
        if key in seen:
            continue
        item = _work_item(row=row, status_count=per_status_count.get(status, 0) + 1)
        selected.append(item)
        seen.add(key)
        per_status_count[status] = per_status_count.get(status, 0) + 1

    return tuple(sorted(selected, key=lambda item: item.priority, reverse=True))


def write_alpha_promotion_worklist_csv(
    rows: tuple[AlphaPromotionWorkItem, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "work_id",
                "work_kind",
                "priority",
                "source_frontier_id",
                "lane",
                "asset",
                "action",
                "status",
                "why_now",
                "required_evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.work_id,
                    row.work_kind,
                    f"{row.priority:.8f}",
                    row.source_frontier_id,
                    row.lane,
                    row.asset,
                    row.action,
                    row.status,
                    row.why_now,
                    row.required_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_promotion_worklist_md(
    rows: tuple[AlphaPromotionWorkItem, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Promotion Worklist\n\n")
        handle.write(
            "This worklist chooses non-duplicate next actions from the promotion frontier. "
            "It is a research execution list, not a live trading list.\n\n"
        )
        handle.write("| work | kind | priority | asset | action | status | required evidence | next step |\n")
        handle.write("| --- | --- | ---: | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.work_id} | "
                f"{row.work_kind} | "
                f"{row.priority:.4f} | "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.status} | "
                f"{_escape(row.required_evidence)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _work_item(*, row: dict[str, str], status_count: int) -> AlphaPromotionWorkItem:
    status = row.get("status", "")
    work_kind = _work_kind(status)
    priority = _priority(row=row, work_kind=work_kind, status_count=status_count)
    asset = row.get("asset", "")
    action = row.get("action", "")
    work_id = f"{work_kind}-{status_count:02d}-{_slug(status)}-{_slug(asset)}-{_slug(action)}"
    return AlphaPromotionWorkItem(
        work_id=work_id,
        work_kind=work_kind,
        priority=priority,
        source_frontier_id=row.get("frontier_id", ""),
        lane=row.get("lane", ""),
        asset=asset,
        action=action,
        status=status,
        why_now=_why_now(row=row, work_kind=work_kind),
        required_evidence=row.get("blocker", ""),
        next_step=row.get("next_step", ""),
    )


def _work_kind(status: str) -> str:
    if status == "paper_cost_survival_watchlist":
        return "repeat_fill_risk_probe"
    if status == "duplicate_dedupe_required":
        return "dedupe_cluster"
    if status == "repeat_conflict_split_required":
        return "split_conflicting_sources"
    if status.startswith("options_"):
        return "options_path_check"
    if status in {"direct_exchange_inflow_data_required", "chain_proxy_alpha_needs_label"}:
        return "flow_label_or_data_check"
    if status.startswith("crowding_"):
        return "crowding_unwind_label"
    if status.startswith("news_"):
        return "news_source_quality_check"
    if status.startswith("event_hedge_"):
        return "event_alignment_check"
    if status == "maker_adverse_selection_blocked":
        return "maker_queue_label_check"
    if status == "capacity_blocked":
        return "resize_or_depth_check"
    return "promotion_review"


def _priority(*, row: dict[str, str], work_kind: str, status_count: int) -> float:
    base = {
        "repeat_fill_risk_probe": 1200.0,
        "dedupe_cluster": 1150.0,
        "split_conflicting_sources": 1100.0,
        "options_path_check": 950.0,
        "flow_label_or_data_check": 850.0,
        "crowding_unwind_label": 780.0,
        "news_source_quality_check": 740.0,
        "event_alignment_check": 700.0,
        "maker_queue_label_check": 660.0,
        "resize_or_depth_check": 620.0,
    }.get(work_kind, 500.0)
    score = min(_float(row.get("frontier_score")), 500.0)
    rank_penalty = status_count * 10.0
    return base + score - rank_penalty


def _why_now(*, row: dict[str, str], work_kind: str) -> str:
    if work_kind == "repeat_fill_risk_probe":
        return "positive cost-survival row exists; next blocker is realized fill, stop, and adverse excursion"
    if work_kind == "dedupe_cluster":
        return "many rows may share one underlying move; dedupe before adding more paper tickets"
    if work_kind == "split_conflicting_sources":
        return "repeat evidence conflicts; split source lanes before treating the cluster as alpha"
    if work_kind == "options_path_check":
        return "cheap-vol candidate exists, but the trade path is the missing evidence"
    if work_kind == "flow_label_or_data_check":
        return "flow context is visible, but the alpha label or direct data source is missing"
    if work_kind == "crowding_unwind_label":
        return "crowding context is visible, but continuation versus unwind is not labeled"
    if work_kind == "news_source_quality_check":
        return "news move needs forward archive or independent-source support"
    if work_kind == "event_alignment_check":
        return "crypto move must align with event-price movement before promotion"
    if work_kind == "maker_queue_label_check":
        return "maker full-fill optimism is blocked by adverse-selection fill proxy"
    if work_kind == "resize_or_depth_check":
        return "paper edge is positive but current size/depth assumption blocks promotion"
    return row.get("blocker", "")


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


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part) or "none"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-path", type=Path, default=FRONTIER_PATH)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_promotion_worklist.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_alpha_promotion_worklist.md")
    args = parser.parse_args()

    rows = build_alpha_promotion_worklist(args.frontier_path)
    write_alpha_promotion_worklist_csv(rows, output_path=args.output_path)
    write_alpha_promotion_worklist_md(rows, output_path=args.md_output_path)
    for row in rows[:12]:
        print(row.work_id, row.asset, row.status, f"{row.priority:.4f}")


if __name__ == "__main__":
    main()
