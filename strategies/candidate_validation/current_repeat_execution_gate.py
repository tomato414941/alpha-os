from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RepeatExecutionGateRow:
    asset: str
    source: str
    venue: str
    label_count: int
    hit_rate_15m: float
    mean_dir15: float
    mean_dir15_bps: float
    spread_bps: float | None
    near_depth_10bps_notional: float | None
    visible_depth_usage_1k: float | None
    annualized_funding: float | None
    rough_net15_bps: float | None
    gate_action: str
    reason: str
    next_step: str


def build_repeat_execution_gate_rows(
    *,
    summary_path: Path = ROOT / "current_followup_repeat_history_summary.csv",
    hl_context_path: Path = ROOT / "current_followup_execution_context.csv",
    okx_context_path: Path = ROOT / "current_followup_okx_execution_context.csv",
    taker_cost_bps: float = 8.0,
) -> tuple[RepeatExecutionGateRow, ...]:
    contexts = _venue_contexts(
        hl_context_path=hl_context_path,
        okx_context_path=okx_context_path,
    )
    rows: list[RepeatExecutionGateRow] = []
    for summary in _read_rows(summary_path):
        if summary.get("action") != "repeat_priority" or summary.get("group_type") != "asset_source":
            continue
        asset, source = _split_asset_source(summary.get("group_key", ""))
        if not asset or not source:
            continue
        for venue in ("HL", "OKX"):
            rows.append(
                _build_gate_row(
                    summary=summary,
                    asset=asset,
                    source=source,
                    venue=venue,
                    context=contexts.get((venue, asset)),
                    taker_cost_bps=taker_cost_bps,
                )
            )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_repeat_execution_gate_csv(
    rows: tuple[RepeatExecutionGateRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "source",
                "venue",
                "label_count",
                "hit_rate_15m",
                "mean_dir15",
                "mean_dir15_bps",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage_1k",
                "annualized_funding",
                "rough_net15_bps",
                "gate_action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.source,
                    row.venue,
                    row.label_count,
                    f"{row.hit_rate_15m:.6f}",
                    f"{row.mean_dir15:.8f}",
                    f"{row.mean_dir15_bps:.4f}",
                    "" if row.spread_bps is None else f"{row.spread_bps:.8f}",
                    ""
                    if row.near_depth_10bps_notional is None
                    else f"{row.near_depth_10bps_notional:.8f}",
                    "" if row.visible_depth_usage_1k is None else f"{row.visible_depth_usage_1k:.8f}",
                    "" if row.annualized_funding is None else f"{row.annualized_funding:.8f}",
                    "" if row.rough_net15_bps is None else f"{row.rough_net15_bps:.4f}",
                    row.gate_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_repeat_execution_gate_md(
    rows: tuple[RepeatExecutionGateRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Repeat Execution Gate\n\n")
        handle.write(
            "This joins repeat-priority labels to current HL/OKX public execution context. "
            "It is a paper-check queue, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | source | venue | labels | hit15 | mean15 bps | spread bps | depth 10bps USD | rough net15 bps | gate | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.venue} | "
                f"{row.label_count} | "
                f"{row.hit_rate_15m:.3f} | "
                f"{row.mean_dir15_bps:.2f} | "
                f"{'' if row.spread_bps is None else f'{row.spread_bps:.2f}'} | "
                f"{'' if row.near_depth_10bps_notional is None else f'{row.near_depth_10bps_notional:.0f}'} | "
                f"{'' if row.rough_net15_bps is None else f'{row.rough_net15_bps:.2f}'} | "
                f"{row.gate_action} | "
                f"{row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`small_repeat_paper_check` means repeated 15m labels are still positive "
            "after a rough spread plus taker-cost haircut and visible 10bps depth is "
            "not obviously blocking a 1k paper check. It still needs 1h confirmation, "
            "real fills, funding PnL, stop behavior, and adverse-selection checks.\n"
        )
    return output_path


def _build_gate_row(
    *,
    summary: dict[str, str],
    asset: str,
    source: str,
    venue: str,
    context: dict[str, str] | None,
    taker_cost_bps: float,
) -> RepeatExecutionGateRow:
    label_count = _int(summary.get("labeled_rows"))
    hit_rate = _float(summary.get("hit_rate_15m"))
    mean_dir15 = _float(summary.get("mean_dir15"))
    mean_bps = mean_dir15 * 10_000.0
    if context is None:
        gate_action = "missing_venue_context"
        reason = "no current execution context for this venue"
        return RepeatExecutionGateRow(
            asset=asset,
            source=source,
            venue=venue,
            label_count=label_count,
            hit_rate_15m=hit_rate,
            mean_dir15=mean_dir15,
            mean_dir15_bps=mean_bps,
            spread_bps=None,
            near_depth_10bps_notional=None,
            visible_depth_usage_1k=None,
            annualized_funding=None,
            rough_net15_bps=None,
            gate_action=gate_action,
            reason=reason,
            next_step=f"collect {venue} execution context for {asset}/{source}",
        )
    context_sources = set(_split_sources(context.get("source", "")))
    spread = _float(context.get("spread_bps"))
    depth = _float(context.get("near_depth_10bps_notional"))
    depth_usage = _float(context.get("visible_depth_usage_1k"))
    funding = _float(context.get("annualized_funding"))
    rough_net = mean_bps - spread - taker_cost_bps
    gate_action = _gate_action(
        source=source,
        context_sources=context_sources,
        rough_net=rough_net,
        depth=depth,
        depth_usage=depth_usage,
    )
    return RepeatExecutionGateRow(
        asset=asset,
        source=source,
        venue=venue,
        label_count=label_count,
        hit_rate_15m=hit_rate,
        mean_dir15=mean_dir15,
        mean_dir15_bps=mean_bps,
        spread_bps=spread,
        near_depth_10bps_notional=depth,
        visible_depth_usage_1k=depth_usage,
        annualized_funding=funding,
        rough_net15_bps=rough_net,
        gate_action=gate_action,
        reason=_reason(
            source=source,
            context_sources=context_sources,
            rough_net=rough_net,
            depth=depth,
            depth_usage=depth_usage,
        ),
        next_step=_next_step(asset=asset, source=source, venue=venue, gate_action=gate_action),
    )


def _venue_contexts(
    *,
    hl_context_path: Path,
    okx_context_path: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    contexts: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_rows(hl_context_path):
        if row.get("action") == "tradable_context_ok":
            contexts[("HL", row.get("asset", ""))] = row
    for row in _read_rows(okx_context_path):
        if row.get("action") == "okx_context_ok":
            contexts[("OKX", row.get("asset", ""))] = row
    return contexts


def _gate_action(
    *,
    source: str,
    context_sources: set[str],
    rough_net: float,
    depth: float,
    depth_usage: float,
) -> str:
    if source not in context_sources:
        return "source_not_in_current_context"
    if depth < 5_000.0 or depth_usage > 0.25:
        return "thin_depth_repeat_watch"
    if rough_net > 0.0:
        return "small_repeat_paper_check"
    return "rough_cost_blocked"


def _reason(
    *,
    source: str,
    context_sources: set[str],
    rough_net: float,
    depth: float,
    depth_usage: float,
) -> str:
    if source not in context_sources:
        return f"{source} is not part of the current venue context"
    if depth < 5_000.0:
        return "visible 10bps depth is thin for even small paper checks"
    if depth_usage > 0.25:
        return "1k paper size would consume too much visible 10bps depth"
    if rough_net <= 0.0:
        return "mean 15m repeat is weaker than rough spread plus taker-cost haircut"
    return "repeat label survives rough public execution haircut for a small paper check"


def _next_step(*, asset: str, source: str, venue: str, gate_action: str) -> str:
    if gate_action == "small_repeat_paper_check":
        return f"paper-check {venue} {asset}/{source} with 1h label, fill, funding, stop, and adverse-excursion logs"
    if gate_action == "thin_depth_repeat_watch":
        return f"repeat {venue} {asset}/{source} only at smaller size or find deeper venue"
    if gate_action == "rough_cost_blocked":
        return f"keep {venue} {asset}/{source} as label evidence until spread/cost improves"
    if gate_action == "source_not_in_current_context":
        return f"refresh {venue} context for {asset}/{source} before paper-checking"
    return f"collect execution context for {venue} {asset}/{source}"


def _split_asset_source(group_key: str) -> tuple[str, str]:
    parts = group_key.split("/", 1)
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _split_sources(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def _sort_key(row: RepeatExecutionGateRow) -> tuple[int, float, float, int]:
    action_rank = {
        "small_repeat_paper_check": 4,
        "thin_depth_repeat_watch": 3,
        "rough_cost_blocked": 2,
        "source_not_in_current_context": 1,
        "missing_venue_context": 0,
    }.get(row.gate_action, 0)
    return (
        action_rank,
        -1.0 if row.rough_net15_bps is None else row.rough_net15_bps,
        row.mean_dir15_bps,
        row.label_count,
    )


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


def _int(value: object) -> int:
    try:
        return int(float(value or 0.0))
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=Path, default=ROOT / "current_followup_repeat_history_summary.csv")
    parser.add_argument("--hl-context-path", type=Path, default=ROOT / "current_followup_execution_context.csv")
    parser.add_argument("--okx-context-path", type=Path, default=ROOT / "current_followup_okx_execution_context.csv")
    parser.add_argument("--taker-cost-bps", type=float, default=8.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_repeat_execution_gate.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_repeat_execution_gate.md")
    args = parser.parse_args()

    rows = build_repeat_execution_gate_rows(
        summary_path=args.summary_path,
        hl_context_path=args.hl_context_path,
        okx_context_path=args.okx_context_path,
        taker_cost_bps=args.taker_cost_bps,
    )
    write_repeat_execution_gate_csv(rows, output_path=args.output_path)
    write_repeat_execution_gate_md(rows, output_path=args.md_output_path)
    for row in rows[:12]:
        print(
            row.asset,
            row.source,
            row.venue,
            row.gate_action,
            "" if row.rough_net15_bps is None else f"net15={row.rough_net15_bps:.2f}bps",
        )


if __name__ == "__main__":
    main()
