from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.current_symbol_opportunity_map import _symbols_for_stack_row


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SymbolClusterConflictRow:
    symbol: str
    status: str
    cluster_score: float
    symbol_cluster_score: float
    source_count: int
    candidate_count: int
    long_count: int
    short_count: int
    relative_value_count: int
    yield_count: int
    risk_or_avoid_count: int
    neutral_count: int
    dominant_bias: str
    top_opportunities: str
    conflicts: str
    next_step: str


def build_symbol_cluster_conflict_rows(
    *,
    stack_path: Path = ROOT / "current_alpha_stack.csv",
    symbol_map_path: Path = ROOT / "current_symbol_opportunity_map.csv",
) -> tuple[SymbolClusterConflictRow, ...]:
    symbol_map = {row.get("symbol", ""): row for row in _read_rows(symbol_map_path)}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(stack_path):
        for symbol in _symbols_for_stack_row(row):
            grouped.setdefault(symbol, []).append(row)
    rows = tuple(
        _build_cluster_conflict_row(
            symbol=symbol,
            rows=cluster_rows,
            symbol_map_row=symbol_map.get(symbol, {}),
        )
        for symbol, cluster_rows in grouped.items()
    )
    return tuple(sorted(rows, key=lambda row: row.cluster_score, reverse=True))


def write_symbol_cluster_conflict_csv(
    rows: tuple[SymbolClusterConflictRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "cluster_score",
                "symbol_cluster_score",
                "source_count",
                "candidate_count",
                "long_count",
                "short_count",
                "relative_value_count",
                "yield_count",
                "risk_or_avoid_count",
                "neutral_count",
                "dominant_bias",
                "top_opportunities",
                "conflicts",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.status,
                    f"{row.cluster_score:.8f}",
                    f"{row.symbol_cluster_score:.8f}",
                    row.source_count,
                    row.candidate_count,
                    row.long_count,
                    row.short_count,
                    row.relative_value_count,
                    row.yield_count,
                    row.risk_or_avoid_count,
                    row.neutral_count,
                    row.dominant_bias,
                    row.top_opportunities,
                    row.conflicts,
                    row.next_step,
                )
            )
    return output_path


def write_symbol_cluster_conflict_md(
    rows: tuple[SymbolClusterConflictRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Symbol Cluster Conflicts\n\n")
        handle.write(
            "This separates symbol clusters that confirm one direction from clusters that mix "
            "directional, relative-value, yield, and risk-avoidance ideas. It is a conflict screen, "
            "not a trade list.\n\n"
        )
        handle.write(
            "| symbol | status | score | sources | candidates | bias counts | dominant | top opportunities | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            bias_counts = (
                f"L={row.long_count}, S={row.short_count}, RV={row.relative_value_count}, "
                f"Y={row.yield_count}, R={row.risk_or_avoid_count}, N={row.neutral_count}"
            )
            handle.write(
                f"| {row.symbol} | {row.status} | {row.cluster_score:.4f} | "
                f"{row.source_count} | {row.candidate_count} | {bias_counts} | "
                f"{row.dominant_bias} | {_escape(row.top_opportunities)} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`confirmed_*` rows are the cleanest next paper-label candidates. "
            "`mixed_*` rows are often more interesting, but they need a label that separates "
            "which lane is actually driving returns before any action.\n"
        )
    return output_path


def _build_cluster_conflict_row(
    *,
    symbol: str,
    rows: list[dict[str, str]],
    symbol_map_row: dict[str, str],
) -> SymbolClusterConflictRow:
    sorted_rows = sorted(rows, key=lambda row: _float(row.get("priority_score")), reverse=True)
    biases = tuple(_bias_for_row(row) for row in sorted_rows)
    counts = {bias: biases.count(bias) for bias in {"long", "short", "relative_value", "yield", "risk_or_avoid", "neutral"}}
    source_count = _int(symbol_map_row.get("source_count")) or len(
        {source.strip() for row in sorted_rows for source in row.get("sources", "").split("+") if source.strip()}
    )
    candidate_count = _int(symbol_map_row.get("candidate_count")) or len(sorted_rows)
    symbol_cluster_score = _float(symbol_map_row.get("cluster_score"))
    dominant_bias = _dominant_bias(counts)
    status = _status_for_counts(counts=counts, source_count=source_count, candidate_count=candidate_count)
    cluster_score = _cluster_score(
        symbol_cluster_score=symbol_cluster_score,
        source_count=source_count,
        candidate_count=candidate_count,
        status=status,
    )
    opportunities = tuple(row.get("opportunity", "") for row in sorted_rows[:5])
    conflicts = tuple(_unique(row.get("conflict", "") for row in sorted_rows if row.get("conflict", "")))[:3]
    return SymbolClusterConflictRow(
        symbol=symbol,
        status=status,
        cluster_score=cluster_score,
        symbol_cluster_score=symbol_cluster_score,
        source_count=source_count,
        candidate_count=candidate_count,
        long_count=counts["long"],
        short_count=counts["short"],
        relative_value_count=counts["relative_value"],
        yield_count=counts["yield"],
        risk_or_avoid_count=counts["risk_or_avoid"],
        neutral_count=counts["neutral"],
        dominant_bias=dominant_bias,
        top_opportunities=", ".join(opportunities),
        conflicts="; ".join(conflicts),
        next_step=_next_step_for_status(symbol=symbol, status=status, dominant_bias=dominant_bias),
    )


def _bias_for_row(row: dict[str, str]) -> str:
    side = row.get("side", "").lower()
    if "long_token_or_relative_value" in side:
        return "long"
    if any(token in side for token in ("long_mstr_short", "short_future_long", "long_future_short")):
        return "relative_value"
    if side.startswith("short_") or side == "short":
        return "short"
    if side.startswith("long_") or side == "long":
        return "long"
    value = " ".join(
        (
            side,
            row.get("status", ""),
            row.get("opportunity", ""),
            row.get("next_step", ""),
        )
    ).lower()
    if any(token in value for token in ("relative_value", "basis", "calendar_spread", "spread")):
        return "relative_value"
    if any(token in value for token in ("yield", "lending", "borrow", "apy")):
        return "yield"
    if any(token in value for token in ("avoid", "risk", "depeg", "premium_reversion", "repeg")):
        return "risk_or_avoid"
    if any(token in value for token in ("short", "outflow", "unlock", "sell")):
        return "short"
    if any(token in value for token in ("long", "inflow", "growth", "momentum", "squeeze")):
        return "long"
    return "neutral"


def _status_for_counts(*, counts: dict[str, int], source_count: int, candidate_count: int) -> str:
    directional_sides = int(counts["long"] > 0) + int(counts["short"] > 0)
    structure_sides = int(counts["relative_value"] > 0) + int(counts["yield"] > 0) + int(counts["risk_or_avoid"] > 0)
    if counts["long"] > 0 and counts["short"] > 0:
        return "mixed_direction_conflict"
    if directional_sides and structure_sides:
        return "mixed_structure_conflict"
    if counts["long"] >= 2:
        return "confirmed_long_cluster"
    if counts["short"] >= 2:
        return "confirmed_short_cluster"
    if counts["relative_value"] >= 2:
        return "relative_value_cluster"
    if counts["yield"] >= 2:
        return "yield_cluster"
    if counts["risk_or_avoid"] >= 2:
        return "risk_resolution_cluster"
    if source_count >= 2 or candidate_count >= 2:
        return "multi_source_watch"
    return "single_candidate_watch"


def _dominant_bias(counts: dict[str, int]) -> str:
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_bias, top_count = ordered[0]
    if top_count == 0:
        return "neutral"
    if len(ordered) > 1 and ordered[1][1] == top_count:
        return "mixed"
    return top_bias


def _cluster_score(*, symbol_cluster_score: float, source_count: int, candidate_count: int, status: str) -> float:
    conflict_bonus = 8.0 if status in {"mixed_direction_conflict", "mixed_structure_conflict"} else 0.0
    confirmation_bonus = 5.0 if status.startswith("confirmed_") else 0.0
    return symbol_cluster_score + min(source_count * 2.0, 8.0) + min(candidate_count, 8.0) + conflict_bonus + confirmation_bonus


def _next_step_for_status(*, symbol: str, status: str, dominant_bias: str) -> str:
    if status in {"mixed_direction_conflict", "mixed_structure_conflict"}:
        return f"split {symbol} labels by lane before trading; do not collapse conflicting ideas into one action"
    if status.startswith("confirmed_"):
        return f"paper-label {symbol} {dominant_bias} setup against forward return, costs, depth, and failure regime"
    if status in {"relative_value_cluster", "yield_cluster", "risk_resolution_cluster"}:
        return f"validate {symbol} {dominant_bias} mechanics, venue access, liquidity, fees, and unwind path"
    return f"collect more {symbol} snapshots before treating this as a cluster"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _unique(values: object) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _int(value: str | None) -> int:
    return int(value) if value else 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--symbol-map-path", type=Path, default=ROOT / "current_symbol_opportunity_map.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_symbol_cluster_conflicts.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_symbol_cluster_conflicts.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_symbol_cluster_conflict_rows(stack_path=args.stack_path, symbol_map_path=args.symbol_map_path)
    write_symbol_cluster_conflict_csv(rows, output_path=args.output_path)
    write_symbol_cluster_conflict_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.status,
            row.symbol,
            f"bias={row.dominant_bias}",
            f"score={row.cluster_score:.4f}",
        )


if __name__ == "__main__":
    main()
