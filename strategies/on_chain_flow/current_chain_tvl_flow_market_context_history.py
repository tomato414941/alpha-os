from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChainTvlFlowMarketContextHistoryRow:
    snapshot_timestamp: str
    venue: str
    token_symbol: str
    action: str
    direction: int
    directional_return_15m: float | None
    annualized_funding: float | None
    funding_support: float | None
    liquidity_usd: float | None
    friction_bps: float | None
    okx_liquidation_action: str
    okx_liquidation_score: float | None
    context_score: float
    note: str


@dataclass(frozen=True)
class ChainTvlFlowMarketContextSummaryRow:
    group_type: str
    group_key: str
    observations: int
    hit_rate_15m: float | None
    mean_dir15: float | None
    mean_funding_support: float | None
    mean_context_score: float
    action: str
    evidence: str


def build_chain_tvl_flow_market_context_history_rows(
    *,
    history_path: Path = ROOT / "chain_tvl_flow_market_context_history.csv",
    current_path: Path = ROOT / "current_chain_tvl_flow_market_context.csv",
    top_rows: int = 25,
) -> tuple[ChainTvlFlowMarketContextHistoryRow, ...]:
    existing = tuple(_history_row(row) for row in _read_rows(history_path))
    current = tuple(
        _current_row(
            row=row,
            fallback_timestamp=datetime.now(UTC).isoformat(),
        )
        for row in _read_rows(current_path)[:top_rows]
    )
    rows_by_key = {_key(row): row for row in existing}
    for row in current:
        rows_by_key.setdefault(_key(row), row)
    return tuple(sorted(rows_by_key.values(), key=lambda row: row.snapshot_timestamp))


def build_chain_tvl_flow_market_context_summary_rows(
    rows: tuple[ChainTvlFlowMarketContextHistoryRow, ...],
) -> tuple[ChainTvlFlowMarketContextSummaryRow, ...]:
    grouped: dict[tuple[str, str], list[ChainTvlFlowMarketContextHistoryRow]] = {}
    for row in rows:
        for group_type, group_key in _group_keys(row):
            grouped.setdefault((group_type, group_key), []).append(row)
    summary = tuple(
        _summary_row(group_type=group_type, group_key=group_key, rows=group_rows)
        for (group_type, group_key), group_rows in grouped.items()
    )
    return tuple(sorted(summary, key=_summary_sort_key, reverse=True))


def write_chain_tvl_flow_market_context_history_csv(
    rows: tuple[ChainTvlFlowMarketContextHistoryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "snapshot_timestamp",
                "venue",
                "token_symbol",
                "action",
                "direction",
                "directional_return_15m",
                "annualized_funding",
                "funding_support",
                "liquidity_usd",
                "friction_bps",
                "okx_liquidation_action",
                "okx_liquidation_score",
                "context_score",
                "note",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.snapshot_timestamp,
                    row.venue,
                    row.token_symbol,
                    row.action,
                    row.direction,
                    _format_float(row.directional_return_15m),
                    _format_float(row.annualized_funding),
                    _format_float(row.funding_support),
                    _format_float(row.liquidity_usd),
                    _format_float(row.friction_bps),
                    row.okx_liquidation_action,
                    _format_float(row.okx_liquidation_score),
                    f"{row.context_score:.8f}",
                    row.note,
                )
            )
    return output_path


def write_chain_tvl_flow_market_context_history_md(
    rows: tuple[ChainTvlFlowMarketContextHistoryRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Chain TVL Flow Market Context History\n\n")
        handle.write(
            "This stores repeated chain-flow market-context snapshots. Current "
            "screens can be regenerated; this file keeps the evidence trail.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- distinct tokens: `{len({row.token_symbol for row in rows})}`\n\n")
        handle.write(
            "| timestamp | venue | token | action | dir15 | funding support | score | note |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in tuple(sorted(rows, key=lambda item: item.snapshot_timestamp, reverse=True))[
            :top
        ]:
            handle.write(
                "| "
                f"{row.snapshot_timestamp} | "
                f"{row.venue} | "
                f"{row.token_symbol} | "
                f"{row.action} | "
                f"{_format_float(row.directional_return_15m)} | "
                f"{_format_float(row.funding_support)} | "
                f"{row.context_score:.6f} | "
                f"{row.note} |\n"
            )
    return output_path


def write_chain_tvl_flow_market_context_summary_csv(
    rows: tuple[ChainTvlFlowMarketContextSummaryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_type",
                "group_key",
                "observations",
                "hit_rate_15m",
                "mean_dir15",
                "mean_funding_support",
                "mean_context_score",
                "action",
                "evidence",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.group_type,
                    row.group_key,
                    row.observations,
                    _format_float(row.hit_rate_15m),
                    _format_float(row.mean_dir15),
                    _format_float(row.mean_funding_support),
                    f"{row.mean_context_score:.8f}",
                    row.action,
                    row.evidence,
                )
            )
    return output_path


def write_chain_tvl_flow_market_context_summary_md(
    rows: tuple[ChainTvlFlowMarketContextSummaryRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Chain TVL Flow Market Context Summary\n\n")
        handle.write(
            "This aggregates repeated chain-flow market-context snapshots. A "
            "single observation is not enough to promote a candidate.\n\n"
        )
        handle.write(
            "| group type | group | obs | hit 15m | mean dir15 | mean funding support | mean score | action | evidence |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.group_type} | "
                f"{row.group_key} | "
                f"{row.observations} | "
                f"{_format_float(row.hit_rate_15m)} | "
                f"{_format_float(row.mean_dir15)} | "
                f"{_format_float(row.mean_funding_support)} | "
                f"{row.mean_context_score:.6f} | "
                f"{row.action} | "
                f"{row.evidence} |\n"
            )
    return output_path


def _summary_row(
    *,
    group_type: str,
    group_key: str,
    rows: list[ChainTvlFlowMarketContextHistoryRow],
) -> ChainTvlFlowMarketContextSummaryRow:
    dir15_values = tuple(
        row.directional_return_15m for row in rows if row.directional_return_15m is not None
    )
    funding_support_values = tuple(
        row.funding_support for row in rows if row.funding_support is not None
    )
    observations = len({row.snapshot_timestamp for row in rows})
    mean_context_score = sum(row.context_score for row in rows) / len(rows)
    hit_rate = (
        None
        if not dir15_values
        else sum(1 for value in dir15_values if value > 0.0) / len(dir15_values)
    )
    mean_dir15 = None if not dir15_values else sum(dir15_values) / len(dir15_values)
    mean_funding_support = (
        None
        if not funding_support_values
        else sum(funding_support_values) / len(funding_support_values)
    )
    return ChainTvlFlowMarketContextSummaryRow(
        group_type=group_type,
        group_key=group_key,
        observations=observations,
        hit_rate_15m=hit_rate,
        mean_dir15=mean_dir15,
        mean_funding_support=mean_funding_support,
        mean_context_score=mean_context_score,
        action=_action(
            observations=observations,
            hit_rate=hit_rate,
            mean_dir15=mean_dir15,
            mean_context_score=mean_context_score,
        ),
        evidence=";".join(
            f"{row.venue}/{row.token_symbol}/{row.context_score:.3f}" for row in rows[:3]
        ),
    )


def _action(
    *,
    observations: int,
    hit_rate: float | None,
    mean_dir15: float | None,
    mean_context_score: float,
) -> str:
    if observations < 2:
        return "collect_repeat"
    if (
        hit_rate is not None
        and mean_dir15 is not None
        and hit_rate >= 0.75
        and mean_dir15 > 0.001
        and mean_context_score > 0.2
    ):
        return "repeat_priority"
    if hit_rate is not None and hit_rate <= 0.25 and mean_context_score < 0.0:
        return "deprioritize"
    return "keep_sampling"


def _group_keys(
    row: ChainTvlFlowMarketContextHistoryRow,
) -> tuple[tuple[str, str], ...]:
    return (
        ("token", row.token_symbol),
        ("venue_token", f"{row.venue}/{row.token_symbol}"),
        ("action", row.action),
        ("venue_action", f"{row.venue}/{row.action}"),
    )


def _summary_sort_key(row: ChainTvlFlowMarketContextSummaryRow) -> tuple[int, float, int]:
    action_rank = {
        "repeat_priority": 3,
        "keep_sampling": 2,
        "collect_repeat": 1,
        "deprioritize": -1,
    }.get(row.action, 0)
    return (action_rank, row.mean_context_score, row.observations)


def _current_row(
    *,
    row: dict[str, str],
    fallback_timestamp: str,
) -> ChainTvlFlowMarketContextHistoryRow:
    return ChainTvlFlowMarketContextHistoryRow(
        snapshot_timestamp=row.get("signal_timestamp") or fallback_timestamp,
        venue=row["venue"],
        token_symbol=row["token_symbol"],
        action=row["action"],
        direction=int(row.get("direction") or "0"),
        directional_return_15m=_float_or_none(row.get("directional_return_15m", "")),
        annualized_funding=_float_or_none(row.get("annualized_funding", "")),
        funding_support=_float_or_none(row.get("funding_support", "")),
        liquidity_usd=_float_or_none(row.get("liquidity_usd", "")),
        friction_bps=_float_or_none(row.get("friction_bps", "")),
        okx_liquidation_action=row.get("okx_liquidation_action", ""),
        okx_liquidation_score=_float_or_none(row.get("okx_liquidation_score", "")),
        context_score=float(row.get("context_score") or "0"),
        note=row.get("note", ""),
    )


def _history_row(row: dict[str, str]) -> ChainTvlFlowMarketContextHistoryRow:
    return ChainTvlFlowMarketContextHistoryRow(
        snapshot_timestamp=row["snapshot_timestamp"],
        venue=row["venue"],
        token_symbol=row["token_symbol"],
        action=row["action"],
        direction=int(row.get("direction") or "0"),
        directional_return_15m=_float_or_none(row.get("directional_return_15m", "")),
        annualized_funding=_float_or_none(row.get("annualized_funding", "")),
        funding_support=_float_or_none(row.get("funding_support", "")),
        liquidity_usd=_float_or_none(row.get("liquidity_usd", "")),
        friction_bps=_float_or_none(row.get("friction_bps", "")),
        okx_liquidation_action=row.get("okx_liquidation_action", ""),
        okx_liquidation_score=_float_or_none(row.get("okx_liquidation_score", "")),
        context_score=float(row.get("context_score") or "0"),
        note=row.get("note", ""),
    )


def _key(row: ChainTvlFlowMarketContextHistoryRow) -> tuple[str, str, str, str]:
    return (row.snapshot_timestamp, row.venue, row.token_symbol, row.action)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "chain_tvl_flow_market_context_history.csv",
    )
    parser.add_argument(
        "--history-md-output-path",
        type=Path,
        default=ROOT / "chain_tvl_flow_market_context_history.md",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "chain_tvl_flow_market_context_summary.csv",
    )
    parser.add_argument(
        "--summary-md-output-path",
        type=Path,
        default=ROOT / "chain_tvl_flow_market_context_summary.md",
    )
    parser.add_argument("--top-rows", type=int, default=25)
    args = parser.parse_args()

    history_rows = build_chain_tvl_flow_market_context_history_rows(
        history_path=args.history_path,
        top_rows=args.top_rows,
    )
    summary_rows = build_chain_tvl_flow_market_context_summary_rows(history_rows)
    write_chain_tvl_flow_market_context_history_csv(
        history_rows,
        output_path=args.history_path,
    )
    write_chain_tvl_flow_market_context_history_md(
        history_rows,
        output_path=args.history_md_output_path,
    )
    write_chain_tvl_flow_market_context_summary_csv(
        summary_rows,
        output_path=args.summary_output_path,
    )
    write_chain_tvl_flow_market_context_summary_md(
        summary_rows,
        output_path=args.summary_md_output_path,
    )
    for row in summary_rows[:10]:
        print(
            row.group_type,
            row.group_key,
            f"obs={row.observations}",
            f"mean_score={row.mean_context_score:.4f}",
            row.action,
        )


if __name__ == "__main__":
    main()
