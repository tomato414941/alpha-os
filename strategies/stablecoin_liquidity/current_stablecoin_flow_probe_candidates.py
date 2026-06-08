from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StablecoinFlowProbeCandidate:
    candidate_id: str
    subject: str
    candidate_type: str
    status: str
    priority: float
    chain: str
    token_symbol: str
    flow_direction: str
    week_change_usd: str
    week_change_pct: str
    market_context_score: str
    required_record: str
    next_step: str


def build_stablecoin_flow_probe_candidates(
    *,
    readiness_path: Path = ROOT / "current_exchange_stablecoin_inflow_readiness.csv",
    market_context_path: Path = ROOT.parent / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
) -> tuple[StablecoinFlowProbeCandidate, ...]:
    contexts = _best_context_by_symbol(market_context_path)
    rows = tuple(
        _candidate_row(row, context=contexts.get(row.get("token_symbol", ""), {}))
        for row in _read_rows(readiness_path)
        if row.get("status") in {"direct_exchange_inflow_data_required", "proxy_label_candidate_not_exchange_inflow"}
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_stablecoin_flow_probe_candidates_csv(
    rows: tuple[StablecoinFlowProbeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "subject",
                "candidate_type",
                "status",
                "priority",
                "chain",
                "token_symbol",
                "flow_direction",
                "week_change_usd",
                "week_change_pct",
                "market_context_score",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.subject,
                    row.candidate_type,
                    row.status,
                    f"{row.priority:.8f}",
                    row.chain,
                    row.token_symbol,
                    row.flow_direction,
                    row.week_change_usd,
                    row.week_change_pct,
                    row.market_context_score,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_stablecoin_flow_probe_candidates_md(
    rows: tuple[StablecoinFlowProbeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Stablecoin Flow Probe Candidates\n\n")
        handle.write(
            "This separates direct exchange-flow data probes from chain-liquidity proxy labels. "
            "Chain stablecoin supply is not treated as exchange inflow.\n\n"
        )
        handle.write(
            "| candidate | subject | type | status | priority | flow | week change | context | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.candidate_id} | "
                f"{row.subject} | "
                f"{row.candidate_type} | "
                f"{row.status} | "
                f"{row.priority:.4f} | "
                f"{row.flow_direction} | "
                f"{row.week_change_usd} | "
                f"{row.market_context_score} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _candidate_row(row: dict[str, str], *, context: dict[str, str]) -> StablecoinFlowProbeCandidate:
    status = row.get("status", "")
    candidate_type = _candidate_type(status)
    context_score = context.get("context_score", "")
    return StablecoinFlowProbeCandidate(
        candidate_id=f"stablecoin-flow-{_slug(row.get('subject', ''))}",
        subject=row.get("subject", ""),
        candidate_type=candidate_type,
        status=status,
        priority=_priority(row=row, context=context),
        chain=row.get("chain", ""),
        token_symbol=row.get("token_symbol", ""),
        flow_direction=row.get("flow_direction", ""),
        week_change_usd=row.get("week_change_usd", ""),
        week_change_pct=row.get("week_change_pct", ""),
        market_context_score=context_score,
        required_record=_required_record(candidate_type),
        next_step=_next_step(candidate_type=candidate_type, row=row),
    )


def _candidate_type(status: str) -> str:
    if status == "direct_exchange_inflow_data_required":
        return "direct_exchange_flow_data_probe"
    return "chain_liquidity_proxy_label"


def _priority(*, row: dict[str, str], context: dict[str, str]) -> float:
    return _float(row.get("readiness_score")) + _float(context.get("context_score")) * 20.0


def _required_record(candidate_type: str) -> str:
    if candidate_type == "direct_exchange_flow_data_probe":
        return "exchange-tagged stablecoin deposits, timestamp, exchange/entity label, asset mapping, 1h return label"
    return "chain supply timestamp, bridge route, venue coverage, beta control, funding PnL, spread/depth, forward label"


def _next_step(*, candidate_type: str, row: dict[str, str]) -> str:
    token = row.get("token_symbol", "")
    if candidate_type == "direct_exchange_flow_data_probe":
        return f"collect or emulate exchange-tagged stablecoin netflow for {token}; do not use chain supply as the label"
    return f"open a chain-liquidity proxy label for {token} and compare it against beta, funding, spread, and depth"


def _best_context_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(path):
        symbol = row.get("token_symbol", "")
        if symbol:
            grouped.setdefault(symbol, []).append(row)
    return {
        symbol: max(rows, key=lambda row: _float(row.get("context_score")))
        for symbol, rows in grouped.items()
    }


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
    return value.lower().replace("/", "-").replace("_", "-").replace(" ", "-")


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readiness-path",
        type=Path,
        default=ROOT / "current_exchange_stablecoin_inflow_readiness.csv",
    )
    parser.add_argument(
        "--market-context-path",
        type=Path,
        default=ROOT.parent / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_stablecoin_flow_probe_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_stablecoin_flow_probe_candidates.md")
    args = parser.parse_args()

    rows = build_stablecoin_flow_probe_candidates(
        readiness_path=args.readiness_path,
        market_context_path=args.market_context_path,
    )
    write_stablecoin_flow_probe_candidates_csv(rows, output_path=args.output_path)
    write_stablecoin_flow_probe_candidates_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.candidate_type, row.subject, f"{row.priority:.4f}", row.next_step)


if __name__ == "__main__":
    main()
