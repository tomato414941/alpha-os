from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIRECT_PROVIDER_EVIDENCE = (
    "CryptoQuant stablecoin exchange flow API: https://cryptoquant.com/en/docs; "
    "Glassnode exchange-flow metrics: https://docs.glassnode.com/data/metric-catalog; "
    "Arkham transfer/entity API: https://docs.intel.arkm.com/openapi/transfers"
)
RESEARCH_REFERENCE = "https://arxiv.org/abs/2411.06327"


@dataclass(frozen=True)
class ExchangeStablecoinInflowReadinessRow:
    subject: str
    status: str
    alpha_kind: str
    readiness_score: float
    chain: str
    token_symbol: str
    flow_direction: str
    week_change_usd: float
    week_change_pct: float
    directional_return_1h: str
    directional_return_4h: str
    provider_evidence: str
    current_proxy_evidence: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_exchange_stablecoin_inflow_readiness_rows(
    *,
    proxy_path: Path = ROOT / "current_stablecoin_exchange_inflow_proxy.csv",
) -> tuple[ExchangeStablecoinInflowReadinessRow, ...]:
    rows = tuple(_row_from_proxy(row) for row in _read_rows(proxy_path))
    return tuple(sorted(rows, key=lambda row: row.readiness_score, reverse=True))


def write_exchange_stablecoin_inflow_readiness_csv(
    rows: tuple[ExchangeStablecoinInflowReadinessRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "subject",
                "status",
                "alpha_kind",
                "readiness_score",
                "chain",
                "token_symbol",
                "flow_direction",
                "week_change_usd",
                "week_change_pct",
                "directional_return_1h",
                "directional_return_4h",
                "provider_evidence",
                "current_proxy_evidence",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.subject,
                    row.status,
                    row.alpha_kind,
                    f"{row.readiness_score:.8f}",
                    row.chain,
                    row.token_symbol,
                    row.flow_direction,
                    f"{row.week_change_usd:.8f}",
                    f"{row.week_change_pct:.8f}",
                    row.directional_return_1h,
                    row.directional_return_4h,
                    row.provider_evidence,
                    row.current_proxy_evidence,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_exchange_stablecoin_inflow_readiness_md(
    rows: tuple[ExchangeStablecoinInflowReadinessRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Exchange Stablecoin Inflow Readiness\n\n")
        handle.write(
            "This separates direct exchange-stablecoin-inflow alpha from chain-level stablecoin migration proxies. "
            "The current local data can support proxy labels, but direct exchange-inflow alpha needs tagged exchange deposits.\n\n"
        )
        handle.write(
            "| subject | status | alpha kind | score | flow | week change | week % | proxy label | next probe |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |\n")
        for row in rows[:30]:
            label = f"1h={row.directional_return_1h or '-'}, 4h={row.directional_return_4h or '-'}"
            handle.write(
                "| "
                f"{row.subject} | "
                f"{row.status} | "
                f"{row.alpha_kind} | "
                f"{row.readiness_score:.4f} | "
                f"{row.flow_direction} | "
                f"{row.week_change_usd:.0f} | "
                f"{row.week_change_pct:.6f} | "
                f"{label} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _row_from_proxy(row: dict[str, str]) -> ExchangeStablecoinInflowReadinessRow:
    chain = row.get("chain", "")
    token = row.get("token_symbol", "")
    status = _status(row)
    alpha_kind = _alpha_kind(status)
    week_change_usd = _float(row.get("week_change_usd"))
    week_change_pct = _float(row.get("week_change_pct"))
    proxy_priority = _float(row.get("priority"))
    return ExchangeStablecoinInflowReadinessRow(
        subject=f"{chain}/{token or 'unmapped'}",
        status=status,
        alpha_kind=alpha_kind,
        readiness_score=_readiness_score(status=status, proxy_priority=proxy_priority, week_change_usd=week_change_usd),
        chain=chain,
        token_symbol=token,
        flow_direction=row.get("stablecoin_flow_direction", ""),
        week_change_usd=week_change_usd,
        week_change_pct=week_change_pct,
        directional_return_1h=row.get("directional_return_1h", ""),
        directional_return_4h=row.get("directional_return_4h", ""),
        provider_evidence=DIRECT_PROVIDER_EVIDENCE,
        current_proxy_evidence=(
            f"proxy_status={row.get('status', '')}; "
            f"exchange_interpretation={row.get('exchange_inflow_interpretation', '')}; "
            f"chain_interpretation={row.get('chain_liquidity_interpretation', '')}"
        ),
        missing_data=_missing_data(status),
        next_probe=_next_probe(status=status, chain=chain, token=token),
    )


def _status(row: dict[str, str]) -> str:
    proxy_status = row.get("status", "")
    token = row.get("token_symbol", "")
    if proxy_status == "needs_exchange_wallet_map_before_exchange_inflow_alpha":
        return "direct_exchange_inflow_data_required"
    if proxy_status == "chain_liquidity_proxy_alpha_candidate":
        return "proxy_label_candidate_not_exchange_inflow"
    if not token:
        return "unmapped_chain_context_not_alpha"
    return "chain_proxy_watch_not_exchange_inflow"


def _alpha_kind(status: str) -> str:
    if status == "direct_exchange_inflow_data_required":
        return "direct_exchange_stablecoin_inflow"
    if status == "proxy_label_candidate_not_exchange_inflow":
        return "chain_liquidity_proxy"
    return "context_only"


def _readiness_score(*, status: str, proxy_priority: float, week_change_usd: float) -> float:
    status_bonus = {
        "direct_exchange_inflow_data_required": 120.0,
        "proxy_label_candidate_not_exchange_inflow": 75.0,
        "chain_proxy_watch_not_exchange_inflow": 35.0,
        "unmapped_chain_context_not_alpha": 10.0,
    }.get(status, 0.0)
    size_score = min(abs(week_change_usd) / 100_000_000.0, 30.0)
    return status_bonus + proxy_priority * 0.35 + size_score


def _missing_data(status: str) -> str:
    if status == "direct_exchange_inflow_data_required":
        return "tagged exchange stablecoin deposits, provider access token, exchange/entity labels, and 1h return labels"
    if status == "proxy_label_candidate_not_exchange_inflow":
        return "chain proxy forward labels, beta control, bridge route, funding PnL, spread/depth, and venue mapping"
    return "tradable token mapping and forward labels"


def _next_probe(*, status: str, chain: str, token: str) -> str:
    if status == "direct_exchange_inflow_data_required":
        return f"obtain or emulate exchange-tagged stablecoin netflow for {token}; keep {chain} chain supply out of the direct alpha label"
    if status == "proxy_label_candidate_not_exchange_inflow":
        return f"label {token} as chain-liquidity proxy, then compare against beta and costs before any trade"
    return f"keep {chain} as context until mapped to a tradable token and labeled"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_exchange_stablecoin_inflow_readiness.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_exchange_stablecoin_inflow_readiness.md")
    args = parser.parse_args()

    rows = build_exchange_stablecoin_inflow_readiness_rows()
    write_exchange_stablecoin_inflow_readiness_csv(rows, output_path=args.output_path)
    write_exchange_stablecoin_inflow_readiness_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.subject, f"{row.readiness_score:.4f}")


if __name__ == "__main__":
    main()
