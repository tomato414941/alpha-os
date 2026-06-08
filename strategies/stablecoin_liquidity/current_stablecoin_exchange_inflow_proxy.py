from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCHANGE_INFLOW_RESEARCH_REFERENCE = "https://arxiv.org/abs/2411.06327"
DIRECT_EXCHANGE_INFLOW_ASSETS = {"BTC", "ETH"}


@dataclass(frozen=True)
class StablecoinExchangeInflowProxy:
    chain: str
    token_symbol: str
    migration_status: str
    stablecoin_flow_direction: str
    week_change_usd: float
    week_change_pct: float
    top_asset: str
    directional_return_1h: str
    directional_return_4h: str
    exchange_inflow_interpretation: str
    chain_liquidity_interpretation: str
    status: str
    priority: float
    missing_data: str
    next_probe: str
    research_reference: str


def build_stablecoin_exchange_inflow_proxies(
    *,
    migration_path: Path = ROOT / "current_chain_stablecoin_migration.csv",
    label_path: Path = ROOT / "current_chain_stablecoin_migration_forward_labels.csv",
    top: int = 20,
) -> tuple[StablecoinExchangeInflowProxy, ...]:
    labels = {(row.get("chain", ""), row.get("token_symbol", "")): row for row in _read_rows(label_path)}
    rows = tuple(
        _build_proxy(row=row, label=labels.get((row.get("chain", ""), row.get("token_symbol", "")), {}))
        for row in _read_rows(migration_path)[:top]
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_stablecoin_exchange_inflow_proxies_csv(
    rows: tuple[StablecoinExchangeInflowProxy, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "chain",
                "token_symbol",
                "migration_status",
                "stablecoin_flow_direction",
                "week_change_usd",
                "week_change_pct",
                "top_asset",
                "directional_return_1h",
                "directional_return_4h",
                "exchange_inflow_interpretation",
                "chain_liquidity_interpretation",
                "status",
                "priority",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.chain,
                    row.token_symbol,
                    row.migration_status,
                    row.stablecoin_flow_direction,
                    f"{row.week_change_usd:.8f}",
                    f"{row.week_change_pct:.8f}",
                    row.top_asset,
                    row.directional_return_1h,
                    row.directional_return_4h,
                    row.exchange_inflow_interpretation,
                    row.chain_liquidity_interpretation,
                    row.status,
                    f"{row.priority:.8f}",
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_stablecoin_exchange_inflow_proxies_md(
    rows: tuple[StablecoinExchangeInflowProxy, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Stablecoin Exchange Inflow Proxy\n\n")
        handle.write(
            "This separates exchange-inflow alpha from chain-level stablecoin liquidity migration. "
            "Current DeFiLlama chain supply rows are not direct exchange-deposit flow.\n\n"
        )
        handle.write(
            "| chain | token | flow | week change | week % | status | priority | exchange interpretation | chain interpretation | next probe |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.chain} | "
                f"{row.token_symbol or '-'} | "
                f"{row.stablecoin_flow_direction} | "
                f"{row.week_change_usd:.0f} | "
                f"{row.week_change_pct:.6f} | "
                f"{row.status} | "
                f"{row.priority:.4f} | "
                f"{_escape(row.exchange_inflow_interpretation)} | "
                f"{_escape(row.chain_liquidity_interpretation)} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _build_proxy(*, row: dict[str, str], label: dict[str, str]) -> StablecoinExchangeInflowProxy:
    token = row.get("token_symbol", "")
    migration_status = row.get("status", "")
    week_change_usd = _float(row.get("week_change_usd"))
    week_change_pct = _float(row.get("week_change_pct"))
    flow_direction = _flow_direction(week_change_usd)
    exchange_interpretation = _exchange_interpretation(token=token)
    chain_interpretation = _chain_interpretation(migration_status=migration_status, token=token)
    status = _status(exchange_interpretation=exchange_interpretation, migration_status=migration_status, token=token)
    return StablecoinExchangeInflowProxy(
        chain=row.get("chain", ""),
        token_symbol=token,
        migration_status=migration_status,
        stablecoin_flow_direction=flow_direction,
        week_change_usd=week_change_usd,
        week_change_pct=week_change_pct,
        top_asset=row.get("top_asset", ""),
        directional_return_1h=label.get("directional_return_1h", ""),
        directional_return_4h=label.get("directional_return_4h", ""),
        exchange_inflow_interpretation=exchange_interpretation,
        chain_liquidity_interpretation=chain_interpretation,
        status=status,
        priority=_priority(week_change_usd=week_change_usd, week_change_pct=week_change_pct, status=status),
        missing_data=_missing_data(status),
        next_probe=_next_probe(status=status, token=token, chain=row.get("chain", "")),
        research_reference=EXCHANGE_INFLOW_RESEARCH_REFERENCE,
    )


def _flow_direction(week_change_usd: float) -> str:
    if week_change_usd > 0.0:
        return "stablecoin_supply_inflow"
    if week_change_usd < 0.0:
        return "stablecoin_supply_outflow"
    return "stablecoin_supply_flat"


def _exchange_interpretation(*, token: str) -> str:
    if token in DIRECT_EXCHANGE_INFLOW_ASSETS:
        return "exchange_inflow_research_target_but_wallet_map_missing"
    if token:
        return "chain_token_liquidity_proxy_not_exchange_inflow"
    return "chain_activity_proxy_not_exchange_inflow"


def _chain_interpretation(*, migration_status: str, token: str) -> str:
    if migration_status == "paper_chain_stablecoin_inflow_watch" and token:
        return "chain_liquidity_inflow_alpha_candidate"
    if migration_status == "paper_chain_stablecoin_outflow_watch" and token:
        return "chain_liquidity_outflow_alpha_candidate"
    if migration_status == "chain_stablecoin_flow_reversal_watch" and token:
        return "chain_liquidity_reversal_control"
    return "chain_liquidity_context_only"


def _status(*, exchange_interpretation: str, migration_status: str, token: str) -> str:
    if exchange_interpretation == "exchange_inflow_research_target_but_wallet_map_missing":
        return "needs_exchange_wallet_map_before_exchange_inflow_alpha"
    if migration_status.startswith("paper_chain_") and token:
        return "chain_liquidity_proxy_alpha_candidate"
    if token:
        return "chain_liquidity_proxy_watch"
    return "unmapped_chain_liquidity_context"


def _priority(*, week_change_usd: float, week_change_pct: float, status: str) -> float:
    status_bonus = {
        "needs_exchange_wallet_map_before_exchange_inflow_alpha": 70.0,
        "chain_liquidity_proxy_alpha_candidate": 60.0,
        "chain_liquidity_proxy_watch": 35.0,
        "unmapped_chain_liquidity_context": 20.0,
    }.get(status, 0.0)
    flow_score = min(abs(week_change_usd) / 50_000_000.0, 30.0)
    pct_score = min(abs(week_change_pct) * 120.0, 20.0)
    return status_bonus + flow_score + pct_score


def _missing_data(status: str) -> str:
    if status == "needs_exchange_wallet_map_before_exchange_inflow_alpha":
        return "exchange wallet map, tagged stablecoin deposits, and 1h exchange-inflow labels"
    if status == "chain_liquidity_proxy_alpha_candidate":
        return "chain-level forward labels, bridge route, venue coverage, funding, and spread/depth"
    return "tradable token mapping, forward labels, and venue coverage"


def _next_probe(*, status: str, token: str, chain: str) -> str:
    if status == "needs_exchange_wallet_map_before_exchange_inflow_alpha":
        return f"collect exchange-tagged stablecoin netflow for {token}; do not treat {chain} supply as exchange inflow yet"
    if status == "chain_liquidity_proxy_alpha_candidate":
        return f"label {token} as a chain-liquidity proxy and compare against funding, spread, and beta controls"
    return f"keep {chain} stablecoin migration as context until token mapping and labels improve"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_stablecoin_exchange_inflow_proxy.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_stablecoin_exchange_inflow_proxy.md")
    args = parser.parse_args()

    rows = build_stablecoin_exchange_inflow_proxies()
    write_stablecoin_exchange_inflow_proxies_csv(rows, output_path=args.output_path)
    write_stablecoin_exchange_inflow_proxies_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.chain, row.token_symbol, f"{row.priority:.4f}")


if __name__ == "__main__":
    main()
