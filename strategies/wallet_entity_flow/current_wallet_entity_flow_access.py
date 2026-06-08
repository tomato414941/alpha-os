from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class WalletEntityFlowAccessRow:
    source: str
    status: str
    endpoint: str
    requires_secret: str
    probe_result: str
    tradable_use: str
    limitation: str
    next_step: str


def build_wallet_entity_flow_access() -> tuple[WalletEntityFlowAccessRow, ...]:
    rows = [
        _hyperliquid_user_fills_access(),
        _hyperliquid_user_state_access(),
        _the_graph_hyperliquid_activity_access(),
        _arkham_entity_flow_access(),
        _chain_tvl_flow_proxy_access(),
    ]
    return tuple(rows)


def write_wallet_entity_flow_access_csv(
    rows: tuple[WalletEntityFlowAccessRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source",
                "status",
                "endpoint",
                "requires_secret",
                "probe_result",
                "tradable_use",
                "limitation",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.source,
                    row.status,
                    row.endpoint,
                    row.requires_secret,
                    row.probe_result,
                    row.tradable_use,
                    row.limitation,
                    row.next_step,
                )
            )
    return output_path


def write_wallet_entity_flow_access_md(
    rows: tuple[WalletEntityFlowAccessRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Wallet Entity Flow Access\n\n")
        handle.write(
            "This checks whether wallet/entity-flow data is reachable enough to become "
            "a concrete alpha lane. It is not a wallet-following strategy.\n\n"
        )
        handle.write("| source | status | secret | probe | tradable use | limitation | next step |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.source} | "
                f"{row.status} | "
                f"{row.requires_secret} | "
                f"{_escape(row.probe_result)} | "
                f"{_escape(row.tradable_use)} | "
                f"{_escape(row.limitation)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _hyperliquid_user_fills_access() -> WalletEntityFlowAccessRow:
    result = _post_hyperliquid({"type": "userFills", "user": ZERO_ADDRESS, "aggregateByTime": True})
    return WalletEntityFlowAccessRow(
        source="hyperliquid_user_fills",
        status="access_ok" if result.startswith("http_200") else "access_problem",
        endpoint=HYPERLIQUID_INFO_URL,
        requires_secret="no",
        probe_result=result,
        tradable_use="wallet-level fill history can become order-flow or entity-flow labels once seed wallets exist",
        limitation="public endpoint needs concrete wallet addresses; zero-address probe only checks reachability",
        next_step="collect seed wallets from explicit sources before building wallet-follow labels",
    )


def _hyperliquid_user_state_access() -> WalletEntityFlowAccessRow:
    result = _post_hyperliquid({"type": "clearinghouseState", "user": ZERO_ADDRESS})
    return WalletEntityFlowAccessRow(
        source="hyperliquid_user_state",
        status="access_ok" if result.startswith("http_200") else "access_problem",
        endpoint=HYPERLIQUID_INFO_URL,
        requires_secret="no",
        probe_result=result,
        tradable_use="wallet-level positions can expose crowded whale direction once seed wallets exist",
        limitation="no entity labels, no wallet selection, and no causal trade rule yet",
        next_step="pair user state with a curated wallet universe and forward labels",
    )


def _the_graph_hyperliquid_activity_access() -> WalletEntityFlowAccessRow:
    has_key = bool(os.environ.get("THE_GRAPH_TOKEN_API_KEY"))
    return WalletEntityFlowAccessRow(
        source="the_graph_hyperliquid_activity",
        status="secret_available" if has_key else "needs_api_key",
        endpoint="https://token-api.thegraph.com/v1/hyperliquid/markets/activity",
        requires_secret="THE_GRAPH_TOKEN_API_KEY",
        probe_result="not_called_without_key",
        tradable_use="chronological fills by coin/user can support wallet-flow and liquidation-flow labels",
        limitation="API key required; endpoint still needs cost, latency, and wallet/entity selection discipline",
        next_step="if a token API key is available, pull recent HYPE/SOL/NEAR activity and label wallet-flow pressure",
    )


def _arkham_entity_flow_access() -> WalletEntityFlowAccessRow:
    has_key = bool(os.environ.get("ARKHAM_API_KEY"))
    return WalletEntityFlowAccessRow(
        source="arkham_entity_flow",
        status="secret_available" if has_key else "needs_api_key",
        endpoint="https://api.arkm.com / Arkham Intel API",
        requires_secret="ARKHAM_API_KEY",
        probe_result="not_called_without_key",
        tradable_use="entity-labeled exchange flows can support ETF, whale, market-maker, and treasury-flow labels",
        limitation="paid/keyed entity data; privacy and label quality must be treated as model risk",
        next_step="if an Arkham key is available, start with exchange inflow/outflow for BTC/ETH/SOL and label forward returns",
    )


def _chain_tvl_flow_proxy_access() -> WalletEntityFlowAccessRow:
    proxy_path = Path(__file__).resolve().parents[1] / "on_chain_flow" / "current_chain_tvl_flow.csv"
    exists = proxy_path.exists()
    return WalletEntityFlowAccessRow(
        source="chain_tvl_flow_proxy",
        status="implemented_proxy" if exists else "missing_proxy",
        endpoint=str(proxy_path),
        requires_secret="no",
        probe_result="local_proxy_exists" if exists else "missing",
        tradable_use="chain-level TVL flow is a coarse proxy for capital movement",
        limitation="not wallet/entity flow; cannot distinguish whales, exchanges, market makers, or treasury actions",
        next_step="keep this as proxy context, but do not treat it as wallet/entity-flow alpha",
    )


def _post_hyperliquid(payload: dict[str, object]) -> str:
    try:
        response = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=20)
    except requests.RequestException as exc:
        return f"request_error:{type(exc).__name__}"
    try:
        body = response.json()
    except ValueError:
        return f"http_{response.status_code}:non_json"
    if isinstance(body, list):
        return f"http_{response.status_code}:list_len={len(body)}"
    if isinstance(body, dict):
        keys = ",".join(sorted(body.keys())[:5])
        return f"http_{response.status_code}:dict_keys={keys}"
    return f"http_{response.status_code}:{type(body).__name__}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_wallet_entity_flow_access.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_wallet_entity_flow_access.md")
    args = parser.parse_args()
    rows = build_wallet_entity_flow_access()
    write_wallet_entity_flow_access_csv(rows, output_path=args.output_path)
    write_wallet_entity_flow_access_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.source, row.status, row.probe_result)


if __name__ == "__main__":
    main()
