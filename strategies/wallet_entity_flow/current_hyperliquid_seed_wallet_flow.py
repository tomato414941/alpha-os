from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class SeedWallet:
    label: str
    address: str
    source: str
    source_url: str
    caveat: str


@dataclass(frozen=True)
class SeedWalletFlowRow:
    timestamp_ms: int
    wallet_label: str
    address: str
    source: str
    coin: str
    fills: int
    buy_notional: float
    sell_notional: float
    net_buy_notional: float
    closed_pnl: float
    fees: float
    net_closed_pnl_after_fees: float
    current_position: float
    current_position_notional: float
    action: str
    score: float
    caveat: str
    next_step: str


SEED_WALLETS = (
    SeedWallet(
        label="public_hypertracker_example",
        address="0x831ea8a4a4d7ea2657ba48f8c074d69bdaece05c",
        source="public_reddit_thread",
        source_url="https://www.reddit.com/r/algoprojects/comments/1jjy4bm/i_built_a_hyperliquid_trader_tracking_app/",
        caveat="public address example, not a verified profitable trader or entity label",
    ),
    SeedWallet(
        label="public_live_bot_example",
        address="0xec917F0F6c8d4AE7fFEFD5856D9ad802DD5F094b",
        source="public_reddit_thread",
        source_url="https://www.reddit.com/r/hyperliquid/comments/1k3sll2/i_built_a_live_bot_to_trade_on_hyperliquid_and/",
        caveat="public address example, not a verified profitable trader or entity label",
    ),
)


def build_seed_wallet_flow_rows(seed_wallets: tuple[SeedWallet, ...] = SEED_WALLETS) -> tuple[SeedWalletFlowRow, ...]:
    rows: list[SeedWalletFlowRow] = []
    for seed in seed_wallets:
        fills = _user_fills(seed.address)
        state = _clearinghouse_state(seed.address)
        rows.extend(_rows_for_seed(seed=seed, fills=fills, state=state))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_seed_wallet_flow_csv(rows: tuple[SeedWalletFlowRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp_ms",
                "wallet_label",
                "address",
                "source",
                "coin",
                "fills",
                "buy_notional",
                "sell_notional",
                "net_buy_notional",
                "closed_pnl",
                "fees",
                "net_closed_pnl_after_fees",
                "current_position",
                "current_position_notional",
                "action",
                "score",
                "caveat",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp_ms,
                    row.wallet_label,
                    row.address,
                    row.source,
                    row.coin,
                    row.fills,
                    f"{row.buy_notional:.8f}",
                    f"{row.sell_notional:.8f}",
                    f"{row.net_buy_notional:.8f}",
                    f"{row.closed_pnl:.8f}",
                    f"{row.fees:.8f}",
                    f"{row.net_closed_pnl_after_fees:.8f}",
                    f"{row.current_position:.8f}",
                    f"{row.current_position_notional:.8f}",
                    row.action,
                    f"{row.score:.8f}",
                    row.caveat,
                    row.next_step,
                )
            )
    return output_path


def write_seed_wallet_flow_md(rows: tuple[SeedWalletFlowRow, ...], *, output_path: Path, top: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Seed Wallet Flow\n\n")
        handle.write(
            "This turns a small public seed-wallet set into wallet-flow observations. "
            "It is not a copy-trading rule and the seed wallets are not verified entities.\n\n"
        )
        handle.write(
            "| wallet | coin | action | fills | net buy USD | closed PnL | fees | net PnL | position | score | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.wallet_label} | "
                f"{row.coin} | "
                f"{row.action} | "
                f"{row.fills} | "
                f"{row.net_buy_notional:.2f} | "
                f"{row.closed_pnl:.2f} | "
                f"{row.fees:.2f} | "
                f"{row.net_closed_pnl_after_fees:.2f} | "
                f"{row.current_position:.6f} | "
                f"{row.score:.2f} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "These rows are useful only as seed observations. A wallet-flow alpha still needs "
            "entity selection, survivorship checks, forward labels, costs, and anti-copycat risk controls.\n"
        )
    return output_path


def _rows_for_seed(
    *,
    seed: SeedWallet,
    fills: tuple[dict[str, Any], ...],
    state: dict[str, Any],
) -> tuple[SeedWalletFlowRow, ...]:
    fills_by_coin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fill in fills:
        coin = str(fill.get("coin", ""))
        if coin:
            fills_by_coin[coin].append(fill)
    positions = _positions_by_coin(state)
    rows: list[SeedWalletFlowRow] = []
    for coin, coin_fills in fills_by_coin.items():
        buy_notional = sum(_notional(fill) for fill in coin_fills if fill.get("side") == "B")
        sell_notional = sum(_notional(fill) for fill in coin_fills if fill.get("side") == "A")
        closed_pnl = sum(_float(fill.get("closedPnl")) for fill in coin_fills)
        fees = sum(_float(fill.get("fee")) for fill in coin_fills)
        net_closed_pnl = closed_pnl - fees
        current_position, current_position_notional = positions.get(coin, (0.0, 0.0))
        timestamp_ms = max((_int(fill.get("time")) for fill in coin_fills), default=0)
        action = _action(
            net_buy_notional=buy_notional - sell_notional,
            net_closed_pnl=net_closed_pnl,
            current_position=current_position,
        )
        score = _score(
            fills=len(coin_fills),
            net_buy_notional=buy_notional - sell_notional,
            net_closed_pnl=net_closed_pnl,
            current_position_notional=current_position_notional,
            action=action,
        )
        rows.append(
            SeedWalletFlowRow(
                timestamp_ms=timestamp_ms,
                wallet_label=seed.label,
                address=seed.address,
                source=seed.source,
                coin=coin,
                fills=len(coin_fills),
                buy_notional=buy_notional,
                sell_notional=sell_notional,
                net_buy_notional=buy_notional - sell_notional,
                closed_pnl=closed_pnl,
                fees=fees,
                net_closed_pnl_after_fees=net_closed_pnl,
                current_position=current_position,
                current_position_notional=current_position_notional,
                action=action,
                score=score,
                caveat=seed.caveat,
                next_step=_next_step(action=action, coin=coin, seed=seed),
            )
        )
    return tuple(rows)


def _user_fills(address: str) -> tuple[dict[str, Any], ...]:
    body = _post_hyperliquid({"type": "userFills", "user": address, "aggregateByTime": True})
    if isinstance(body, list):
        return tuple(fill for fill in body if isinstance(fill, dict))
    return ()


def _clearinghouse_state(address: str) -> dict[str, Any]:
    body = _post_hyperliquid({"type": "clearinghouseState", "user": address})
    return body if isinstance(body, dict) else {}


def _post_hyperliquid(payload: dict[str, object]) -> Any:
    response = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _positions_by_coin(state: dict[str, Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for item in state.get("assetPositions", []) or []:
        position = item.get("position", {}) if isinstance(item, dict) else {}
        coin = str(position.get("coin", ""))
        if not coin:
            continue
        size = _float(position.get("szi"))
        notional = _float(position.get("positionValue"))
        positions[coin] = (size, notional)
    return positions


def _action(*, net_buy_notional: float, net_closed_pnl: float, current_position: float) -> str:
    if net_closed_pnl <= 0.0:
        return "reject_negative_wallet_pnl"
    if current_position > 0.0:
        return "watch_wallet_long_pressure"
    if current_position < 0.0:
        return "watch_wallet_short_pressure"
    if net_buy_notional > 0.0:
        return "watch_recent_wallet_buy_flow"
    if net_buy_notional < 0.0:
        return "watch_recent_wallet_sell_flow"
    return "context_only"


def _score(
    *,
    fills: int,
    net_buy_notional: float,
    net_closed_pnl: float,
    current_position_notional: float,
    action: str,
) -> float:
    if action == "reject_negative_wallet_pnl":
        return min(net_closed_pnl, 0.0)
    return (
        min(abs(net_buy_notional) / 1_000.0, 50.0)
        + min(max(net_closed_pnl, 0.0) / 10.0, 50.0)
        + min(abs(current_position_notional) / 1_000.0, 30.0)
        + min(fills / 10.0, 20.0)
    )


def _next_step(*, action: str, coin: str, seed: SeedWallet) -> str:
    if action == "reject_negative_wallet_pnl":
        return f"do not follow {seed.label} on {coin}; keep only as negative-control wallet-flow sample"
    if action == "context_only":
        return f"keep {seed.label}/{coin} as context until current position or directional flow is explicit"
    return (
        f"label {seed.label}/{coin} wallet-flow pressure over 15m/1h/4h, "
        "then compare against market-wide flow and execution costs"
    )


def _notional(fill: dict[str, Any]) -> float:
    return _float(fill.get("px")) * _float(fill.get("sz"))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {"", None} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(value) if value not in {"", None} else 0
    except (TypeError, ValueError):
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_hyperliquid_seed_wallet_flow.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_hyperliquid_seed_wallet_flow.md")
    args = parser.parse_args()
    rows = build_seed_wallet_flow_rows()
    write_seed_wallet_flow_csv(rows, output_path=args.output_path)
    write_seed_wallet_flow_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.wallet_label, row.coin, row.action, f"{row.score:.2f}")


if __name__ == "__main__":
    main()
