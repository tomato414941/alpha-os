from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


GECKOTERMINAL_TRENDING_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/trending_pools"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DexPoolFlowRow:
    timestamp: str
    pool_id: str
    network: str
    dex: str
    name: str
    pool_created_at: str
    reserve_usd: float
    fdv_usd: float
    market_cap_usd: float
    volume_m5_usd: float
    volume_m15_usd: float
    volume_h1_usd: float
    volume_h24_usd: float
    price_change_m15: float
    price_change_h1: float
    price_change_h6: float
    price_change_h24: float
    buys_h1: int
    sells_h1: int
    buyers_h1: int
    sellers_h1: int
    buy_sell_imbalance_h1: float
    volume_reserve_ratio_h1: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def fetch_trending_pools(url: str = GECKOTERMINAL_TRENDING_POOLS_URL) -> tuple[dict[str, object], ...]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return tuple(response.json().get("data") or ())


def build_dex_pool_flow_rows(
    raw_rows: tuple[dict[str, object], ...],
    *,
    timestamp: str | None = None,
) -> tuple[DexPoolFlowRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows = tuple(_build_row(raw=row, timestamp=observed_at) for row in raw_rows)
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_dex_pool_flow_csv(rows: tuple[DexPoolFlowRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "pool_id",
                "network",
                "dex",
                "name",
                "pool_created_at",
                "reserve_usd",
                "fdv_usd",
                "market_cap_usd",
                "volume_m5_usd",
                "volume_m15_usd",
                "volume_h1_usd",
                "volume_h24_usd",
                "price_change_m15",
                "price_change_h1",
                "price_change_h6",
                "price_change_h24",
                "buys_h1",
                "sells_h1",
                "buyers_h1",
                "sellers_h1",
                "buy_sell_imbalance_h1",
                "volume_reserve_ratio_h1",
                "score",
                "status",
                "side",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.pool_id,
                    row.network,
                    row.dex,
                    row.name,
                    row.pool_created_at,
                    f"{row.reserve_usd:.8f}",
                    f"{row.fdv_usd:.8f}",
                    f"{row.market_cap_usd:.8f}",
                    f"{row.volume_m5_usd:.8f}",
                    f"{row.volume_m15_usd:.8f}",
                    f"{row.volume_h1_usd:.8f}",
                    f"{row.volume_h24_usd:.8f}",
                    f"{row.price_change_m15:.8f}",
                    f"{row.price_change_h1:.8f}",
                    f"{row.price_change_h6:.8f}",
                    f"{row.price_change_h24:.8f}",
                    row.buys_h1,
                    row.sells_h1,
                    row.buyers_h1,
                    row.sellers_h1,
                    f"{row.buy_sell_imbalance_h1:.8f}",
                    f"{row.volume_reserve_ratio_h1:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_dex_pool_flow_md(rows: tuple[DexPoolFlowRow, ...], *, output_path: Path, top: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current GeckoTerminal DEX Pool Flow\n\n")
        handle.write(
            "This screen reads GeckoTerminal trending pools and scores DEX pool activity. "
            "It is a pool-flow screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.network} | {row.dex} | {row.name} | {row.status} | "
                f"{row.reserve_usd:.0f} | {row.volume_h1_usd:.0f} | "
                f"{row.volume_reserve_ratio_h1:.4f} | {row.price_change_h1:.4f} | "
                f"{row.price_change_h24:.4f} | {row.buy_sell_imbalance_h1:.4f} | "
                f"{row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _build_row(*, raw: dict[str, object], timestamp: str) -> DexPoolFlowRow:
    attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    relationships = raw.get("relationships") if isinstance(raw.get("relationships"), dict) else {}
    transactions = attributes.get("transactions") if isinstance(attributes.get("transactions"), dict) else {}
    volume = attributes.get("volume_usd") if isinstance(attributes.get("volume_usd"), dict) else {}
    price_change = (
        attributes.get("price_change_percentage")
        if isinstance(attributes.get("price_change_percentage"), dict)
        else {}
    )
    h1_transactions = transactions.get("h1") if isinstance(transactions.get("h1"), dict) else {}
    reserve_usd = _float(attributes.get("reserve_in_usd"))
    volume_h1 = _float(volume.get("h1"))
    buys_h1 = _int(h1_transactions.get("buys"))
    sells_h1 = _int(h1_transactions.get("sells"))
    buy_sell_imbalance = _imbalance(buys=buys_h1, sells=sells_h1)
    volume_reserve_ratio = volume_h1 / reserve_usd if reserve_usd > 0.0 else 0.0
    price_change_h1 = _float(price_change.get("h1"))
    price_change_h24 = _float(price_change.get("h24"))
    score = _score(
        reserve_usd=reserve_usd,
        volume_h1=volume_h1,
        volume_h24=_float(volume.get("h24")),
        volume_reserve_ratio=volume_reserve_ratio,
        buy_sell_imbalance=buy_sell_imbalance,
        price_change_h1=price_change_h1,
        price_change_h24=price_change_h24,
    )
    status, side, reason = _status_side_reason(
        reserve_usd=reserve_usd,
        volume_h1=volume_h1,
        volume_reserve_ratio=volume_reserve_ratio,
        buy_sell_imbalance=buy_sell_imbalance,
        price_change_h1=price_change_h1,
        price_change_h24=price_change_h24,
    )
    network = _relationship_id(relationships, "network")
    dex = _relationship_id(relationships, "dex")
    name = str(attributes.get("name") or "")
    return DexPoolFlowRow(
        timestamp=timestamp,
        pool_id=str(raw.get("id") or ""),
        network=network,
        dex=dex,
        name=name,
        pool_created_at=str(attributes.get("pool_created_at") or ""),
        reserve_usd=reserve_usd,
        fdv_usd=_float(attributes.get("fdv_usd")),
        market_cap_usd=_float(attributes.get("market_cap_usd")),
        volume_m5_usd=_float(volume.get("m5")),
        volume_m15_usd=_float(volume.get("m15")),
        volume_h1_usd=volume_h1,
        volume_h24_usd=_float(volume.get("h24")),
        price_change_m15=_float(price_change.get("m15")),
        price_change_h1=price_change_h1,
        price_change_h6=_float(price_change.get("h6")),
        price_change_h24=price_change_h24,
        buys_h1=buys_h1,
        sells_h1=sells_h1,
        buyers_h1=_int(h1_transactions.get("buyers")),
        sellers_h1=_int(h1_transactions.get("sellers")),
        buy_sell_imbalance_h1=buy_sell_imbalance,
        volume_reserve_ratio_h1=volume_reserve_ratio,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=f"check {network}/{dex} {name} route depth, slippage, gas, MEV, token restrictions, and repeat flow",
    )


def _status_side_reason(
    *,
    reserve_usd: float,
    volume_h1: float,
    volume_reserve_ratio: float,
    buy_sell_imbalance: float,
    price_change_h1: float,
    price_change_h24: float,
) -> tuple[str, str, str]:
    if reserve_usd < 50_000.0:
        return "dex_microcap_liquidity_watch", "none", "pool is too thin for direct action"
    if volume_h1 >= 50_000.0 and volume_reserve_ratio >= 0.5 and price_change_h24 >= 100.0:
        return "paper_dex_reversal_risk_watch", "watch_reversal_or_no_trade", "extreme pool turnover after large 24h move"
    if volume_h1 >= 30_000.0 and price_change_h1 >= 5.0 and buy_sell_imbalance >= 0.05:
        return "paper_dex_pool_momentum_watch", "watch_pool_momentum", "short-term pool flow and price are aligned"
    if volume_reserve_ratio >= 0.5:
        return "dex_liquidity_stress_watch", "none", "pool turnover is high relative to reserves"
    return "dex_pool_context_watch", "none", "pool flow is context but not yet actionable"


def _score(
    *,
    reserve_usd: float,
    volume_h1: float,
    volume_h24: float,
    volume_reserve_ratio: float,
    buy_sell_imbalance: float,
    price_change_h1: float,
    price_change_h24: float,
) -> float:
    reserve_score = min(reserve_usd / 100_000.0, 20.0)
    volume_score = min(volume_h1 / 10_000.0, 25.0) + min(volume_h24 / 500_000.0, 10.0)
    turnover_score = min(volume_reserve_ratio * 10.0, 25.0)
    flow_score = max(buy_sell_imbalance, -0.5) * 10.0
    move_score = min(abs(price_change_h1), 50.0) * 0.3 + min(abs(price_change_h24), 200.0) * 0.05
    thin_penalty = 20.0 if reserve_usd < 50_000.0 else 0.0
    return reserve_score + volume_score + turnover_score + flow_score + move_score - thin_penalty


def _relationship_id(relationships: dict[str, object], key: str) -> str:
    relationship = relationships.get(key)
    if not isinstance(relationship, dict):
        return ""
    data = relationship.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("id") or "")


def _imbalance(*, buys: int, sells: int) -> float:
    total = buys + sells
    return (buys - sells) / total if total else 0.0


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(value) if value not in {None, ""} else 0
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_geckoterminal_pool_flow.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_geckoterminal_pool_flow.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_dex_pool_flow_rows(fetch_trending_pools())
    write_dex_pool_flow_csv(rows, output_path=args.output_path)
    write_dex_pool_flow_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.network, row.dex, row.name, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
