from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log10
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent
DEFILLAMA_FEES_URL = (
    "https://api.llama.fi/overview/fees"
    "?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
)


@dataclass(frozen=True)
class ProtocolTokenMapping:
    token_symbol: str
    protocol_group: str
    thesis: str


@dataclass(frozen=True)
class ProtocolFeeRow:
    slug: str
    name: str
    category: str
    token_symbol: str
    protocol_group: str
    total_24h: float
    total_7d: float
    total_30d: float
    change_7d_over_7d: float
    change_30d_over_30d: float
    annualized_1y: float
    funding: float
    day_volume_usd: float
    open_interest_usd: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


TOKEN_MAPPINGS = {
    "hyperliquid-perps": ProtocolTokenMapping("HYPE", "hyperliquid", "perp exchange fee growth"),
    "hyper-foundation-hype-staking": ProtocolTokenMapping("HYPE", "hyperliquid", "staking fee capture context"),
    "aave-v3": ProtocolTokenMapping("AAVE", "aave", "lending fee growth"),
    "uniswap-v3": ProtocolTokenMapping("UNI", "uniswap", "DEX fee growth"),
    "uniswap-v4": ProtocolTokenMapping("UNI", "uniswap", "DEX fee growth"),
    "curve-dex": ProtocolTokenMapping("CRV", "curve", "DEX fee growth"),
    "ethena-usde": ProtocolTokenMapping("ENA", "ethena", "basis product fee growth"),
    "jupiter-perpetual-exchange": ProtocolTokenMapping("JUP", "jupiter", "perp exchange fee growth"),
    "jupiter": ProtocolTokenMapping("JUP", "jupiter", "routing and exchange activity"),
    "aerodrome-slipstream": ProtocolTokenMapping("AERO", "aerodrome", "Base DEX fee growth"),
    "aerodrome": ProtocolTokenMapping("AERO", "aerodrome", "Base DEX fee growth"),
    "pancakeswap": ProtocolTokenMapping("CAKE", "pancakeswap", "DEX fee growth"),
    "pendle": ProtocolTokenMapping("PENDLE", "pendle", "yield-trading fee growth"),
    "gmx": ProtocolTokenMapping("GMX", "gmx", "perp exchange fee growth"),
    "dydx": ProtocolTokenMapping("DYDX", "dydx", "perp exchange fee growth"),
    "morpho-blue": ProtocolTokenMapping("MORPHO", "morpho", "lending fee growth"),
    "ether.fi-liquid": ProtocolTokenMapping("ETHFI", "etherfi", "liquid staking and allocator fees"),
    "lido": ProtocolTokenMapping("LDO", "lido", "liquid staking fee base"),
    "sky-lending": ProtocolTokenMapping("SKY", "sky", "lending fee base"),
    "solana": ProtocolTokenMapping("SOL", "solana", "chain fee base"),
    "ethereum": ProtocolTokenMapping("ETH", "ethereum", "chain fee base"),
    "bsc": ProtocolTokenMapping("BNB", "bnb-chain", "chain fee base"),
}


def build_protocol_fee_rows(
    *,
    fees_url: str = DEFILLAMA_FEES_URL,
    max_rows: int = 30,
) -> tuple[ProtocolFeeRow, ...]:
    protocols = _fetch_protocols(fees_url)
    okx_context = _market_context_by_symbol(STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure.csv")
    hl_context = _market_context_by_symbol(STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv")
    rows: list[ProtocolFeeRow] = []
    for protocol in protocols:
        slug = str(protocol.get("slug") or "")
        mapping = TOKEN_MAPPINGS.get(slug)
        if not mapping:
            continue
        token_context = okx_context.get(mapping.token_symbol) or hl_context.get(mapping.token_symbol) or {}
        rows.append(_build_row(protocol=protocol, mapping=mapping, token_context=token_context))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True)[:max_rows])


def write_protocol_fee_csv(rows: tuple[ProtocolFeeRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "slug",
                "name",
                "category",
                "token_symbol",
                "protocol_group",
                "total_24h",
                "total_7d",
                "total_30d",
                "change_7d_over_7d",
                "change_30d_over_30d",
                "annualized_1y",
                "funding",
                "day_volume_usd",
                "open_interest_usd",
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
                    row.slug,
                    row.name,
                    row.category,
                    row.token_symbol,
                    row.protocol_group,
                    f"{row.total_24h:.8f}",
                    f"{row.total_7d:.8f}",
                    f"{row.total_30d:.8f}",
                    f"{row.change_7d_over_7d:.8f}",
                    f"{row.change_30d_over_30d:.8f}",
                    f"{row.annualized_1y:.8f}",
                    f"{row.funding:.8f}",
                    f"{row.day_volume_usd:.8f}",
                    f"{row.open_interest_usd:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_md(rows: tuple[ProtocolFeeRow, ...], *, output_path: Path, top: int = 15) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Screen\n\n")
        handle.write(
            "This converts DeFiLlama protocol fees into tradable-token research candidates. "
            "It is a non-price context screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| token | protocol | category | status | side | 7d fees | 30d fees | 7d growth | funding | score | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.token_symbol} | {row.name} | {row.category} | {row.status} | {row.side} | "
                f"{row.total_7d:.0f} | {row.total_30d:.0f} | {row.change_7d_over_7d:.2f} | "
                f"{row.funding:.4f} | {row.score:.4f} | {row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Protocol fees are not a direct token valuation model. They are useful when fee growth overlaps with "
            "tradable market structure, funding, unlocks, attention, or sector flow and then survives forward labels.\n"
        )
    return output_path


def _fetch_protocols(fees_url: str) -> tuple[dict[str, object], ...]:
    response = requests.get(fees_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return tuple(payload.get("protocols") or ())


def _build_row(
    *,
    protocol: dict[str, object],
    mapping: ProtocolTokenMapping,
    token_context: dict[str, str],
) -> ProtocolFeeRow:
    slug = str(protocol.get("slug") or "")
    name = str(protocol.get("name") or "")
    category = str(protocol.get("category") or "")
    total_24h = _float(protocol.get("total24h"))
    total_7d = _float(protocol.get("total7d"))
    total_30d = _float(protocol.get("total30d"))
    change_7d = _float(protocol.get("change_7dover7d"))
    change_30d = _float(protocol.get("change_30dover30d"))
    annualized_1y = _float(protocol.get("annualized1y"))
    funding = _float(token_context.get("annualized_funding"))
    day_volume = _float(token_context.get("day_volume_usd") or token_context.get("day_notional_volume"))
    open_interest = _float(token_context.get("open_interest_usd") or token_context.get("open_interest_notional"))
    score = _score(
        total_7d=total_7d,
        total_30d=total_30d,
        change_7d=change_7d,
        change_30d=change_30d,
        day_volume=day_volume,
        open_interest=open_interest,
        funding=funding,
    )
    status, side, reason = _status_side_reason(
        change_7d=change_7d,
        total_30d=total_30d,
        day_volume=day_volume,
        funding=funding,
        thesis=mapping.thesis,
    )
    return ProtocolFeeRow(
        slug=slug,
        name=name,
        category=category,
        token_symbol=mapping.token_symbol,
        protocol_group=mapping.protocol_group,
        total_24h=total_24h,
        total_7d=total_7d,
        total_30d=total_30d,
        change_7d_over_7d=change_7d,
        change_30d_over_30d=change_30d,
        annualized_1y=annualized_1y,
        funding=funding,
        day_volume_usd=day_volume,
        open_interest_usd=open_interest,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=f"label {mapping.token_symbol} returns after protocol fee-growth snapshots and join to funding/unlock context",
    )


def _status_side_reason(
    *,
    change_7d: float,
    total_30d: float,
    day_volume: float,
    funding: float,
    thesis: str,
) -> tuple[str, str, str]:
    if total_30d < 5_000_000:
        return "watch", "none", "fee base is still small for a strong standalone candidate"
    if day_volume <= 0.0:
        return "watch", "none", "fee growth exists but tradable perp context is missing"
    if change_7d >= 75.0 and funding <= 0.2:
        return "paper_long_context", "long_token_or_relative_value", f"{thesis} accelerated and funding is not too expensive"
    if change_7d >= 75.0:
        return "funding_crowded_watch", "watch", f"{thesis} accelerated but funding may be crowded"
    if change_7d <= -50.0:
        return "fee_decay_watch", "watch_or_short", f"{thesis} decelerated sharply"
    return "watch", "none", f"{thesis} is material but not a clean growth/decay setup"


def _score(
    *,
    total_7d: float,
    total_30d: float,
    change_7d: float,
    change_30d: float,
    day_volume: float,
    open_interest: float,
    funding: float,
) -> float:
    fee_scale = log10(max(total_30d, 1.0))
    current_scale = log10(max(total_7d, 1.0))
    growth = max(min(change_7d, 250.0), -100.0) / 10.0
    month_growth = max(min(change_30d, 250.0), -100.0) / 25.0
    liquidity = min(log10(max(day_volume, 1.0)), 9.0)
    oi = min(log10(max(open_interest, 1.0)), 9.0)
    funding_penalty = max(funding - 0.2, 0.0) * 10.0
    return fee_scale + current_scale + growth + month_growth + liquidity + oi - funding_penalty


def _market_context_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = row.get("asset") or row.get("token_symbol") or row.get("symbol")
        if symbol and symbol not in output:
            output[symbol] = row
    return output


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_screen.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_protocol_fee_screen.md")
    args = parser.parse_args()

    rows = build_protocol_fee_rows()
    write_protocol_fee_csv(rows, output_path=args.output_path)
    write_protocol_fee_md(rows, output_path=args.markdown_output_path)
    for row in rows[:10]:
        print(row.status, row.token_symbol, row.side, f"score={row.score:.4f}", row.name, row.reason)


if __name__ == "__main__":
    main()
