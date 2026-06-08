from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent
COINGECKO_PUBLIC_TREASURY_URL = "https://api.coingecko.com/api/v3/companies/public_treasury/{asset_id}"

ASSETS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
}


@dataclass(frozen=True)
class PublicTreasuryContextRow:
    timestamp: str
    asset_id: str
    asset: str
    source_status: str
    total_holdings: float
    total_value_usd: float
    market_cap_dominance: float
    top_holder_name: str
    top_holder_symbol: str
    top_holder_holdings: float
    top_holder_supply_pct: float
    annualized_funding: float
    open_interest_notional: float
    day_notional_volume: float
    action: str
    side: str
    score: float
    reason: str
    next_step: str


def build_public_treasury_context_rows(
    *,
    asset_ids: tuple[str, ...] = tuple(ASSETS),
    hyperliquid_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    request_delay_seconds: float = 2.0,
    timestamp: str | None = None,
) -> tuple[PublicTreasuryContextRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    market_by_asset = {row.get("asset", ""): row for row in _read_rows(hyperliquid_path)}
    rows: list[PublicTreasuryContextRow] = []
    for index, asset_id in enumerate(asset_ids):
        if index > 0 and request_delay_seconds > 0.0:
            time.sleep(request_delay_seconds)
        rows.append(
            _build_row(
                asset_id=asset_id,
                payload=_fetch_public_treasury(asset_id),
                market=market_by_asset.get(ASSETS.get(asset_id, ""), {}),
                timestamp=observed_at,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_public_treasury_context_csv(
    rows: tuple[PublicTreasuryContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset_id",
                "asset",
                "source_status",
                "total_holdings",
                "total_value_usd",
                "market_cap_dominance",
                "top_holder_name",
                "top_holder_symbol",
                "top_holder_holdings",
                "top_holder_supply_pct",
                "annualized_funding",
                "open_interest_notional",
                "day_notional_volume",
                "action",
                "side",
                "score",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset_id,
                    row.asset,
                    row.source_status,
                    f"{row.total_holdings:.8f}",
                    f"{row.total_value_usd:.2f}",
                    f"{row.market_cap_dominance:.8f}",
                    row.top_holder_name,
                    row.top_holder_symbol,
                    f"{row.top_holder_holdings:.8f}",
                    f"{row.top_holder_supply_pct:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.open_interest_notional:.2f}",
                    f"{row.day_notional_volume:.2f}",
                    row.action,
                    row.side,
                    f"{row.score:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_public_treasury_context_md(
    rows: tuple[PublicTreasuryContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Public Treasury Context\n\n")
        handle.write(
            "This joins public-company crypto treasury holdings to current perp funding. "
            "It is institutional demand context, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | dominance | top holder | top supply pct | funding | action | score | reason | next step |\n"
        )
        handle.write("| --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.asset} | {row.source_status} | {row.market_cap_dominance:.4f} | "
                f"{row.top_holder_name}/{row.top_holder_symbol} | "
                f"{row.top_holder_supply_pct:.4f} | {row.annualized_funding:.6f} | "
                f"{row.action} | {row.score:.4f} | {row.reason} | {row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Public treasury holdings are slower structural demand or concentration context. "
            "They need forward labels, financing/issuance checks, and separation from equity-proxy noise before promotion.\n"
        )
    return output_path


def _fetch_public_treasury(asset_id: str) -> dict[str, object]:
    try:
        response = requests.get(
            COINGECKO_PUBLIC_TREASURY_URL.format(asset_id=asset_id),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        if response.status_code == 429:
            return {"source_status": "source_rate_limited"}
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"source_status": f"source_error:{type(exc).__name__}"}
    payload = response.json()
    payload["source_status"] = "source_ok"
    return payload


def _build_row(
    *,
    asset_id: str,
    payload: dict[str, object],
    market: dict[str, str],
    timestamp: str,
) -> PublicTreasuryContextRow:
    asset = ASSETS.get(asset_id, asset_id.upper())
    source_status = str(payload.get("source_status") or "source_unknown")
    companies = payload.get("companies") if source_status == "source_ok" else ()
    top_holder = companies[0] if isinstance(companies, list) and companies else {}
    market_cap_dominance = _float(payload.get("market_cap_dominance"))
    top_supply_pct = _float(top_holder.get("percentage_of_total_supply")) if isinstance(top_holder, dict) else 0.0
    annualized_funding = _float(market.get("annualized_funding"))
    action, side, reason, next_step = _action_side_reason(
        source_status=source_status,
        asset=asset,
        market_cap_dominance=market_cap_dominance,
        top_supply_pct=top_supply_pct,
        annualized_funding=annualized_funding,
    )
    return PublicTreasuryContextRow(
        timestamp=timestamp,
        asset_id=asset_id,
        asset=asset,
        source_status=source_status,
        total_holdings=_float(payload.get("total_holdings")),
        total_value_usd=_float(payload.get("total_value_usd")),
        market_cap_dominance=market_cap_dominance,
        top_holder_name=str(top_holder.get("name") or "") if isinstance(top_holder, dict) else "",
        top_holder_symbol=str(top_holder.get("symbol") or "") if isinstance(top_holder, dict) else "",
        top_holder_holdings=_float(top_holder.get("total_holdings")) if isinstance(top_holder, dict) else 0.0,
        top_holder_supply_pct=top_supply_pct,
        annualized_funding=annualized_funding,
        open_interest_notional=_float(market.get("open_interest_notional")),
        day_notional_volume=_float(market.get("day_notional_volume")),
        action=action,
        side=side,
        score=_score(
            source_status=source_status,
            market_cap_dominance=market_cap_dominance,
            top_supply_pct=top_supply_pct,
            annualized_funding=annualized_funding,
        ),
        reason=reason,
        next_step=next_step,
    )


def _action_side_reason(
    *,
    source_status: str,
    asset: str,
    market_cap_dominance: float,
    top_supply_pct: float,
    annualized_funding: float,
) -> tuple[str, str, str, str]:
    if source_status != "source_ok":
        return (
            "public_treasury_source_blocked",
            "none",
            f"CoinGecko treasury source is {source_status}",
            "retry later and do not promote this source until fresh treasury rows are available",
        )
    if market_cap_dominance >= 3.0 and top_supply_pct >= 0.5 and annualized_funding < 0.0:
        return (
            "public_treasury_accumulation_vs_short_perp_watch",
            f"long_{asset.lower()}",
            "large public treasury concentration overlaps with short-perp funding",
            f"label {asset} 4h/24h/5d outcomes when treasury concentration overlaps negative perp funding",
        )
    if market_cap_dominance >= 3.0 and annualized_funding > 0.2:
        return (
            "public_treasury_crowded_long_watch",
            f"watch_or_hedge_{asset.lower()}",
            "large public treasury concentration overlaps with positive leveraged funding",
            f"check whether {asset} treasury concentration plus high funding is a crowded-long unwind signal",
        )
    if market_cap_dominance >= 3.0:
        return (
            "public_treasury_concentration_watch",
            f"{asset.lower()}_structural_demand_context",
            "public treasury holdings are material but current perp positioning is not a clean divergence",
            f"join {asset} treasury concentration to equity proxy, issuance news, funding, and forward returns",
        )
    return (
        "public_treasury_context_only",
        "none",
        "public treasury holdings are not large enough for current triage",
        "keep as context only until holdings become material or overlap another tradable pressure",
    )


def _score(
    *,
    source_status: str,
    market_cap_dominance: float,
    top_supply_pct: float,
    annualized_funding: float,
) -> float:
    if source_status != "source_ok":
        return 0.0
    return market_cap_dominance * 10.0 + top_supply_pct * 5.0 + min(abs(annualized_funding) * 2.0, 10.0)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="bitcoin,ethereum")
    parser.add_argument(
        "--hyperliquid-path",
        type=Path,
        default=STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_public_treasury_context.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_public_treasury_context.md",
    )
    parser.add_argument("--request-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    asset_ids = tuple(asset.strip() for asset in args.assets.split(",") if asset.strip())
    rows = build_public_treasury_context_rows(
        asset_ids=asset_ids,
        hyperliquid_path=args.hyperliquid_path,
        request_delay_seconds=args.request_delay_seconds,
    )
    write_public_treasury_context_csv(rows, output_path=args.output_path)
    write_public_treasury_context_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(row.asset, row.action, f"dominance={row.market_cap_dominance:.4f}", f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
