from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ChainTvlFlowMarketContextRow:
    venue: str
    token_symbol: str
    action: str
    direction: int
    directional_return_15m: float | None
    annualized_funding: float | None
    funding_support: float | None
    premium: float | None
    liquidity_usd: float | None
    open_interest_usd: float | None
    friction_bps: float | None
    okx_liquidation_action: str
    okx_liquidation_score: float | None
    okx_liquidation_imbalance: float | None
    context_score: float
    note: str


def build_chain_tvl_flow_market_context_rows(
    *,
    label_path: Path = ROOT / "current_chain_tvl_flow_forward_labels.csv",
    hl_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    okx_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure.csv",
    liquidation_path: Path = STRATEGIES_ROOT
    / "liquidation_flow"
    / "current_okx_liquidation_flow.csv",
) -> tuple[ChainTvlFlowMarketContextRow, ...]:
    hl_by_asset = {row["asset"]: row for row in _read_rows(hl_path)}
    okx_by_asset = {row["asset"]: row for row in _read_rows(okx_path)}
    liquidation_by_asset = {row["asset"]: row for row in _read_rows(liquidation_path)}
    rows = tuple(
        _build_row(
            label=row,
            hl_row=hl_by_asset.get(row["token_symbol"]),
            okx_row=okx_by_asset.get(row["token_symbol"]),
            liquidation_row=liquidation_by_asset.get(row["token_symbol"]),
        )
        for row in _read_rows(label_path)
    )
    return tuple(sorted(rows, key=lambda row: row.context_score, reverse=True))


def write_chain_tvl_flow_market_context_csv(
    rows: tuple[ChainTvlFlowMarketContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "venue",
                "token_symbol",
                "action",
                "direction",
                "directional_return_15m",
                "annualized_funding",
                "funding_support",
                "premium",
                "liquidity_usd",
                "open_interest_usd",
                "friction_bps",
                "okx_liquidation_action",
                "okx_liquidation_score",
                "okx_liquidation_imbalance",
                "context_score",
                "note",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.venue,
                    row.token_symbol,
                    row.action,
                    row.direction,
                    _format_float(row.directional_return_15m),
                    _format_float(row.annualized_funding),
                    _format_float(row.funding_support),
                    _format_float(row.premium),
                    _format_float(row.liquidity_usd),
                    _format_float(row.open_interest_usd),
                    _format_float(row.friction_bps),
                    row.okx_liquidation_action,
                    _format_float(row.okx_liquidation_score),
                    _format_float(row.okx_liquidation_imbalance),
                    f"{row.context_score:.8f}",
                    row.note,
                )
            )
    return output_path


def write_chain_tvl_flow_market_context_md(
    rows: tuple[ChainTvlFlowMarketContextRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain TVL Flow Market Context\n\n")
        handle.write(
            "This joins chain TVL flow forward labels with current perp funding, "
            "liquidity, and OKX liquidation context. It is still a research screen, "
            "not a deployable strategy.\n\n"
        )
        handle.write(
            "| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.venue} | "
                f"{row.token_symbol} | "
                f"{row.action} | "
                f"{_format_float(row.directional_return_15m)} | "
                f"{_format_float(row.funding_support)} | "
                f"{_format_float(row.annualized_funding)} | "
                f"{row.okx_liquidation_action} | "
                f"{_format_float(row.okx_liquidation_score)} | "
                f"{row.context_score:.6f} | "
                f"{row.note} |\n"
            )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    hl_row: dict[str, str] | None,
    okx_row: dict[str, str] | None,
    liquidation_row: dict[str, str] | None,
) -> ChainTvlFlowMarketContextRow:
    venue = label["venue"]
    direction = int(label.get("direction") or "0")
    market_row = hl_row if venue == "HL" else okx_row
    annualized_funding = _float_or_none((market_row or {}).get("annualized_funding", ""))
    funding_support = (
        None if annualized_funding is None else -float(direction) * annualized_funding
    )
    directional_return_15m = _float_or_none(label.get("directional_return_15m", ""))
    liquidation_score = _float_or_none((liquidation_row or {}).get("cascade_score", ""))
    liquidation_imbalance = _float_or_none(
        (liquidation_row or {}).get("forced_buy_sell_imbalance", "")
    )
    context_score = _context_score(
        directional_return_15m=directional_return_15m,
        funding_support=funding_support,
        liquidation_score=liquidation_score,
        liquidity_usd=_liquidity_usd(venue=venue, market_row=market_row),
        friction_bps=_friction_bps(venue=venue, market_row=market_row),
    )
    return ChainTvlFlowMarketContextRow(
        venue=venue,
        token_symbol=label["token_symbol"],
        action=label["action"],
        direction=direction,
        directional_return_15m=directional_return_15m,
        annualized_funding=annualized_funding,
        funding_support=funding_support,
        premium=_float_or_none((market_row or {}).get("premium", "")),
        liquidity_usd=_liquidity_usd(venue=venue, market_row=market_row),
        open_interest_usd=_open_interest_usd(venue=venue, market_row=market_row),
        friction_bps=_friction_bps(venue=venue, market_row=market_row),
        okx_liquidation_action=(liquidation_row or {}).get("action", ""),
        okx_liquidation_score=liquidation_score,
        okx_liquidation_imbalance=liquidation_imbalance,
        context_score=context_score,
        note=_note(
            directional_return_15m=directional_return_15m,
            funding_support=funding_support,
            liquidation_score=liquidation_score,
        ),
    )


def _context_score(
    *,
    directional_return_15m: float | None,
    funding_support: float | None,
    liquidation_score: float | None,
    liquidity_usd: float | None,
    friction_bps: float | None,
) -> float:
    score = 0.0 if directional_return_15m is None else directional_return_15m * 100.0
    if funding_support is not None:
        score += min(funding_support, 0.5)
    if liquidation_score is not None:
        score += min(liquidation_score * 10.0, 0.5)
    if liquidity_usd is not None and liquidity_usd < 1_000_000.0:
        score -= 0.5
    if friction_bps is not None:
        score -= min(max(friction_bps, 0.0) / 20.0, 0.5)
    return score


def _note(
    *,
    directional_return_15m: float | None,
    funding_support: float | None,
    liquidation_score: float | None,
) -> str:
    parts: list[str] = []
    if directional_return_15m is not None and directional_return_15m > 0.0:
        parts.append("price label positive")
    if funding_support is not None and funding_support > 0.0:
        parts.append("funding helps direction")
    if liquidation_score is not None and liquidation_score > 0.01:
        parts.append("has recent liquidation context")
    return "; ".join(parts) if parts else "weak current context"


def _liquidity_usd(*, venue: str, market_row: dict[str, str] | None) -> float | None:
    if market_row is None:
        return None
    if venue == "HL":
        return _float_or_none(market_row.get("day_notional_volume", ""))
    return _float_or_none(market_row.get("day_volume_usd", ""))


def _open_interest_usd(*, venue: str, market_row: dict[str, str] | None) -> float | None:
    if market_row is None:
        return None
    if venue == "HL":
        return _float_or_none(market_row.get("open_interest_notional", ""))
    return _float_or_none(market_row.get("open_interest_usd", ""))


def _friction_bps(*, venue: str, market_row: dict[str, str] | None) -> float | None:
    if market_row is None:
        return None
    if venue == "HL":
        impact_spread = _float_or_none(market_row.get("impact_spread", ""))
        return None if impact_spread is None else impact_spread * 10000.0
    return _float_or_none(market_row.get("spread_bps", ""))


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
        "--output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_market_context.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow_market_context.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_chain_tvl_flow_market_context_rows()
    write_chain_tvl_flow_market_context_csv(rows, output_path=args.output_path)
    write_chain_tvl_flow_market_context_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.venue,
            row.token_symbol,
            f"dir15={_format_float(row.directional_return_15m)}",
            f"funding_support={_format_float(row.funding_support)}",
            f"score={row.context_score:.4f}",
            row.note,
        )


if __name__ == "__main__":
    main()
