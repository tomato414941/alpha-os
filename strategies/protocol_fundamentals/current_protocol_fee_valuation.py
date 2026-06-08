from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"


@dataclass(frozen=True)
class CoinGeckoMapping:
    coin_id: str


@dataclass(frozen=True)
class ProtocolFeeValuationRow:
    token_symbol: str
    protocol: str
    fee_status: str
    annualized_fees: float
    market_cap: float
    fdv: float
    fee_to_market_cap: float
    fee_to_fdv: float
    change_7d_over_7d: float
    funding: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


COINGECKO_IDS = {
    "AAVE": CoinGeckoMapping("aave"),
    "AERO": CoinGeckoMapping("aerodrome-finance"),
    "CRV": CoinGeckoMapping("curve-dao-token"),
    "ENA": CoinGeckoMapping("ethena"),
    "ETH": CoinGeckoMapping("ethereum"),
    "HYPE": CoinGeckoMapping("hyperliquid"),
    "JUP": CoinGeckoMapping("jupiter-exchange-solana"),
    "MORPHO": CoinGeckoMapping("morpho"),
    "PENDLE": CoinGeckoMapping("pendle"),
    "SOL": CoinGeckoMapping("solana"),
    "UNI": CoinGeckoMapping("uniswap"),
}


def build_protocol_fee_valuation_rows(
    *,
    fee_path: Path = ROOT / "current_protocol_fee_screen.csv",
) -> tuple[ProtocolFeeValuationRow, ...]:
    fee_rows = tuple(row for row in _read_rows(fee_path) if row.get("token_symbol") in COINGECKO_IDS)
    markets = _fetch_markets(tuple(COINGECKO_IDS[row.get("token_symbol", "")].coin_id for row in fee_rows))
    output = tuple(_build_row(fee=row, market=markets.get(row.get("token_symbol", ""), {})) for row in fee_rows)
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_protocol_fee_valuation_csv(
    rows: tuple[ProtocolFeeValuationRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "token_symbol",
                "protocol",
                "fee_status",
                "annualized_fees",
                "market_cap",
                "fdv",
                "fee_to_market_cap",
                "fee_to_fdv",
                "change_7d_over_7d",
                "funding",
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
                    row.token_symbol,
                    row.protocol,
                    row.fee_status,
                    f"{row.annualized_fees:.8f}",
                    f"{row.market_cap:.8f}",
                    f"{row.fdv:.8f}",
                    f"{row.fee_to_market_cap:.8f}",
                    f"{row.fee_to_fdv:.8f}",
                    f"{row.change_7d_over_7d:.8f}",
                    f"{row.funding:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_valuation_md(
    rows: tuple[ProtocolFeeValuationRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Valuation\n\n")
        handle.write(
            "This joins DeFiLlama annualized protocol fees to CoinGecko market cap and FDV. "
            "It is a valuation context screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| token | protocol | status | fee / mcap | fee / fdv | growth 7d | funding | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.token_symbol} | {row.protocol} | {row.status} | "
                f"{row.fee_to_market_cap:.4f} | {row.fee_to_fdv:.4f} | "
                f"{row.change_7d_over_7d:.2f} | {row.funding:.4f} | "
                f"{row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _fetch_markets(coin_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    response = requests.get(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "ids": ",".join(sorted(set(coin_ids))),
            "per_page": 250,
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    by_symbol: dict[str, dict[str, object]] = {}
    reverse = {mapping.coin_id: token for token, mapping in COINGECKO_IDS.items()}
    for row in response.json():
        token = reverse.get(str(row.get("id") or ""))
        if token:
            by_symbol[token] = row
    return by_symbol


def _build_row(
    *,
    fee: dict[str, str],
    market: dict[str, object],
) -> ProtocolFeeValuationRow:
    token = fee.get("token_symbol", "")
    annualized_fees = _float(fee.get("annualized_1y"))
    market_cap = _float(market.get("market_cap"))
    fdv = _float(market.get("fully_diluted_valuation"))
    fee_to_market_cap = annualized_fees / market_cap if market_cap > 0.0 else 0.0
    fee_to_fdv = annualized_fees / fdv if fdv > 0.0 else 0.0
    growth = _float(fee.get("change_7d_over_7d"))
    funding = _float(fee.get("funding"))
    score = _score(
        fee_to_market_cap=fee_to_market_cap,
        fee_to_fdv=fee_to_fdv,
        growth=growth,
        funding=funding,
    )
    status, side, reason = _status_side_reason(
        fee_to_market_cap=fee_to_market_cap,
        fee_to_fdv=fee_to_fdv,
        growth=growth,
        funding=funding,
    )
    return ProtocolFeeValuationRow(
        token_symbol=token,
        protocol=fee.get("name", ""),
        fee_status=fee.get("status", ""),
        annualized_fees=annualized_fees,
        market_cap=market_cap,
        fdv=fdv,
        fee_to_market_cap=fee_to_market_cap,
        fee_to_fdv=fee_to_fdv,
        change_7d_over_7d=growth,
        funding=funding,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=f"label {token} fee-yield valuation snapshots against forward returns and funding costs",
    )


def _score(
    *,
    fee_to_market_cap: float,
    fee_to_fdv: float,
    growth: float,
    funding: float,
) -> float:
    return (
        min(fee_to_market_cap * 100.0, 30.0)
        + min(fee_to_fdv * 100.0, 30.0)
        + max(min(growth, 250.0), -100.0) / 20.0
        - max(funding - 0.2, 0.0) * 5.0
    )


def _status_side_reason(
    *,
    fee_to_market_cap: float,
    fee_to_fdv: float,
    growth: float,
    funding: float,
) -> tuple[str, str, str]:
    if fee_to_market_cap >= 0.15 and fee_to_fdv >= 0.05 and growth >= 50.0 and funding <= 0.2:
        return "paper_value_growth_candidate", "long_token_or_relative_value", "fee yield and growth are both strong"
    if fee_to_market_cap >= 0.10 and growth >= 50.0:
        return "paper_value_watch", "long_token_or_relative_value", "fee yield is material and fees are growing"
    if fee_to_fdv < 0.02 and growth < 0.0:
        return "expensive_fee_decay_watch", "watch_or_short", "fee yield is low and fees are decelerating"
    return "watch", "none", "valuation context is not decisive"


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_valuation.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_valuation.md",
    )
    args = parser.parse_args()

    try:
        rows = build_protocol_fee_valuation_rows()
    except requests.RequestException as exc:
        if args.output_path.exists() and args.markdown_output_path.exists():
            print(f"preserving existing protocol fee valuation after market fetch failure: {exc}")
            return
        raise
    write_protocol_fee_valuation_csv(rows, output_path=args.output_path)
    write_protocol_fee_valuation_md(rows, output_path=args.markdown_output_path)
    for row in rows[:10]:
        print(row.status, row.token_symbol, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
