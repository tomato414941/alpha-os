from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    label: str
    group: str
    risk_direction: int


@dataclass(frozen=True)
class AssetSnapshot:
    symbol: str
    label: str
    group: str
    last_date: str
    close: float
    return_1d: float
    return_5d: float
    return_20d: float
    risk_score: float


@dataclass(frozen=True)
class MacroCryptoTicket:
    name: str
    status: str
    side: str
    score: float
    reason: str


ASSETS = (
    AssetConfig("BTC-USD", "BTC", "crypto", 1),
    AssetConfig("ETH-USD", "ETH", "crypto", 1),
    AssetConfig("QQQ", "Nasdaq 100 ETF", "equity_growth", 1),
    AssetConfig("SPY", "S&P 500 ETF", "equity_broad", 1),
    AssetConfig("IWM", "Russell 2000 ETF", "equity_small_cap", 1),
    AssetConfig("HYG", "High-yield credit ETF", "credit", 1),
    AssetConfig("TLT", "Long Treasury ETF", "duration", -1),
    AssetConfig("UUP", "Dollar ETF", "dollar", 1),
    AssetConfig("GLD", "Gold ETF", "gold", 0),
    AssetConfig("^VIX", "VIX", "volatility", 1),
)


def build_current_context(
    *,
    start_date: date,
    end_date: date,
) -> tuple[tuple[AssetSnapshot, ...], tuple[MacroCryptoTicket, ...]]:
    session = requests.Session()
    snapshots = tuple(
        _fetch_asset_snapshot(asset, start_date=start_date, end_date=end_date, session=session)
        for asset in ASSETS
    )
    return snapshots, _build_tickets(snapshots)


def write_snapshots_csv(snapshots: tuple[AssetSnapshot, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "label",
                "group",
                "last_date",
                "close",
                "return_1d",
                "return_5d",
                "return_20d",
                "risk_score",
            )
        )
        for row in snapshots:
            writer.writerow(
                (
                    row.symbol,
                    row.label,
                    row.group,
                    row.last_date,
                    f"{row.close:.8f}",
                    f"{row.return_1d:.8f}",
                    f"{row.return_5d:.8f}",
                    f"{row.return_20d:.8f}",
                    f"{row.risk_score:.8f}",
                )
            )
    return output_path


def write_tickets_csv(tickets: tuple[MacroCryptoTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("name", "status", "side", "score", "reason"))
        for ticket in tickets:
            writer.writerow(
                (
                    ticket.name,
                    ticket.status,
                    ticket.side,
                    f"{ticket.score:.8f}",
                    ticket.reason,
                )
            )
    return output_path


def write_markdown(
    snapshots: tuple[AssetSnapshot, ...],
    tickets: tuple[MacroCryptoTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Macro Crypto Context\n\n")
        handle.write(
            "This screen checks whether crypto is aligned with or diverging from broad risk assets. "
            "It is a research screen, not a trade instruction.\n\n"
        )
        handle.write("## Paper Tickets\n\n")
        handle.write("| name | status | side | score | reason |\n")
        handle.write("| --- | --- | --- | ---: | --- |\n")
        for ticket in tickets:
            handle.write(
                f"| {ticket.name} | {ticket.status} | {ticket.side} | "
                f"{ticket.score:.4f} | {ticket.reason} |\n"
            )
        handle.write("\n## Asset Context\n\n")
        handle.write("| symbol | group | close | 1d | 5d | 20d | risk score |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in snapshots:
            handle.write(
                f"| {row.symbol} | {row.group} | {row.close:.4f} | "
                f"{row.return_1d:.4f} | {row.return_5d:.4f} | "
                f"{row.return_20d:.4f} | {row.risk_score:.4f} |\n"
            )
    return output_path


def _fetch_asset_snapshot(
    asset: AssetConfig,
    *,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> AssetSnapshot:
    print(f"fetch {asset.symbol}", flush=True)
    prices = _fetch_yahoo_closes(
        asset.symbol,
        start_date=start_date,
        end_date=end_date,
        session=session,
    )
    if len(prices) < 2:
        raise ValueError(f"Not enough Yahoo rows for {asset.symbol}")
    latest_index = len(prices) - 1
    close = prices[latest_index][1]
    return AssetSnapshot(
        symbol=asset.symbol,
        label=asset.label,
        group=asset.group,
        last_date=prices[latest_index][0],
        close=close,
        return_1d=_return_at(prices, latest_index, 1),
        return_5d=_return_at(prices, latest_index, 5),
        return_20d=_return_at(prices, latest_index, 20),
        risk_score=asset.risk_direction * _return_at(prices, latest_index, 5),
    )


def _fetch_yahoo_closes(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> tuple[tuple[str, float], ...]:
    response = session.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={
            "period1": _date_to_seconds(start_date),
            "period2": _date_to_seconds(end_date + timedelta(days=1)),
            "interval": "1d",
            "events": "history",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    timestamps = payload["timestamp"]
    closes = payload["indicators"]["quote"][0]["close"]
    rows: list[tuple[str, float]] = []
    for timestamp, close in zip(timestamps, closes, strict=True):
        if close is None:
            continue
        rows.append((datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat(), float(close)))
    return tuple(rows)


def _build_tickets(snapshots: tuple[AssetSnapshot, ...]) -> tuple[MacroCryptoTicket, ...]:
    by_symbol = {row.symbol: row for row in snapshots}
    btc = by_symbol["BTC-USD"]
    eth = by_symbol["ETH-USD"]
    risk_assets = tuple(row for row in snapshots if row.group in {"equity_growth", "equity_broad", "credit"})
    defensive_assets = tuple(row for row in snapshots if row.group in {"dollar", "volatility", "duration"})

    risk_on_score = sum(row.risk_score for row in risk_assets) / len(risk_assets)
    defensive_pressure = sum(row.risk_score for row in defensive_assets) / len(defensive_assets)
    crypto_score = (btc.return_5d + eth.return_5d) / 2.0
    crypto_vs_risk = crypto_score - risk_on_score

    tickets = [
        _risk_on_catchup_ticket(risk_on_score=risk_on_score, crypto_vs_risk=crypto_vs_risk),
        _risk_off_crypto_short_ticket(
            defensive_pressure=defensive_pressure,
            crypto_score=crypto_score,
            btc=btc,
            eth=eth,
        ),
        _eth_btc_rotation_ticket(btc=btc, eth=eth, risk_on_score=risk_on_score),
    ]
    return tuple(sorted(tickets, key=lambda ticket: abs(ticket.score), reverse=True))


def _risk_on_catchup_ticket(*, risk_on_score: float, crypto_vs_risk: float) -> MacroCryptoTicket:
    score = risk_on_score - crypto_vs_risk
    if risk_on_score > 0.01 and crypto_vs_risk < -0.01:
        return MacroCryptoTicket(
            name="crypto_risk_on_catchup",
            status="paper_long_candidate",
            side="long_btc_eth",
            score=score,
            reason="equities/credit are risk-on while crypto lags",
        )
    return MacroCryptoTicket(
        name="crypto_risk_on_catchup",
        status="watch",
        side="none",
        score=score,
        reason="risk-on catch-up condition is not strong enough",
    )


def _risk_off_crypto_short_ticket(
    *,
    defensive_pressure: float,
    crypto_score: float,
    btc: AssetSnapshot,
    eth: AssetSnapshot,
) -> MacroCryptoTicket:
    score = defensive_pressure - crypto_score
    if defensive_pressure > 0.01 and crypto_score > -0.01:
        return MacroCryptoTicket(
            name="crypto_risk_off_lagged_short",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=score,
            reason="dollar/VIX/duration pressure is risk-off while crypto has not repriced enough",
        )
    return MacroCryptoTicket(
        name="crypto_risk_off_lagged_short",
        status="watch",
        side="none",
        score=score,
        reason=f"risk-off pressure is visible, but crypto already repriced; btc5d={btc.return_5d:.4f}, eth5d={eth.return_5d:.4f}",
    )


def _eth_btc_rotation_ticket(
    *,
    btc: AssetSnapshot,
    eth: AssetSnapshot,
    risk_on_score: float,
) -> MacroCryptoTicket:
    eth_vs_btc = eth.return_5d - btc.return_5d
    if risk_on_score > 0.0 and eth_vs_btc < -0.02:
        return MacroCryptoTicket(
            name="eth_beta_catchup",
            status="paper_relative_value_candidate",
            side="long_eth_short_btc",
            score=abs(eth_vs_btc) + risk_on_score,
            reason="risk assets are firm while ETH lags BTC materially",
        )
    if risk_on_score < 0.0 and eth_vs_btc > 0.02:
        return MacroCryptoTicket(
            name="eth_beta_deleveraging",
            status="paper_relative_value_candidate",
            side="short_eth_long_btc",
            score=abs(eth_vs_btc) + abs(risk_on_score),
            reason="risk assets are weak while ETH outperforms BTC materially",
        )
    return MacroCryptoTicket(
        name="eth_btc_macro_rotation",
        status="watch",
        side="none",
        score=abs(eth_vs_btc),
        reason="ETH/BTC relative move is not aligned with a strong macro regime",
    )


def _return_at(prices: tuple[tuple[str, float], ...], latest_index: int, lookback: int) -> float:
    previous_index = max(0, latest_index - lookback)
    previous = prices[previous_index][1]
    if previous == 0.0:
        return 0.0
    return prices[latest_index][1] / previous - 1.0


def _date_to_seconds(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--snapshot-output-path",
        type=Path,
        default=ROOT / "current_macro_crypto_context.csv",
    )
    parser.add_argument(
        "--ticket-output-path",
        type=Path,
        default=ROOT / "current_macro_crypto_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_macro_crypto_context.md",
    )
    args = parser.parse_args()

    end_date = datetime.now(tz=UTC).date()
    start_date = end_date - timedelta(days=args.days)
    snapshots, tickets = build_current_context(start_date=start_date, end_date=end_date)
    write_snapshots_csv(snapshots, output_path=args.snapshot_output_path)
    write_tickets_csv(tickets, output_path=args.ticket_output_path)
    write_markdown(snapshots, tickets, output_path=args.markdown_output_path)
    for ticket in tickets:
        print(ticket.status, ticket.side, f"score={ticket.score:.4f}", ticket.name, ticket.reason)


if __name__ == "__main__":
    main()
