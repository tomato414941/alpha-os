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
class ProxyAsset:
    symbol: str
    label: str
    group: str


@dataclass(frozen=True)
class ProxySnapshot:
    symbol: str
    label: str
    group: str
    last_date: str
    close: float
    return_1d: float
    return_5d: float
    return_20d: float
    vs_btc_1d: float
    vs_btc_5d: float


@dataclass(frozen=True)
class CryptoEquityProxyTicket:
    name: str
    status: str
    side: str
    score: float
    reason: str


ASSETS = (
    ProxyAsset("BTC-USD", "BTC", "crypto"),
    ProxyAsset("ETH-USD", "ETH", "crypto"),
    ProxyAsset("MSTR", "MicroStrategy", "btc_treasury_equity"),
    ProxyAsset("COIN", "Coinbase", "exchange_equity"),
    ProxyAsset("HOOD", "Robinhood", "broker_equity"),
    ProxyAsset("IBIT", "iShares Bitcoin Trust", "spot_btc_etf"),
    ProxyAsset("BITO", "ProShares Bitcoin Strategy ETF", "btc_futures_etf"),
    ProxyAsset("MARA", "MARA Holdings", "miner"),
    ProxyAsset("RIOT", "Riot Platforms", "miner"),
    ProxyAsset("CLSK", "CleanSpark", "miner"),
)


def build_current_context(
    *,
    start_date: date,
    end_date: date,
) -> tuple[tuple[ProxySnapshot, ...], tuple[CryptoEquityProxyTicket, ...]]:
    session = requests.Session()
    raw = {
        asset.symbol: _fetch_yahoo_closes(
            asset.symbol,
            start_date=start_date,
            end_date=end_date,
            session=session,
        )
        for asset in ASSETS
    }
    btc = _snapshot_base_returns(raw["BTC-USD"])
    snapshots = tuple(_snapshot(asset, raw[asset.symbol], btc_return_1d=btc[0], btc_return_5d=btc[1]) for asset in ASSETS)
    return snapshots, _build_tickets(snapshots)


def write_snapshots_csv(snapshots: tuple[ProxySnapshot, ...], *, output_path: Path) -> Path:
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
                "vs_btc_1d",
                "vs_btc_5d",
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
                    f"{row.vs_btc_1d:.8f}",
                    f"{row.vs_btc_5d:.8f}",
                )
            )
    return output_path


def write_tickets_csv(
    tickets: tuple[CryptoEquityProxyTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("name", "status", "side", "score", "reason"))
        for ticket in tickets:
            writer.writerow((ticket.name, ticket.status, ticket.side, f"{ticket.score:.8f}", ticket.reason))
    return output_path


def write_markdown(
    snapshots: tuple[ProxySnapshot, ...],
    tickets: tuple[CryptoEquityProxyTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Equity Proxy Context\n\n")
        handle.write(
            "This compares BTC/ETH with crypto-linked equities and ETFs. "
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
        handle.write("\n## Proxy Context\n\n")
        handle.write("| symbol | group | close | 1d | 5d | 20d | vs BTC 1d | vs BTC 5d |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in snapshots:
            handle.write(
                f"| {row.symbol} | {row.group} | {row.close:.4f} | "
                f"{row.return_1d:.4f} | {row.return_5d:.4f} | "
                f"{row.return_20d:.4f} | {row.vs_btc_1d:.4f} | {row.vs_btc_5d:.4f} |\n"
            )
    return output_path


def _fetch_yahoo_closes(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> tuple[tuple[str, float], ...]:
    print(f"fetch {symbol}", flush=True)
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


def _snapshot(
    asset: ProxyAsset,
    prices: tuple[tuple[str, float], ...],
    *,
    btc_return_1d: float,
    btc_return_5d: float,
) -> ProxySnapshot:
    if len(prices) < 2:
        raise ValueError(f"Not enough Yahoo rows for {asset.symbol}")
    latest_index = len(prices) - 1
    return_1d = _return_at(prices, latest_index, 1)
    return_5d = _return_at(prices, latest_index, 5)
    return ProxySnapshot(
        symbol=asset.symbol,
        label=asset.label,
        group=asset.group,
        last_date=prices[latest_index][0],
        close=prices[latest_index][1],
        return_1d=return_1d,
        return_5d=return_5d,
        return_20d=_return_at(prices, latest_index, 20),
        vs_btc_1d=return_1d - btc_return_1d,
        vs_btc_5d=return_5d - btc_return_5d,
    )


def _snapshot_base_returns(prices: tuple[tuple[str, float], ...]) -> tuple[float, float]:
    latest_index = len(prices) - 1
    return _return_at(prices, latest_index, 1), _return_at(prices, latest_index, 5)


def _build_tickets(snapshots: tuple[ProxySnapshot, ...]) -> tuple[CryptoEquityProxyTicket, ...]:
    by_symbol = {row.symbol: row for row in snapshots}
    btc = by_symbol["BTC-USD"]
    mstr = by_symbol["MSTR"]
    coin = by_symbol["COIN"]
    hood = by_symbol["HOOD"]
    ibit = by_symbol["IBIT"]
    miners = tuple(row for row in snapshots if row.group == "miner")
    equity_proxies = (mstr, coin, hood, ibit, *miners)
    avg_proxy_vs_btc_5d = sum(row.vs_btc_5d for row in equity_proxies) / len(equity_proxies)
    avg_miner_vs_btc_5d = sum(row.vs_btc_5d for row in miners) / len(miners)
    tickets = (
        _proxy_lead_ticket(avg_proxy_vs_btc_5d=avg_proxy_vs_btc_5d, btc=btc),
        _mstr_dislocation_ticket(mstr=mstr, btc=btc),
        _miner_stress_ticket(avg_miner_vs_btc_5d=avg_miner_vs_btc_5d),
        _exchange_beta_ticket(coin=coin, hood=hood, btc=btc),
    )
    return tuple(sorted(tickets, key=lambda ticket: abs(ticket.score), reverse=True))


def _proxy_lead_ticket(*, avg_proxy_vs_btc_5d: float, btc: ProxySnapshot) -> CryptoEquityProxyTicket:
    score = avg_proxy_vs_btc_5d
    if avg_proxy_vs_btc_5d > 0.04 and btc.return_5d < 0.0:
        return CryptoEquityProxyTicket(
            name="crypto_equity_proxy_lead_long",
            status="paper_long_candidate",
            side="long_btc_eth",
            score=score,
            reason="crypto-linked equities outperform BTC while BTC is down",
        )
    if avg_proxy_vs_btc_5d < -0.04 and btc.return_5d > -0.03:
        return CryptoEquityProxyTicket(
            name="crypto_equity_proxy_lead_short",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=score,
            reason="crypto-linked equities underperform BTC while BTC has not repriced much",
        )
    return CryptoEquityProxyTicket(
        name="crypto_equity_proxy_lead",
        status="watch",
        side="none",
        score=score,
        reason="proxy basket does not show a clean lead/lag trade",
    )


def _mstr_dislocation_ticket(*, mstr: ProxySnapshot, btc: ProxySnapshot) -> CryptoEquityProxyTicket:
    score = mstr.vs_btc_5d
    if score > 0.08:
        return CryptoEquityProxyTicket(
            name="mstr_btc_dislocation",
            status="paper_relative_value_watch",
            side="long_btc_short_mstr",
            score=score,
            reason="MSTR strongly outperforms BTC; check announcement/news and borrow before relative-value action",
        )
    if score < -0.08:
        return CryptoEquityProxyTicket(
            name="mstr_btc_dislocation",
            status="paper_relative_value_watch",
            side="long_mstr_short_btc",
            score=score,
            reason="MSTR strongly underperforms BTC; check announcement/news and equity-market risk before action",
        )
    return CryptoEquityProxyTicket(
        name="mstr_btc_dislocation",
        status="watch",
        side="none",
        score=score,
        reason=f"MSTR/BTC spread is not extreme; mstr5d={mstr.return_5d:.4f}, btc5d={btc.return_5d:.4f}",
    )


def _miner_stress_ticket(*, avg_miner_vs_btc_5d: float) -> CryptoEquityProxyTicket:
    if avg_miner_vs_btc_5d < -0.08:
        return CryptoEquityProxyTicket(
            name="miner_stress_vs_btc",
            status="paper_risk_context",
            side="de_risk_alt_beta",
            score=avg_miner_vs_btc_5d,
            reason="miners materially underperform BTC, which can indicate equity-market stress around crypto beta",
        )
    return CryptoEquityProxyTicket(
        name="miner_stress_vs_btc",
        status="watch",
        side="none",
        score=avg_miner_vs_btc_5d,
        reason="miner basket does not show severe stress versus BTC",
    )


def _exchange_beta_ticket(
    *,
    coin: ProxySnapshot,
    hood: ProxySnapshot,
    btc: ProxySnapshot,
) -> CryptoEquityProxyTicket:
    exchange_vs_btc = ((coin.vs_btc_5d + hood.vs_btc_5d) / 2.0)
    if exchange_vs_btc > 0.06 and btc.return_5d < 0.0:
        return CryptoEquityProxyTicket(
            name="exchange_beta_lead",
            status="paper_long_candidate",
            side="long_btc_eth",
            score=exchange_vs_btc,
            reason="COIN/HOOD outperform BTC while BTC is down; check whether equities are anticipating crypto recovery",
        )
    if exchange_vs_btc < -0.06 and btc.return_5d > -0.03:
        return CryptoEquityProxyTicket(
            name="exchange_beta_lead",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=exchange_vs_btc,
            reason="COIN/HOOD underperform BTC while BTC has not repriced much",
        )
    return CryptoEquityProxyTicket(
        name="exchange_beta_lead",
        status="watch",
        side="none",
        score=exchange_vs_btc,
        reason="exchange equity beta does not show a clean crypto lead signal",
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
        default=ROOT / "current_crypto_equity_proxy_context.csv",
    )
    parser.add_argument(
        "--ticket-output-path",
        type=Path,
        default=ROOT / "current_crypto_equity_proxy_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_crypto_equity_proxy_context.md",
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
