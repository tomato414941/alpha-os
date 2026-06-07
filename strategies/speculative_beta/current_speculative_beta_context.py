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
class SpeculativeAsset:
    symbol: str
    label: str
    group: str
    risk_direction: int


@dataclass(frozen=True)
class SpeculativeSnapshot:
    symbol: str
    label: str
    group: str
    last_date: str
    close: float
    return_1d: float
    return_5d: float
    return_20d: float
    vs_btc_5d: float
    risk_score_5d: float


@dataclass(frozen=True)
class SpeculativeBetaTicket:
    name: str
    status: str
    side: str
    score: float
    reason: str


ASSETS = (
    SpeculativeAsset("BTC-USD", "BTC", "crypto", 1),
    SpeculativeAsset("ETH-USD", "ETH", "crypto", 1),
    SpeculativeAsset("ARKK", "ARK Innovation ETF", "high_beta_growth", 1),
    SpeculativeAsset("SOXX", "iShares Semiconductor ETF", "semiconductor", 1),
    SpeculativeAsset("SMH", "VanEck Semiconductor ETF", "semiconductor", 1),
    SpeculativeAsset("NVDA", "Nvidia", "ai_equity", 1),
    SpeculativeAsset("AVGO", "Broadcom", "ai_equity", 1),
    SpeculativeAsset("TSLA", "Tesla", "high_beta_growth", 1),
    SpeculativeAsset("PLTR", "Palantir", "ai_equity", 1),
    SpeculativeAsset("QQQ", "Nasdaq 100 ETF", "growth_index", 1),
    SpeculativeAsset("IWM", "Russell 2000 ETF", "small_cap", 1),
    SpeculativeAsset("^VIX", "VIX", "volatility", 1),
)


def build_current_context(
    *,
    start_date: date,
    end_date: date,
) -> tuple[tuple[SpeculativeSnapshot, ...], tuple[SpeculativeBetaTicket, ...]]:
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
    btc_return_5d = _return_at_latest(raw["BTC-USD"], 5)
    snapshots = tuple(_snapshot(asset, raw[asset.symbol], btc_return_5d=btc_return_5d) for asset in ASSETS)
    return snapshots, _build_tickets(snapshots)


def write_snapshots_csv(snapshots: tuple[SpeculativeSnapshot, ...], *, output_path: Path) -> Path:
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
                "vs_btc_5d",
                "risk_score_5d",
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
                    f"{row.vs_btc_5d:.8f}",
                    f"{row.risk_score_5d:.8f}",
                )
            )
    return output_path


def write_tickets_csv(tickets: tuple[SpeculativeBetaTicket, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("name", "status", "side", "score", "reason"))
        for ticket in tickets:
            writer.writerow((ticket.name, ticket.status, ticket.side, f"{ticket.score:.8f}", ticket.reason))
    return output_path


def write_markdown(
    snapshots: tuple[SpeculativeSnapshot, ...],
    tickets: tuple[SpeculativeBetaTicket, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Speculative Beta Context\n\n")
        handle.write(
            "This compares crypto with AI, semiconductor, and high-beta growth proxies. "
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
        handle.write("\n## Current Context\n\n")
        handle.write("| symbol | group | close | 1d | 5d | 20d | vs BTC 5d | risk score 5d |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in snapshots:
            handle.write(
                f"| {row.symbol} | {row.group} | {row.close:.4f} | "
                f"{row.return_1d:.4f} | {row.return_5d:.4f} | "
                f"{row.return_20d:.4f} | {row.vs_btc_5d:.4f} | {row.risk_score_5d:.4f} |\n"
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
    asset: SpeculativeAsset,
    prices: tuple[tuple[str, float], ...],
    *,
    btc_return_5d: float,
) -> SpeculativeSnapshot:
    if len(prices) < 2:
        raise ValueError(f"Not enough Yahoo rows for {asset.symbol}")
    latest_index = len(prices) - 1
    return_5d = _return_at(prices, latest_index, 5)
    return SpeculativeSnapshot(
        symbol=asset.symbol,
        label=asset.label,
        group=asset.group,
        last_date=prices[latest_index][0],
        close=prices[latest_index][1],
        return_1d=_return_at(prices, latest_index, 1),
        return_5d=return_5d,
        return_20d=_return_at(prices, latest_index, 20),
        vs_btc_5d=return_5d - btc_return_5d,
        risk_score_5d=asset.risk_direction * return_5d,
    )


def _build_tickets(snapshots: tuple[SpeculativeSnapshot, ...]) -> tuple[SpeculativeBetaTicket, ...]:
    by_symbol = {row.symbol: row for row in snapshots}
    btc = by_symbol["BTC-USD"]
    eth = by_symbol["ETH-USD"]
    vix = by_symbol["^VIX"]
    high_beta = tuple(row for row in snapshots if row.group in {"high_beta_growth", "ai_equity", "semiconductor"})
    ai = tuple(row for row in snapshots if row.group == "ai_equity")
    semis = tuple(row for row in snapshots if row.group == "semiconductor")

    high_beta_score = _mean(row.risk_score_5d for row in high_beta)
    ai_vs_btc = _mean(row.vs_btc_5d for row in ai)
    semis_vs_btc = _mean(row.vs_btc_5d for row in semis)
    crypto_score = (btc.return_5d + eth.return_5d) / 2.0

    tickets = (
        _high_beta_lead_ticket(high_beta_score=high_beta_score, crypto_score=crypto_score, btc=btc),
        _ai_crypto_divergence_ticket(ai_vs_btc=ai_vs_btc, btc=btc, eth=eth),
        _semiconductor_crypto_divergence_ticket(semis_vs_btc=semis_vs_btc, btc=btc),
        _vix_air_pocket_ticket(vix=vix, btc=btc, eth=eth, high_beta_score=high_beta_score),
    )
    return tuple(sorted(tickets, key=lambda ticket: abs(ticket.score), reverse=True))


def _high_beta_lead_ticket(
    *,
    high_beta_score: float,
    crypto_score: float,
    btc: SpeculativeSnapshot,
) -> SpeculativeBetaTicket:
    score = high_beta_score - crypto_score
    if high_beta_score > 0.03 and crypto_score < 0.0:
        return SpeculativeBetaTicket(
            name="speculative_beta_lead_long",
            status="paper_long_candidate",
            side="long_btc_eth",
            score=score,
            reason="AI/high-beta equities are risk-on while crypto is down",
        )
    if high_beta_score < -0.03 and btc.return_5d > -0.03:
        return SpeculativeBetaTicket(
            name="speculative_beta_air_pocket_short",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=score,
            reason="AI/high-beta equities are risk-off while BTC has not repriced much",
        )
    return SpeculativeBetaTicket(
        name="speculative_beta_lead",
        status="watch",
        side="none",
        score=score,
        reason="speculative beta does not show a clean lead/lag trade",
    )


def _ai_crypto_divergence_ticket(
    *,
    ai_vs_btc: float,
    btc: SpeculativeSnapshot,
    eth: SpeculativeSnapshot,
) -> SpeculativeBetaTicket:
    if ai_vs_btc > 0.08 and eth.return_5d < btc.return_5d:
        return SpeculativeBetaTicket(
            name="ai_beta_crypto_catchup",
            status="paper_long_candidate",
            side="long_eth_or_crypto_beta",
            score=ai_vs_btc,
            reason="AI equities strongly outperform BTC while ETH also lags BTC",
        )
    if ai_vs_btc < -0.08 and btc.return_5d > -0.03:
        return SpeculativeBetaTicket(
            name="ai_beta_crypto_warning",
            status="paper_short_candidate",
            side="short_crypto_beta",
            score=ai_vs_btc,
            reason="AI equities strongly underperform BTC while BTC has not repriced much",
        )
    return SpeculativeBetaTicket(
        name="ai_beta_crypto_divergence",
        status="watch",
        side="none",
        score=ai_vs_btc,
        reason="AI/BTC divergence is not actionable without repeated labels",
    )


def _semiconductor_crypto_divergence_ticket(
    *,
    semis_vs_btc: float,
    btc: SpeculativeSnapshot,
) -> SpeculativeBetaTicket:
    if semis_vs_btc > 0.06 and btc.return_5d < 0.0:
        return SpeculativeBetaTicket(
            name="semis_crypto_catchup",
            status="paper_long_candidate",
            side="long_btc_eth",
            score=semis_vs_btc,
            reason="semiconductors outperform BTC while BTC is down",
        )
    if semis_vs_btc < -0.06 and btc.return_5d > -0.03:
        return SpeculativeBetaTicket(
            name="semis_crypto_warning",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=semis_vs_btc,
            reason="semiconductors underperform BTC while BTC has not repriced much",
        )
    return SpeculativeBetaTicket(
        name="semis_crypto_divergence",
        status="watch",
        side="none",
        score=semis_vs_btc,
        reason="semiconductor/BTC divergence is not clean enough",
    )


def _vix_air_pocket_ticket(
    *,
    vix: SpeculativeSnapshot,
    btc: SpeculativeSnapshot,
    eth: SpeculativeSnapshot,
    high_beta_score: float,
) -> SpeculativeBetaTicket:
    crypto_score = (btc.return_5d + eth.return_5d) / 2.0
    score = vix.return_5d - crypto_score - high_beta_score
    if vix.return_5d > 0.15 and high_beta_score < -0.03 and crypto_score > -0.03:
        return SpeculativeBetaTicket(
            name="vix_high_beta_air_pocket",
            status="paper_short_candidate",
            side="short_btc_eth",
            score=score,
            reason="VIX and high-beta equities are risk-off while crypto has not repriced much",
        )
    return SpeculativeBetaTicket(
        name="vix_high_beta_air_pocket",
        status="watch",
        side="none",
        score=score,
        reason="risk-off shock is visible, but crypto may already have repriced",
    )


def _return_at_latest(prices: tuple[tuple[str, float], ...], lookback: int) -> float:
    return _return_at(prices, len(prices) - 1, lookback)


def _return_at(prices: tuple[tuple[str, float], ...], latest_index: int, lookback: int) -> float:
    previous_index = max(0, latest_index - lookback)
    previous = prices[previous_index][1]
    if previous == 0.0:
        return 0.0
    return prices[latest_index][1] / previous - 1.0


def _mean(values: object) -> float:
    rows = tuple(float(value) for value in values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)


def _date_to_seconds(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--snapshot-output-path",
        type=Path,
        default=ROOT / "current_speculative_beta_context.csv",
    )
    parser.add_argument(
        "--ticket-output-path",
        type=Path,
        default=ROOT / "current_speculative_beta_paper_tickets.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_speculative_beta_context.md",
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
