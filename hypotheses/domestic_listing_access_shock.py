"""Domestic listing access shock hypothesis.

HYPOTHESIS: When a Japanese exchange (Coincheck/bitFlyer/GMO/bitbank)
            announces it will list an already-globally-traded coin,
            the coin shows positive market-adjusted returns in the
            days after the announcement, driven by newly enabled
            domestic retail demand.
VERDICT:    REJECTED (2026-06-11). Pre-registered window close(t0) ->
            close(t0+2): mean abnormal return +0.04%, p = 0.51, N = 42.
            The entire apparent effect (+3.9% measured from t0-1) sits
            on/before the announcement day and is dominated by two ENJ
            outliers from the Jan-Mar 2021 mania. At daily resolution,
            anyone entering after reading the news captures nothing.
COUNTERPARTY: Japanese retail gaining app access. Assumed barrier:
            Japanese-language announcements, capacity too small for
            funds. Rejection shows the move completes intraday -
            the market is faster than the daily close.
TEST:       50 listing announcements 2018-2025 reconstructed from
            CoinPost archives (cleaning: drop delistings, order-book
            migrations, margin additions, digests). Binance daily
            closes, BTC-adjusted. Bootstrap from random same-coin
            dates; reject if mean <= 95th pct or < 0.3% friction.
REVIVAL:    An intraday version (enter within minutes of the
            announcement timestamp) is a SEPARATE, untested hypothesis
            requiring announcement timestamps and intraday data. Do
            not treat this rejection as covering it.

Run: PYTHONPATH=src python hypotheses/domestic_listing_access_shock.py
Event list is frozen below (re-harvesting CoinPost may drift); prices
are re-fetched from public APIs, so the verdict re-derives.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import statistics
import time
import urllib.request
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

# (announcement date, exchange, symbol) - frozen 2026-06-11 harvest
EVENTS = [
    ("2018-01-31", "bitflyer", "LSK"),
    ("2019-05-31", "coincheck", "MONA"),
    ("2019-12-02", "bitflyer", "XRP"),
    ("2020-07-13", "coincheck", "BAT"),
    ("2020-08-06", "bitflyer", "XEM"),
    ("2020-08-06", "bitflyer", "XLM"),
    ("2020-09-10", "bitbank", "XLM"),
    ("2020-12-08", "bitflyer", "XTZ"),
    ("2021-01-19", "coincheck", "ENJ"),
    ("2021-01-20", "bitbank", "QTUM"),
    ("2021-03-03", "gmo", "ENJ"),
    ("2021-04-07", "coincheck", "OMG"),
    ("2021-06-29", "bitflyer", "DOT"),
    ("2021-10-19", "bitbank", "OMG"),
    ("2021-10-19", "bitbank", "XYM"),
    ("2021-12-03", "bitbank", "LINK"),
    ("2022-01-19", "bitbank", "DAI"),
    ("2022-01-19", "bitbank", "MKR"),
    ("2022-01-26", "gmo", "ADA"),
    ("2022-02-17", "gmo", "DAI"),
    ("2022-02-17", "gmo", "MKR"),
    ("2022-03-03", "gmo", "LINK"),
    ("2022-03-09", "bitbank", "BOBA"),
    ("2022-05-24", "coincheck", "SAND"),
    ("2022-06-08", "bitbank", "MATIC"),
    ("2022-07-06", "bitflyer", "MKR"),
    ("2022-07-06", "bitflyer", "MATIC"),
    ("2022-07-29", "bitbank", "DOGE"),
    ("2022-07-29", "bitbank", "DOT"),
    ("2022-11-04", "bitbank", "AVAX"),
    ("2022-12-02", "bitbank", "AXS"),
    ("2023-03-03", "bitbank", "APE"),
    ("2023-03-03", "bitbank", "CHZ"),
    ("2023-03-03", "bitbank", "GALA"),
    ("2023-03-16", "coincheck", "FNCT"),
    ("2023-06-12", "bitbank", "GRT"),
    ("2023-10-06", "coincheck", "WBTC"),
    ("2023-10-19", "bitbank", "BNB"),
    ("2023-12-11", "bitbank", "ARB"),
    ("2023-12-11", "bitbank", "IMX"),
    ("2023-12-11", "bitbank", "OP"),
    ("2024-02-27", "bitflyer", "DOGE"),
    ("2024-08-26", "bitflyer", "MASK"),
    ("2024-10-29", "gmo", "AVAX"),
    ("2025-01-15", "coincheck", "DOGE"),
    ("2025-02-04", "bitbank", "TRX"),
    ("2025-04-25", "coincheck", "PEPE"),
    ("2025-05-13", "bitbank", "ATOM"),
    ("2025-08-08", "bitbank", "SUI"),
    ("2025-12-11", "coincheck", "SOL"),
]

# ---------- contract types ----------


@dataclass(frozen=True)
class ListingEventInput:
    days_since_announcement: int  # 0 = announcement day


@dataclass(frozen=True)
class TargetPosition:
    weight: float  # 1.0 = long vs BTC hedge, 0.0 = flat


class ListingMomentumStrategy:
    """Enter at the first close after the announcement, exit two days later."""

    def decide(self, strategy_input: ListingEventInput) -> TargetPosition:
        if 0 <= strategy_input.days_since_announcement < 2:
            return TargetPosition(1.0)
        return TargetPosition(0.0)


_CONTRACT_CHECK: TradingStrategy[ListingEventInput, TargetPosition] = (
    ListingMomentumStrategy()
)

# ---------- data (impure) ----------


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def binance_daily(symbol: str) -> dict[dt.date, float]:
    out: dict[dt.date, float] = {}
    t = 1420070400000
    while True:
        ks = fetch_json(
            "https://data-api.binance.vision/api/v3/klines"
            f"?symbol={symbol}USDT&interval=1d&limit=1000&startTime={t}"
        )
        if not ks:
            break
        for k in ks:
            out[dt.date.fromtimestamp(k[0] / 1000)] = float(k[4])
        if len(ks) < 1000:
            break
        t = ks[-1][0] + 1
    return out


# ---------- verdict harness ----------


def main() -> None:
    btc = binance_daily("BTC")
    series: dict[str, dict[dt.date, float]] = {}
    for _, _, s in EVENTS:
        if s in series:
            continue
        try:
            d = binance_daily(s)
            if len(d) > 100:
                series[s] = d
        except Exception:
            pass
        time.sleep(0.25)

    def ar(sym: str, t0: dt.date, pre: int, post: int) -> float | None:
        px = series[sym]
        d0, d1 = t0 - dt.timedelta(days=pre), t0 + dt.timedelta(days=post)
        if d0 not in px or d1 not in px or d0 not in btc or d1 not in btc:
            return None
        return (px[d1] / px[d0] - 1) - (btc[d1] / btc[d0] - 1)

    random.seed(42)

    def study(pre: int, post: int) -> None:
        obs, used = [], []
        for date, _, s in EVENTS:
            if s not in series:
                continue
            t0 = dt.date.fromisoformat(date)
            if (t0 - min(series[s])).days < 90:
                continue  # must be globally traded well before listing
            a = ar(s, t0, pre, post)
            if a is not None:
                obs.append(a)
                used.append(s)
        pools: dict[str, list[float]] = {}
        for s in set(used):
            pool: list[float] = []
            days_list = sorted(series[s])
            while len(pool) < 300:
                a = ar(s, random.choice(days_list), pre, post)
                if a is not None:
                    pool.append(a)
            pools[s] = pool
        boots = sorted(
            statistics.mean(random.choice(pools[s]) for s in used)
            for _ in range(4000)
        )
        mean_obs = statistics.mean(obs)
        p95 = boots[int(0.95 * len(boots))]
        pval = sum(1 for b in boots if b >= mean_obs) / len(boots)
        verdict = "SUPPORTED" if (mean_obs > p95 and mean_obs > 0.003) else "REJECTED"
        print(
            f"  window t-{pre}->t+{post}: N={len(obs)} mean={mean_obs * 100:+.2f}%"
            f" median={statistics.median(obs) * 100:+.2f}%"
            f" p95={p95 * 100:+.2f}% p={pval:.4f} -> {verdict}"
        )

    print("=== pre-registered: close(t0) -> close(t0+2), BTC-adjusted ===")
    study(0, 2)
    print("=== context windows (not the registered test) ===")
    study(1, 2)  # includes pre-announcement day: where the apparent effect lives
    study(-1, 3)  # enter one day after announcement
    print(
        "\nNote: the t-1 window 'effect' is not capturable at daily resolution."
        "\nIntraday reaction to announcement timestamps = separate hypothesis."
    )


if __name__ == "__main__":
    main()
