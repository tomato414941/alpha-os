"""December JPY discount hypothesis.

HYPOTHESIS: JPY-quoted retail-heavy coins (XRP) trade at a systematic
            discount vs USD venues in Dec 15-31 driven by Japanese
            tax-loss selling (no wash-sale rule), closing by Jan 10.
VERDICT:    REJECTED 0/8 years (2026-06-10). Window means were mostly
            on the premium side; the discount never materialized.
COUNTERPARTY: Japanese retail forced by the calendar-year tax boundary.
            Assumed barrier to arbitrage: JPY accounts and JPY rails.
            Rejection suggests domestic arbitrage keeps venues connected.
TEST:       bitbank XRP/JPY x Binance XRP/USDT x USDJPY, 2018-2025.
            Dec 15-31 mean discount vs 5th percentile of same-year
            17-day rolling means outside the window. Pre-registered
            rule: reject if fewer than 4 of 8 years hit.
REVIVAL:    Re-run only if JPY/USD venue connectivity degrades
            (structural widening of cross-venue spreads).

Run: PYTHONPATH=src python hypotheses/december_jpy_discount.py
The VERDICT above is a copy of this script's output; re-running
re-derives it from public APIs (bitbank, Binance data mirror, ECB FX).
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import urllib.request
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

# ---------- contract types ----------


@dataclass(frozen=True)
class JpyDiscountInput:
    date: dt.date
    discount: float  # (jpy_px / usdjpy) / usd_px - 1
    baseline_mean: float  # trailing discount mean, window days excluded
    baseline_std: float
    current_weight: float


@dataclass(frozen=True)
class TargetBasisPosition:
    weight: float  # 1.0 = long JPY leg / short USD leg, 0.0 = flat


class DecemberJpyDiscountStrategy:
    """Enter when the December tax-loss window pushes the discount below a z threshold."""

    def __init__(self, entry_z: float = -1.5) -> None:
        self._entry_z = entry_z

    def decide(self, strategy_input: JpyDiscountInput) -> TargetBasisPosition:
        d = strategy_input.date
        in_entry = d.month == 12 and d.day >= 15
        in_hold = in_entry or (d.month == 1 and d.day <= 10)
        if not in_hold:
            return TargetBasisPosition(0.0)
        if strategy_input.current_weight > 0:
            if strategy_input.discount >= 0:
                return TargetBasisPosition(0.0)
            return TargetBasisPosition(1.0)
        if in_entry and strategy_input.baseline_std > 0:
            z = (
                strategy_input.discount - strategy_input.baseline_mean
            ) / strategy_input.baseline_std
            if z <= self._entry_z:
                return TargetBasisPosition(1.0)
        return TargetBasisPosition(0.0)


_CONTRACT_CHECK: TradingStrategy[JpyDiscountInput, TargetBasisPosition] = (
    DecemberJpyDiscountStrategy()
)

# ---------- data (impure, outside the contract) ----------


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def bitbank_daily(pair: str, years: range) -> dict[dt.date, float]:
    out: dict[dt.date, float] = {}
    for y in years:
        try:
            d = fetch_json(f"https://public.bitbank.cc/{pair}/candlestick/1day/{y}")
        except Exception:
            continue
        for o in d["data"]["candlestick"][0]["ohlcv"]:
            out[dt.date.fromtimestamp(o[5] / 1000)] = float(o[3])
    return out


def binance_daily(symbol: str, start: dt.date) -> dict[dt.date, float]:
    out: dict[dt.date, float] = {}
    t = int(dt.datetime(start.year, start.month, start.day).timestamp() * 1000)
    while True:
        ks = fetch_json(
            "https://data-api.binance.vision/api/v3/klines"
            f"?symbol={symbol}&interval=1d&limit=1000&startTime={t}"
        )
        if not ks:
            break
        for k in ks:
            out[dt.date.fromtimestamp(k[0] / 1000)] = float(k[4])
        if len(ks) < 1000:
            break
        t = ks[-1][0] + 1
    return out


def usdjpy_daily(start: str, end: str) -> dict[dt.date, float]:
    d = fetch_json(f"https://api.frankfurter.dev/v1/{start}..{end}?base=USD&symbols=JPY")
    raw = {dt.date.fromisoformat(k): v["JPY"] for k, v in d["rates"].items()}
    out: dict[dt.date, float] = {}
    last = None
    day = min(raw)
    while day <= dt.date.fromisoformat(end):
        last = raw.get(day, last)
        if last:
            out[day] = last
        day += dt.timedelta(days=1)
    return out


# ---------- verdict harness ----------


def in_window(d: dt.date) -> bool:
    return (d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 10)


def main() -> None:
    today = dt.date.today().isoformat()
    jpy = bitbank_daily("xrp_jpy", range(2018, dt.date.today().year + 1))
    usd = binance_daily("XRPUSDT", dt.date(2018, 5, 4))
    fx = usdjpy_daily("2018-01-01", today)
    days = sorted(set(jpy) & set(usd) & set(fx))
    disc = {d: (jpy[d] / fx[d]) / usd[d] - 1 for d in days}
    print(f"common days: {len(days)} ({days[0]} - {days[-1]})")

    print("\n=== effect test: Dec 15-31 mean vs same-year 5th pct of 17d windows ===")
    hits = 0
    nyears = 0
    for y in range(2018, dt.date.today().year):
        win = [disc[d] for d in days if d.year == y and d.month == 12 and d.day >= 15]
        rest = [disc[d] for d in days if d.year == y and not in_window(d)]
        if len(win) < 10 or len(rest) < 200:
            continue
        nyears += 1
        wmean = statistics.mean(win)
        rolls = sorted(
            statistics.mean(rest[i : i + 17]) for i in range(len(rest) - 17)
        )
        p5 = rolls[int(0.05 * len(rolls))]
        hit = wmean < p5
        hits += hit
        print(
            f"  {y}: window {wmean * 1e4:+7.1f}bps  5th pct {p5 * 1e4:+7.1f}bps"
            f"  -> {'HIT' if hit else 'miss'}"
        )
    verdict = "SUPPORTED" if hits >= 4 else "REJECTED"
    print(f"\n  VERDICT: {verdict} {hits}/{nyears} (pre-registered: reject if < 4)")

    print("\n=== strategy backtest via contract (0.3% round-trip friction) ===")
    strat = DecemberJpyDiscountStrategy()
    cost = 0.0030
    weight = 0.0
    entry_disc = 0.0
    pnl_by_year: dict[int, float] = {}
    hist: list[float] = []
    for d in days:
        hist.append(disc[d])
        base = hist[-104:-14] if len(hist) > 120 else []
        if len(base) < 60:
            continue
        decision = strat.decide(
            JpyDiscountInput(
                d, disc[d], statistics.mean(base), statistics.pstdev(base), weight
            )
        )
        if decision.weight != weight:
            if decision.weight > 0:
                entry_disc = disc[d]
                pnl_by_year[d.year] = pnl_by_year.get(d.year, 0.0) - cost
            else:
                pnl_by_year[d.year] = pnl_by_year.get(d.year, 0.0) + (
                    disc[d] - entry_disc
                )
            weight = decision.weight
    total = 0.0
    for y in sorted(pnl_by_year):
        print(f"  {y}: {pnl_by_year[y] * 100:+.2f}%")
        total += pnl_by_year[y]
    print(f"  total: {total * 100:+.2f}% (per-event positions, no compounding)")
    print("\nNote: if the effect test rejects, backtest PnL is noise. Do not cherry-pick.")


if __name__ == "__main__":
    main()
