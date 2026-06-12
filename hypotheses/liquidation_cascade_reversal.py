"""Liquidation cascade reversal hypothesis.

HYPOTHESIS: After a long-liquidation cascade (minute long-liq ratio
            >= 0.9 for >= 3 consecutive minutes with a price drop of
            >= 30 bps over the run), price mean-reverts within 30
            minutes by more than retail friction (0.10% round trip).
            Counterparty: the liquidated longs - forced,
            price-insensitive sellers. The payment is for providing
            liquidity at the moment nobody wants to.
VERDICT:    REJECTED as tradable (2026-06-11), pre-registered rule:
            >= 3 of 4 sibling symbols net-positive required, got 2/4.
              btc: 210 events net +3.6bps (boot95 -6.5, p<0.001)
              eth: 294 events net +1.2bps (boot95 -6.4, p<0.001)
              sol: 218 events net +8.1bps (boot95 -5.7, p<0.001)
              xrp: 111 events net -2.4bps (boot95 -4.4, p=0.017)
              bnb:  50 events net -3.2bps (boot95 -2.8, p=0.065)
SCIENTIFIC NOTE: the effect itself is real - ALL five symbols beat
            random entries (excess +8 to +14 bps gross vs bootstrap
            mean). It is simply thinner than our 10 bps cost floor.
            This is the measured confirmation of the project's core
            diagnosis: the pond has edge, but it sits below the
            retail friction line and belongs to sub-bp-cost players.
RULES FIXED IN ADVANCE: thresholds (0.9 / 3min / 30bps / 30min /
            10bps friction) were registered before results and must
            not be tuned post hoc. A different parameterization is a
            NEW hypothesis requiring fresh registration.
DATA:       (1) own liquidation-side-ratio stream via signal-noise
            API (private; SIGNAL_NOISE_API_KEY + Tailscale required);
            magnitudes were discarded at collection - only the
            per-minute long/(long+short) ratio survives.
            (2) Binance USD-M futures 1m klines via fapi.binance.com
            - geo-blocked from US IPs; run from a non-US host
            (e.g. the signal-noise VPS).
REVIVAL:    If execution friction drops an order of magnitude
            (maker rebates / colocated access), the measured excess
            clears the bar. Re-run as-is and re-judge.

Run: PYTHONPATH=src python hypotheses/liquidation_cascade_reversal.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

RATIO_TH = 0.9
RUN_MIN = 3
DROP_BPS = 30
HORIZON_MIN = 30
FRICTION = 0.0010
SYMS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT",
        "xrp": "XRPUSDT", "bnb": "BNBUSDT"}
SN_BASE = "https://signal-noise.taildd87b4.ts.net"

# ---------- contract types ----------


@dataclass(frozen=True)
class CascadeInput:
    cascade_run_minutes: int   # consecutive minutes with ratio >= RATIO_TH
    run_drop_bps: float        # price change over the run, in bps
    minutes_since_end: int     # 0 = cascade just ended


@dataclass(frozen=True)
class TargetPosition:
    weight: float  # +1.0 = long for HORIZON_MIN minutes, 0.0 = flat


class CascadeReversalStrategy:
    """Buy the end of a qualifying long-liquidation cascade (rejected at 10bps)."""

    def decide(self, strategy_input: CascadeInput) -> TargetPosition:
        qualifies = (
            strategy_input.cascade_run_minutes >= RUN_MIN
            and strategy_input.run_drop_bps <= -DROP_BPS
            and 0 <= strategy_input.minutes_since_end < HORIZON_MIN
        )
        return TargetPosition(1.0 if qualifies else 0.0)


_CONTRACT_CHECK: TradingStrategy[CascadeInput, TargetPosition] = (
    CascadeReversalStrategy()
)

# ---------- data (impure, private + geo-restricted) ----------


def fetch_json(req: urllib.request.Request, timeout: int, retries: int = 5):
    for i in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))


def sn_api_key() -> str:
    path = os.path.expanduser("~/.secrets/signal-noise-env")
    for line in open(path):
        if line.startswith("SIGNAL_NOISE_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("SIGNAL_NOISE_API_KEY not found")


def liq_minutes(key: str) -> dict[dt.datetime, float]:
    req = urllib.request.Request(
        f"{SN_BASE}/signals/liq_stream_{key}/data?limit=200000",
        headers={"X-API-Key": sn_api_key()},
    )
    rows = fetch_json(req, timeout=180)
    out = {}
    for r in rows:
        t = dt.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        out[t.replace(second=0, microsecond=0, tzinfo=None)] = float(r["value"])
    return out


def futures_1m(symbol: str, start_ms: int, end_ms: int) -> dict[dt.datetime, float]:
    px = {}
    t = start_ms
    while t < end_ms:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
               f"&interval=1m&limit=1500&startTime={t}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ks = fetch_json(req, timeout=30)
        if not ks:
            break
        for k in ks:
            ts = dt.datetime.fromtimestamp(k[0] / 1000, dt.UTC)
            px[ts.replace(second=0, microsecond=0, tzinfo=None)] = float(k[4])
        t = ks[-1][0] + 60000
        time.sleep(0.15)
    return px


# ---------- verdict harness ----------


def main() -> None:
    random.seed(7)
    net_positive = {}
    print(f"params: ratio>={RATIO_TH} run>={RUN_MIN}m drop>={DROP_BPS}bps "
          f"horizon={HORIZON_MIN}m friction={FRICTION * 1e4:.0f}bps")
    for key, sym in SYMS.items():
        liq = liq_minutes(key)
        ts_sorted = sorted(liq)
        px = futures_1m(
            sym,
            int(ts_sorted[0].replace(tzinfo=dt.UTC).timestamp() * 1000) - 3600000,
            int(ts_sorted[-1].replace(tzinfo=dt.UTC).timestamp() * 1000) + 3600000,
        )
        events = []
        run = 0
        for t in ts_sorted:
            if liq[t] >= RATIO_TH:
                run += 1
            else:
                if run >= RUN_MIN:
                    t_end = t - dt.timedelta(minutes=1)
                    t_start = t_end - dt.timedelta(minutes=run - 1)
                    p0 = px.get(t_start - dt.timedelta(minutes=1))
                    p1 = px.get(t_end)
                    if p0 and p1 and (p1 / p0 - 1) * 1e4 <= -DROP_BPS:
                        events.append(t_end)
                run = 0
        rets = []
        for e in events:
            p_in = px.get(e)
            p_out = px.get(e + dt.timedelta(minutes=HORIZON_MIN))
            if p_in and p_out:
                rets.append(p_out / p_in - 1 - FRICTION)
        if len(rets) < 10:
            print(f"{key}: events={len(rets)} insufficient")
            continue
        mean = sum(rets) / len(rets)
        pool = [t for t in ts_sorted
                if t in px and (t + dt.timedelta(minutes=HORIZON_MIN)) in px]
        boots = []
        for _ in range(2000):
            s = sum(px[t + dt.timedelta(minutes=HORIZON_MIN)] / px[t] - 1 - FRICTION
                    for t in (random.choice(pool) for _ in range(len(rets))))
            boots.append(s / len(rets))
        boots.sort()
        p95 = boots[int(0.95 * len(boots))]
        pval = sum(1 for b in boots if b >= mean) / len(boots)
        net_positive[key] = mean > 0
        print(f"{key}: events={len(rets)} mean_net={mean * 1e4:+.1f}bps "
              f"boot95={p95 * 1e4:+.1f}bps p={pval:.4f}")
    sibs = [k for k in SYMS if k != "btc" and k in net_positive]
    agree = sum(1 for k in sibs if net_positive[k])
    supported = net_positive.get("btc", False) and agree >= 3
    print(f"\nsibling net>0: {agree}/{len(sibs)} (rule: need >=3)"
          f"\nVERDICT: {'SUPPORTED' if supported else 'REJECTED'}")


if __name__ == "__main__":
    main()
