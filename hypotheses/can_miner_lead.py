"""CAN miner-equity lead hypothesis.

HYPOTHESIS: Bitcoin-miner equities carry information beyond BTC's own
            move: their daily change (orthogonalized to same-day BTC)
            negatively predicts next-day BTC. Surfaced by an FDR
            screen (q=0.05) over 3,878 signal-noise series where CAN
            (Canaan Inc.) was the only non-leak survivor
            (rho=-0.137, partial=-0.129, n=1177, both halves negative).
VERDICT:    REJECTED (2026-06-11). Sibling test kills the mechanism:
            all 10 other miner equities (MARA, RIOT, CLSK, HUT, BITF,
            WULF, CIFR, BTBT, CORZ, IREN) show partials collapsing to
            ~0 (max |0.038|) once same-day BTC is controlled. Their
            weak raw negatives were echoes of BTC's own next-day
            reversal in this sample. A single-name anomaly with no
            mechanism and no sibling confirmation is treated as a
            multiple-testing artifact.
RULE-DESIGN CONFESSION: the pre-registered sibling rule was written
            on RAW sign majority and technically passed (10/11
            negative). That was a specification error: CAN's own
            survival evidence was the partial, so the sibling test
            had to be on partials too. Lesson: pre-register rules on
            the decisive statistic, or weak rules pass wrong things.
DATA:       signal-noise API (private). Requires Tailscale access and
            SIGNAL_NOISE_API_KEY in ~/.secrets/signal-noise-env.
            First graveyard entry that depends on our own data.
REVIVAL:    Only if a concrete mechanism for a CAN-specific channel is
            proposed, or the effect replicates on an independent
            period/venue.

Run: PYTHONPATH=src python hypotheses/can_miner_lead.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import urllib.request
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

BASE = "https://signal-noise.taildd87b4.ts.net"
MINERS = [
    "can", "mara", "riot", "clsk", "hut", "bitf",
    "wulf", "cifr", "btbt", "corz", "iren",
]

# ---------- contract types ----------


@dataclass(frozen=True)
class MinerLeadInput:
    can_residual_z: float  # CAN move orthogonal to same-day BTC, z-scored


@dataclass(frozen=True)
class TargetPosition:
    weight: float  # -1.0 = short BTC next day, +1.0 = long, 0.0 = flat


class CanMinerLeadStrategy:
    """Fade next-day BTC in the direction CAN's residual points (rejected)."""

    def __init__(self, entry_z: float = 1.0) -> None:
        self._entry_z = entry_z

    def decide(self, strategy_input: MinerLeadInput) -> TargetPosition:
        z = strategy_input.can_residual_z
        if z >= self._entry_z:
            return TargetPosition(-1.0)
        if z <= -self._entry_z:
            return TargetPosition(1.0)
        return TargetPosition(0.0)


_CONTRACT_CHECK: TradingStrategy[MinerLeadInput, TargetPosition] = (
    CanMinerLeadStrategy()
)

# ---------- data (impure, private API) ----------


def api_key() -> str:
    path = os.path.expanduser("~/.secrets/signal-noise-env")
    for line in open(path):
        if line.startswith("SIGNAL_NOISE_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("SIGNAL_NOISE_API_KEY not found")


def series(name: str, limit: int = 60000) -> dict[dt.date, float]:
    req = urllib.request.Request(
        f"{BASE}/signals/{name}/data?limit={limit}",
        headers={"X-API-Key": api_key()},
    )
    rows = json.loads(urllib.request.urlopen(req, timeout=60).read())
    out: dict[dt.date, float] = {}
    for r in rows:
        if r.get("value") is not None:
            # last value per date wins (rows are time-ascending)
            out[dt.date.fromisoformat(r["timestamp"][:10])] = float(r["value"])
    return out


# ---------- statistics ----------


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


# ---------- verdict harness ----------


def main() -> None:
    btc = series("btc_ohlcv")
    ds = sorted(btc)
    ret: dict[dt.date, float] = {}
    for a, b in zip(ds, ds[1:]):
        if (b - a).days == 1:
            ret[a] = math.log(btc[b] / btc[a])

    print(f"{'sym':6} {'n':>5} {'rho_next':>9} {'partial':>8} {'half1':>7} {'half2':>7}")
    partials: dict[str, float] = {}
    can_halves = (0.0, 0.0)
    for sym in MINERS:
        f = series(sym, limit=5000)
        fd = sorted(f)
        trip = []
        for a, b in zip(fd, fd[1:]):
            if (b - a).days == 1 and b in ret and a in btc and b in btc:
                trip.append((f[b] - f[a], math.log(btc[b] / btc[a]), ret[b]))
        if len(trip) < 120:
            print(f"{sym:6} n={len(trip)} insufficient")
            continue
        x = [t[0] for t in trip]
        c = [t[1] for t in trip]
        y = [t[2] for t in trip]
        rho = spearman(x, y)
        rx, rc, ry = ranks(x), ranks(c), ranks(y)
        rxy, rxc, rcy = pearson(rx, ry), pearson(rx, rc), pearson(rc, ry)
        part = (rxy - rxc * rcy) / math.sqrt((1 - rxc**2) * (1 - rcy**2))
        h = len(trip) // 2
        r1, r2 = spearman(x[:h], y[:h]), spearman(x[h:], y[h:])
        partials[sym] = part
        if sym == "can":
            can_halves = (r1, r2)
        print(f"{sym:6} {len(trip):5d} {rho:+9.4f} {part:+8.4f} {r1:+7.4f} {r2:+7.4f}")

    # corrected rule: support requires sibling agreement ON PARTIALS
    siblings = [s for s in MINERS if s != "can" and s in partials]
    agree = sum(1 for s in siblings if partials[s] < -0.05)
    can_ok = (
        partials.get("can", 0) < -0.05
        and can_halves[0] < 0
        and can_halves[1] < 0
    )
    supported = can_ok and agree >= len(siblings) * 0.6
    print(
        f"\nsibling partials < -0.05: {agree}/{len(siblings)}"
        f"  can_ok={can_ok}"
        f"\nVERDICT: {'SUPPORTED' if supported else 'REJECTED'}"
        " (rule: CAN partial < -0.05 in both halves AND >=60% of"
        " sibling partials < -0.05)"
    )


if __name__ == "__main__":
    main()
