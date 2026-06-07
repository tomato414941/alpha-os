from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"


@dataclass(frozen=True)
class YieldCandidate:
    chain: str
    project: str
    symbol: str
    pool: str
    tvl_usd: float
    apy: float
    apy_base: float | None
    apy_reward: float | None
    apy_mean_30d: float | None
    stablecoin: bool
    il_risk: str
    exposure: str
    outlier: bool
    score: float


def fetch_yield_pools(url: str = DEFILLAMA_POOLS_URL) -> tuple[dict[str, object], ...]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return tuple(response.json()["data"])


def screen_stable_yields(
    pools: tuple[dict[str, object], ...],
    *,
    min_tvl_usd: float = 10_000_000.0,
    max_apy: float = 30.0,
) -> tuple[YieldCandidate, ...]:
    candidates = []
    for pool in pools:
        tvl = float(pool.get("tvlUsd") or 0.0)
        apy = float(pool.get("apy") or 0.0)
        stablecoin = bool(pool.get("stablecoin"))
        il_risk = str(pool.get("ilRisk") or "")
        exposure = str(pool.get("exposure") or "")
        outlier = bool(pool.get("outlier"))
        if not stablecoin or il_risk != "no" or exposure != "single" or outlier:
            continue
        if tvl < min_tvl_usd or apy <= 0.0 or apy > max_apy:
            continue
        apy_mean_30d = _optional_float(pool.get("apyMean30d"))
        stability_penalty = abs(apy - apy_mean_30d) if apy_mean_30d is not None else 0.0
        score = apy - (0.25 * stability_penalty)
        candidates.append(
            YieldCandidate(
                chain=str(pool.get("chain") or ""),
                project=str(pool.get("project") or ""),
                symbol=str(pool.get("symbol") or ""),
                pool=str(pool.get("pool") or ""),
                tvl_usd=tvl,
                apy=apy,
                apy_base=_optional_float(pool.get("apyBase")),
                apy_reward=_optional_float(pool.get("apyReward")),
                apy_mean_30d=apy_mean_30d,
                stablecoin=stablecoin,
                il_risk=il_risk,
                exposure=exposure,
                outlier=outlier,
                score=score,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.score, reverse=True))


def write_yield_candidates(
    candidates: tuple[YieldCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "chain",
                "project",
                "symbol",
                "pool",
                "tvl_usd",
                "apy",
                "apy_base",
                "apy_reward",
                "apy_mean_30d",
                "stablecoin",
                "il_risk",
                "exposure",
                "outlier",
                "score",
            )
        )
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.chain,
                    candidate.project,
                    candidate.symbol,
                    candidate.pool,
                    f"{candidate.tvl_usd:.2f}",
                    f"{candidate.apy:.6f}",
                    _format_optional(candidate.apy_base),
                    _format_optional(candidate.apy_reward),
                    _format_optional(candidate.apy_mean_30d),
                    candidate.stablecoin,
                    candidate.il_risk,
                    candidate.exposure,
                    candidate.outlier,
                    f"{candidate.score:.6f}",
                )
            )
    return output_path


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-tvl-usd", type=float, default=10_000_000.0)
    parser.add_argument("--max-apy", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_yield_screen.csv",
    )
    args = parser.parse_args()

    candidates = screen_stable_yields(
        fetch_yield_pools(),
        min_tvl_usd=args.min_tvl_usd,
        max_apy=args.max_apy,
    )
    write_yield_candidates(candidates, output_path=args.output_path)
    for candidate in candidates[: args.top]:
        print(
            candidate.chain,
            candidate.project,
            candidate.symbol,
            f"{candidate.tvl_usd:.0f}",
            f"{candidate.apy:.4f}",
            f"{candidate.score:.4f}",
        )


if __name__ == "__main__":
    main()
