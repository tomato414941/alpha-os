from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.defi_yield.current_yield_screen import (
    YieldCandidate,
    fetch_yield_pools,
    screen_stable_yields,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class YieldQualityRow:
    chain: str
    project: str
    symbol: str
    pool: str
    tvl_usd: float
    apy: float
    apy_base: float
    apy_reward: float
    apy_mean_30d: float
    reward_share: float
    apy_deviation_30d: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def build_yield_quality_rows(candidates: tuple[YieldCandidate, ...]) -> tuple[YieldQualityRow, ...]:
    rows = tuple(_build_row(candidate) for candidate in candidates)
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_yield_quality_csv(rows: tuple[YieldQualityRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
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
                "reward_share",
                "apy_deviation_30d",
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
                    row.chain,
                    row.project,
                    row.symbol,
                    row.pool,
                    f"{row.tvl_usd:.2f}",
                    f"{row.apy:.6f}",
                    f"{row.apy_base:.6f}",
                    f"{row.apy_reward:.6f}",
                    f"{row.apy_mean_30d:.6f}",
                    f"{row.reward_share:.8f}",
                    f"{row.apy_deviation_30d:.6f}",
                    f"{row.score:.6f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_yield_quality_md(rows: tuple[YieldQualityRow, ...], *, output_path: Path, top: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current DeFi Yield Quality\n\n")
        handle.write(
            "This separates base stablecoin yield from reward-heavy APY. "
            "It is a carry-source screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| chain | project | symbol | status | tvl USD | apy | base | reward | reward share | dev 30d | score | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.chain} | {row.project} | {row.symbol} | {row.status} | "
                f"{row.tvl_usd:.0f} | {row.apy:.4f} | {row.apy_base:.4f} | "
                f"{row.apy_reward:.4f} | {row.reward_share:.4f} | "
                f"{row.apy_deviation_30d:.4f} | {row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _build_row(candidate: YieldCandidate) -> YieldQualityRow:
    apy_base = candidate.apy_base or 0.0
    apy_reward = candidate.apy_reward or 0.0
    apy_mean_30d = candidate.apy_mean_30d or candidate.apy
    reward_share = apy_reward / candidate.apy if candidate.apy > 0.0 else 0.0
    apy_deviation = abs(candidate.apy - apy_mean_30d)
    score = _score(
        tvl_usd=candidate.tvl_usd,
        apy=candidate.apy,
        apy_base=apy_base,
        reward_share=reward_share,
        apy_deviation=apy_deviation,
    )
    status, side, reason = _status_side_reason(
        tvl_usd=candidate.tvl_usd,
        apy=candidate.apy,
        apy_base=apy_base,
        reward_share=reward_share,
        apy_deviation=apy_deviation,
    )
    return YieldQualityRow(
        chain=candidate.chain,
        project=candidate.project,
        symbol=candidate.symbol,
        pool=candidate.pool,
        tvl_usd=candidate.tvl_usd,
        apy=candidate.apy,
        apy_base=apy_base,
        apy_reward=apy_reward,
        apy_mean_30d=apy_mean_30d,
        reward_share=reward_share,
        apy_deviation_30d=apy_deviation,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=(
            f"check {candidate.chain}/{candidate.project} custody, withdrawal path, "
            "APY source, capacity, and exit liquidity"
        ),
    )


def _score(
    *,
    tvl_usd: float,
    apy: float,
    apy_base: float,
    reward_share: float,
    apy_deviation: float,
) -> float:
    tvl_score = min(tvl_usd / 50_000_000.0, 10.0)
    base_bonus = apy_base * 1.2
    total_yield = apy * 0.4
    reward_penalty = reward_share * 4.0
    instability_penalty = apy_deviation * 0.6
    return tvl_score + base_bonus + total_yield - reward_penalty - instability_penalty


def _status_side_reason(
    *,
    tvl_usd: float,
    apy: float,
    apy_base: float,
    reward_share: float,
    apy_deviation: float,
) -> tuple[str, str, str]:
    if tvl_usd >= 50_000_000.0 and apy_base >= 8.0 and reward_share <= 0.15 and apy_deviation <= 3.0:
        return "paper_base_yield_watch", "allocate_stablecoin_capital", "base APY is material and not reward-heavy"
    if tvl_usd >= 20_000_000.0 and apy >= 9.0 and reward_share > 0.15:
        return "paper_incentive_yield_watch", "watch_incentive_decay", "APY is material but reward-heavy"
    if apy_deviation > 5.0:
        return "yield_decay_watch", "none", "current APY differs materially from 30d mean"
    return "yield_context_watch", "none", "yield is context but not yet actionable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_yield_quality_screen.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_yield_quality_screen.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_yield_quality_rows(screen_stable_yields(fetch_yield_pools()))
    write_yield_quality_csv(rows, output_path=args.output_path)
    write_yield_quality_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.chain, row.project, row.symbol, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
