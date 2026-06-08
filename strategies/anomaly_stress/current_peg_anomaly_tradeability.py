from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PegAnomalyTradeabilityRow:
    symbol: str
    status: str
    side: str
    score: float
    peg_deviation: float
    price: float
    current_supply_usd: float
    dex_pool_match_count: int
    best_pool: str
    best_pool_reserve_usd: float
    best_pool_volume_h24_usd: float
    yield_conflict_count: int
    source_status: str
    reason: str
    next_step: str


def build_peg_anomaly_tradeability_rows(root: Path = ROOT) -> tuple[PegAnomalyTradeabilityRow, ...]:
    anomalies = tuple(
        row
        for row in _read_rows(root / "anomaly_stress" / "current_cross_market_stress_anomaly.csv")
        if row.get("source_lane") == "stablecoin_liquidity"
        and row.get("status") == "cross_market_peg_stress_anomaly"
    )
    peg_by_symbol = {
        row.get("symbol", "").lower(): row
        for row in _read_rows(root / "stablecoin_liquidity" / "current_peg_stress_screen.csv")
    }
    pool_rows = _read_rows(root / "dex_pool_flow" / "current_geckoterminal_pool_flow.csv")
    yield_rows = _read_rows(root / "defi_yield" / "current_yield_peg_risk_join.csv")

    output: list[PegAnomalyTradeabilityRow] = []
    for anomaly in anomalies[:12]:
        symbol = anomaly.get("subject", "")
        peg = peg_by_symbol.get(symbol.lower(), {})
        matched_pools = _matched_pool_rows(symbol=symbol, pool_rows=pool_rows)
        best_pool = max(matched_pools, key=lambda row: _float(row.get("reserve_usd")), default={})
        yield_conflicts = tuple(
            row
            for row in yield_rows
            if row.get("peg_symbol", "").lower() == symbol.lower()
            and row.get("status")
            in {
                "paper_yield_premium_conflict_watch",
                "paper_yield_depeg_conflict_watch",
                "yield_supply_stress_watch",
            }
        )
        price = _float(peg.get("price"))
        peg_deviation = _float(peg.get("peg_deviation"))
        current_supply = _float(peg.get("current_supply_usd"))
        pool_reserve = _float(best_pool.get("reserve_usd"))
        pool_volume_h24 = _float(best_pool.get("volume_h24_usd"))
        status, side, reason = _status_side_reason(
            peg_deviation=peg_deviation,
            pool_reserve_usd=pool_reserve,
            pool_volume_h24_usd=pool_volume_h24,
            yield_conflict_count=len(yield_conflicts),
            current_supply_usd=current_supply,
        )
        output.append(
            PegAnomalyTradeabilityRow(
                symbol=symbol,
                status=status,
                side=side,
                score=_score(
                    status=status,
                    peg_deviation=peg_deviation,
                    pool_reserve_usd=pool_reserve,
                    pool_volume_h24_usd=pool_volume_h24,
                    yield_conflict_count=len(yield_conflicts),
                ),
                peg_deviation=peg_deviation,
                price=price,
                current_supply_usd=current_supply,
                dex_pool_match_count=len(matched_pools),
                best_pool=_pool_label(best_pool),
                best_pool_reserve_usd=pool_reserve,
                best_pool_volume_h24_usd=pool_volume_h24,
                yield_conflict_count=len(yield_conflicts),
                source_status=anomaly.get("side", ""),
                reason=reason,
                next_step=_next_step(symbol=symbol, status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_peg_anomaly_tradeability_csv(
    rows: tuple[PegAnomalyTradeabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "side",
                "score",
                "peg_deviation",
                "price",
                "current_supply_usd",
                "dex_pool_match_count",
                "best_pool",
                "best_pool_reserve_usd",
                "best_pool_volume_h24_usd",
                "yield_conflict_count",
                "source_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.peg_deviation:.8f}",
                    f"{row.price:.8f}",
                    f"{row.current_supply_usd:.2f}",
                    row.dex_pool_match_count,
                    row.best_pool,
                    f"{row.best_pool_reserve_usd:.8f}",
                    f"{row.best_pool_volume_h24_usd:.8f}",
                    row.yield_conflict_count,
                    row.source_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_peg_anomaly_tradeability_md(
    rows: tuple[PegAnomalyTradeabilityRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Peg Anomaly Tradeability\n\n")
        handle.write(
            "This separates a peg anomaly from a currently routeable trade candidate. "
            "A missing pool match is not proof that a route does not exist; it means this "
            "current public snapshot did not validate execution yet.\n\n"
        )
        handle.write(
            "| symbol | status | side | score | peg deviation | pool matches | best pool | "
            "pool reserve USD | pool vol 24h | yield conflicts | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.side} | {row.score:.4f} | "
                f"{row.peg_deviation:.6f} | {row.dex_pool_match_count} | {_escape(row.best_pool)} | "
                f"{row.best_pool_reserve_usd:.0f} | {row.best_pool_volume_h24_usd:.0f} | "
                f"{row.yield_conflict_count} | {_escape(row.reason)} |\n"
            )
    return output_path


def _matched_pool_rows(*, symbol: str, pool_rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    if not symbol or symbol.lower() in {"usd", "usdt", "usdc"}:
        return ()
    token = re.escape(symbol.lower().replace("$", ""))
    pattern = re.compile(rf"(?<![a-z0-9]){token}(?![a-z0-9])")
    return tuple(row for row in pool_rows if pattern.search(row.get("name", "").lower().replace("$", "")))


def _status_side_reason(
    *,
    peg_deviation: float,
    pool_reserve_usd: float,
    pool_volume_h24_usd: float,
    yield_conflict_count: int,
    current_supply_usd: float,
) -> tuple[str, str, str]:
    if pool_reserve_usd >= 500_000.0 and pool_volume_h24_usd >= 100_000.0 and abs(peg_deviation) >= 0.005:
        return (
            "peg_anomaly_tradeability_candidate",
            "paper_route_check",
            "peg anomaly has a current pool match with enough public depth for a paper route check",
        )
    if yield_conflict_count > 0 and abs(peg_deviation) >= 0.005:
        return (
            "peg_anomaly_mechanics_watch",
            "paper_mechanics_check",
            "peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade",
        )
    if current_supply_usd >= 500_000_000.0 and abs(peg_deviation) >= 0.005:
        return (
            "peg_anomaly_mechanics_watch",
            "paper_mechanics_check",
            "large-supply peg anomaly needs issuer, redemption, and venue mechanics before execution",
        )
    if abs(peg_deviation) >= 0.005:
        return (
            "peg_anomaly_stale_or_unrouted",
            "no_trade_until_route",
            "material peg anomaly lacks a current route/depth confirmation in the joined public snapshots",
        )
    return "peg_anomaly_deprioritize", "none", "peg deviation is not material after route screening"


def _score(
    *,
    status: str,
    peg_deviation: float,
    pool_reserve_usd: float,
    pool_volume_h24_usd: float,
    yield_conflict_count: int,
) -> float:
    if status == "peg_anomaly_tradeability_candidate":
        return min(
            95.0,
            65.0
            + abs(peg_deviation) * 100.0
            + min(pool_reserve_usd / 250_000.0, 12.0)
            + min(pool_volume_h24_usd / 100_000.0, 8.0),
        )
    if status == "peg_anomaly_mechanics_watch":
        return min(82.0, 55.0 + abs(peg_deviation) * 50.0 + min(yield_conflict_count * 4.0, 12.0))
    if status == "peg_anomaly_stale_or_unrouted":
        return min(45.0, 30.0 + abs(peg_deviation) * 20.0)
    return 20.0


def _pool_label(row: dict[str, str]) -> str:
    if not row:
        return ""
    return f"{row.get('network', '')}/{row.get('dex', '')} {row.get('name', '')}"


def _next_step(*, symbol: str, status: str) -> str:
    if status == "peg_anomaly_tradeability_candidate":
        return f"paper-check {symbol} pool route depth, slippage, redemption path, venue access, and repeat peg quote"
    if status == "peg_anomaly_mechanics_watch":
        return f"check {symbol} issuer/redemption mechanics, yield linkage, venue access, and quote freshness"
    if status == "peg_anomaly_stale_or_unrouted":
        return f"do not promote {symbol}; first find a live route, quote freshness, redemption path, and executable depth"
    return f"deprioritize {symbol} until a material peg deviation and route evidence appear"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_peg_anomaly_tradeability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=LANE_ROOT / "current_peg_anomaly_tradeability.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_peg_anomaly_tradeability_rows()
    write_peg_anomaly_tradeability_csv(rows, output_path=args.output_path)
    write_peg_anomaly_tradeability_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.symbol, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
