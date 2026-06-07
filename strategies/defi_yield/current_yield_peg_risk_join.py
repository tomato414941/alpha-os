from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class YieldPegRiskRow:
    chain: str
    project: str
    symbol: str
    yield_status: str
    yield_side: str
    tvl_usd: float
    apy: float
    apy_base: float
    reward_share: float
    yield_score: float
    peg_symbol: str
    peg_status: str
    peg_side: str
    peg_price: float
    peg_deviation: float
    peg_score: float
    match_kind: str
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def build_yield_peg_risk_rows(
    yield_rows: tuple[dict[str, str], ...],
    peg_rows: tuple[dict[str, str], ...],
) -> tuple[YieldPegRiskRow, ...]:
    peg_index = _peg_index(peg_rows)
    rows = tuple(_build_row(row, peg_index) for row in yield_rows if _is_material_yield(row))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_yield_peg_risk_csv(rows: tuple[YieldPegRiskRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "chain",
                "project",
                "symbol",
                "yield_status",
                "yield_side",
                "tvl_usd",
                "apy",
                "apy_base",
                "reward_share",
                "yield_score",
                "peg_symbol",
                "peg_status",
                "peg_side",
                "peg_price",
                "peg_deviation",
                "peg_score",
                "match_kind",
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
                    row.yield_status,
                    row.yield_side,
                    f"{row.tvl_usd:.2f}",
                    f"{row.apy:.6f}",
                    f"{row.apy_base:.6f}",
                    f"{row.reward_share:.8f}",
                    f"{row.yield_score:.6f}",
                    row.peg_symbol,
                    row.peg_status,
                    row.peg_side,
                    f"{row.peg_price:.8f}",
                    f"{row.peg_deviation:.8f}",
                    f"{row.peg_score:.8f}",
                    row.match_kind,
                    f"{row.score:.6f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_yield_peg_risk_md(
    rows: tuple[YieldPegRiskRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Yield Peg Risk Join\n\n")
        handle.write(
            "This joins stable-yield candidates with stablecoin peg stress. "
            "It is a cross-lane risk screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| chain | project | symbol | status | apy | base | tvl USD | peg symbol | peg status | price | peg deviation | score | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.chain} | {row.project} | {row.symbol} | {row.status} | "
                f"{row.apy:.4f} | {row.apy_base:.4f} | {row.tvl_usd:.0f} | "
                f"{row.peg_symbol or '-'} | {row.peg_status or '-'} | "
                f"{row.peg_price:.6f} | {row.peg_deviation:.6f} | {row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _build_row(row: dict[str, str], peg_index: dict[str, dict[str, str]]) -> YieldPegRiskRow:
    symbol = row.get("symbol", "")
    peg, match_kind = _match_peg_row(symbol, peg_index)
    yield_score = _float(row.get("score"))
    peg_deviation = _float(peg.get("peg_deviation") if peg else "")
    peg_score = _float(peg.get("score") if peg else "")
    status, side, reason, score = _status_side_reason_score(
        yield_status=row.get("status", ""),
        yield_score=yield_score,
        peg_status=peg.get("status", "") if peg else "",
        peg_deviation=peg_deviation,
        peg_score=peg_score,
        match_kind=match_kind,
    )
    next_step = _next_step(row, peg, status)
    return YieldPegRiskRow(
        chain=row.get("chain", ""),
        project=row.get("project", ""),
        symbol=symbol,
        yield_status=row.get("status", ""),
        yield_side=row.get("side", ""),
        tvl_usd=_float(row.get("tvl_usd")),
        apy=_float(row.get("apy")),
        apy_base=_float(row.get("apy_base")),
        reward_share=_float(row.get("reward_share")),
        yield_score=yield_score,
        peg_symbol=peg.get("symbol", "") if peg else "",
        peg_status=peg.get("status", "") if peg else "",
        peg_side=peg.get("side", "") if peg else "",
        peg_price=_float(peg.get("price") if peg else ""),
        peg_deviation=peg_deviation,
        peg_score=peg_score,
        match_kind=match_kind,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=next_step,
    )


def _peg_index(rows: tuple[dict[str, str], ...]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = _normalize_symbol(row.get("symbol", ""))
        if symbol:
            output[symbol] = row
    return output


def _match_peg_row(symbol: str, peg_index: dict[str, dict[str, str]]) -> tuple[dict[str, str] | None, str]:
    normalized = _normalize_symbol(symbol)
    if normalized in peg_index:
        return peg_index[normalized], "exact_symbol"
    if normalized.startswith("S") and len(normalized) > 4:
        underlying = normalized[1:]
        if underlying in peg_index:
            return peg_index[underlying], "leading_s_wrapper"
    return None, "none"


def _normalize_symbol(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def _is_material_yield(row: dict[str, str]) -> bool:
    return row.get("status") in {"paper_base_yield_watch", "paper_incentive_yield_watch", "yield_context_watch"} and (
        _float(row.get("apy_base")) >= 3.0 or _float(row.get("apy")) >= 8.0
    )


def _status_side_reason_score(
    *,
    yield_status: str,
    yield_score: float,
    peg_status: str,
    peg_deviation: float,
    peg_score: float,
    match_kind: str,
) -> tuple[str, str, str, float]:
    peg_risk_bonus = min(abs(peg_deviation) * 500.0, 30.0)
    supply_bonus = min(peg_score / 100.0, 8.0)
    if peg_status == "paper_depeg_repeg_watch":
        return (
            "paper_yield_depeg_conflict_watch",
            "avoid_or_repeg_research",
            "yield may be compensation for below-peg, redemption, or issuer risk",
            yield_score + peg_risk_bonus + supply_bonus,
        )
    if peg_status == "paper_premium_mean_reversion_watch":
        return (
            "paper_yield_premium_conflict_watch",
            "avoid_or_premium_reversion_research",
            "yield asset trades above peg, so carry can be offset by premium reversion",
            yield_score + peg_risk_bonus + supply_bonus,
        )
    if peg_status == "peg_supply_stress_watch":
        return (
            "yield_supply_stress_watch",
            "watch_supply_or_redemption_risk",
            "yield asset has material supply stress even though price is near peg",
            yield_score + supply_bonus,
        )
    if peg_status:
        return (
            "paper_yield_without_peg_stress_watch",
            "allocate_stablecoin_capital_after_ops_check",
            "yield candidate has no current material peg stress in the peg screen",
            yield_score,
        )
    if match_kind == "none" and yield_status == "paper_base_yield_watch":
        return (
            "yield_peg_unresolved_watch",
            "research_redemption_and_peg_source",
            "yield candidate has no matching peg-stress row, so peg source is unresolved",
            yield_score,
        )
    return (
        "yield_context_watch",
        "none",
        "yield is context but peg linkage is not yet actionable",
        yield_score,
    )


def _next_step(row: dict[str, str], peg: dict[str, str] | None, status: str) -> str:
    chain = row.get("chain", "")
    project = row.get("project", "")
    symbol = row.get("symbol", "")
    if status in {"paper_yield_depeg_conflict_watch", "paper_yield_premium_conflict_watch"} and peg:
        return (
            f"check {symbol} redemption route, tradable venues, exit liquidity, "
            f"and whether {peg.get('symbol', '')} peg risk explains the APY"
        )
    if status == "yield_supply_stress_watch" and peg:
        return f"check {peg.get('symbol', '')} supply change, issuer updates, redemption route, and pool exit liquidity"
    return f"check {chain}/{project} custody, APY source, withdrawal path, capacity, and exit liquidity"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yield-path", type=Path, default=ROOT / "current_yield_quality_screen.csv")
    parser.add_argument(
        "--peg-path",
        type=Path,
        default=STRATEGIES_ROOT / "stablecoin_liquidity" / "current_peg_stress_screen.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_yield_peg_risk_join.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_yield_peg_risk_join.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_yield_peg_risk_rows(_read_rows(args.yield_path), _read_rows(args.peg_path))
    write_yield_peg_risk_csv(rows, output_path=args.output_path)
    write_yield_peg_risk_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.chain, row.project, row.symbol, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
