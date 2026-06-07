from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ExchangeCatalystMarketJoinRow:
    timestamp: str
    published_at: str
    symbol: str
    catalyst_kind: str
    direction_hint: int
    catalyst_score: float
    annualized_funding: float
    mark_oracle_diff: float
    impact_spread: float
    carry_reversion_action: str
    carry_reversion_observations: int
    carry_reversion_score: float
    action: str
    score: float
    title: str
    reason: str


def build_exchange_catalyst_market_join_rows(
    *,
    catalyst_path: Path = ROOT / "current_exchange_catalyst_snapshot.csv",
    perp_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    carry_reversion_path: Path = (
        STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv"
    ),
) -> tuple[ExchangeCatalystMarketJoinRow, ...]:
    catalysts = _read_rows(catalyst_path)
    perp_by_symbol = _rows_by_symbol(perp_snapshot_path)
    carry_by_symbol = _rows_by_symbol(carry_reversion_path)
    rows = tuple(
        _build_row(
            catalyst=row,
            perp=perp_by_symbol.get(row["symbol"].upper()),
            carry=carry_by_symbol.get(row["symbol"].upper()),
        )
        for row in catalysts
        if row["symbol"].upper() in perp_by_symbol
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_exchange_catalyst_market_join_csv(
    rows: tuple[ExchangeCatalystMarketJoinRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "published_at",
                "symbol",
                "catalyst_kind",
                "direction_hint",
                "catalyst_score",
                "annualized_funding",
                "mark_oracle_diff",
                "impact_spread",
                "carry_reversion_action",
                "carry_reversion_observations",
                "carry_reversion_score",
                "action",
                "score",
                "title",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.published_at,
                    row.symbol,
                    row.catalyst_kind,
                    row.direction_hint,
                    f"{row.catalyst_score:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.impact_spread:.12f}",
                    row.carry_reversion_action,
                    row.carry_reversion_observations,
                    f"{row.carry_reversion_score:.8f}",
                    row.action,
                    f"{row.score:.8f}",
                    row.title,
                    row.reason,
                )
            )
    return output_path


def write_exchange_catalyst_market_join_md(
    rows: tuple[ExchangeCatalystMarketJoinRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Exchange Catalyst Market Join\n\n")
        handle.write(
            "This joins exchange-announcement catalysts to current Hyperliquid "
            "perp state. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| published | symbol | kind | dir | funding | carry action | obs | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.published_at} | "
                f"{row.symbol} | "
                f"{row.catalyst_kind} | "
                f"{row.direction_hint} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.carry_reversion_action} | "
                f"{row.carry_reversion_observations} | "
                f"{row.score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows here are externally triggered candidates. They still need "
            "forward labels, spread/fee checks, and repeated event evidence.\n"
        )
    return output_path


def _build_row(
    *,
    catalyst: dict[str, str],
    perp: dict[str, str] | None,
    carry: dict[str, str] | None,
) -> ExchangeCatalystMarketJoinRow:
    direction_hint = int(catalyst.get("direction_hint") or "0")
    catalyst_score = float(catalyst.get("score") or "0")
    annualized_funding = _float((perp or {}).get("annualized_funding"))
    mark_oracle_diff = _float((perp or {}).get("mark_oracle_diff"))
    impact_spread = _float((perp or {}).get("impact_spread"))
    carry_action = (carry or {}).get("action", "")
    carry_observations = int((carry or {}).get("observations") or "0")
    carry_score = _float((carry or {}).get("mean_score"))
    action = _action(
        direction_hint=direction_hint,
        catalyst_kind=catalyst.get("catalyst_kind", ""),
        annualized_funding=annualized_funding,
        carry_action=carry_action,
        carry_observations=carry_observations,
        carry_score=carry_score,
    )
    return ExchangeCatalystMarketJoinRow(
        timestamp=catalyst["timestamp"],
        published_at=catalyst["published_at"],
        symbol=catalyst["symbol"].upper(),
        catalyst_kind=catalyst["catalyst_kind"],
        direction_hint=direction_hint,
        catalyst_score=catalyst_score,
        annualized_funding=annualized_funding,
        mark_oracle_diff=mark_oracle_diff,
        impact_spread=impact_spread,
        carry_reversion_action=carry_action,
        carry_reversion_observations=carry_observations,
        carry_reversion_score=carry_score,
        action=action,
        score=_score(
            catalyst_score=catalyst_score,
            annualized_funding=annualized_funding,
            impact_spread=impact_spread,
            carry_observations=carry_observations,
            carry_score=carry_score,
        ),
        title=catalyst["title"],
        reason=_reason(action),
    )


def _action(
    *,
    direction_hint: int,
    catalyst_kind: str,
    annualized_funding: float,
    carry_action: str,
    carry_observations: int,
    carry_score: float,
) -> str:
    if direction_hint == 0:
        return "exchange_catalyst_context"
    if carry_observations >= 6 and carry_score >= 5.0:
        return "exchange_catalyst_carry_overlap"
    if abs(annualized_funding) >= 0.2:
        return "exchange_catalyst_funding_overlap"
    if catalyst_kind in {"perp_listing_watch", "exchange_removal_watch"}:
        return "exchange_catalyst_watch"
    if carry_action:
        return "exchange_catalyst_context"
    return "exchange_catalyst_low_context"


def _score(
    *,
    catalyst_score: float,
    annualized_funding: float,
    impact_spread: float,
    carry_observations: int,
    carry_score: float,
) -> float:
    carry_bonus = carry_score + carry_observations if carry_observations >= 3 else 0.0
    funding_bonus = abs(annualized_funding) * 2.0
    friction_penalty = impact_spread * 100.0
    return catalyst_score + carry_bonus + funding_bonus - friction_penalty


def _reason(action: str) -> str:
    if action == "exchange_catalyst_carry_overlap":
        return "exchange catalyst overlaps persistent current perp carry/reversion state"
    if action == "exchange_catalyst_funding_overlap":
        return "exchange catalyst overlaps material current funding state"
    if action == "exchange_catalyst_watch":
        return "exchange catalyst is directly tradable on Hyperliquid"
    if action == "exchange_catalyst_context":
        return "exchange catalyst is tradable but has weak current perp context"
    return "exchange catalyst has low current perp context"


def _rows_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("asset", row.get("symbol", "")).upper(): row for row in _read_rows(path)}


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value or "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalyst-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_snapshot.csv",
    )
    parser.add_argument(
        "--perp-snapshot-path",
        type=Path,
        default=STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--carry-reversion-path",
        type=Path,
        default=STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_market_join.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_exchange_catalyst_market_join.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_exchange_catalyst_market_join_rows(
        catalyst_path=args.catalyst_path,
        perp_snapshot_path=args.perp_snapshot_path,
        carry_reversion_path=args.carry_reversion_path,
    )
    write_exchange_catalyst_market_join_csv(rows, output_path=args.csv_output_path)
    write_exchange_catalyst_market_join_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.action, f"score={row.score:.4f}", row.catalyst_kind)


if __name__ == "__main__":
    main()
