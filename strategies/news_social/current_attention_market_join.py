from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class AttentionMarketJoinRow:
    symbol: str
    name: str
    attention_rank: int
    attention_24h_change: float
    annualized_funding: float
    mark_oracle_diff: float
    impact_spread: float
    carry_reversion_action: str
    carry_reversion_observations: int
    carry_reversion_score: float
    action: str
    score: float
    reason: str


def build_attention_market_join_rows(
    *,
    attention_path: Path = ROOT / "current_attention_snapshot.csv",
    perp_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    carry_reversion_path: Path = (
        STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv"
    ),
) -> tuple[AttentionMarketJoinRow, ...]:
    attention_rows = _attention_rows(attention_path)
    perp_by_symbol = _rows_by_symbol(perp_snapshot_path)
    carry_by_symbol = _rows_by_symbol(carry_reversion_path)
    rows = tuple(
        _build_row(
            attention=row,
            perp=perp_by_symbol.get(row["symbol"].upper()),
            carry=carry_by_symbol.get(row["symbol"].upper()),
        )
        for row in attention_rows
        if row["symbol"].upper() in perp_by_symbol
    )
    candidates = tuple(row for row in rows if row.action != "ignore")
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_attention_market_join_csv(
    rows: tuple[AttentionMarketJoinRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "name",
                "attention_rank",
                "attention_24h_change",
                "annualized_funding",
                "mark_oracle_diff",
                "impact_spread",
                "carry_reversion_action",
                "carry_reversion_observations",
                "carry_reversion_score",
                "action",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.name,
                    row.attention_rank,
                    f"{row.attention_24h_change:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.impact_spread:.12f}",
                    row.carry_reversion_action,
                    row.carry_reversion_observations,
                    f"{row.carry_reversion_score:.8f}",
                    row.action,
                    f"{row.score:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_attention_market_join_md(
    rows: tuple[AttentionMarketJoinRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Attention Market Join\n\n")
        handle.write(
            "This joins CoinGecko trending attention to current Hyperliquid perp "
            "market state. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | name | rank | 24h change | funding | mark/oracle | carry action | obs | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.name} | "
                f"{row.attention_rank} | "
                f"{row.attention_24h_change:.4f} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.mark_oracle_diff:.6f} | "
                f"{row.carry_reversion_action} | "
                f"{row.carry_reversion_observations} | "
                f"{row.score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows here combine attention and perp-market state. A row is useful only "
            "as a research candidate; it still needs future-return labels and execution checks.\n"
        )
    return output_path


def _attention_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            row for row in csv.DictReader(handle)
            if row["source"] == "coingecko_trending"
        )


def _rows_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    return {row.get("asset", row.get("symbol", "")).upper(): row for row in rows}


def _build_row(
    *,
    attention: dict[str, str],
    perp: dict[str, str] | None,
    carry: dict[str, str] | None,
) -> AttentionMarketJoinRow:
    symbol = attention["symbol"].upper()
    attention_rank = int(attention["rank"])
    attention_24h_change = _float(attention["value"])
    annualized_funding = _float((perp or {}).get("annualized_funding"))
    mark_oracle_diff = _float((perp or {}).get("mark_oracle_diff"))
    impact_spread = _float((perp or {}).get("impact_spread"))
    carry_action = (carry or {}).get("action", "")
    carry_observations = int((carry or {}).get("observations") or "0")
    carry_score = _float((carry or {}).get("mean_score"))
    action = _action(
        attention_rank=attention_rank,
        attention_24h_change=attention_24h_change,
        carry_observations=carry_observations,
        carry_score=carry_score,
        annualized_funding=annualized_funding,
    )
    return AttentionMarketJoinRow(
        symbol=symbol,
        name=attention["name"],
        attention_rank=attention_rank,
        attention_24h_change=attention_24h_change,
        annualized_funding=annualized_funding,
        mark_oracle_diff=mark_oracle_diff,
        impact_spread=impact_spread,
        carry_reversion_action=carry_action,
        carry_reversion_observations=carry_observations,
        carry_reversion_score=carry_score,
        action=action,
        score=_score(
            action=action,
            attention_rank=attention_rank,
            attention_24h_change=attention_24h_change,
            carry_observations=carry_observations,
            carry_score=carry_score,
            annualized_funding=annualized_funding,
            impact_spread=impact_spread,
        ),
        reason=_reason(action),
    )


def _action(
    *,
    attention_rank: int,
    attention_24h_change: float,
    carry_observations: int,
    carry_score: float,
    annualized_funding: float,
) -> str:
    if attention_rank <= 15 and carry_observations >= 6 and carry_score >= 5.0:
        return "attention_carry_reversion_watch"
    if attention_rank <= 15 and abs(attention_24h_change) >= 5.0 and abs(annualized_funding) >= 0.2:
        return "attention_funding_watch"
    return "ignore"


def _score(
    *,
    action: str,
    attention_rank: int,
    attention_24h_change: float,
    carry_observations: int,
    carry_score: float,
    annualized_funding: float,
    impact_spread: float,
) -> float:
    if action == "ignore":
        return float("-inf")
    attention_score = max(16 - attention_rank, 0)
    move_score = min(abs(attention_24h_change), 25.0) / 5.0
    carry_persistence = carry_observations + carry_score
    funding_score = abs(annualized_funding) * 2.0
    friction_penalty = impact_spread * 100.0
    return attention_score + move_score + carry_persistence + funding_score - friction_penalty


def _reason(action: str) -> str:
    if action == "attention_carry_reversion_watch":
        return "trending asset overlaps with persistent carry/reversion perp state"
    if action == "attention_funding_watch":
        return "trending asset has material price move and large funding state"
    return "attention is not joined to a strong current perp state"


def _float(value: str | None) -> float:
    return float(value or "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-path",
        type=Path,
        default=ROOT / "current_attention_snapshot.csv",
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
        default=ROOT / "current_attention_market_join.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_attention_market_join.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_attention_market_join_rows(
        attention_path=args.attention_path,
        perp_snapshot_path=args.perp_snapshot_path,
        carry_reversion_path=args.carry_reversion_path,
    )
    write_attention_market_join_csv(rows, output_path=args.csv_output_path)
    write_attention_market_join_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.action, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
