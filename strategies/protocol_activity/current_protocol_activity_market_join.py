from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ProtocolActivityMarketJoinRow:
    timestamp: str
    symbol: str
    name: str
    protocol_action: str
    protocol_score: float
    commit_count_4_weeks: int
    telegram_users: int
    annualized_funding: float
    mark_oracle_diff: float
    impact_spread: float
    carry_reversion_action: str
    carry_reversion_observations: int
    carry_reversion_score: float
    action: str
    score: float
    reason: str


def build_protocol_activity_market_join_rows(
    *,
    protocol_path: Path = ROOT / "current_coingecko_protocol_activity.csv",
    perp_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    carry_reversion_path: Path = (
        STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_summary.csv"
    ),
) -> tuple[ProtocolActivityMarketJoinRow, ...]:
    protocol_rows = _read_rows(protocol_path)
    perp_by_symbol = _rows_by_symbol(perp_snapshot_path)
    carry_by_symbol = _rows_by_symbol(carry_reversion_path)
    rows = tuple(
        _build_row(
            protocol=row,
            perp=perp_by_symbol.get(row["symbol"].upper()),
            carry=carry_by_symbol.get(row["symbol"].upper()),
        )
        for row in protocol_rows
        if row["symbol"].upper() in perp_by_symbol
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_protocol_activity_market_join_csv(
    rows: tuple[ProtocolActivityMarketJoinRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "name",
                "protocol_action",
                "protocol_score",
                "commit_count_4_weeks",
                "telegram_users",
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
                    row.timestamp,
                    row.symbol,
                    row.name,
                    row.protocol_action,
                    f"{row.protocol_score:.8f}",
                    row.commit_count_4_weeks,
                    row.telegram_users,
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


def write_protocol_activity_market_join_md(
    rows: tuple[ProtocolActivityMarketJoinRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Activity Market Join\n\n")
        handle.write(
            "This joins protocol developer/community activity to current "
            "Hyperliquid perp state. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | action | commits 4w | telegram | funding | carry action | obs | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.action} | "
                f"{row.commit_count_4_weeks} | "
                f"{row.telegram_users} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.carry_reversion_action} | "
                f"{row.carry_reversion_observations} | "
                f"{row.score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is a context join. It does not prove developer activity causes "
            "returns; it only identifies tradable names where protocol activity "
            "and current perp state overlap.\n"
        )
    return output_path


def _build_row(
    *,
    protocol: dict[str, str],
    perp: dict[str, str] | None,
    carry: dict[str, str] | None,
) -> ProtocolActivityMarketJoinRow:
    protocol_score = float(protocol.get("score") or "0")
    commits = int(protocol.get("commit_count_4_weeks") or "0")
    telegram = int(protocol.get("telegram_users") or "0")
    annualized_funding = _float((perp or {}).get("annualized_funding"))
    mark_oracle_diff = _float((perp or {}).get("mark_oracle_diff"))
    impact_spread = _float((perp or {}).get("impact_spread"))
    carry_action = (carry or {}).get("action", "")
    carry_observations = int((carry or {}).get("observations") or "0")
    carry_score = _float((carry or {}).get("mean_score"))
    action = _action(
        protocol_action=protocol.get("action", ""),
        carry_observations=carry_observations,
        carry_score=carry_score,
        annualized_funding=annualized_funding,
    )
    return ProtocolActivityMarketJoinRow(
        timestamp=protocol["timestamp"],
        symbol=protocol["symbol"].upper(),
        name=protocol["name"],
        protocol_action=protocol["action"],
        protocol_score=protocol_score,
        commit_count_4_weeks=commits,
        telegram_users=telegram,
        annualized_funding=annualized_funding,
        mark_oracle_diff=mark_oracle_diff,
        impact_spread=impact_spread,
        carry_reversion_action=carry_action,
        carry_reversion_observations=carry_observations,
        carry_reversion_score=carry_score,
        action=action,
        score=_score(
            protocol_score=protocol_score,
            annualized_funding=annualized_funding,
            carry_observations=carry_observations,
            carry_score=carry_score,
            impact_spread=impact_spread,
        ),
        reason=_reason(action),
    )


def _action(
    *,
    protocol_action: str,
    carry_observations: int,
    carry_score: float,
    annualized_funding: float,
) -> str:
    if carry_observations >= 6 and carry_score >= 5.0:
        return "protocol_activity_carry_overlap"
    if abs(annualized_funding) >= 0.2 and protocol_action != "low_activity_context":
        return "protocol_activity_funding_overlap"
    if protocol_action in {"developer_attention_watch", "community_attention_watch"}:
        return "protocol_activity_watch"
    return "protocol_activity_context"


def _score(
    *,
    protocol_score: float,
    annualized_funding: float,
    carry_observations: int,
    carry_score: float,
    impact_spread: float,
) -> float:
    carry_bonus = carry_score + carry_observations if carry_observations >= 3 else 0.0
    funding_bonus = abs(annualized_funding) * 2.0
    friction_penalty = impact_spread * 100.0
    return protocol_score + carry_bonus + funding_bonus - friction_penalty


def _reason(action: str) -> str:
    if action == "protocol_activity_carry_overlap":
        return "protocol activity overlaps persistent carry/reversion perp state"
    if action == "protocol_activity_funding_overlap":
        return "protocol activity overlaps material current funding state"
    if action == "protocol_activity_watch":
        return "protocol has strong current non-price activity and is tradable"
    return "protocol activity is tradable context only"


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
        "--protocol-path",
        type=Path,
        default=ROOT / "current_coingecko_protocol_activity.csv",
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
        default=ROOT / "current_protocol_activity_market_join.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_protocol_activity_market_join.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_protocol_activity_market_join_rows(
        protocol_path=args.protocol_path,
        perp_snapshot_path=args.perp_snapshot_path,
        carry_reversion_path=args.carry_reversion_path,
    )
    write_protocol_activity_market_join_csv(rows, output_path=args.csv_output_path)
    write_protocol_activity_market_join_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.action, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
