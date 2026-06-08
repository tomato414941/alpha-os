from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TokenUnlockActionabilityRow:
    symbol: str
    name: str
    side: str
    status: str
    action: str
    score: float
    ticket_status: str
    ticket_score: float
    days_until: int
    unlock_value_usd: float
    percent_supply: float
    annualized_funding: float
    day_notional_volume: float
    open_interest_notional: float
    impact_spread: float
    market_action: str
    market_score: float
    reason: str
    next_step: str


def build_token_unlock_actionability_rows(root: Path = ROOT) -> tuple[TokenUnlockActionabilityRow, ...]:
    market_by_symbol = {
        row.get("symbol", ""): row
        for row in _read_rows(root / "current_token_unlock_market_join.csv")
        if row.get("symbol")
    }
    output: list[TokenUnlockActionabilityRow] = []
    for row in _read_rows(root / "current_token_unlock_paper_tickets.csv"):
        if row.get("status") not in {
            "paper_short_candidate",
            "crowded_short_risk",
            "wide_impact_watch",
            "too_illiquid",
        }:
            continue
        market = market_by_symbol.get(row.get("symbol", ""), {})
        status, action, reason = _status_action_reason(row)
        symbol = row.get("symbol", "")
        output.append(
            TokenUnlockActionabilityRow(
                symbol=symbol,
                name=row.get("name", ""),
                side=row.get("side", ""),
                status=status,
                action=action,
                score=_score(row=row, status=status, market_score=_float(market.get("score"))),
                ticket_status=row.get("status", ""),
                ticket_score=_float(row.get("score")),
                days_until=_int(row.get("days_until")),
                unlock_value_usd=_float(row.get("unlock_value_usd")),
                percent_supply=_float(row.get("percent_supply")),
                annualized_funding=_float(row.get("annualized_funding")),
                day_notional_volume=_float(row.get("day_notional_volume")),
                open_interest_notional=_float(row.get("open_interest_notional")),
                impact_spread=_float(row.get("impact_spread")),
                market_action=market.get("action", ""),
                market_score=_float(market.get("score")),
                reason=reason,
                next_step=_next_step(symbol=symbol, status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_token_unlock_actionability_csv(
    rows: tuple[TokenUnlockActionabilityRow, ...],
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
                "side",
                "status",
                "action",
                "score",
                "ticket_status",
                "ticket_score",
                "days_until",
                "unlock_value_usd",
                "percent_supply",
                "annualized_funding",
                "day_notional_volume",
                "open_interest_notional",
                "impact_spread",
                "market_action",
                "market_score",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.name,
                    row.side,
                    row.status,
                    row.action,
                    f"{row.score:.8f}",
                    row.ticket_status,
                    f"{row.ticket_score:.8f}",
                    row.days_until,
                    f"{row.unlock_value_usd:.2f}",
                    f"{row.percent_supply:.4f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.open_interest_notional:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.market_action,
                    f"{row.market_score:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_token_unlock_actionability_md(
    rows: tuple[TokenUnlockActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Actionability\n\n")
        handle.write(
            "This separates scheduled supply events from tradable candidates. "
            "Without event-window labels, an unlock is context, not an alpha candidate.\n\n"
        )
        handle.write(
            "| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.action} | {row.score:.4f} | "
                f"{row.ticket_status} {row.ticket_score:.2f} | {row.days_until} | "
                f"{row.unlock_value_usd:.0f} | {row.percent_supply:.4f} | "
                f"{row.annualized_funding:.8f} | {row.day_notional_volume:.0f} | "
                f"{row.impact_spread:.8f} | {_escape(row.reason)} |\n"
            )
    return output_path


def _status_action_reason(row: dict[str, str]) -> tuple[str, str, str]:
    status = row.get("status", "")
    if status == "too_illiquid":
        return "unlock_event_not_tradeable", "do_not_probe", "perp venue volume is too low for a paper probe"
    if status == "wide_impact_watch":
        return "unlock_event_execution_blocked", "wait_for_tighter_depth", "visible impact spread is too wide"
    if status == "crowded_short_risk" or _float(row.get("annualized_funding")) < 0.0:
        return (
            "unlock_event_crowded_squeeze_watch",
            "label_before_short",
            "supply shock overlaps crowded short or negative funding risk",
        )
    return (
        "unlock_event_label_pending",
        "create_event_window_label",
        "unlock short thesis has no event-window forward label yet",
    )


def _score(*, row: dict[str, str], status: str, market_score: float) -> float:
    status_base = {
        "unlock_event_label_pending": 42.0,
        "unlock_event_crowded_squeeze_watch": 36.0,
        "unlock_event_execution_blocked": 26.0,
        "unlock_event_not_tradeable": 10.0,
    }.get(status, 0.0)
    urgency = max(0.0, 30.0 - _float(row.get("days_until"))) / 30.0
    supply = min(_float(row.get("percent_supply")), 20.0) / 4.0
    value = min(_float(row.get("unlock_value_usd")) / 100_000_000.0, 3.0)
    volume = min(_float(row.get("day_notional_volume")) / 20_000_000.0, 3.0)
    impact_penalty = min(_float(row.get("impact_spread")) * 1000.0, 8.0)
    return status_base + urgency * 4.0 + supply + value + volume + min(market_score / 30.0, 2.0) - impact_penalty


def _next_step(*, symbol: str, status: str) -> str:
    if status == "unlock_event_label_pending":
        return f"label {symbol} unlock event window before treating the supply event as a tradable alpha"
    if status == "unlock_event_crowded_squeeze_watch":
        return f"label {symbol} both short-pressure and squeeze outcomes before any short probe"
    if status == "unlock_event_execution_blocked":
        return f"wait for tighter {symbol} depth or use a different route before any paper probe"
    return f"drop {symbol} unlock from paper priority until venue liquidity improves"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _int(value: str | None) -> int:
    return int(float(value)) if value else 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_token_unlock_actionability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_token_unlock_actionability.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_token_unlock_actionability_rows()
    write_token_unlock_actionability_csv(rows, output_path=args.output_path)
    write_token_unlock_actionability_md(rows, output_path=args.markdown_output_path, top=args.top)

    if rows:
        best = rows[0]
        print("best_token_unlock_actionability", best.symbol, best.status, f"score={best.score:.4f}")


if __name__ == "__main__":
    main()
