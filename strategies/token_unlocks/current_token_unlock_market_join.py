from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class UnlockMarketContext:
    symbol: str
    name: str
    unlock_action: str
    days_until: str
    unlock_value_usd: float
    percent_supply: float
    annualized_funding: float | None
    day_notional_volume: float | None
    open_interest_notional: float | None
    action: str
    score: float


def join_unlocks_to_market(
    *,
    unlock_path: Path,
    hyperliquid_path: Path,
) -> tuple[UnlockMarketContext, ...]:
    unlocks = _read_rows(unlock_path)
    market_by_asset = {row["asset"]: row for row in _read_rows(hyperliquid_path)}
    contexts: list[UnlockMarketContext] = []
    for row in unlocks:
        market = market_by_asset.get(row["symbol"])
        if market is None:
            continue
        funding = _float_or_none(market.get("annualized_funding", ""))
        volume = _float_or_none(market.get("day_notional_volume", ""))
        open_interest = _float_or_none(market.get("open_interest_notional", ""))
        score = _float(row["score"])
        if funding is not None:
            score += min(abs(funding), 1.0)
        if open_interest is not None and volume is not None and volume > 0.0:
            score += min(open_interest / volume, 3.0)
        contexts.append(
            UnlockMarketContext(
                symbol=row["symbol"],
                name=row["name"],
                unlock_action=row["action"],
                days_until=row["days_until"],
                unlock_value_usd=_float(row["unlock_value_usd"]),
                percent_supply=_float(row["percent_supply"]),
                annualized_funding=funding,
                day_notional_volume=volume,
                open_interest_notional=open_interest,
                action=_context_action(row["action"], funding=funding),
                score=score,
            )
        )
    return tuple(sorted(contexts, key=lambda context: context.score, reverse=True))


def write_contexts(
    contexts: tuple[UnlockMarketContext, ...],
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
                "unlock_action",
                "days_until",
                "unlock_value_usd",
                "percent_supply",
                "annualized_funding",
                "day_notional_volume",
                "open_interest_notional",
                "action",
                "score",
            )
        )
        for context in contexts:
            writer.writerow(
                (
                    context.symbol,
                    context.name,
                    context.unlock_action,
                    context.days_until,
                    f"{context.unlock_value_usd:.2f}",
                    f"{context.percent_supply:.4f}",
                    _format_float(context.annualized_funding),
                    _format_float(context.day_notional_volume),
                    _format_float(context.open_interest_notional),
                    context.action,
                    f"{context.score:.8f}",
                )
            )
    return output_path


def write_markdown(
    contexts: tuple[UnlockMarketContext, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Market Join\n\n")
        handle.write(
            "This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
        for context in contexts[:30]:
            handle.write(
                f"| {context.symbol} | {context.name} | {context.unlock_action} | "
                f"{context.days_until} | {context.unlock_value_usd:.2f} | "
                f"{context.percent_supply:.4f} | {_format_float(context.annualized_funding)} | "
                f"{_format_float(context.open_interest_notional)} | {context.action} | "
                f"{context.score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.\n"
        )
    return output_path


def _context_action(unlock_action: str, *, funding: float | None) -> str:
    if unlock_action == "unlock_supply_shock_watch" and funding is not None:
        if funding > 0.0:
            return "unlock_short_pressure_funding_overlap"
        if funding < 0.0:
            return "unlock_supply_shock_crowded_short_overlap"
    return unlock_action


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value else 0.0


def _float_or_none(value: str) -> float | None:
    return float(value) if value else None


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unlock-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_token_unlock_snapshot.csv",
    )
    parser.add_argument(
        "--hyperliquid-path",
        type=Path,
        default=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_token_unlock_market_join.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_token_unlock_market_join.md",
    )
    args = parser.parse_args()

    contexts = join_unlocks_to_market(
        unlock_path=args.unlock_path,
        hyperliquid_path=args.hyperliquid_path,
    )
    write_contexts(contexts, output_path=args.output_path)
    write_markdown(contexts, output_path=args.markdown_output_path)
    for context in contexts[:10]:
        print(
            context.symbol,
            context.action,
            f"value={context.unlock_value_usd:.2f}",
            f"supply={context.percent_supply:.2f}",
            f"funding={_format_float(context.annualized_funding)}",
            f"score={context.score:.4f}",
        )


if __name__ == "__main__":
    main()
