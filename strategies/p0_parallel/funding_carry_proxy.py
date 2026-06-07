from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarryCandidate:
    date: str
    symbol: str
    sum_funding_rate: float
    mean_premium_close: float
    next_mean_premium_close: float
    premium_change: float
    perp_direction: int
    funding_pnl: float
    basis_pnl: float
    gross_proxy_pnl: float
    net_proxy_pnl: float


def build_funding_carry_candidates(
    *,
    history_path: Path,
    round_trip_cost_bps: float,
) -> tuple[CarryCandidate, ...]:
    rows = _read_history_rows(history_path)
    by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], []).append(row)

    candidates: list[CarryCandidate] = []
    cost = round_trip_cost_bps / 10_000.0
    for symbol_rows in by_symbol.values():
        ordered_rows = sorted(symbol_rows, key=lambda row: row["date"])
        for index, row in enumerate(ordered_rows[:-1]):
            next_row = ordered_rows[index + 1]
            sum_funding_rate = float(row["sum_funding_rate"])
            if sum_funding_rate == 0.0:
                continue
            mean_premium_close = float(row["mean_premium_close"])
            next_mean_premium_close = float(next_row["mean_premium_close"])
            premium_change = next_mean_premium_close - mean_premium_close
            perp_direction = -1 if sum_funding_rate > 0.0 else 1
            funding_pnl = abs(sum_funding_rate)
            basis_pnl = perp_direction * premium_change
            gross_proxy_pnl = funding_pnl + basis_pnl
            candidates.append(
                CarryCandidate(
                    date=row["date"],
                    symbol=row["symbol"],
                    sum_funding_rate=sum_funding_rate,
                    mean_premium_close=mean_premium_close,
                    next_mean_premium_close=next_mean_premium_close,
                    premium_change=premium_change,
                    perp_direction=perp_direction,
                    funding_pnl=funding_pnl,
                    basis_pnl=basis_pnl,
                    gross_proxy_pnl=gross_proxy_pnl,
                    net_proxy_pnl=gross_proxy_pnl - cost,
                )
            )
    return tuple(
        sorted(candidates, key=lambda candidate: candidate.net_proxy_pnl, reverse=True)
    )


def write_funding_carry_candidates(
    candidates: tuple[CarryCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "date",
                "symbol",
                "sum_funding_rate",
                "mean_premium_close",
                "next_mean_premium_close",
                "premium_change",
                "perp_direction",
                "funding_pnl",
                "basis_pnl",
                "gross_proxy_pnl",
                "net_proxy_pnl",
            )
        )
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.date,
                    candidate.symbol,
                    f"{candidate.sum_funding_rate:.12f}",
                    f"{candidate.mean_premium_close:.12f}",
                    f"{candidate.next_mean_premium_close:.12f}",
                    f"{candidate.premium_change:.12f}",
                    candidate.perp_direction,
                    f"{candidate.funding_pnl:.12f}",
                    f"{candidate.basis_pnl:.12f}",
                    f"{candidate.gross_proxy_pnl:.12f}",
                    f"{candidate.net_proxy_pnl:.12f}",
                )
            )
    return output_path


def summarize_candidates(candidates: tuple[CarryCandidate, ...]) -> dict[str, float]:
    if not candidates:
        return {
            "observations": 0.0,
            "mean_net_proxy_pnl": 0.0,
            "hit_rate": 0.0,
            "best_net_proxy_pnl": 0.0,
            "worst_net_proxy_pnl": 0.0,
        }
    net_values = tuple(candidate.net_proxy_pnl for candidate in candidates)
    return {
        "observations": float(len(candidates)),
        "mean_net_proxy_pnl": sum(net_values) / len(net_values),
        "hit_rate": sum(1.0 for value in net_values if value > 0.0) / len(net_values),
        "best_net_proxy_pnl": max(net_values),
        "worst_net_proxy_pnl": min(net_values),
    }


def _read_history_rows(history_path: Path) -> tuple[dict[str, str], ...]:
    with history_path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path(__file__).resolve().parent / "binance_derivatives_history.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "binance_funding_carry_candidates.csv",
    )
    parser.add_argument("--round-trip-cost-bps", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    candidates = build_funding_carry_candidates(
        history_path=args.history_path,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    write_funding_carry_candidates(candidates, output_path=args.output_path)
    summary = summarize_candidates(candidates)
    print(
        "summary",
        f"observations={summary['observations']:.0f}",
        f"mean_net={summary['mean_net_proxy_pnl']:.8f}",
        f"hit_rate={summary['hit_rate']:.4f}",
        f"best={summary['best_net_proxy_pnl']:.8f}",
        f"worst={summary['worst_net_proxy_pnl']:.8f}",
    )
    for candidate in candidates[: args.top]:
        print(
            candidate.date,
            candidate.symbol,
            f"direction={candidate.perp_direction}",
            f"funding={candidate.funding_pnl:.8f}",
            f"basis={candidate.basis_pnl:.8f}",
            f"net={candidate.net_proxy_pnl:.8f}",
        )


if __name__ == "__main__":
    main()
