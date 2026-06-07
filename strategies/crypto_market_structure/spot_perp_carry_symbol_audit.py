from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto.data import LOCAL_DATASET_DIR as SPOT_DATASET_DIR
from strategies.crypto_market_structure.data import (
    DEFAULT_SYMBOLS,
    LOCAL_DATASET_DIR as PERP_DATASET_DIR,
    MarketStructureDay,
    load_market_structure_days,
)
from strategies.crypto_market_structure.funding_carry import (
    FundingCarryDecisionInput,
    PositiveFundingCarryStrategy,
)
from strategies.crypto_market_structure.spot_perp_carry import SpotDay, load_spot_days


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SymbolContribution:
    candidate: str
    symbol: str
    held_steps: int
    mean_weight_when_held: float
    gross_contribution: float
    funding_contribution: float
    basis_contribution: float
    mean_funding_rate_sum_when_held: float
    mean_pair_return_when_held: float


@dataclass
class _SymbolAccumulator:
    held_steps: int = 0
    weight_sum: float = 0.0
    gross_contribution: float = 0.0
    funding_contribution: float = 0.0
    basis_contribution: float = 0.0
    funding_sum: float = 0.0
    pair_return_sum: float = 0.0


def build_symbol_audit(
    *,
    spot_rows_by_symbol: dict[str, tuple[SpotDay, ...]],
    perp_rows_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
    min_funding_rate_sum: float = 0.0002,
    top_n_values: tuple[int, ...] = (1, 2, 3),
    rebalance_days_values: tuple[int, ...] = (14,),
    capital_per_notional: float = 2.0,
) -> tuple[SymbolContribution, ...]:
    spot_by_symbol_and_timestamp = {
        symbol: {row.timestamp: row for row in rows}
        for symbol, rows in spot_rows_by_symbol.items()
    }
    perp_by_symbol_and_timestamp = {
        symbol: {row.timestamp: row for row in rows}
        for symbol, rows in perp_rows_by_symbol.items()
    }
    symbols = tuple(sorted(set(spot_by_symbol_and_timestamp) & set(perp_by_symbol_and_timestamp)))
    timestamps = sorted(
        set.union(
            *(
                set(spot_by_symbol_and_timestamp[symbol])
                | set(perp_by_symbol_and_timestamp[symbol])
                for symbol in symbols
            ),
        )
    )
    rows = [
        row
        for top_n in top_n_values
        for rebalance_days in rebalance_days_values
        for row in _audit_candidate(
            spot_by_symbol_and_timestamp,
            perp_by_symbol_and_timestamp,
            symbols=symbols,
            timestamps=timestamps,
            candidate=f"spot_perp_positive_funding_top_{top_n}_{rebalance_days}d",
            min_funding_rate_sum=min_funding_rate_sum,
            top_n=top_n,
            rebalance_days=rebalance_days,
            capital_per_notional=capital_per_notional,
        )
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.candidate, row.gross_contribution),
            reverse=True,
        )
    )


def write_symbol_audit_csv(
    rows: tuple[SymbolContribution, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "symbol",
                "held_steps",
                "mean_weight_when_held",
                "gross_contribution",
                "funding_contribution",
                "basis_contribution",
                "mean_funding_rate_sum_when_held",
                "mean_pair_return_when_held",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate,
                    row.symbol,
                    row.held_steps,
                    f"{row.mean_weight_when_held:.10f}",
                    f"{row.gross_contribution:.10f}",
                    f"{row.funding_contribution:.10f}",
                    f"{row.basis_contribution:.10f}",
                    f"{row.mean_funding_rate_sum_when_held:.10f}",
                    f"{row.mean_pair_return_when_held:.10f}",
                )
            )
    return output_path


def write_symbol_audit_md(
    rows: tuple[SymbolContribution, ...],
    *,
    output_path: Path,
    top: int = 8,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_candidate: dict[str, list[SymbolContribution]] = {}
    for row in rows:
        by_candidate.setdefault(row.candidate, []).append(row)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Spot/Perp Carry Symbol Audit\n\n")
        handle.write(
            "This decomposes 14-day spot/perp carry candidates by symbol. Gross "
            "contribution excludes transaction costs; funding and basis contributions "
            "show whether the candidate is earning funding or relying on spot/perp "
            "basis movement.\n\n"
        )
        for candidate in sorted(by_candidate):
            handle.write(f"## {candidate}\n\n")
            handle.write(
                "| symbol | held steps | mean weight | gross contribution | funding contribution | basis contribution | mean funding | mean pair return |\n"
            )
            handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for row in sorted(
                by_candidate[candidate],
                key=lambda item: item.gross_contribution,
                reverse=True,
            )[:top]:
                handle.write(
                    "| "
                    f"{row.symbol} | "
                    f"{row.held_steps} | "
                    f"{row.mean_weight_when_held:.6f} | "
                    f"{row.gross_contribution:.6f} | "
                    f"{row.funding_contribution:.6f} | "
                    f"{row.basis_contribution:.6f} | "
                    f"{row.mean_funding_rate_sum_when_held:.6f} | "
                    f"{row.mean_pair_return_when_held:.6f} |\n"
                )
            handle.write("\n")
        handle.write("## Interpretation\n\n")
        handle.write(
            "The useful follow-up is not just the best aggregate candidate; it is the "
            "symbols where funding contribution remains positive without depending on "
            "large favorable basis moves.\n"
        )
    return output_path


def _audit_candidate(
    spot_by_symbol_and_timestamp: dict[str, dict[str, SpotDay]],
    perp_by_symbol_and_timestamp: dict[str, dict[str, MarketStructureDay]],
    *,
    symbols: tuple[str, ...],
    timestamps: list[str],
    candidate: str,
    min_funding_rate_sum: float,
    top_n: int,
    rebalance_days: int,
    capital_per_notional: float,
) -> tuple[SymbolContribution, ...]:
    accumulators: dict[str, _SymbolAccumulator] = {}
    current_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    strategy = PositiveFundingCarryStrategy(
        min_funding_rate_sum=min_funding_rate_sum,
        top_n=top_n,
    )
    for index, timestamp in enumerate(timestamps[:-1]):
        next_timestamp = timestamps[index + 1]
        available_symbols = tuple(
            symbol
            for symbol in symbols
            if timestamp in spot_by_symbol_and_timestamp[symbol]
            and next_timestamp in spot_by_symbol_and_timestamp[symbol]
            and timestamp in perp_by_symbol_and_timestamp[symbol]
            and next_timestamp in perp_by_symbol_and_timestamp[symbol]
        )
        if index % rebalance_days == 0:
            target_weights = strategy.decide(
                FundingCarryDecisionInput(
                    rows_by_symbol={
                        symbol: perp_by_symbol_and_timestamp[symbol][timestamp]
                        for symbol in available_symbols
                    },
                    current_weights={
                        symbol: weight
                        for symbol, weight in current_weights.items()
                        if symbol in available_symbols
                    },
                )
            ).target_weights
        target_weights = {
            symbol: weight
            for symbol, weight in target_weights.items()
            if symbol in available_symbols
        }
        for symbol, weight in target_weights.items():
            current_spot = spot_by_symbol_and_timestamp[symbol][timestamp]
            next_spot = spot_by_symbol_and_timestamp[symbol][next_timestamp]
            current_perp = perp_by_symbol_and_timestamp[symbol][timestamp]
            next_perp = perp_by_symbol_and_timestamp[symbol][next_timestamp]
            funding = next_perp.funding_rate_sum / capital_per_notional
            basis = _spot_minus_perp_return(
                current_spot=current_spot,
                next_spot=next_spot,
                current_perp=current_perp,
                next_perp=next_perp,
            ) / capital_per_notional
            pair_return = funding + basis
            accumulator = accumulators.setdefault(symbol, _SymbolAccumulator())
            accumulator.held_steps += 1
            accumulator.weight_sum += weight
            accumulator.gross_contribution += weight * pair_return
            accumulator.funding_contribution += weight * funding
            accumulator.basis_contribution += weight * basis
            accumulator.funding_sum += next_perp.funding_rate_sum
            accumulator.pair_return_sum += pair_return
        current_weights = dict(target_weights)
    return tuple(
        SymbolContribution(
            candidate=candidate,
            symbol=symbol,
            held_steps=accumulator.held_steps,
            mean_weight_when_held=(
                accumulator.weight_sum / accumulator.held_steps
                if accumulator.held_steps
                else 0.0
            ),
            gross_contribution=accumulator.gross_contribution,
            funding_contribution=accumulator.funding_contribution,
            basis_contribution=accumulator.basis_contribution,
            mean_funding_rate_sum_when_held=(
                accumulator.funding_sum / accumulator.held_steps
                if accumulator.held_steps
                else 0.0
            ),
            mean_pair_return_when_held=(
                accumulator.pair_return_sum / accumulator.held_steps
                if accumulator.held_steps
                else 0.0
            ),
        )
        for symbol, accumulator in accumulators.items()
    )


def _spot_minus_perp_return(
    *,
    current_spot: SpotDay,
    next_spot: SpotDay,
    current_perp: MarketStructureDay,
    next_perp: MarketStructureDay,
) -> float:
    spot_return = (next_spot.close / current_spot.close) - 1.0
    perp_return = (next_perp.close / current_perp.close) - 1.0
    return spot_return - perp_return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dataset-dir", type=Path, default=SPOT_DATASET_DIR)
    parser.add_argument("--perp-dataset-dir", type=Path, default=PERP_DATASET_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--min-funding-rate-sum", type=float, default=0.0002)
    parser.add_argument("--top-n-values", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rebalance-days-values", nargs="+", type=int, default=[14])
    parser.add_argument("--capital-per-notional", type=float, default=2.0)
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_symbol_audit.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_symbol_audit.md",
    )
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    spot_rows_by_symbol = load_spot_days(
        dataset_dir=args.spot_dataset_dir,
        symbols=tuple(args.symbols),
    )
    perp_rows_by_symbol = load_market_structure_days(
        dataset_dir=args.perp_dataset_dir,
        symbols=tuple(args.symbols),
    )
    rows = build_symbol_audit(
        spot_rows_by_symbol=spot_rows_by_symbol,
        perp_rows_by_symbol=perp_rows_by_symbol,
        min_funding_rate_sum=args.min_funding_rate_sum,
        top_n_values=tuple(args.top_n_values),
        rebalance_days_values=tuple(args.rebalance_days_values),
        capital_per_notional=args.capital_per_notional,
    )
    write_symbol_audit_csv(rows, output_path=args.csv_output_path)
    write_symbol_audit_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.candidate,
            row.symbol,
            f"held={row.held_steps}",
            f"gross={row.gross_contribution:.6f}",
            f"funding={row.funding_contribution:.6f}",
            f"basis={row.basis_contribution:.6f}",
        )


if __name__ == "__main__":
    main()
