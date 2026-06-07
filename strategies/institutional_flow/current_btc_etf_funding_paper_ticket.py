from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VenueCandidate:
    venue: str
    instrument: str
    side: str
    annualized_funding: float
    carry_side: str
    volume_usd_24h: float
    open_interest_usd: float
    spread_or_impact: float
    basis_or_premium: float
    score: float
    status: str
    reason: str


def build_paper_ticket(
    *,
    current_candidate_path: Path,
    hyperliquid_path: Path,
    okx_path: Path,
) -> tuple[VenueCandidate, ...]:
    current_rows = _read_rows(current_candidate_path)
    if not current_rows or current_rows[0].get("status") != "active_paper_watch":
        return ()
    rows: list[VenueCandidate] = []
    hl_btc = _row_by_value(hyperliquid_path, field="asset", value="BTC")
    if hl_btc:
        rows.append(_hyperliquid_candidate(hl_btc))
    okx_btc = _row_by_value(okx_path, field="asset", value="BTC")
    if okx_btc:
        rows.append(_okx_candidate(okx_btc))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_ticket_csv(
    rows: tuple[VenueCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "venue",
                "instrument",
                "side",
                "annualized_funding",
                "carry_side",
                "volume_usd_24h",
                "open_interest_usd",
                "spread_or_impact",
                "basis_or_premium",
                "score",
                "status",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.venue,
                    row.instrument,
                    row.side,
                    f"{row.annualized_funding:.8f}",
                    row.carry_side,
                    f"{row.volume_usd_24h:.8f}",
                    f"{row.open_interest_usd:.8f}",
                    f"{row.spread_or_impact:.12f}",
                    f"{row.basis_or_premium:.12f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.reason,
                )
            )
    return output_path


def write_ticket_md(
    rows: tuple[VenueCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current BTC ETF Funding Paper Ticket\n\n")
        handle.write(
            "This compares current venues for the active BTC ETF-flow/funding paper watch. "
            "It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.venue} | {row.instrument} | {row.side} | "
                f"{row.annualized_funding:.8f} | {row.carry_side} | "
                f"{row.volume_usd_24h:.0f} | {row.open_interest_usd:.0f} | "
                f"{row.spread_or_impact:.8f} | {row.basis_or_premium:.8f} | "
                f"{row.score:.6f} | {row.status} | {row.reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. "
            "The current score only ranks visible public venue context.\n"
        )
    return output_path


def _hyperliquid_candidate(row: dict[str, str]) -> VenueCandidate:
    annualized_funding = _float(row["annualized_funding"])
    volume = _float(row["day_notional_volume"])
    open_interest = _float(row["open_interest_notional"])
    impact_spread = _float(row["impact_spread"])
    basis = _float(row["mark_oracle_diff"])
    carry_side = row["carry_side"]
    status, reason = _status_reason(
        annualized_funding=annualized_funding,
        carry_side=carry_side,
        volume=volume,
        spread_or_impact=impact_spread,
        basis_or_premium=basis,
    )
    return VenueCandidate(
        venue="Hyperliquid",
        instrument="BTC-USD perpetual",
        side="short",
        annualized_funding=annualized_funding,
        carry_side=carry_side,
        volume_usd_24h=volume,
        open_interest_usd=open_interest,
        spread_or_impact=impact_spread,
        basis_or_premium=basis,
        score=_score(
            annualized_funding=annualized_funding,
            volume=volume,
            spread_or_impact=impact_spread,
            basis_or_premium=basis,
        ),
        status=status,
        reason=reason,
    )


def _okx_candidate(row: dict[str, str]) -> VenueCandidate:
    annualized_funding = _float(row["annualized_funding"])
    volume = _float(row["day_volume_usd"])
    open_interest = _float(row["open_interest_usd"])
    spread_bps = _float(row["spread_bps"])
    spread = spread_bps / 10_000.0
    premium = _float(row["premium"])
    carry_side = row["carry_side"]
    status, reason = _status_reason(
        annualized_funding=annualized_funding,
        carry_side=carry_side,
        volume=volume,
        spread_or_impact=spread,
        basis_or_premium=premium,
    )
    return VenueCandidate(
        venue="OKX",
        instrument=row["inst_id"],
        side="short",
        annualized_funding=annualized_funding,
        carry_side=carry_side,
        volume_usd_24h=volume,
        open_interest_usd=open_interest,
        spread_or_impact=spread,
        basis_or_premium=premium,
        score=_score(
            annualized_funding=annualized_funding,
            volume=volume,
            spread_or_impact=spread,
            basis_or_premium=premium,
        ),
        status=status,
        reason=reason,
    )


def _score(
    *,
    annualized_funding: float,
    volume: float,
    spread_or_impact: float,
    basis_or_premium: float,
) -> float:
    liquidity_score = min(volume / 1_000_000_000.0, 5.0)
    return annualized_funding + liquidity_score - (spread_or_impact * 100.0) - abs(basis_or_premium)


def _status_reason(
    *,
    annualized_funding: float,
    carry_side: str,
    volume: float,
    spread_or_impact: float,
    basis_or_premium: float,
) -> tuple[str, str]:
    if carry_side != "short_perp_receives_funding":
        return "reject", "short side does not receive funding"
    if annualized_funding <= 0.0:
        return "reject", "funding is not positive for short carry"
    if volume < 500_000_000.0:
        return "reject", "24h volume is too low for BTC paper venue priority"
    if spread_or_impact > 0.002:
        return "watch_only", "visible spread or impact is high"
    if abs(basis_or_premium) > 0.002:
        return "watch_only", "basis or premium is large enough to require mark/index checks"
    return "paper_venue_candidate", "short carry, liquidity, and visible friction are acceptable for paper watch"


def _row_by_value(path: Path, *, field: str, value: str) -> dict[str, str] | None:
    return next((row for row in _read_rows(path) if row.get(field) == value), None)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-candidate-path",
        type=Path,
        default=ROOT / "current_btc_etf_funding_candidate.csv",
    )
    parser.add_argument(
        "--hyperliquid-path",
        type=Path,
        default=ROOT.parents[0] / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--okx-path",
        type=Path,
        default=ROOT.parents[0] / "perp_market_map" / "current_okx_perp_pressure.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_btc_etf_funding_paper_ticket.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_btc_etf_funding_paper_ticket.md",
    )
    args = parser.parse_args()

    rows = build_paper_ticket(
        current_candidate_path=args.current_candidate_path,
        hyperliquid_path=args.hyperliquid_path,
        okx_path=args.okx_path,
    )
    write_ticket_csv(rows, output_path=args.output_path)
    write_ticket_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(
            row.venue,
            row.status,
            f"funding={row.annualized_funding:.6f}",
            f"score={row.score:.6f}",
            row.reason,
        )


if __name__ == "__main__":
    main()
