from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


COINGECKO_DERIVATIVES_URL = "https://api.coingecko.com/api/v3/derivatives"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DerivativesPositioningRow:
    timestamp: str
    market: str
    symbol: str
    index_id: str
    contract_type: str
    price: float
    index_price: float
    price_change_24h: float
    basis: float
    spread: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    oi_volume_ratio: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def fetch_derivatives(url: str = COINGECKO_DERIVATIVES_URL) -> tuple[dict[str, object], ...]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return tuple(response.json())


def build_derivatives_positioning_rows(
    raw_rows: tuple[dict[str, object], ...],
    *,
    timestamp: str | None = None,
    min_open_interest: float = 50_000_000.0,
    min_volume_24h: float = 10_000_000.0,
) -> tuple[DerivativesPositioningRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows = tuple(
        _build_row(raw=row, timestamp=observed_at)
        for row in raw_rows
        if _float(row.get("open_interest")) >= min_open_interest
        and _float(row.get("volume_24h")) >= min_volume_24h
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_derivatives_positioning_csv(
    rows: tuple[DerivativesPositioningRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "market",
                "symbol",
                "index_id",
                "contract_type",
                "price",
                "index_price",
                "price_change_24h",
                "basis",
                "spread",
                "funding_rate",
                "open_interest",
                "volume_24h",
                "oi_volume_ratio",
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
                    row.timestamp,
                    row.market,
                    row.symbol,
                    row.index_id,
                    row.contract_type,
                    f"{row.price:.8f}",
                    f"{row.index_price:.8f}",
                    f"{row.price_change_24h:.8f}",
                    f"{row.basis:.8f}",
                    f"{row.spread:.8f}",
                    f"{row.funding_rate:.8f}",
                    f"{row.open_interest:.8f}",
                    f"{row.volume_24h:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_derivatives_positioning_md(
    rows: tuple[DerivativesPositioningRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current CoinGecko Derivatives Positioning\n\n")
        handle.write(
            "This screen scores multi-venue derivatives open interest, volume, funding, basis, and spread. "
            "It is a positioning screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| market | symbol | status | OI USD | volume 24h | OI/vol | funding | basis | spread | chg 24h | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.market} | {row.symbol} | {row.status} | "
                f"{row.open_interest:.0f} | {row.volume_24h:.0f} | {row.oi_volume_ratio:.4f} | "
                f"{row.funding_rate:.6f} | {row.basis:.6f} | {row.spread:.4f} | "
                f"{row.price_change_24h:.4f} | {row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _build_row(*, raw: dict[str, object], timestamp: str) -> DerivativesPositioningRow:
    open_interest = _float(raw.get("open_interest"))
    volume_24h = _float(raw.get("volume_24h"))
    basis = _float(raw.get("basis"))
    funding_rate = _float(raw.get("funding_rate"))
    spread = _float(raw.get("spread"))
    price_change = _float(raw.get("price_percentage_change_24h"))
    oi_volume_ratio = open_interest / volume_24h if volume_24h > 0.0 else 0.0
    score = _score(
        open_interest=open_interest,
        volume_24h=volume_24h,
        funding_rate=funding_rate,
        basis=basis,
        spread=spread,
        price_change_24h=price_change,
        oi_volume_ratio=oi_volume_ratio,
    )
    status, side, reason = _status_side_reason(
        funding_rate=funding_rate,
        basis=basis,
        oi_volume_ratio=oi_volume_ratio,
        price_change_24h=price_change,
    )
    market = str(raw.get("market") or "")
    symbol = str(raw.get("symbol") or "")
    return DerivativesPositioningRow(
        timestamp=timestamp,
        market=market,
        symbol=symbol,
        index_id=str(raw.get("index_id") or ""),
        contract_type=str(raw.get("contract_type") or ""),
        price=_float(raw.get("price")),
        index_price=_float(raw.get("index")),
        price_change_24h=price_change,
        basis=basis,
        spread=spread,
        funding_rate=funding_rate,
        open_interest=open_interest,
        volume_24h=volume_24h,
        oi_volume_ratio=oi_volume_ratio,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=f"label {market} {symbol} forward returns, funding PnL, depth, fees, and margin constraints",
    )


def _status_side_reason(
    *,
    funding_rate: float,
    basis: float,
    oi_volume_ratio: float,
    price_change_24h: float,
) -> tuple[str, str, str]:
    if oi_volume_ratio >= 1.0 and abs(funding_rate) >= 0.02:
        side = "watch_short_crowded_long" if funding_rate > 0 else "watch_long_crowded_short"
        return "paper_oi_funding_crowding_watch", side, "high OI/volume with material funding"
    if abs(basis) >= 0.5 and abs(funding_rate) >= 0.01:
        side = "watch_basis_funding_reversion"
        return "paper_basis_funding_dislocation_watch", side, "basis and funding are both stretched"
    if abs(price_change_24h) >= 10.0 and oi_volume_ratio >= 0.5:
        return "paper_derivatives_momentum_risk_watch", "watch_continuation_or_reversal", "large move with meaningful OI"
    return "derivatives_positioning_context", "none", "positioning context is visible but not actionable yet"


def _score(
    *,
    open_interest: float,
    volume_24h: float,
    funding_rate: float,
    basis: float,
    spread: float,
    price_change_24h: float,
    oi_volume_ratio: float,
) -> float:
    oi_score = min(open_interest / 500_000_000.0, 20.0)
    volume_score = min(volume_24h / 500_000_000.0, 15.0)
    funding_score = min(abs(funding_rate) * 600.0, 25.0)
    basis_score = min(abs(basis) * 2.0, 15.0)
    crowding_score = min(oi_volume_ratio * 8.0, 20.0)
    move_score = min(abs(price_change_24h), 30.0) * 0.3
    spread_penalty = min(spread * 5.0, 10.0)
    return oi_score + volume_score + funding_score + basis_score + crowding_score + move_score - spread_penalty


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_coingecko_derivatives_positioning.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_coingecko_derivatives_positioning.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_derivatives_positioning_rows(fetch_derivatives())
    write_derivatives_positioning_csv(rows, output_path=args.output_path)
    write_derivatives_positioning_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.market, row.symbol, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
