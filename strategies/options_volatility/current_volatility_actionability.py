from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VolatilityActionabilityRow:
    currency: str
    expiry: str
    structure: str
    status: str
    side: str
    score: float
    days_to_expiry: float
    atm_iv: float
    realized_vol_24h: float
    iv_premium_24h: float
    quote_spread_pct: float
    max_loss_pct: float
    realized_move_pct: float
    premium_to_realized_move: float
    top_ask_premium_depth_usd: float
    source_status: str
    reason: str
    next_step: str


def build_volatility_actionability_rows(root: Path = ROOT) -> tuple[VolatilityActionabilityRow, ...]:
    output: list[VolatilityActionabilityRow] = []
    for row in _read_rows(root / "options_volatility" / "current_options_volatility_paper_tickets.csv"):
        if row.get("structure") not in {"long_atm_straddle", "short_put_spread", "calendar_spread"}:
            continue
        dte = _float(row.get("days_to_expiry"))
        quote_spread = _float(row.get("quote_spread_pct"))
        max_loss = _float(row.get("max_loss_pct"))
        premium_to_realized = _float(row.get("premium_to_realized_move"))
        top_depth = _float(row.get("top_ask_premium_depth_usd"))
        status, side, reason = _status_side_reason(
            source_status=row.get("status", ""),
            quote_status=row.get("quote_status", ""),
            days_to_expiry=dte,
            quote_spread_pct=quote_spread,
            max_loss_pct=max_loss,
            premium_to_realized_move=premium_to_realized,
            top_ask_premium_depth_usd=top_depth,
        )
        output.append(
            VolatilityActionabilityRow(
                currency=row.get("currency", ""),
                expiry=row.get("expiry", ""),
                structure=row.get("structure", ""),
                status=status,
                side=side,
                score=_score(
                    status=status,
                    iv_premium_24h=_float(row.get("iv_premium_24h")),
                    quote_spread_pct=quote_spread,
                    max_loss_pct=max_loss,
                    premium_to_realized_move=premium_to_realized,
                    top_ask_premium_depth_usd=top_depth,
                ),
                days_to_expiry=dte,
                atm_iv=_float(row.get("atm_iv")),
                realized_vol_24h=_float(row.get("realized_vol_24h")),
                iv_premium_24h=_float(row.get("iv_premium_24h")),
                quote_spread_pct=quote_spread,
                max_loss_pct=max_loss,
                realized_move_pct=_float(row.get("realized_move_pct")),
                premium_to_realized_move=premium_to_realized,
                top_ask_premium_depth_usd=top_depth,
                source_status=row.get("status", ""),
                reason=reason,
                next_step=_next_step(
                    currency=row.get("currency", ""),
                    expiry=row.get("expiry", ""),
                    structure=row.get("structure", ""),
                    status=status,
                ),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_volatility_actionability_csv(
    rows: tuple[VolatilityActionabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "currency",
                "expiry",
                "structure",
                "status",
                "side",
                "score",
                "days_to_expiry",
                "atm_iv",
                "realized_vol_24h",
                "iv_premium_24h",
                "quote_spread_pct",
                "max_loss_pct",
                "realized_move_pct",
                "premium_to_realized_move",
                "top_ask_premium_depth_usd",
                "source_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.currency,
                    row.expiry,
                    row.structure,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.days_to_expiry:.4f}",
                    f"{row.atm_iv:.4f}",
                    f"{row.realized_vol_24h:.4f}",
                    f"{row.iv_premium_24h:.4f}",
                    f"{row.quote_spread_pct:.6f}",
                    f"{row.max_loss_pct:.6f}",
                    f"{row.realized_move_pct:.6f}",
                    f"{row.premium_to_realized_move:.6f}",
                    f"{row.top_ask_premium_depth_usd:.2f}",
                    row.source_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_volatility_actionability_md(
    rows: tuple[VolatilityActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volatility Actionability\n\n")
        handle.write(
            "This separates cheap-vol surface anomalies from option structures that are ready "
            "for a paper hedge/fill check. It still uses public Deribit quotes and top-of-book "
            "depth, not a multi-level sweep or live hedge model.\n\n"
        )
        handle.write(
            "| currency | expiry | structure | status | side | score | dte | iv premium 24h | "
            "quote spread | max loss % | prem/rv move | top depth USD | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.currency} | {row.expiry} | {row.structure} | {row.status} | {row.side} | "
                f"{row.score:.4f} | {row.days_to_expiry:.2f} | {row.iv_premium_24h:.2f} | "
                f"{row.quote_spread_pct:.4f} | {row.max_loss_pct:.2f} | "
                f"{row.premium_to_realized_move:.2f} | {row.top_ask_premium_depth_usd:.0f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _status_side_reason(
    *,
    source_status: str,
    quote_status: str,
    days_to_expiry: float,
    quote_spread_pct: float,
    max_loss_pct: float,
    premium_to_realized_move: float,
    top_ask_premium_depth_usd: float,
) -> tuple[str, str, str]:
    if source_status == "paper_long_vol_quote_candidate":
        if days_to_expiry < 3.0:
            return "volatility_short_expiry_hedge_watch", "paper_hedge_check", "expiry is close, so gamma and hedge timing dominate"
        if quote_spread_pct <= 0.06 and max_loss_pct <= 12.0 and premium_to_realized_move <= 0.65 and top_ask_premium_depth_usd >= 25_000.0:
            return (
                "volatility_candidate_after_hedge_check",
                "paper_long_vol_after_hedge_check",
                "cheap IV has visible top depth and capped premium before hedge and sweep checks",
            )
        return (
            "volatility_quote_mechanics_watch",
            "paper_quote_check",
            "quote exists, but spread, premium size, depth, or breakeven move still needs mechanics review",
        )
    if source_status == "paper_long_vol_quote_blocked":
        return (
            "volatility_premium_or_depth_blocked",
            "no_trade_until_structure",
            "cheap IV is blocked by premium size, quote quality, depth, or structure constraints",
        )
    if source_status in {"paper_short_put_spread_candidate", "paper_calendar_spread_watch"}:
        return (
            "volatility_structure_mechanics_watch",
            "paper_structure_check",
            "structure needs explicit spread legs, max loss, margin, and hedge plan before promotion",
        )
    if quote_status in {"quote_missing", "quote_too_wide", "top_depth_too_thin"}:
        return "volatility_quote_blocked", "no_trade_until_quote", "quote quality is not good enough for a paper ticket"
    return "volatility_deprioritize", "none", "volatility row is not actionable after quote and structure checks"


def _score(
    *,
    status: str,
    iv_premium_24h: float,
    quote_spread_pct: float,
    max_loss_pct: float,
    premium_to_realized_move: float,
    top_ask_premium_depth_usd: float,
) -> float:
    cheapness = max(-iv_premium_24h, 0.0)
    if status == "volatility_candidate_after_hedge_check":
        return min(
            95.0,
            64.0
            + min(cheapness / 2.0, 18.0)
            + min(top_ask_premium_depth_usd / 25_000.0, 8.0)
            - max(quote_spread_pct - 0.03, 0.0) * 100.0
            - max(max_loss_pct - 8.0, 0.0) * 0.6
            - max(premium_to_realized_move - 0.5, 0.0) * 10.0,
        )
    if status == "volatility_quote_mechanics_watch":
        return min(72.0, 48.0 + min(cheapness / 3.0, 14.0) + min(top_ask_premium_depth_usd / 50_000.0, 5.0))
    if status == "volatility_short_expiry_hedge_watch":
        return min(62.0, 42.0 + min(cheapness / 3.0, 12.0))
    if status == "volatility_structure_mechanics_watch":
        return min(58.0, 42.0 + min(abs(iv_premium_24h) / 4.0, 10.0))
    if status in {"volatility_premium_or_depth_blocked", "volatility_quote_blocked"}:
        return min(38.0, 24.0 + min(cheapness / 5.0, 10.0))
    return 20.0


def _next_step(*, currency: str, expiry: str, structure: str, status: str) -> str:
    subject = f"{currency} {expiry} {structure}"
    if status == "volatility_candidate_after_hedge_check":
        return f"paper-check {subject} multi-level sweep, delta hedge schedule, max premium loss, margin, and exit bid"
    if status == "volatility_quote_mechanics_watch":
        return f"check {subject} spread, top depth, premium-at-risk, breakeven move, and hedge feasibility"
    if status == "volatility_short_expiry_hedge_watch":
        return f"check {subject} gamma timing, hedge frequency, and event/expiry handling before paper promotion"
    if status == "volatility_structure_mechanics_watch":
        return f"define {subject} legs, max loss, margin, hedge plan, and exit rule before paper promotion"
    if status in {"volatility_premium_or_depth_blocked", "volatility_quote_blocked"}:
        return f"do not promote {subject}; wait for better quote, depth, premium size, or structure"
    return f"deprioritize {subject} until quote and hedge checks produce a paper candidate"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=LANE_ROOT / "current_volatility_actionability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=LANE_ROOT / "current_volatility_actionability.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_volatility_actionability_rows()
    write_volatility_actionability_csv(rows, output_path=args.output_path)
    write_volatility_actionability_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.currency, row.expiry, row.structure, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
