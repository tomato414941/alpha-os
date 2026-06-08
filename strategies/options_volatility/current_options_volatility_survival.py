from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://doi.org/10.1007/s00500-025-10980-7"


@dataclass(frozen=True)
class OptionsVolatilitySurvivalRow:
    candidate_id: str
    status: str
    currency: str
    expiry: str
    structure: str
    survival_score: float
    actionability_score: float
    atm_iv: float
    realized_vol_24h: float
    iv_premium_24h: float
    quote_spread_pct: float
    max_loss_pct: float
    premium_to_realized_move: float
    top_ask_premium_depth_usd: float
    evidence: str
    missing_work: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_options_volatility_survival_rows(
    *,
    actionability_path: Path = ROOT / "current_volatility_actionability.csv",
    paper_tickets_path: Path = ROOT / "current_options_volatility_paper_tickets.csv",
) -> tuple[OptionsVolatilitySurvivalRow, ...]:
    tickets = {
        (row.get("currency", ""), row.get("expiry", ""), row.get("structure", "")): row
        for row in _read_rows(paper_tickets_path)
    }
    rows = tuple(
        _row_from_actionability(
            row=row,
            ticket=tickets.get((row.get("currency", ""), row.get("expiry", ""), row.get("structure", "")), {}),
        )
        for row in _read_rows(actionability_path)
    )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_options_volatility_survival_csv(rows: tuple[OptionsVolatilitySurvivalRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "status",
                "currency",
                "expiry",
                "structure",
                "survival_score",
                "actionability_score",
                "atm_iv",
                "realized_vol_24h",
                "iv_premium_24h",
                "quote_spread_pct",
                "max_loss_pct",
                "premium_to_realized_move",
                "top_ask_premium_depth_usd",
                "evidence",
                "missing_work",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.status,
                    row.currency,
                    row.expiry,
                    row.structure,
                    f"{row.survival_score:.8f}",
                    f"{row.actionability_score:.8f}",
                    f"{row.atm_iv:.8f}",
                    f"{row.realized_vol_24h:.8f}",
                    f"{row.iv_premium_24h:.8f}",
                    f"{row.quote_spread_pct:.8f}",
                    f"{row.max_loss_pct:.8f}",
                    f"{row.premium_to_realized_move:.8f}",
                    f"{row.top_ask_premium_depth_usd:.8f}",
                    row.evidence,
                    row.missing_work,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_options_volatility_survival_md(rows: tuple[OptionsVolatilitySurvivalRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Options Volatility Survival\n\n")
        handle.write(
            "This separates cheap-IV observations from option structures that can survive quote, premium, "
            "depth, and hedge-path checks. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| candidate | status | score | IV | RV24 | IV premium | spread | max loss | premium/RV move | depth USD | next probe |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:30]:
            handle.write(
                "| "
                f"{row.candidate_id} | "
                f"{row.status} | "
                f"{row.survival_score:.4f} | "
                f"{row.atm_iv:.4f} | "
                f"{row.realized_vol_24h:.4f} | "
                f"{row.iv_premium_24h:.4f} | "
                f"{row.quote_spread_pct:.6f} | "
                f"{row.max_loss_pct:.4f} | "
                f"{row.premium_to_realized_move:.4f} | "
                f"{row.top_ask_premium_depth_usd:.2f} | "
                f"{_escape(row.next_probe)} |\n"
            )
    return output_path


def _row_from_actionability(*, row: dict[str, str], ticket: dict[str, str]) -> OptionsVolatilitySurvivalRow:
    currency = row.get("currency", "")
    expiry = row.get("expiry", "")
    structure = row.get("structure", "")
    actionability_score = _float(row.get("score"))
    atm_iv = _float(row.get("atm_iv"))
    realized_vol_24h = _float(row.get("realized_vol_24h"))
    iv_premium_24h = _float(row.get("iv_premium_24h"))
    quote_spread_pct = _float(row.get("quote_spread_pct"))
    max_loss_pct = _float(row.get("max_loss_pct"))
    premium_to_realized_move = _float(row.get("premium_to_realized_move"))
    depth = _float(row.get("top_ask_premium_depth_usd"))
    status = _status(row=row, ticket=ticket)
    survival_score = _survival_score(
        status=status,
        actionability_score=actionability_score,
        iv_premium_24h=iv_premium_24h,
        quote_spread_pct=quote_spread_pct,
        max_loss_pct=max_loss_pct,
        premium_to_realized_move=premium_to_realized_move,
        depth=depth,
    )
    candidate_id = f"{currency}_{expiry}_{structure}".lower()
    evidence = (
        f"actionability={row.get('status', '')}; "
        f"source={row.get('source_status', '')}; "
        f"quote={ticket.get('quote_status', '')}; "
        f"ticket_status={ticket.get('status', '')}; "
        f"reason={row.get('reason', '')}"
    )
    return OptionsVolatilitySurvivalRow(
        candidate_id=candidate_id,
        status=status,
        currency=currency,
        expiry=expiry,
        structure=structure,
        survival_score=survival_score,
        actionability_score=actionability_score,
        atm_iv=atm_iv,
        realized_vol_24h=realized_vol_24h,
        iv_premium_24h=iv_premium_24h,
        quote_spread_pct=quote_spread_pct,
        max_loss_pct=max_loss_pct,
        premium_to_realized_move=premium_to_realized_move,
        top_ask_premium_depth_usd=depth,
        evidence=evidence,
        missing_work=_missing_work(status),
        next_probe=_next_probe(status=status, currency=currency, expiry=expiry, structure=structure),
    )


def _status(*, row: dict[str, str], ticket: dict[str, str]) -> str:
    actionability = row.get("status", "")
    quote_status = ticket.get("quote_status", "")
    max_loss = _float(row.get("max_loss_pct"))
    depth = _float(row.get("top_ask_premium_depth_usd"))
    if actionability == "volatility_candidate_needs_sweep_hedge" and quote_status == "quote_executable_watch":
        return "long_vol_hedge_path_required"
    if actionability == "volatility_short_expiry_hedge_watch":
        return "short_expiry_gamma_timing_required"
    if max_loss > 12.0:
        return "premium_size_blocks_survival"
    if depth < 1_000.0:
        return "top_depth_blocks_survival"
    if actionability == "volatility_quote_mechanics_watch":
        return "quote_mechanics_required"
    return "volatility_context_not_tradeable"


def _survival_score(
    *,
    status: str,
    actionability_score: float,
    iv_premium_24h: float,
    quote_spread_pct: float,
    max_loss_pct: float,
    premium_to_realized_move: float,
    depth: float,
) -> float:
    status_bonus = {
        "long_vol_hedge_path_required": 120.0,
        "short_expiry_gamma_timing_required": 70.0,
        "quote_mechanics_required": 55.0,
        "premium_size_blocks_survival": -40.0,
        "top_depth_blocks_survival": -70.0,
        "volatility_context_not_tradeable": -90.0,
    }.get(status, 0.0)
    cheap_iv_score = max(-iv_premium_24h, 0.0) * 0.9
    depth_score = min(depth / 25_000.0, 20.0)
    premium_efficiency = max(1.0 - premium_to_realized_move, 0.0) * 35.0
    penalties = quote_spread_pct * 160.0 + max(max_loss_pct - 8.0, 0.0) * 4.0
    return status_bonus + actionability_score + cheap_iv_score + depth_score + premium_efficiency - penalties


def _missing_work(status: str) -> str:
    if status == "long_vol_hedge_path_required":
        return "multi-level sweep, delta hedge schedule, exit bid, margin, and realized hedge PnL"
    if status == "short_expiry_gamma_timing_required":
        return "gamma timing, hedge frequency, event risk, and expiry handling"
    if status == "quote_mechanics_required":
        return "spread, top depth, premium-at-risk, breakeven move, and hedge feasibility"
    if status == "premium_size_blocks_survival":
        return "premium size is too large for the current structure"
    if status == "top_depth_blocks_survival":
        return "top ask depth is too thin for a clean paper ticket"
    return "option row is context only until quote and hedge mechanics improve"


def _next_probe(*, status: str, currency: str, expiry: str, structure: str) -> str:
    subject = f"{currency} {expiry} {structure}"
    if status == "long_vol_hedge_path_required":
        return f"paper-check {subject} with sweep depth, delta hedge marks, exit bid, max loss, and margin"
    if status == "short_expiry_gamma_timing_required":
        return f"check {subject} gamma timing and hedge frequency before any paper promotion"
    if status == "quote_mechanics_required":
        return f"review {subject} spread, premium-at-risk, breakeven move, and hedge feasibility"
    return f"do not promote {subject}; wait for better quote, depth, premium size, or structure"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_options_volatility_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_options_volatility_survival.md")
    args = parser.parse_args()

    rows = build_options_volatility_survival_rows()
    write_options_volatility_survival_csv(rows, output_path=args.output_path)
    write_options_volatility_survival_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.status, row.candidate_id, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
